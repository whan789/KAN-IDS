import os
import glob
import re
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import logging
from model.eTransformer import eTransformer
# from model.eRNN import eRNN
from memory_profiler import profile
from collections import Counter

# --- 외부 파일에서 필요한 클래스 및 함수 임포트 ---
# from model.eTransformer import eTransformer
from model.eRNN import eRNN
from fvcore.nn import FlopCountAnalysis
# --- 새로 만든 메트릭 파일에서 함수 임포트 ---
from forwards_utils import calculate_earliness, calculate_erde, calculate_f_latency, calculate_tap


CLASS_LABELS = {
    0: 'benign',
    1: 'Spoofing',
    2: 'Brute Force',
    3: 'Web based',
    4: 'Recon',
}

# [수정됨] get_args 함수
def get_args():
    parser = argparse.ArgumentParser(description="Student Model Evaluation with Various Optimization Techniques")
    
    # --- 경로 설정 ---
    parser.add_argument('--model_dir', type=str, default='/home/whan_i/IoT/ckpt_final/checkpoints_teacher_ori', help='Directory containing the trained teacher model checkpoints')

    parser.add_argument('--npz_path', type=str, default='/data/whan_i/IoT/data/DL_data_final/final_test_data_merged.npz', help='Path to the test .npz file')
    parser.add_argument('--log_path', type=str, default='./ckpt_final/test_student_eTransformer.log', help='Path to save the evaluation log file')
    
    parser.add_argument('--start_epoch', type=int, default=20, help='Starting epoch number to evaluate')
    parser.add_argument('--end_epoch', type=int, default=50, help='Ending epoch number to evaluate')
    parser.add_argument('--epoch_step', type=int, default=5, help='Step between epochs to evaluate')
    
    # --- Student 모델 파라미터 (모델 로드를 위해 필요) ---
    parser.add_argument('--student_architecture', type=str, default='transformer', choices=['transformer', 'mlp', 'kan'], help="Architecture of the student model.")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for evaluation')
    parser.add_argument('--input_dim', type=int, default=514, help='Input feature dimension')
    parser.add_argument('--seq_length', type=int, default=67, help='Sequence length')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of attack classes')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    # Transformer 전용
    parser.add_argument('--student_d_model', type=int, default=128, help='Model dimension d_model (Transformer only)')
    parser.add_argument('--student_nhead', type=int, default=8, help='Attention heads (Transformer only)')
    parser.add_argument('--student_layers', type=int, default=2, help='Number of layers (Transformer only)')
    parser.add_argument('--student_ffn_dim', type=int, default=64, help='FFN hidden dimension (Transformer only)')
    parser.add_argument('--use_linear_attention', default = True, help='Use Linear Attention (Transformer only)')
    parser.add_argument('--use_kan_ffn', default = True, help='Use KAN for FFN in Transformer (Transformer only)')
    # MLP & KAN 전용
    parser.add_argument('--mlp_kan_hidden_dim', type=int, default=256, help='The hidden dimension "h" for MLP and KAN.')
    # KAN 분류기 전용
    parser.add_argument('--classifier_hidden_dim', type=int, default=64, help='Hidden dimension for the final KAN classifier head')
    
    # --- Pruning 설정 ---
    parser.add_argument('--pruning_amount', type=float, default=0.3, help='Amount of global unstructured pruning to apply')

    # --- 조기 탐지 평가 파라미터 ---
    parser.add_argument('--early_detection_threshold', type=float, default=0.95, help='Confidence threshold for early detection.')
    parser.add_argument('--param_o', type=float, default=5, help='Parameter "o" for ERDE calculation (FP cost).')
    parser.add_argument('--param_lambda', type=float, default=0.1, help='Parameter "lambda" for TaP calculation.')
    
    return parser.parse_args()


def setup_logging(log_path):
    # (이 함수는 변경 없음)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path) # 'w' 모드로 시작 시 로그 파일 초기화
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# ... PacketDataset, evaluate_early_detection, profile_and_evaluate, parse_memory_log 함수는 변경 없음 ...
class PacketDataset(Dataset):
    def __init__(self, npz_path):
        with np.load(npz_path) as data: self.data, self.labels, self.time = data['X'], data['y'], data['T_relative']
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.data[idx]).float(), 
                torch.from_numpy(self.time[idx]).float(), 
                torch.from_numpy(np.array(self.labels[idx])).long())

def evaluate_early_detection(model, test_loader, device, args):
    model.eval()
    final_preds, true_labels = [], []
    all_final_probs_list = []
    detection_times_t, true_lengths_T = [], []
    total_inference_time = 0.0
    all_logits_seqs = []

    with torch.no_grad():
        for data, time_info, labels in tqdm(test_loader, desc="Early Detection Evaluation"):
            data, time_info = data.to(device), time_info.to(device)
            B = data.shape[0]
            for i in range(B):
                full_flow_data, full_flow_time = data[i], time_info[i]
                T = (full_flow_time >= 0).sum().item()
                if T == 0:
                    continue
                
                current_label = labels[i].item()

                found_positive = False
                last_probs = None
                current_flow_logits = []

                for k in range(1, T + 1):
                    sub_flow_data, sub_flow_time = full_flow_data[:k, :].unsqueeze(0), full_flow_time[:k].unsqueeze(0)
                    padding_mask = (sub_flow_time == -1)
                    
                    start_time = time.time()
                    logits, _, _ = model(sub_flow_data)
                    total_inference_time += time.time() - start_time
                    
                    current_flow_logits.append(logits.squeeze(0))
                    
                    probs = F.softmax(logits, dim=1).squeeze(0)
                    last_probs = probs
                    
                    current_max_prob = probs.max().item()
                    
                    if current_label != 0 and current_max_prob >= args.early_detection_threshold:
                        final_pred, t = probs.argmax().item(), k
                        found_positive = True
                        break
                
                if not found_positive:
                    final_pred, t = last_probs.argmax().item(), T

                final_preds.append(final_pred)
                true_labels.append(current_label)
                detection_times_t.append(t)
                true_lengths_T.append(T)
                
                all_logits_seqs.append(torch.stack(current_flow_logits))
                
                final_probs_dist = probs.cpu().numpy() if found_positive else last_probs.cpu().numpy()
                all_final_probs_list.append(final_probs_dist)

    # --- 전체 성능 평가 (모든 레이블 포함) ---
    true_labels_np = np.array(true_labels)
    final_preds_np = np.array(final_preds)
    all_final_probs_np = np.array(all_final_probs_list)

    report_dict = classification_report(
        true_labels_np, final_preds_np, 
        target_names=list(CLASS_LABELS.values()), 
        output_dict=True, 
        zero_division=0
    )
    
    overall_accuracy = report_dict.pop('accuracy')
    performance_df = pd.DataFrame(report_dict).transpose()
    
    # AUROC 계산
    auroc_values = {}
    if len(np.unique(true_labels_np)) > 1 and all_final_probs_np.ndim == 2:
        try:
            per_class_auroc = roc_auc_score(true_labels_np, all_final_probs_np, multi_class='ovr', average=None)
            weighted_auroc = roc_auc_score(true_labels_np, all_final_probs_np, multi_class='ovr', average='weighted')
            class_names = list(CLASS_LABELS.values())
            
            for i, class_name in enumerate(class_names):
                if class_name in performance_df.index:
                    performance_df.loc[class_name, 'auroc'] = per_class_auroc[i]

            performance_df.loc['weighted avg', 'auroc'] = weighted_auroc
            performance_df.loc['macro avg', 'auroc'] = np.mean(per_class_auroc)

        except ValueError as e:
            logging.warning(f"Could not calculate AUROC: {e}")

    performance_df.rename(columns={'recall': 'recall (class_accuracy)'}, inplace=True)
    performance_df['support'] = performance_df['support'].astype(int)
    
    # --- 조기 탐지 관련 지표 계산 (정상 레이블 제외) ---
    # ##### <<< 수정된 부분 3 시작 #####
    
    # 실제 레이블이 0이 아닌 샘플들만 필터링하기 위한 마스크 생성
    non_normal_mask = (true_labels_np != 0)
    
    # 조기 탐지 지표를 계산할 비정상 샘플이 있는지 확인
    avg_packets_for_detection = 0.0 # 기본값 초기화
    
    if np.any(non_normal_mask):
        # 마스크를 사용하여 비정상 샘플에 대한 데이터만 추출
        filtered_detection_times = np.array(detection_times_t)[non_normal_mask]
        filtered_true_lengths = np.array(true_lengths_T)[non_normal_mask]
        filtered_true_labels = true_labels_np[non_normal_mask]
        filtered_final_preds = final_preds_np[non_normal_mask]

        # [추가] 조기 탐지에 사용된 평균 패킷 수 계산
        avg_packets_for_detection = np.mean(filtered_detection_times)

        # 기존 조기 탐지 지표 계산
        earliness = calculate_earliness(filtered_detection_times, filtered_true_lengths, filtered_true_labels, filtered_final_preds)
        erde = calculate_erde(filtered_true_labels, filtered_final_preds, filtered_detection_times, filtered_true_lengths, args.param_o)
        tap = calculate_tap(filtered_true_labels, filtered_final_preds, filtered_detection_times, args.param_o, args.param_lambda)
    else:
        logging.warning("No non-normal samples found for early detection metric calculation.")
        earliness, erde, tap = 0.0, 0.0, 0.0
        
    avg_latency_ms = (total_inference_time / len(true_labels)) * 1000 if true_labels else 0
    f1_latency_score = calculate_f_latency(all_logits_seqs, true_labels_np)

    other_metrics = {
        "Accuracy": overall_accuracy,
        "Latency (ms/sample)": avg_latency_ms,
        "Avg Packets for Detection": avg_packets_for_detection, # [추가] 딕셔너리에 새 지표 추가
        "Earliness_Score_TP_Only": earliness, 
        "ERDE": erde, 
        "TaP": tap,
        "F1-Latency": f1_latency_score
    }
    
    # ##### >>> 수정된 부분 끝 <<< #####

    return performance_df, other_metrics

def evaluate_full_flow_only(model, test_loader, device, num_classes):
    model.eval()
    final_preds, true_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for data, time_info, labels in tqdm(test_loader, desc="Full-Flow Only Evaluation"):
            data, time_info, labels = data.to(device), time_info.to(device), labels.to(device)
            
            #padding_mask = (time_info == -1)
            
            logits, _, _ = model(data)
            
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            final_preds.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    true_labels_np = np.array(true_labels)
    final_preds_np = np.array(final_preds)
    all_probs_np = np.array(all_probs)

    # classification_report 생성
    class_names = list(CLASS_LABELS.values())
    report_dict = classification_report(
        true_labels_np, final_preds_np, 
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )
    accuracy = report_dict.pop('accuracy')
    performance_df = pd.DataFrame(report_dict).transpose()
    
    # ##### <<< 수정된 부분 시작 #####

    # AUROC 계산을 위한 새 컬럼 추가
    performance_df['auroc'] = np.nan

    try:
        # 1. 각 클래스별 AUROC 계산 (average=None)
        per_class_auroc = roc_auc_score(true_labels_np, all_probs_np, multi_class='ovr', average=None)
        
        # 2. Macro Average AUROC 계산
        macro_auroc = roc_auc_score(true_labels_np, all_probs_np, multi_class='ovr', average='macro')
        weight_auroc = roc_auc_score(true_labels_np, all_probs_np, multi_class='ovr', average='weighted')

        # 3. 계산된 AUROC 값을 DataFrame에 채우기
        # roc_auc_score는 클래스 인덱스 순서대로 결과를 반환합니다.
        for i, class_name in enumerate(class_names):
            if class_name in performance_df.index:
                performance_df.loc[class_name, 'auroc'] = per_class_auroc[i]
        
        # macro avg 행에 AUROC 값 추가
        performance_df.loc['macro avg', 'auroc'] = macro_auroc
        performance_df.loc['weighted avg', 'auroc'] = weight_auroc
        
    except ValueError as e:
        # y_true에 클래스가 하나만 존재할 경우 AUROC를 계산할 수 없어 ValueError 발생
        logging.warning(f"Could not calculate AUROC for full-flow: {e}")
        
    # ##### >>> 수정된 부분 끝 #####
        
    logging.info("\n\n" + "="*80 + "\n--- Full-Flow (Max Accuracy) Performance ---\n" + "="*80)
    logging.info(f"\nOverall Accuracy: {accuracy:.4f}\n")
    logging.info("\n" + performance_df.to_string(float_format="%.4f"))
    
    return accuracy, performance_df

def evaluate_aggregate_subflows(model, test_loader, device, num_classes):
    # ... (계산 로직은 이전과 동일)
    model.eval()
    all_subflow_preds, all_subflow_labels, all_subflow_probs = [], [], []
    with torch.no_grad():
        for data, time_info, labels in tqdm(test_loader, desc="Aggregate Subflow Evaluation"):
            data, time_info = data.to(device), time_info.to(device)
            B = data.shape[0]
            for i in range(B):
                full_flow_data, full_flow_time = data[i], time_info[i]
                true_label = labels[i].item()
                T = (full_flow_time >= 0).sum().item()
                if T == 0: continue
                for k in range(1, T + 1):
                    sub_flow_data = full_flow_data[:k, :].unsqueeze(0)
                    sub_flow_time = full_flow_time[:k].unsqueeze(0)
                    padding_mask = (sub_flow_time == -1)
                    logits, _, _ = model(sub_flow_data, time_info=sub_flow_time, src_mask=padding_mask)
                    probs = F.softmax(logits, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    all_subflow_preds.append(pred)
                    all_subflow_labels.append(true_label)
                    all_subflow_probs.append(probs.cpu().numpy().squeeze())

    # --- 지표 계산 ---
    logging.info("Calculating aggregate performance across all subflows...")
    true_labels_np = np.array(all_subflow_labels)
    final_preds_np = np.array(all_subflow_preds)
    all_probs_np = np.array(all_subflow_probs)
    report_dict = classification_report(true_labels_np, final_preds_np, target_names=list(CLASS_LABELS.values()), output_dict=True, zero_division=0)
    accuracy = report_dict.pop('accuracy')
    performance_df = pd.DataFrame(report_dict).transpose()
    auroc_values = {}
    if len(np.unique(true_labels_np)) > 1:
        try:
            per_class_auroc = roc_auc_score(true_labels_np, all_probs_np, multi_class='ovr', average=None)
            weighted_auroc = roc_auc_score(true_labels_np, all_probs_np, multi_class='ovr', average='weighted')
            for i, class_name in CLASS_LABELS.items():
                if i < len(per_class_auroc): auroc_values[class_name] = per_class_auroc[i]
            auroc_values['weighted avg'] = weighted_auroc
            auroc_values['macro avg'] = np.mean(per_class_auroc)
        except ValueError as e:
            logging.warning(f"Could not calculate AUROC for aggregate subflows: {e}")
    performance_df['auroc'] = pd.Series(auroc_values)
    performance_df.rename(columns={'recall': 'recall (class_accuracy)'}, inplace=True)
    performance_df['support'] = performance_df['support'].astype(int)

    # [추가] 함수가 직접 자신의 상세 리포트를 로깅
    logging.info("\n\n" + "="*80 + "\n--- Aggregate Performance of All Subflows ---\n" + "="*80)
    logging.info(f"\nOverall Accuracy (on all subflows): {accuracy:.4f}\n")
    ordered_cols = ['precision', 'recall (class_accuracy)', 'f1-score', 'auroc', 'support']
    existing_cols = [col for col in ordered_cols if col in performance_df.columns]
    logging.info("\n" + performance_df[existing_cols].to_string(float_format="%.4f"))

    # [수정] 이제 이 함수는 아무것도 반환할 필요가 없음
    # return accuracy, performance_df

def evaluate_full_flow_by_vote(model, test_loader, device, num_classes):
    """
    각 샘플의 모든 서브플로우(k=1..T) 예측에 대한 다수결 투표로 최종 예측을 결정하고,
    이를 바탕으로 전체 테스트셋의 성능을 평가합니다. (계산 비용이 매우 높음)
    """
    model.eval()
    final_preds, true_labels, all_final_probs = [], [], []
    
    with torch.no_grad():
        for data, time_info, labels in tqdm(test_loader, desc="Full-Flow by Vote Evaluation"):
            data, time_info = data.to(device), time_info.to(device)
            B = data.shape[0]
            
            for i in range(B):
                full_flow_data, full_flow_time = data[i], time_info[i]
                T = (full_flow_time >= 0).sum().item()
                if T == 0: continue
                
                subflow_preds_history = []
                last_probs = None

                for k in range(1, T + 1):
                    sub_flow_data = full_flow_data[:k, :].unsqueeze(0)
                    sub_flow_time = full_flow_time[:k].unsqueeze(0)
                    #padding_mask = (sub_flow_time == -1)
                    
                    logits, _, _ = model(sub_flow_data)
                    
                    probs = F.softmax(logits, dim=1).squeeze(0)
                    pred = probs.argmax().item()
                    subflow_preds_history.append(pred)
                    
                    if k == T:
                        last_probs = probs.cpu().numpy()

                vote_counts = Counter(subflow_preds_history)
                final_pred = vote_counts.most_common(1)[0][0]
                
                final_preds.append(final_pred)
                true_labels.append(labels[i].item())
                
                if last_probs is not None:
                    all_final_probs.append(last_probs)

    true_labels_np = np.array(true_labels)
    final_preds_np = np.array(final_preds)
    all_probs_np = np.array(all_final_probs)
    
    class_names = list(CLASS_LABELS.values())
    report_dict = classification_report(
        true_labels_np, final_preds_np, 
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )
    accuracy = report_dict.pop('accuracy')
    performance_df = pd.DataFrame(report_dict).transpose()
    
    # ##### <<< 수정된 부분 시작 #####

    # AUROC 계산을 위한 새 컬럼 추가
    performance_df['auroc'] = np.nan

    try:
        # 1. 각 클래스별 AUROC 계산
        per_class_auroc = roc_auc_score(true_labels_np, all_probs_np, multi_class='ovr', average=None)
        
        # 2. Macro 및 Weighted Average AUROC 계산
        weighted_auroc = roc_auc_score(true_labels_np, all_probs_np, multi_class='ovr', average='weighted')
        macro_auroc = roc_auc_score(true_labels_np, all_probs_np, multi_class='ovr', average='macro')

        # 3. 계산된 AUROC 값을 DataFrame에 채우기
        for i, class_name in enumerate(class_names):
            if class_name in performance_df.index:
                performance_df.loc[class_name, 'auroc'] = per_class_auroc[i]
        
        # 평균 AUROC 값 추가
        performance_df.loc['weighted avg', 'auroc'] = weighted_auroc
        performance_df.loc['macro avg', 'auroc'] = macro_auroc
        
    except ValueError as e:
        logging.warning(f"Could not calculate AUROC for vote-based evaluation: {e}")
        
    # ##### >>> 수정된 부분 끝 #####
    
    logging.info("\n\n" + "="*80 + "\n--- Full-Flow Performance (by Majority Vote) ---\n" + "="*80)
    logging.info(f"\nOverall Accuracy: {accuracy:.4f}\n")
    logging.info("\n" + performance_df.to_string(float_format="%.4f"))

    return accuracy, performance_df

@profile(stream=open('student_memory_profile.log', 'w+'))
def profile_and_evaluate(model, dataloader, device, args):
    """메모리 프로파일링을 위해 evaluate 함수를 감싸는 래퍼 함수"""
    return evaluate_early_detection(model, dataloader, device, args)

def parse_memory_log(log_path):
    peak_mem = -1.0
    mem_regex = re.compile(r'(\d+\.?\d*)\s+MiB')
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = mem_regex.search(line)
                if match:
                    mem_val = float(match.group(1))
                    if mem_val > peak_mem:
                        peak_mem = mem_val
    except FileNotFoundError:
        logging.warning(f"Memory profile log not found: {log_path}")
    except Exception as e:
        logging.warning(f"Error parsing memory profile log: {e}")
        
    return peak_mem


# --- [새로 추가된 함수] 단일 체크포인트 평가 로직 ---
def evaluate_checkpoint(args, DEVICE, model_path, test_loader):
    """단일 체크포인트 파일에 대한 전체 평가를 수행하고 결과를 로깅합니다."""
    
    logging.info("\n" + "#"*100)
    logging.info(f"# Evaluating Checkpoint: {os.path.basename(model_path)}")
    logging.info("#"*100 + "\n")

    # 모델 로드
    model = eTransformer(d_model=514, seq_len=67, num_classes=5, num_heads=2, key_dim=32, dropout=0.2).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 1. 정적 특성 및 리소스 측정
    logging.info("Calculating model static and resource statistics...")
    total_params = sum(p.numel() for p in model.parameters())
    torch.save(model.state_dict(), "temp_student.p")
    model_size_mb = os.path.getsize("temp_student.p") / 1e6
    os.remove("temp_student.p")
    
    dummy_input = torch.randn(1, args.seq_length, args.input_dim).to(DEVICE)
    #dummy_time_info = torch.randn(1, args.seq_length).to(DEVICE)
    flops = FlopCountAnalysis(model, (dummy_input,))
    gflops = flops.total() / 1e9
    
    # 2. 동적 특성 측정 (성능)
    logging.info("Running Early Detection performance evaluation...")
    performance_df, other_metrics = profile_and_evaluate(model, test_loader, DEVICE, args)
    peak_mem_mb = parse_memory_log('student_memory_profile.log')
    
    evaluate_full_flow_only(model, test_loader, DEVICE, args.num_classes)
    #evaluate_full_flow_by_vote(model, test_loader, DEVICE, args.num_classes)
    #evaluate_aggregate_subflows(model, test_loader, DEVICE, args.num_classes)
    
    # 3. 모든 결과 취합 및 출력
    summary_data = {
        "Total Params": f"{total_params:,}",
        "Model Size (MB)": f"{model_size_mb:.2f}",
        "Memory Footprint (MB)": f"{peak_mem_mb:.2f}",
        "GFLOPs": f"{gflops:.4f}"
    }
    summary_data.update({key: f"{val:.4f}" for key, val in other_metrics.items()})
    summary_df = pd.DataFrame([summary_data])
    
    logging.info("\n\n" + "="*80 + "\n--- General & Resource Summary ---\n" + "="*80)
    logging.info("\n" + summary_df.to_string(index=False))
    
    logging.info("\n\n" + "="*80 + "\n--- Early detection Performance ---\n" + "="*80)
    ordered_cols = ['precision', 'recall (class_accuracy)', 'f1-score', 'auroc', 'support']
    existing_cols = [col for col in ordered_cols if col in performance_df.columns]
    logging.info("\n" + performance_df[existing_cols].to_string(float_format="%.4f"))


# --- [수정됨] main 함수 ---
def main(args):
    setup_logging(args.log_path)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting evaluation run with settings:")
    logging.info(f" -> Device: {DEVICE}")
    logging.info(f" -> Model Directory: {args.model_dir}")
    logging.info(f" -> Epoch Range: {args.start_epoch} to {args.end_epoch} (step: {args.epoch_step})")
    logging.info(f" -> Log File: {args.log_path}")

    # 데이터 로더는 한 번만 생성
    logging.info(f"Loading test data from {args.npz_path}")
    test_dataset = PacketDataset(args.npz_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 지정된 에포크 범위에 대해 순회
    for epoch in range(args.start_epoch, args.end_epoch + 1, args.epoch_step):
        checkpoint_name = f"teacher_epoch_{epoch}.pth"
        model_path = os.path.join(args.model_dir, checkpoint_name)
        
        if os.path.exists(model_path):
            evaluate_checkpoint(args, DEVICE, model_path, test_loader)
        else:
            logging.warning(f"Checkpoint not found, skipping: {model_path}")
            
    logging.info("\n\n" + "#"*100)
    logging.info("All specified epochs have been evaluated.")
    logging.info("#"*100)


if __name__ == '__main__':
    args = get_args()
    main(args)