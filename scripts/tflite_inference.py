import os
import re
import glob # <<< [추가된 부분] 파일 검색을 위해 glob 임포트
import tflite_runtime.interpreter as tflite
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
import time
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import logging
from memory_profiler import profile
from collections import Counter

# 외부 유틸리티 함수 임포트 (이 forwards_utils.py 파일이 존재해야 합니다)
from forwards_utils import calculate_earliness, calculate_erde, calculate_f_latency, calculate_tap

# 클래스 레이블 정의
CLASS_LABELS = {
    0: 'benign',
    1: 'Spoofing',
    2: 'Brute Force',
    3: 'Web based',
    4: 'Recon',
}

def get_args():
    """명령줄 인자를 파싱하는 함수"""
    parser = argparse.ArgumentParser(description="TFLite Model Evaluation Script")
    
    # 경로 설정
    # get_args() 함수 내부
    parser.add_argument('--model_path', default='./student_final.tflite', type=str, required=True, help='Path to a single .tflite file OR a directory containing them')
    parser.add_argument('--npz_path', type=str, default=None, help='Path to the test .npz file (not needed if --use_dummy_data is set)')
    parser.add_argument('--log_path', type=str, default='./tflite_evaluation.log', help='Path to save the evaluation log file')
    
    # <<< [삭제] 에포크 관련 인자들 모두 삭제 ---
    # parser.add_argument('--start_epoch', ...)
    # parser.add_argument('--end_epoch', ...)
    # parser.add_argument('--epoch_step', ...)
    
    # 데이터 및 평가 파라미터
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--input_dim', type=int, default=514, help='Input feature dimension')
    parser.add_argument('--seq_length', type=int, default=67, help='Sequence length')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of attack classes')
    
    # 조기 탐지 평가 파라미터
    parser.add_argument('--early_detection_threshold', type=float, default=0.95, help='Confidence threshold for early detection')
    parser.add_argument('--param_o', type=float, default=5, help='Parameter "o" for ERDE calculation (FP cost)')
    parser.add_argument('--param_lambda', type=float, default=0.1, help='Parameter "lambda" for TaP calculation')

    # 더미 데이터 사용을 위한 플래그(flag) 추가
    parser.add_argument('--use_dummy_data', action='store_true', 
                        help='If set, generate and use random dummy data instead of loading from npz_path.')
    
    args = parser.parse_args()
    
    if not args.use_dummy_data and args.npz_path is None:
        parser.error("--npz_path is required unless --use_dummy_data is set.")
        
    return args

def setup_logging(log_path):
    """로깅 설정을 초기화하는 함수"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def create_dummy_dataset_loader(args):
    """테스트를 위한 더미 tf.data.Dataset 객체를 생성하고 반환하는 함수"""
    logging.info("--- GENERATING DUMMY DATA FOR TESTING ---")
    NUM_SAMPLES = 512
    X = np.random.rand(NUM_SAMPLES, args.seq_length, args.input_dim).astype(np.float32)
    y = np.random.randint(0, args.num_classes, size=(NUM_SAMPLES,)).astype(np.int64)
    T_relative = np.zeros((NUM_SAMPLES, args.seq_length), dtype=np.float32)
    for i in range(NUM_SAMPLES):
        true_length = np.random.randint(10, args.seq_length + 1)
        T_relative[i, :true_length] = np.sort(np.random.rand(true_length) * 100)
        T_relative[i, true_length:] = -1.0
    dummy_dataset = tf.data.Dataset.from_tensor_slices((X, T_relative, y))
    return dummy_dataset.batch(args.batch_size)

class PacketDataset:
    """NPZ 파일에서 데이터를 로드하는 클래스"""
    def __init__(self, npz_path):
        with np.load(npz_path) as data:
            self.data = data['X'].astype(np.float32)
            self.time = data['T_relative'].astype(np.float32)
            self.labels = data['y'].astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.time[idx], self.labels[idx])

def softmax(x, axis=-1):
    """NumPy 기반 Softmax 함수"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def evaluate_early_detection(interpreter, test_loader, args):
    """조기 탐지 성능을 평가하는 함수"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    final_preds, true_labels = [], []
    all_final_probs_list = []
    detection_times_t, true_lengths_T = [], []
    total_inference_time = 0.0
    all_logits_seqs = []
    for data, time_info, labels in tqdm(test_loader, desc="Early Detection Evaluation"):
        data, time_info, labels = data.numpy(), time_info.numpy(), labels.numpy()
        B = data.shape[0]
        for i in range(B):
            full_flow_data, full_flow_time = data[i], time_info[i]
            T = (full_flow_time >= 0).sum().item()
            if T == 0: continue
            current_label = labels[i].item()
            found_positive, last_probs, current_flow_logits = False, None, []
            for k in range(1, T + 1):
                sub_flow_data = full_flow_data[:k, :][np.newaxis, ...]
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], sub_flow_data)
                interpreter.invoke()
                logits = interpreter.get_tensor(output_details[0]['index'])
                total_inference_time += time.time() - start_time
                current_flow_logits.append(logits.squeeze(0))
                probs = softmax(logits).squeeze(0)
                last_probs = probs
                if current_label != 0 and np.max(probs) >= args.early_detection_threshold:
                    final_pred, t = np.argmax(probs), k
                    found_positive = True
                    break
            if not found_positive:
                final_pred, t = np.argmax(last_probs), T
            final_preds.append(final_pred)
            true_labels.append(current_label)
            detection_times_t.append(t)
            true_lengths_T.append(T)
            all_logits_seqs.append(np.stack(current_flow_logits))
            all_final_probs_list.append(probs if found_positive else last_probs)
    true_labels_np, final_preds_np, all_final_probs_np = np.array(true_labels), np.array(final_preds), np.array(all_final_probs_list)
    report_dict = classification_report(true_labels_np, final_preds_np, target_names=list(CLASS_LABELS.values()), output_dict=True, zero_division=0)
    overall_accuracy = report_dict.pop('accuracy')
    performance_df = pd.DataFrame(report_dict).transpose()
    if len(np.unique(true_labels_np)) > 1 and all_final_probs_np.ndim == 2:
        try:
            per_class_auroc = roc_auc_score(true_labels_np, all_final_probs_np, multi_class='ovr', average=None)
            weighted_auroc = roc_auc_score(true_labels_np, all_final_probs_np, multi_class='ovr', average='weighted')
            for i, class_name in enumerate(CLASS_LABELS.values()):
                if class_name in performance_df.index:
                    performance_df.loc[class_name, 'auroc'] = per_class_auroc[i]
            performance_df.loc['weighted avg', 'auroc'] = weighted_auroc
            performance_df.loc['macro avg', 'auroc'] = np.mean(per_class_auroc)
        except ValueError as e: logging.warning(f"Could not calculate AUROC: {e}")
    non_normal_mask = (true_labels_np != 0)
    avg_packets, earliness, erde, tap = 0.0, 0.0, 0.0, 0.0
    if np.any(non_normal_mask):
        filtered_times, filtered_lengths = np.array(detection_times_t)[non_normal_mask], np.array(true_lengths_T)[non_normal_mask]
        filtered_true, filtered_preds = true_labels_np[non_normal_mask], final_preds_np[non_normal_mask]
        avg_packets = np.mean(filtered_times)
        earliness = calculate_earliness(filtered_times, filtered_lengths, filtered_true, filtered_preds)
        erde = calculate_erde(filtered_true, filtered_preds, filtered_times, filtered_lengths, args.param_o)
        tap = calculate_tap(filtered_true, filtered_preds, filtered_times, args.param_o, args.param_lambda)
    else: logging.warning("No non-normal samples for early detection metrics.")
    avg_latency_ms = (total_inference_time / len(true_labels)) * 1000 if true_labels else 0
    f1_latency_score = calculate_f_latency(all_logits_seqs, true_labels_np)
    return performance_df, {"Accuracy": overall_accuracy, "Latency (ms/sample)": avg_latency_ms, "Avg Packets for Detection": avg_packets, "Earliness_Score_TP_Only": earliness, "ERDE": erde, "TaP": tap, "F1-Latency": f1_latency_score}

def evaluate_full_flow_only(interpreter, test_loader, args):
    """전체 플로우(Full-Flow)에 대한 성능을 평가하는 함수"""
    input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()
    final_preds, true_labels, all_probs = [], [], []
    for data, _, labels in tqdm(test_loader, desc="Full-Flow Only Evaluation"):
        data, labels = data.numpy(), labels.numpy()
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        logits = interpreter.get_tensor(output_details[0]['index'])
        probs_batch = softmax(logits)
        preds_batch = np.argmax(probs_batch, axis=1)
        final_preds.extend(preds_batch); true_labels.extend(labels); all_probs.extend(probs_batch)
    true_labels_np, final_preds_np, all_probs_np = np.array(true_labels), np.array(final_preds), np.array(all_probs)
    report_dict = classification_report(true_labels_np, final_preds_np, target_names=list(CLASS_LABELS.values()), output_dict=True, zero_division=0)
    accuracy = report_dict.pop('accuracy')
    performance_df = pd.DataFrame(report_dict).transpose()
    try:
        per_class_auroc = roc_auc_score(true_labels_np, all_probs_np, multi_class='ovr', average=None)
        macro_auroc = roc_auc_score(true_labels_np, all_probs_np, multi_class='ovr', average='macro')
        weighted_auroc = roc_auc_score(true_labels_np, all_probs_np, multi_class='ovr', average='weighted')
        for i, class_name in enumerate(CLASS_LABELS.values()):
            if class_name in performance_df.index: performance_df.loc[class_name, 'auroc'] = per_class_auroc[i]
        performance_df.loc['macro avg', 'auroc'] = macro_auroc
        performance_df.loc['weighted avg', 'auroc'] = weighted_auroc
    except ValueError as e: logging.warning(f"Could not calculate AUROC for full-flow: {e}")
    logging.info("\n\n" + "="*80 + "\n--- Full-Flow (Max Accuracy) Performance ---\n" + "="*80)
    logging.info(f"\nOverall Accuracy: {accuracy:.4f}\n\n{performance_df.to_string(float_format='%.4f')}")
    return accuracy, performance_df

@profile(stream=open('tflite_memory_profile.log', 'w+'))
def profile_and_evaluate(interpreter, dataloader, args):
    return evaluate_early_detection(interpreter, dataloader, args)

def parse_memory_log(log_path):
    peak_mem = -1.0
    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = re.search(r'(\d+\.?\d*)\s+MiB', line)
                if match and float(match.group(1)) > peak_mem: peak_mem = float(match.group(1))
    except (FileNotFoundError, Exception) as e: logging.warning(f"Could not parse memory log: {e}")
    return peak_mem

def evaluate_checkpoint(args, model_path, test_loader):
    logging.info("\n" + "#"*100 + f"\n# Evaluating TFLite Model: {os.path.basename(model_path)}\n" + "#"*100 + "\n")
    interpreter = tflite.Interpreter(model_path=model_path); interpreter.allocate_tensors()
    logging.info(f"TFLite model loaded successfully from {model_path}")
    model_size_mb = os.path.getsize(model_path) / 1e6
    performance_df, other_metrics = profile_and_evaluate(interpreter, test_loader, args)
    peak_mem_mb = parse_memory_log('tflite_memory_profile.log')
    evaluate_full_flow_only(interpreter, test_loader, args)
    summary_data = {"Total Params": "N/A", "Model Size (MB)": f"{model_size_mb:.2f}", "Memory Footprint (MB)": f"{peak_mem_mb:.2f}", "GFLOPs": "N/A"}
    summary_data.update({key: f"{val:.4f}" for key, val in other_metrics.items()})
    logging.info("\n\n" + "="*80 + "\n--- General & Resource Summary ---\n" + "="*80 + f"\n{pd.DataFrame([summary_data]).to_string(index=False)}")
    logging.info("\n\n" + "="*80 + "\n--- Early detection Performance ---\n" + "="*80)
    logging.info(f"\n{performance_df[['precision', 'recall', 'f1-score', 'auroc', 'support']].to_string(float_format='%.4f')}")

def main(args):
    """메인 실행 함수"""
    setup_logging(args.log_path)
    logging.info(f"Starting TFLite evaluation run with settings: {vars(args)}")
    if args.use_dummy_data:
        test_loader = create_dummy_dataset_loader(args)
    else:
        logging.info(f"Loading test data from {args.npz_path}")
        dataset = PacketDataset(args.npz_path)

        def numpy_batch_generator(dataset, batch_size):
            num_samples = len(dataset)
            for i in range(0, num_samples, batch_size):
                end_index = min(i + batch_size, num_samples)

                batch_data = dataset.data[i:end_index]
                batch_time = dataset.time[i:end_index]
                batch_labels = dataset.labels[i:end_index]

                yield (batch_data, batch_time, batch_labels)
        test_loader = tf_dataset.batch(args.batch_size)
    
    # <<< [수정된 부분] glob을 사용하여 폴더 내 모든 .tflite 파일을 찾음
    model_paths = glob.glob(os.path.join(args.model_dir, '*.tflite'))
    
    if not model_paths:
        logging.error(f"No .tflite models found in directory: {args.model_dir}")
        return

    logging.info(f"Found {len(model_paths)} models to evaluate in '{args.model_dir}'.")

    for model_path in model_paths:
        evaluate_checkpoint(args, model_path, test_loader)
            
    logging.info("\n\n" + "#"*100 + "\nAll specified models have been evaluated.\n" + "#"*100)

if __name__ == '__main__':
    args = get_args()
    main(args)
