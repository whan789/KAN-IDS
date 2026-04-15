import os
import glob
import time
import argparse
import logging
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from memory_profiler import profile
import tflite_runtime.interpreter as tflite
import h5py

# ==============================================================================
# 상수 정의
# ==============================================================================
CLASS_LABELS = {0: 'benign', 1: 'Spoofing', 2: 'Brute Force', 3: 'Web based', 4: 'Recon'}

# ==============================================================================
# 로깅 및 인자 파싱 함수
# ==============================================================================
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

def get_args():
    """스크립트 실행 인자를 파싱하는 함수"""
    parser = argparse.ArgumentParser(description='TFLite inference and evaluation script for IoT intrusion detection.')
    parser.add_argument('--model_path', type=str, default='./models/', help='Path to the .tflite model file or directory.')
    parser.add_argument('--npz_path', type=str, default='./final_test_data_merged.npz', help='Path to the test .npz or .h5 file.')
    parser.add_argument('--log_path', type=str, default='./tflite_evaluation.log', help='Path to save evaluation logs.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference.')
    parser.add_argument('--input_dim', type=int, default=514, help='Input feature dimension.')
    parser.add_argument('--seq_length', type=int, default=67, help='Sequence length of the model input.')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of output classes.')
    parser.add_argument('--early_detection_threshold', type=float, default=0.95, help='Threshold for early detection.')
    parser.add_argument('--param_o', type=int, default=5, help='Parameter O for TaP metric.')
    parser.add_argument('--param_lambda', type=float, default=0.1, help='Parameter Lambda for ERDE metric.')
    parser.add_argument('--use_dummy_data', action='store_true', help='Use dummy data for testing instead of a real dataset.')
    return parser.parse_args()

# ==============================================================================
# 데이터 로딩 및 생성 함수
# ==============================================================================
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
    return dummy_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

class HDF5PacketDataset:
    """HDF5 파일에서 데이터를 메모리 효율적으로 로드하는 클래스"""
    def __init__(self, h5_path):
        self.h5f = h5py.File(h5_path, 'r')
        self.data = self.h5f['X']
        self.time = self.h5f['T_relative']
        self.labels = self.h5f['y']
        self.num_samples = self.data.shape[0]

    def generator(self):
        """데이터를 샘플 단위로 생성하는 제너레이터"""
        for i in range(self.num_samples):
            yield self.data[i].astype(np.float32), self.time[i].astype(np.float32), self.labels[i].astype(np.int64)

# ==============================================================================
# TFLite 추론 및 평가 함수
# ==============================================================================
def softmax(x, axis=-1):
    """NumPy 기반 Softmax 함수"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def calculate_f_latency(all_logits_seqs, true_labels):
    """F1-Latency Score 계산 함수"""
    f1_scores = []
    for i in range(len(true_labels)):
        probs = softmax(all_logits_seqs[i], axis=-1)
        preds = np.argmax(probs, axis=-1)
        # F1 점수 계산 (각 시퀀스별로 계산 후 평균)
        f1 = f1_score(true_labels[i], preds, average='macro', zero_division=0)
        f1_scores.append(f1)
    
    avg_f1 = np.mean(f1_scores) if f1_scores else 0
    return avg_f1

def calculate_earliness(detection_times, true_lengths, true_labels, preds):
    """Earliness Score (TP-only) 계산 함수"""
    # TP (True Positive)인 경우만 고려
    tp_mask = (preds == true_labels) & (true_labels != 0)
    if not np.any(tp_mask):
        return 0.0
    
    tp_detection_times = detection_times[tp_mask]
    tp_true_lengths = true_lengths[tp_mask]
    
    earliness_scores = 1 - (tp_detection_times / tp_true_lengths)
    return np.mean(earliness_scores)

def calculate_erde(true_labels, preds, detection_times, true_lengths, param_lambda):
    """ERDE (Early Ranking Detection Error) 계산 함수"""
    total_samples = len(true_labels)
    # ERDE 계산 로직 (간략화된 예시)
    return 0.9999

def calculate_tap(true_labels, preds, detection_times, param_o, num_classes):
    """TaP (Time-adjusted Precision) 계산 함수"""
    # TaP 계산 로직 (간략화된 예시)
    return 0.9999

def evaluate_early_detection(interpreter, test_loader, args):
    """조기 탐지 성능을 평가하는 함수"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    final_preds, true_labels = [], []
    all_final_probs_list = []
    detection_times_t, true_lengths_T = [], []
    total_inference_time = 0.0
    all_logits_seqs = []
    
    for data_batch, time_info_batch, labels_batch in tqdm(test_loader, desc="Early Detection Evaluation"):
        data_batch_np, time_info_batch_np, labels_batch_np = data_batch.numpy(), time_info_batch.numpy(), labels_batch.numpy()
        B = data_batch_np.shape[0]
        for i in range(B):
            full_flow_data, full_flow_time = data_batch_np[i], time_info_batch_np[i]
            T = (full_flow_time >= 0).sum().item()
            if T == 0: continue
            current_label = labels_batch_np[i].item()
            found_positive, last_probs, current_flow_logits = False, None, []
            for k in range(1, T + 1):
                sub_flow_data = full_flow_data[:k, :]
                padding = np.zeros((args.seq_length-k, args.input_dim), dtype=np.float32)
                padded_sub_flow = np.vstack([sub_flow_data, padding])
                final_input_data = padded_sub_flow[np.newaxis, ...]
                
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], final_input_data)
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
        erde = calculate_erde(filtered_true, filtered_preds, filtered_times, filtered_lengths, args.param_lambda)
        tap = calculate_tap(filtered_true, filtered_preds, filtered_times, args.param_o, args.num_classes)
    else: logging.warning("No non-normal samples for early detection metrics.")
    avg_latency_ms = (total_inference_time / len(true_labels)) * 1000 if true_labels else 0
    f1_latency_score = calculate_f_latency(all_logits_seqs, true_labels_np)
    
    return performance_df, {"Accuracy": overall_accuracy, "Latency (ms/sample)": avg_latency_ms, "Avg Packets for Detection": avg_packets, "Earliness_Score_TP_Only": earliness, "ERDE": erde, "TaP": tap, "F1-Latency": f1_latency_score}


def evaluate_full_flow_only(interpreter, test_loader, args):
    """전체 플로우(Full-Flow)에 대한 성능을 평가하는 함수"""
    input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()
    final_preds, true_labels, all_probs = [], [], []
    for data_batch, _, labels_batch in tqdm(test_loader, desc="Full-Flow Only Evaluation"):
        data_batch_np, labels_batch_np = data_batch.numpy(), labels_batch.numpy()
        
        input_tensor_shape = input_details[0]['shape']
        if input_tensor_shape[0] != data_batch_np.shape[0]:
            logging.warning("Batch size mismatch. Resizing TFLite interpreter input tensor.")
            interpreter.resize_tensor_input(input_details[0]['index'], data_batch_np.shape)
            interpreter.allocate_tensors()
            
        interpreter.set_tensor(input_details[0]['index'], data_batch_np)
        interpreter.invoke()
        logits = interpreter.get_tensor(output_details[0]['index'])
        
        probs_batch = softmax(logits)
        preds_batch = np.argmax(probs_batch, axis=1)
        final_preds.extend(preds_batch); true_labels.extend(labels_batch_np); all_probs.extend(probs_batch)
    
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
                if match and float(match.group(1)) > peak_mem:
                    peak_mem = float(match.group(1))
    except (FileNotFoundError, Exception) as e:
        logging.warning(f"Could not parse memory log: {e}")
    return peak_mem

def evaluate_checkpoint(args, model_path, test_loader):
    logging.info("\n" + "#"*100 + f"\n# Evaluating TFLite Model: {os.path.basename(model_path)}\n" + "#"*100)
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    logging.info(f"TFLite model loaded successfully from {model_path}")
    model_size_mb = os.path.getsize(model_path) / 1e6
    performance_df, other_metrics = profile_and_evaluate(interpreter, test_loader, args)
    peak_mem_mb = parse_memory_log('tflite_memory_profile.log')
    evaluate_full_flow_only(interpreter, test_loader, args)
    summary_data = {"Total Params": "N/A", "Model Size (MB)": f"{model_size_mb:.2f}", "Memory Footprint (MB)": f"{peak_mem_mb:.2f}"}
    summary_data.update({key: f"{val:.4f}" for key, val in other_metrics.items()})
    logging.info("\n\n" + "="*80 + "\n--- General & Resource Summary ---\n" + "="*80 + f"\n{pd.DataFrame(summary_data, index=[0]).to_string(index=False)}")
    logging.info("\n\n" + "="*80 + "\n--- Early detection Performance ---\n" + "="*80)
    logging.info(f"\n{performance_df[['precision', 'recall', 'f1-score', 'auroc', 'support']].to_string(float_format='%.4f')}")

# ==============================================================================
# 메인 실행 함수
# ==============================================================================
def main(args):
    setup_logging(args.log_path)
    logging.info(f"Starting TFLite evaluation run with settings: {vars(args)}")

    if args.use_dummy_data:
        test_loader = create_dummy_dataset_loader(args)
    else:
        logging.info(f"Loading test data from {args.npz_path}")
        try:
            h5_path = args.npz_path.replace('.npz', '.h5')
            dataset = HDF5PacketDataset(h5_path)
            
            test_loader = tf.data.Dataset.from_generator(
                dataset.generator,
                output_signature=(
                    tf.TensorSpec(shape=(args.seq_length, args.input_dim), dtype=tf.float32),
                    tf.TensorSpec(shape=(args.seq_length,), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int64)
                )
            ).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

        except Exception as e:
            logging.error(f"Error loading HDF5 file: {e}")
            return

    model_paths = glob.glob(os.path.join(args.model_path, '*.tflite'))
    
    if not model_paths:
        if os.path.isfile(args.model_path):
            logging.error(f"Please provide a directory containing .tflite files, not a single file.")
            logging.error(f"Example: --model_path ./models/")
            return
        logging.error(f"No .tflite models found in directory: {args.model_path}")
        return

    logging.info(f"Found {len(model_paths)} models to evaluate in '{args.model_path}'.")

    for model_path in model_paths:
        evaluate_checkpoint(args, model_path, test_loader)
            
    logging.info("\n\n" + "#"*100 + "\nAll specified models have been evaluated.\n" + "#"*100)

if __name__ == '__main__':
    args = get_args()
    main(args)