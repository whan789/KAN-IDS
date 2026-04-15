import numpy as np
import os
import torch.nn.functional as F

def augment_flow(x_flow, t_relative_flow, max_flow_len=40, packet_vec_len=448):
    """
    단일 플로우 데이터에 5가지 증강 기법을 순차적으로 적용합니다.

    Args:
        x_flow (np.ndarray): 단일 플로우의 패킷 데이터. Shape: (max_flow_len, packet_vec_len)
        t_relative_flow (np.ndarray): 단일 플로우의 상대 시간 데이터. Shape: (max_flow_len,)
        max_flow_len (int): 최대 플로우 길이.
        packet_vec_len (int): 패킷 벡터의 차원.

    Returns:
        tuple: 증강된 (x_aug, t_relative_aug) 튜플.
    """
    # 원본 데이터를 복사하여 사용
    x_aug = x_flow.copy()
    t_aug = t_relative_flow.copy()

    # 패딩을 제외한 실제 플로우 길이 계산
    # np.where는 조건에 맞는 인덱스의 튜플을 반환하므로, [0]으로 배열을 꺼내고 다시 [0]으로 첫 인덱스를 가져옵니다.
    try:
        n = np.where(t_aug < 0)[0][0]
    except IndexError:
        n = len(t_aug) # 패딩이 없는 경우

    if n <= 1: # 플로우 길이가 너무 짧으면 증강을 건너뜀
        return x_aug, t_aug

    # ====================================================================
    # 1. Jitter Injection (시간 변동 주입)
    # ====================================================================
    # 상대 시간(누적)을 패킷 간 도착 시간(IAT)으로 변환
    iats = np.diff(t_aug[:n], prepend=0)
    
    # 논문에서는 tmin을 이전/다음 패킷과의 최소 시간차로 정의했지만,
    # 여기서는 각 IAT를 기준으로 더 간단하고 강건하게 구현합니다.
    for i in range(1, n): # 첫 패킷(IAT=0)은 제외
        tmin = iats[i]
        jitter = np.random.uniform(-0.7 * tmin, 0.7 * tmin)
        iats[i] = max(0, iats[i] + jitter) # IAT가 음수가 되지 않도록 보정
        
    t_aug[:n] = np.cumsum(iats) # IAT를 다시 누적 상대 시간으로 변환

    # ====================================================================
    # 2. Traffic Scaling (트래픽 스케일링)
    # ====================================================================
    iats = np.diff(t_aug[:n], prepend=0)
    scale_factor = np.random.choice([0.5, 0.75, 1.0, 1.25, 1.5])
    iats *= scale_factor
    t_aug[:n] = np.cumsum(iats)
    
    # ====================================================================
    # 3. Packet Drop
    # ====================================================================
    max_packets_to_drop = max(0, int(0.25 * n - 0.5))
    num_to_drop = np.random.randint(0, max_packets_to_drop + 1)
    
    if num_to_drop > 0:
        indices_to_drop = np.random.choice(range(n), num_to_drop, replace=False)
        x_aug = np.delete(x_aug, indices_to_drop, axis=0)
        t_aug = np.delete(t_aug, indices_to_drop, axis=0)
        
        # --- 수정된 부분 ---
        # 현재 배열의 길이를 기준으로 max_flow_len이 되기 위해 필요한 패딩 길이를 정확히 계산합니다.
        pad_width = max_flow_len - x_aug.shape[0]
        
        x_aug = np.pad(x_aug, ((0, pad_width), (0, 0)), 'constant')
        t_aug = np.pad(t_aug, (0, pad_width), 'constant', constant_values=-1.0)
        n -= num_to_drop

    # ====================================================================
    # 4. Packet Insertion
    # ====================================================================
    max_packets_to_insert = max(0, int(0.15 * n - 0.5))
    num_to_insert = np.random.randint(0, max_packets_to_insert + 1)

    if num_to_insert > 0 and (x_aug.shape[0] + num_to_insert) <= max_flow_len:
        insert_indices = sorted(np.random.choice(range(n + 1), num_to_insert, replace=False))
        zero_packet = np.zeros((1, packet_vec_len))
        
        for offset, idx in enumerate(insert_indices):
            insert_time = t_aug[idx-1] if idx > 0 else 0
            # for 루프 내에서 insert를 여러번 하면 인덱스가 바뀌므로, 계산된 인덱스에 offset을 더해줌
            x_aug = np.insert(x_aug, idx + offset, zero_packet, axis=0)
            t_aug = np.insert(t_aug, idx + offset, insert_time)
        
        # 삽입 후 최대 길이를 초과하면 잘라내고 부족하면 패딩
        x_aug = x_aug[:max_flow_len]
        t_aug = t_aug[:max_flow_len]
        if x_aug.shape[0] < max_flow_len:
            pad_width = max_flow_len - x_aug.shape[0]
            x_aug = np.pad(x_aug, ((0, pad_width), (0, 0)), 'constant')
            t_aug = np.pad(t_aug, (0, pad_width), 'constant', constant_values=-1.0)

        n += num_to_insert

    # ====================================================================
    # 5. Noise Injection (노이즈 주입)
    # ====================================================================
    max_packets_to_modify = int(n / 3)
    num_to_modify = np.random.randint(0, max_packets_to_modify + 1)

    if num_to_modify > 0:
        packet_indices = np.random.choice(range(n), num_to_modify, replace=False)
        
        for packet_idx in packet_indices:
            max_bytes_to_alter = int(packet_vec_len / 100)
            num_to_alter = np.random.randint(0, max_bytes_to_alter + 1)
            
            if num_to_alter > 0:
                byte_indices = np.random.choice(range(packet_vec_len), num_to_alter, replace=False)
                noise = np.random.normal(loc=0.0, scale=0.1, size=num_to_alter)
                x_aug[packet_idx, byte_indices] += noise
    
    # 데이터는 0과 1 사이로 정규화되었으므로, 노이즈 주입 후 범위를 벗어날 수 있음.
    # 따라서 값을 [0, 1] 범위로 다시 클리핑합니다.
    np.clip(x_aug, 0, 1, out=x_aug)

    return x_aug, t_aug

import torch
import torch.nn as nn

def pgd_attack(model, loss_fn, x_clean, time_info, y_true, epsilon, alpha, num_iter, flow_lengths):
    """
    PGD 공격을 사용하여 적대적 예제를 생성합니다. 데이터 증강을 활용해서 학습한 Teacher모델에 PGD를 적용한 데이터에 fine tuning
    PGD는 FGSM을 여러 번 작은 보폭으로 반복하는 다중 스텝(multi-step) 공격입니다. 매 스텝마다 조금씩 노이즈를 추가하고, 그 결과가 허용된 범위를 벗어나지 않도록 프로젝션(projection)을 수행합니다. 
    이는 모델의 손실(loss)을 높이는 방향으로 더 정교하게 최적의 공격 지점을 찾아냅니다.

    Args:
        model (nn.Module): 훈련된 모델.
        loss_fn: 손실 함수 (예: nn.CrossEntropyLoss).
        x_clean (torch.Tensor): 원본 입력 데이터 (배치).
        time_info
        y_true (torch.Tensor): 원본 데이터의 실제 레이블.
        epsilon (float): 최대 허용되는 노이즈의 크기 (L-infinity norm).
        alpha (float): 각 스텝에서 이동할 크기 (스텝 사이즈).
        num_iter (int): 공격 반복 횟수.

    Returns:
        torch.Tensor: 생성된 적대적 예제.
    """
    # 공격 생성 중에는 모델의 가중치가 변하지 않아야 하므로 eval 모드로 설정
    model.eval()
    
    # 원본 데이터에 대한 복사본을 만들어 적대적 예제로 사용
    # requires_grad=True로 설정하여 입력에 대한 그래디언트를 계산할 수 있도록 함
    x_adv = x_clean.clone().detach().requires_grad_(True)

    # PGD 루프 시작
    for _ in range(num_iter):
        # 모델의 이전 그래디언트 초기화
        model.zero_grad()
        
        # 순전파: 현재 적대적 예제에 대한 모델의 예측 계산
        outputs, _, _ = model(x_adv, time_info)
        
        # 손실 계산
        loss = loss_fn(outputs, y_true, flow_lengths)
        
        # 역전파: 입력(x_adv)에 대한 손실의 그래디언트 계산
        loss.backward()

        # 그래디언트의 부호(sign)를 가져와 노이즈의 방향 결정
        grad_sign = x_adv.grad.sign()

        # 적대적 예제 업데이트 (그래디언트 방향으로 alpha만큼 이동)
        x_adv.data = x_adv.data + alpha * grad_sign

        # 프로젝션(Projection) 1: 엡실론(epsilon) 공 안에 머무르도록 조정
        # 원본 데이터로부터의 변화량(eta)이 epsilon을 넘지 않도록 clamp
        eta = torch.clamp(x_adv.data - x_clean.data, -epsilon, epsilon)
        x_adv.data = x_clean.data + eta

        # 프로젝션(Projection) 2: 데이터의 원래 범위(e.g., 0~1)를 벗어나지 않도록 조정
        x_adv.data = torch.clamp(x_adv.data, 0, 1)

    # 훈련에 사용하기 위해 모델을 다시 train 모드로 설정
    model.train()
    
    # 최종적으로 생성된 적대적 예제를 반환 (그래디언트 계산은 더 이상 필요 없음)
    return x_adv.detach()

class EDL_Loss(nn.Module):
    """
    Early Detection Loss (EDL), Focal Loss, 그리고 수동 클래스 가중치를 결합한 커스텀 손실 함수.
    """
    def __init__(self, use_focal_loss=True, gamma=2.0, edl_weight_factor=0.1, class_weights=None):
        """
        Args:
            use_focal_loss (bool): True이면 Focal Loss를 결합하고, False이면 일반 EDL만 사용합니다.
            gamma (float): Focal Loss의 focusing 파라미터.
            edl_weight_factor (float): EDL의 가중치 조절 인자.
            class_weights (torch.Tensor, optional): 클래스별 가중치 텐서. Defaults to None.
        """
        super(EDL_Loss, self).__init__()
        self.use_focal_loss = use_focal_loss
        self.gamma = gamma
        self.edl_weight_factor = edl_weight_factor
        
        # [수정] CrossEntropyLoss 초기화 시 weight 파라미터를 전달합니다.
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights, reduction='none')

    def forward(self, logits, labels, flow_lengths):
        """
        Args:
            logits (torch.Tensor): 모델의 예측 출력. Shape: (batch_size, num_classes)
            labels (torch.Tensor): 실제 정답 레이블. Shape: (batch_size,)
            flow_lengths (torch.Tensor): 각 샘플의 실제 플로우 길이. Shape: (batch_size,)
        
        Returns:
            torch.Tensor: 최종 배치 손실.
        """
        # 1. 표준 Cross-Entropy 손실(클래스 가중치 적용됨)과 EDL 가중치는 공통으로 계산
        ce_loss = self.cross_entropy(logits, labels)
        edl_weights = torch.exp(-self.edl_weight_factor * flow_lengths.float())

        if self.use_focal_loss:
            # 2a. Focal Loss 가중치 계산
            pt = F.softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).squeeze(1)
            focal_modulator = (1 - pt) ** self.gamma
            
            # 최종 손실: EDL 가중치 * Focal 가중치 * CE (클래스 가중치 이미 적용됨)
            combined_loss = edl_weights * focal_modulator * ce_loss
        else:
            # 2b. Focal Loss 없이 EDL 가중치만 적용
            pt = F.softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).squeeze(1)
            focal_modulator = (1 - pt) ** self.gamma
            loss1 = edl_weights * ce_loss
            loss2 = 0.8 * focal_modulator * ce_loss
            combined_loss = loss1 + loss2
            #combined_loss = edl_weights * ce_loss
        
        # 3. 배치 전체의 평균 손실 반환
        return combined_loss.mean()
    
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

def Earliness_score(
    logits_seq: torch.Tensor,   # [B, L, C]: 시점별 logit
    y_true: torch.Tensor,       # [B]: 정답 클래스
    T: torch.Tensor,            # [B]: 전체 패킷 수
    threshold: float = 0.99     # confidence threshold
):
    B, L, C = logits_seq.shape

    # Softmax 확률로 변환
    probs = F.softmax(logits_seq, dim=-1)        # [B, L, C]
    print('확률 :',probs)
    conf, pred = probs.max(dim=-1)               # 각 시점에서 가장 높은 확률값 및 인덱스

    # 조건: 정답 맞춤 & confidence >= threshold
    correct = pred == y_true.unsqueeze(1)        # [B, L]
    confident = conf >= threshold                # [B, L]
    valid = correct & confident                  # [B, L]

    # 초기 t: 못 찾은 경우 T로 설정
    #t = torch.full((B,), fill_value=T.max().item(), dtype=torch.long)
    t = torch.full((B,), fill_value=T.max().item(), dtype=torch.long, device=T.device)

    for i in range(B):
        idx = torch.nonzero(valid[i], as_tuple=False)
        if len(idx) > 0:
            t[i] = idx[0].item()  # 가장 빠른 시점의 index
    
    # Earliness 계산
    T = T.float()
    t = t.float()
    earliness_score = (T - t) / (T - 1)
    
    return earliness_score, t.to(torch.int32)

def ERDE(
    logits_seq: torch.Tensor,  # [B, L, C] — 시점별 logits
    y_true: torch.Tensor,      # [B] — 정답 레이블
    threshold: float = 0.99,   # confidence 임계값
    o: float = 5.0,            # ERDE 허용 지연
    fp_cost: float = None      # 잘못된 조기탐지(FP) 페널티
):
    """
    전체 배치에 대해 ERDE score를 계산
    - 조기 탐지 성공: sigmoid(d - o)
    - 조기 탐지 실패 후 정답 맞춤: sigmoid(T - o)
    - 조기 탐지 실패 + 오답: 1.0
    - 조기 탐지 했지만 오답: fp_cost (기본값: 1/N)
    """

    B, L, C = logits_seq.shape
    device = logits_seq.device

    probs = torch.softmax(logits_seq, dim=-1)  # [B, L, C]
    conf, pred = probs.max(dim=-1)             # [B, L], [B, L]

    # 정답 여부 + confidence 조건
    match = (pred == y_true.unsqueeze(1))      # [B, L]
    confident = (conf >= threshold)            # [B, L]
    valid = match & confident                  # [B, L]

    # 초기화
    delays = torch.full((B,), fill_value=L, dtype=torch.int32, device=device)
    y_pred_at_decision = torch.full((B,), fill_value=-1, dtype=torch.int32, device=device)
    detected = torch.zeros(B, dtype=torch.bool, device=device)

    # 조기 탐지 시점 및 예측 저장
    for i in range(B):
        indices = torch.nonzero(valid[i], as_tuple=False)
        if len(indices) > 0:
            t = indices[0].item()
            delays[i] = t
            y_pred_at_decision[i] = pred[i, t]
            detected[i] = True

    # 최종 시점 예측 (조기탐지 실패 시 fallback)
    y_pred_final = pred[:, -1]

    if fp_cost is None:
        fp_cost = 1.0 / B

    # ERDE 점수 초기화
    erde = torch.zeros(B, dtype=torch.float32, device=device)

    # === 조기 탐지 성공 ===
    tp = detected & (y_pred_at_decision == y_true)
    fp = detected & (y_pred_at_decision != y_true)

    erde[tp] = 1 / (1 + torch.exp(-(delays[tp].float() - o)))  # sigmoid(d - o)
    erde[fp] = fp_cost

    # === 조기 탐지 실패 ===
    fn_late = (~detected) & (y_pred_final == y_true)
    fn_end = (~detected) & (y_pred_final != y_true)

    erde[fn_late] = 1 / (1 + torch.exp(-torch.tensor(L - 1 - o, dtype=torch.float32, device=device)))  # sigmoid(T - o)
    erde[fn_end] = 1.0

    return erde.mean()

def Latency(
    logits,
    y_true,
    attack_class_indices=[1, 2, 3, 4, 5], # 0은 Benign이라 제외
    confidence_threshold=0.99,
    fixed_p=0.1,
    fail_mode="max"
):
    """
    logits → softmax → decision point 추출 → flow 단위 y_pred → F-latency 계산 

    Args:
        logits: Tensor [B, L, C] - 모델 출력 logits
        y_true: torch.Tensor [B] - flow 단위 실제 라벨
        attack_class_indices: list[int], 공격 클래스 인덱스 리스트
        confidence_threshold: decision point를 결정할 확률 기준
        fixed_p: penalty 기울기 값 (None이면 median 기반 계산)
        fail_mode: 'max' → 미탐지를 최대 penalty로, 'exclude' → 제외

    Returns:
        f_latency: float, F-latency 점수
        y_pred: np.ndarray, flow 단위 예측 클래스
        decision_points: list[int|None]
    """
    # 1. Tensor → numpy 변환 (y_true만 먼저 변환)
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.cpu().numpy()
    else:
        y_true_np = np.array(y_true)

    # 2. Softmax로 확률 변환
    probs = F.softmax(logits, dim=-1)  # [B, L, C]

    # 3. Decision point와 예측 클래스 추출 (벡터화)
    B, L, C = probs.shape
    max_probs, max_classes = torch.max(probs, dim=-1)          # [B, L]
    decision_mask = max_probs >= confidence_threshold          # [B, L] bool
    has_decision = decision_mask.any(dim=1)                     # [B] bool
    first_true_idx = torch.argmax(decision_mask.int(), dim=1)   # [B]

    decision_points = []
    y_pred_np = np.zeros(B, dtype=np.int64)

    for i in range(B):
        if has_decision[i]:  # threshold 넘은 시점이 존재
            dp_idx = first_true_idx[i].item()
            decision_points.append(dp_idx + 1)  # 1-based indexing
            y_pred_np[i] = max_classes[i, dp_idx].item()
        else:  # 한 번도 threshold를 넘지 못함 → Benign
            decision_points.append(None)
            y_pred_np[i] = 0

    y_pred = y_pred_np

    # 4. Macro F1 계산
    f1 = f1_score(y_true_np, y_pred, average='macro', zero_division=0)

    # 5. TP 케이스 decision_point 수집
    tp_decision_points = []
    
    # Benign(0)) 제외 & 공격 클래스 예측 & 조건(공격유형) 정확히 일치 시 TP로 취급
    for yt, yp, dp in zip(y_true_np, y_pred, decision_points):
        if yt > 0 and yp > 0 and yt == yp:
            if dp is None:
                if fail_mode == "max":
                    tp_decision_points.append(float('inf'))
                elif fail_mode == "exclude":
                    continue
            else:
                tp_decision_points.append(max(1, dp))

    if not tp_decision_points:
        return 0.0, y_pred, decision_points  # TP 없음

    # 6. p 값 계산 (논문 방식)
    if fixed_p is None:
        valid_points = [dp for dp in tp_decision_points if dp != float('inf')]
        if valid_points:  # 빈 리스트 방지
            median_dp = np.median(valid_points)
            p = np.log(3) / (median_dp - 1) if median_dp > 1 else 0.1
        else:
            p = 0.1 #기본값
    else:
        p = fixed_p

    # 7. Penalty 계산
    penalties = []
    for dp in tp_decision_points:
        if dp == float('inf'):
            penalties.append(1.0)
        else:
            penalty = -1 + (2 / (1 + np.exp(-p * (dp - 1))))
            penalties.append(max(0.0, penalty))

    median_penalty = np.median(penalties)

    # 8. F-latency 계산
    f_latency = max(0.0, f1 * (1 - median_penalty))

    return f_latency, y_pred, decision_points


#============Tap
def compute_tap_from_softmax(
    softmax_seqs: np.ndarray,      # [N, T, C]
    y_true: np.ndarray,            # [N]
    class_names: list,             # 클래스 이름 리스트
    excluded_classes=None,         # 제외할 클래스 이름 리스트
    threshold=0.9,                 # 탐지 confidence 임계값
    o=2,                           # 허용 지연 시점
    lamb=1.0                       # 패널티 함수 경사 λ
):
    def penalty_fn(k, o, lamb):
        """논문 기반 penalty 함수"""
        return 2 * (-1 + 2 / (1 + np.exp(-lamb * (k - o))))

    excluded_classes = excluded_classes or []
    excluded_indices = {class_names.index(c) for c in excluded_classes if c in class_names}

    N = len(y_true)
    tap_vals = []
    delays = []
    outcomes = []

    for i in range(N):
        true_label = y_true[i]
        if true_label in excluded_indices:
            continue  # 제외된 클래스 건너뜀

        probs_seq = softmax_seqs[i]  # shape: [T, C]
        k = None
        pred_class = None

        for t, probs in enumerate(probs_seq):
            top = np.argmax(probs)
            confidence = probs[top]
            if confidence >= threshold:
                pred_class = top
                k = t + 1  # 시간 step 1부터 시작한다고 가정
                break

        if pred_class is None:
            tap_vals.append(0.0)
            outcomes.append("delay")
            delays.append(None)

        elif pred_class != true_label:
            tap_vals.append(-1.0)
            outcomes.append("FP/FN")
            delays.append(k)

        else:
            if k <= o:
                tap_vals.append(1.0)
                outcomes.append("TP (early)")
                delays.append(k)
            else:
                penalty = penalty_fn(k, o, lamb)
                score = max(0.0, 1.0 - penalty)  # 음수 방지
                tap_vals.append(score)
                outcomes.append("TP (late)")
                delays.append(k)

    tap_vals = np.asarray(tap_vals, dtype=float)
    tap_mean = float(tap_vals.mean()) if len(tap_vals) > 0 else float("nan")

    return tap_mean, tap_vals, outcomes, delays

import numpy as np
from sklearn.metrics import f1_score

def calculate_earliness(t: np.ndarray, T: np.ndarray, true_labels: np.ndarray, final_preds: np.ndarray) -> float:
    """ TP 케이스에 대한 평균 Earliness Score를 계산합니다. """
    t, T = np.array(t, dtype=float), np.array(T, dtype=float)
    # 레이블 0을 정상(Benign)으로 가정하고 제외
    tp_mask = (true_labels == final_preds) & (true_labels != 0)
    
    if not np.any(tp_mask):
        return 0.0

    # T-1이 0이 되는 경우 방지
    denominator = T[tp_mask] - 1
    denominator[denominator <= 0] = 1 # 분모가 0 또는 음수면 1로 처리

    earliness_scores = (T[tp_mask] - t[tp_mask]) / denominator
    return np.mean(earliness_scores) if len(earliness_scores) > 0 else 0.0

def calculate_erde(true_labels: np.ndarray, final_preds: np.ndarray, t: np.ndarray, T: np.ndarray, o: float) -> float:
    """ ERDE 점수를 계산합니다. """
    erde_scores = []
    for i in range(len(true_labels)):
        is_correct = (true_labels[i] == final_preds[i])
        
        if is_correct:
            if true_labels[i] == 0:  # TN
                erde_scores.append(0.0)
            else:  # TP
                # 오류 점수는 늦을수록 1에 가까워져야 함
                penalty = 1.0 - (1.0 / (1.0 + np.exp(t[i] - o)))
                erde_scores.append(penalty)
        else: # Incorrect
            erde_scores.append(1.0) # FP와 FN 모두 최대 오류 1.0으로 처리

    return np.mean(erde_scores) if len(erde_scores) > 0 else 0.0

def calculate_f_latency(logits_seqs: list, y_true: np.ndarray, threshold: float = 0.99, p: float = 0.1) -> float:
    """ F1-Latency 점수를 계산합니다. """
    y_true_np = np.array(y_true)
    B = len(y_true_np)
    y_pred_np = np.zeros(B, dtype=np.int64)
    decision_points = []

    # 각 플로우에 대한 최종 예측 및 탐지 시점 결정
    for i in range(B):
        probs_seq = torch.softmax(logits_seqs[i], dim=-1)
        max_probs, max_classes = torch.max(probs_seq, dim=-1)
        decision_mask = max_probs >= threshold
        
        if torch.any(decision_mask):
            dp_idx = torch.nonzero(decision_mask, as_tuple=False)[0].item()
            decision_points.append(dp_idx + 1)
            y_pred_np[i] = max_classes[dp_idx].item()
        else:
            decision_points.append(None)
            y_pred_np[i] = max_classes[-1].item() # 마지막 시점의 예측 사용

    f1 = f1_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
    
    tp_decision_points = []
    for yt, yp, dp in zip(y_true_np, y_pred_np, decision_points):
        if yt > 0 and yt == yp and dp is not None:
            tp_decision_points.append(dp)

    if not tp_decision_points:
        return 0.0

    penalties = [-1 + (2 / (1 + np.exp(-p * (dp - 1)))) for dp in tp_decision_points]
    median_penalty = np.median(penalties)
    
    return max(0.0, f1 * (1 - median_penalty))

def calculate_tap(true_labels: np.ndarray, final_preds: np.ndarray, t: np.ndarray, o: float, lambda_param: float) -> float:
    """ TaP 점수를 계산합니다. """
    tap_scores = []
    for i in range(len(true_labels)):
        is_correct = (true_labels[i] == final_preds[i])
        
        if is_correct:
            if t[i] <= o:
                tap_scores.append(1.0)
            else:
                penalty = 2.0 * (-1.0 + 2.0 / (1.0 + np.exp(-lambda_param * (t[i] - o))))
                tap_scores.append(1.0 - penalty) # max(0.0, ...) 제거
        else:
            tap_scores.append(-1.0)
            
    # 'delay' 케이스는 이 로직에서 발생하지 않음 (모든 플로우는 최종 결정됨)
    return np.mean(tap_scores) if len(tap_scores) > 0 else 0.0

# ====================================================================
# 사용 예시
# ====================================================================
if __name__ == '__main__':
    # 이 예시는 전처리된 .npz 파일이 있다고 가정합니다.
    # '/path/to/your/npz_files'를 실제 경로로 변경하세요.
    NPZ_BASE_PATH = './' 
    # 예시로 사용할 파일 이름을 지정하세요.
    EXAMPLE_NPZ_FILE = 'example.npz' 
    npz_path = os.path.join(NPZ_BASE_PATH, EXAMPLE_NPZ_FILE)

    # 더미 npz 파일 생성
    if not os.path.exists(npz_path):
        print(f"'{npz_path}'를 찾을 수 없어 더미 파일을 생성합니다.")
        dummy_x = np.random.rand(10, 40, 448) # 10개 샘플
        dummy_t = np.sort(np.random.rand(10, 40) * 100, axis=1)
        dummy_t[:, 30:] = -1 # 30개 패킷을 가진 플로우로 가정
        dummy_y = np.random.randint(0, 5, size=10)
        np.savez_compressed(npz_path, X=dummy_x, T_relative=dummy_t, y=dummy_y)

    # NPZ 파일 로드
    data = np.load(npz_path)
    X_data = data['X']
    T_relative_data = data['T_relative']
    y_data = data['y']

    print(f"총 {len(X_data)}개의 샘플을 로드했습니다.")
    
    # 첫 번째 샘플을 가져와 증강 적용
    sample_idx = 0
    original_x = X_data[sample_idx]
    original_t = T_relative_data[sample_idx]

    print("\n[적용 전 원본 데이터]")
    print(f"X shape: {original_x.shape}")
    print(f"T_relative (앞 5개): {original_t[:5]}")
    
    # 증강 함수 호출
    augmented_x, augmented_t = augment_flow(original_x, original_t)
    
    print("\n[적용 후 증강된 데이터]")
    print(f"X shape: {augmented_x.shape}")
    print(f"T_relative (앞 5개): {augmented_t[:5]}")

    # 시간 값이 변경되었는지 확인 (Jitter 또는 Scaling으로 인해)
    time_changed = not np.allclose(original_t[:10], augmented_t[:10])
    print(f"\n시간 정보 변경 여부: {time_changed}")