import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.layers import to_2tuple
import numpy as np

from .KAN import KAN

# ==============================================================================
# Dynamic_Sinusoidal_Positional_Encoding 클래스는 변경 없이 그대로 사용합니다.
# ==============================================================================
class Dynamic_Sinusoidal_Positional_Encoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(position * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, x: torch.Tensor, time_info: torch.Tensor) -> torch.Tensor:
        time_info = time_info.unsqueeze(-1)
        angle = time_info * self.div_term
        pe = torch.zeros_like(x)
        pe[:, :, 0::2] = torch.sin(angle)
        pe[:, :, 1::2] = torch.cos(angle)
        x = x + pe
        return self.dropout(x)

# ---

class Student_Encoder_Layer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, 
                 use_linear_attention: bool = False, use_kan: bool = False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == self.d_model, "d_model must be divisible by nhead"
        self.use_linear_attention = use_linear_attention

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if use_kan:
            # ### 수정된 부분: 새로운 KAN 생성 방식(layers_hidden 리스트)으로 FFN을 대체합니다.
            self.feed_forward = KAN(
                layers_hidden=[d_model, dim_feedforward, d_model],
                base_activation=torch.nn.SiLU # 새로운 KAN의 기본 활성화 함수
            )
        else:
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model)
            )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> tuple:
        # forward 메서드는 변경할 필요가 없습니다.
        src_norm = self.norm1(src)
        q = self.q_proj(src_norm)
        k = self.k_proj(src_norm)
        v = self.v_proj(src_norm)
        
        batch_size, seq_len, _ = src.shape
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        attn_probs = None
        
        if self.use_linear_attention:
            phi_q = F.elu(q) + 1
            phi_k = F.elu(k) + 1
            if src_key_padding_mask is not None:
                # unsqueeze(-1)은 head 차원, unsqueeze(1)은 feature 차원에 해당합니다.
                # 마스크 shape: [B, S] -> [B, 1, S, 1] 이 되어야 합니다.
                # 현재 코드: [B, 1, 1, S] -> [B, H, S, D] 브로드캐스팅 시 오류 가능성 있음
                # 더 명확한 마스킹을 위해 수정합니다.
                mask = src_key_padding_mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.nhead, -1, self.head_dim)
                phi_k = phi_k.masked_fill(mask, 0)

            # K^T V 계산 (효율적)
            kv_context = torch.matmul(phi_k.transpose(-2, -1), v)

            # --- [수정 시작] 안정적인 정규화 ---
            # 1. phi_k의 합을 먼저 계산합니다.
            normalizer = torch.sum(phi_k, dim=-2, keepdim=True)
            
            # 2. clamp를 사용하여 normalizer가 매우 작은 양수(epsilon) 값을 갖도록 보장합니다.
            #    이렇게 하면 0으로 나누는 것을 방지할 수 있습니다.
            #    1e-8은 PyTorch 내부에서도 안정성을 위해 자주 사용되는 작은 값입니다.
            safe_normalizer = normalizer.clamp(min=1e-8)
            
            # 최종 계산
            attn_output = torch.matmul(phi_q, kv_context) / safe_normalizer
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if src_key_padding_mask is not None:
                mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
                attn_scores = attn_scores.masked_fill(mask, -1e4) # -1e9
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        src = src + self.dropout(attn_output)
        
        src_norm2 = self.norm2(src)
        ff_output = self.feed_forward(src_norm2)
        
        src = src + self.dropout(ff_output)
        
        return src, attn_probs

# ---

class Student_model(nn.Module):
    def __init__(self, 
                 input_feature_dim: int, 
                 num_classes: int, 
                 seq_length: int,
                 dropout: float,
                 architecture: str = 'transformer',
                 # Transformer 전용 인자
                 d_model: int = 32, 
                 nhead: int = 4, 
                 num_encoder_layers: int = 4, 
                 dim_feedforward: int = 512, # Transformer의 FFN을 위한 인자
                 use_linear_attention: bool = False,
                 use_kan: bool = False,
                 # MLP & KAN 전용 인자
                 mlp_kan_hidden_dim: int = 32, 
                 # KAN 분류기를 위한 인자
                 classifier_hidden_dim: int = 128,
                 weight_init: str = 'base'):
        
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes

        if architecture == 'transformer':
            self.d_model = d_model
            self.input_linear = nn.Linear(input_feature_dim, d_model)
            self.pos_encoder = Dynamic_Sinusoidal_Positional_Encoding(d_model, dropout)
            self.encoder_layers = nn.ModuleList(
                [Student_Encoder_Layer(d_model, nhead, dim_feedforward, dropout, use_linear_attention, use_kan) 
                 for _ in range(num_encoder_layers)]
            )
            self.final_norm = nn.LayerNorm(d_model)
            self.classifier = nn.Linear(d_model, num_classes)

        elif architecture == 'mlp':
            h = mlp_kan_hidden_dim
            # Flatten()을 제거하고, nn.Linear가 input_feature_dim에서 직접 시작
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_feature_dim, h), # 1번째 Dense Layer
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h, h),                 # 2번째 Dense Layer
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.classifier = nn.Linear(h, num_classes)

        elif architecture == 'kan':
            h = mlp_kan_hidden_dim
            # 하나의 KAN 모듈이 input_feature_dim에서 시작하는 3-레이어 구조를 갖도록 정의
            self.kan_model = KAN(
                layers_hidden=[input_feature_dim, h, h, num_classes]
            )
            
        else:
            raise ValueError(f"Unsupported architecture: '{architecture}'. Choose from 'transformer', 'mlp', 'kan'.")

        if weight_init != 'skip':
            self.init_weights(weight_init)
            
    def init_weights(self, mode: str = ''):
        # ### 수정된 부분: KAN은 자체 초기화를 사용하므로, 다른 레이어만 초기화합니다.
        init_fn = get_init_weights_vit(mode)
        named_apply(init_fn, self)

    def forward(self, src: torch.Tensor, time_info: torch.Tensor, src_mask: torch.Tensor = None) -> tuple:
        if self.architecture == 'transformer':
            projected_src = self.input_linear(src) * math.sqrt(self.d_model)
            pos_encoded_src = self.pos_encoder(projected_src, time_info)
            output = pos_encoded_src
            attention_maps, hidden_states = [], []
            for layer in self.encoder_layers:
                hidden_states.append(output)
                output, attn_probs = layer(output, src_key_padding_mask=src_mask)
                if attn_probs is not None:
                    attention_maps.append(attn_probs)
            output = self.final_norm(output)
            hidden_states.append(output)
            cls_output = output[:, 0, :]
            logits = self.classifier(cls_output)
            return logits, attention_maps, hidden_states

        elif self.architecture == 'mlp':
            # src: [Batch, Seq, Dim]
            features = self.feature_extractor(src) # out: [Batch, Seq, h]
            
            # 1. 시퀀스 정보 통합 (Aggregation) - 평균 풀링 사용
            aggregated_features = torch.mean(features, dim=1) # out: [Batch, h]
            
            logits = self.classifier(aggregated_features)
            return logits, [], [aggregated_features]
        
        elif self.architecture == 'kan':
            # src: [Batch, Seq, in_dim]
            batch_size, seq_len, in_dim = src.shape
            
            # KAN의 2D 입력 요구사항에 맞게 [Batch * Seq, in_dim] 형태로 변환
            x = src.view(-1, in_dim)
            
            hidden_states = []
            num_internal_layers = len(self.kan_model.layers)
            
            # 통합된 kan_model의 내부 레이어를 순회하며 중간 은닉 상태 추출
            for i, internal_layer in enumerate(self.kan_model.layers):
                x = internal_layer(x)
                # 최종 로짓이 아닌, 중간 레이어의 출력만 hidden_states로 저장
                if i < num_internal_layers - 1:
                    # [Batch * Seq, h] -> [Batch, Seq, h] 형태로 복원하여 저장
                    hidden_state_seq = x.view(batch_size, seq_len, -1)
                    hidden_states.append(hidden_state_seq)
            
            logits_flat = x # 마지막 레이어의 출력이 각 시점의 로짓

            # [Batch * Seq, num_classes] -> [Batch, Seq, num_classes] 형태로 복원
            logits_sequence = logits_flat.view(batch_size, seq_len, -1)
            
            # 시퀀스 정보 통합 (Aggregation) - 평균 풀링으로 최종 로짓 계산
            final_logits = torch.mean(logits_sequence, dim=1)
            
            return final_logits, [], hidden_states
    
def named_apply(fn, module, name='', depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn, child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

def init_weights_vit_base(module: nn.Module, name: str = ''):

    # KAN 모듈인 경우 초기화를 건너뛰고 자체 초기화에 맡깁니다.
    if isinstance(module, KAN):
        return

    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def get_init_weights_vit(mode: str = 'base') -> Callable:
    """mode에 맞는 초기화 함수를 반환하는 Factory 함수"""
    if mode == 'base':
        return init_weights_vit_base
    else:
        # 다른 초기화 모드가 필요하다면 여기에 추가
        return init_weights_vit_base

if __name__ == "__main__":
    # --- 1. 테스트용 공통 하이퍼파라미터 ---
    common_params = {
        'input_feature_dim': 514,
        'num_classes': 5,
        'seq_length': 67,
        'dropout': 0.3,
    }

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("--- Student 모델 아키텍처별 파라미터 수 비교 ---")
    
    # --- 2. Transformer 모델 테스트 ---
    transformer_params = {
        'd_model': 128, 
        'nhead': 8, 
        'num_encoder_layers': 2, 
        'dim_feedforward': 512
    }
    
    # 기본 Transformer (FFN = MLP)
    student_transformer_mlp_ffn = Student_model(
        **common_params, 
        architecture='transformer', 
        use_kan=False, 
        **transformer_params
    )
    print(f"Transformer (FFN: MLP) 파라미터 수: {count_parameters(student_transformer_mlp_ffn):,}")
    
    transformer_params2 = {
        'd_model': 128, 
        'nhead': 8, 
        'num_encoder_layers': 2, 
        'dim_feedforward': 64
    }

    # use_kan=True인 Transformer (FFN = KAN)
    student_transformer_kan_ffn = Student_model(
        **common_params, 
        architecture='transformer', 
        use_kan=True, 
        **transformer_params2
    )
    print(f"Transformer (FFN: KAN) 파라미터 수: {count_parameters(student_transformer_kan_ffn):,}")
    ### ------------------ ###

    # --- 3. MLP 모델 테스트 ---
    mlp_params = { 'mlp_kan_hidden_dim': 128 }
    student_mlp = Student_model(**common_params, architecture='mlp', **mlp_params)
    print(f"MLP 파라미터 수: {count_parameters(student_mlp):,}")

    # --- 4. KAN 모델 테스트 ---
    kan_params = { 'mlp_kan_hidden_dim': 16 }
    student_kan = Student_model(**common_params, architecture='kan', **kan_params)
    print(f"KAN 파라미터 수: {count_parameters(student_kan):,}")
