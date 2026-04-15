import torch
import torch.nn as nn

class SimpleDNN(nn.Module):
    def __init__(self, input_dim=448, hidden_dim=256, output_dim=10):
        super(SimpleDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape
        x = x.view(-1, input_dim)          # (batch * seq_len, input_dim)
        x = self.net(x)                    # (batch * seq_len, output_dim)
        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, output_dim)
        return x

import torch
import torch.nn as nn
import ai_edge_torch

# 🧠 Step 1: Define DNN
class SimpleDNN(nn.Module):
    def __init__(self, input_dim=448, hidden_dim=256, output_dim=10):
        super(SimpleDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        x = x.view(-1, input_dim)              # (B * L, D)
        x = self.net(x)                        # (B * L, output_dim)
        x = x.view(batch_size, seq_len, -1)    # (B, L, output_dim)
        return x

# ✅ Step 2: Instantiate model
model = SimpleDNN()
model.eval()  # ⚠️ 반드시 eval() 모드로 전환

# ✅ Step 3: Sample input for tracing
# LiteRT는 실제 형태로 tracing 해야 함 (배치 1 기준)
sample_input = (torch.randn(1, 40, 448),)

# ✅ Step 4: Convert to LiteRT format (.tflite)
edge_model = ai_edge_torch.convert(model, sample_input)

# ✅ Step 5: Export to TFLite flatbuffer
edge_model.export("./simple_dnn.tflite")

print("✅ 모델 변환 완료: simple_dnn.tflite 생성됨")
