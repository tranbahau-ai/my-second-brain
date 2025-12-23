
  

## Giới thiệu
Feed-Forward Networks (FFN) là một thành phần quan trọng trong kiến trúc Transformer và GPT. FFN được sử dụng sau mỗi Multi-Head Attention layer trong mỗi Transformer block.

  

## Vai trò của FFN trong Transformer

FFN có hai vai trò chính:  
1. **Tăng khả năng biểu diễn (Representation Learning)**: FFN giúp mô hình học các biểu diễn phức tạp hơn từ đầu ra của attention mechanism
2. **Xử lý từng position độc lập**: Khác với attention layer xử lý mối quan hệ giữa các token, FFN xử lý mỗi position (token) một cách độc lập
## Kiến trúc FFN  

```

Input (d_model) → Linear Layer 1 (d_model → d_ff) → Activation (GELU) → Linear Layer 2 (d_ff → d_model) → Output

```

Trong đó:
- `d_model`: Dimension của embedding (ví dụ: 768 cho GPT-2 small)
- `d_ff`: Dimension của hidden layer, thường lớn hơn d_model (ví dụ: 3072 = 4 × 768)
### Công thức toán học
```

FFN(x) = W₂ · GELU(W₁ · x + b₁) + b₂

```

Trong đó:
- W₁: Ma trận trọng số layer 1 (d_model × d_ff)
- W₂: Ma trận trọng số layer 2 (d_ff × d_model)
- b₁, b₂: Bias vectors
- GELU: Gaussian Error Linear Unit activation function
## Ví dụ minh họa với code

### Ví dụ 1: Implement FFN cơ bản

```python

import torch

import torch.nn as nn

  

class FeedForward(nn.Module):

def __init__(self, cfg):

"""

cfg: Configuration dict chứa:

- emb_dim: embedding dimension (d_model)

- ff_dim: feedforward dimension (d_ff) - thường là 4 * emb_dim

"""

super().__init__()

self.layers = nn.Sequential(

nn.Linear(cfg["emb_dim"], cfg["ff_dim"]),

nn.GELU(), # Activation function

nn.Linear(cfg["ff_dim"], cfg["emb_dim"])

)

  

def forward(self, x):

return self.layers(x)

  

# Ví dụ sử dụng

cfg = {

"emb_dim": 768, # Giống GPT-2 small

"ff_dim": 3072 # 4 × 768

}

ffn = FeedForward(cfg)

# Input: (batch_size, seq_len, emb_dim)

batch_size = 2
seq_len = 10
x = torch.randn(batch_size, seq_len, cfg["emb_dim"])

# Forward pass

output = ffn(x)

print(f"Input shape: {x.shape}") # torch.Size([2, 10, 768])
print(f"Output shape: {output.shape}") # torch.Size([2, 10, 768])

```

  

**Giải thích:**

- Input có shape `(2, 10, 768)`: 2 câu, mỗi câu 10 tokens, mỗi token được biểu diễn bằng vector 768 chiều

- FFN xử lý từng token độc lập, giữ nguyên shape đầu ra

- Layer 1 mở rộng từ 768 → 3072 chiều (tăng capacity)

- GELU thêm tính phi tuyến

- Layer 2 thu hẹp về 768 chiều (về dimension ban đầu)


Nói cách khác, mỗi token tiếp theo được sinh ra theo quy luật xác suất có điều kiện:
$$
P(w_{t+1} \mid w_1, \dots, w_t) = \text{softmax}(\text{logits}(w_{t+1}))
$$


### Ví dụ 2: FFN với Dropout (theo GPT-2)

  

```python

class FeedForwardWithDropout(nn.Module):

def __init__(self, cfg):

super().__init__()

self.fc1 = nn.Linear(cfg["emb_dim"], cfg["ff_dim"])

self.gelu = nn.GELU()

self.fc2 = nn.Linear(cfg["ff_dim"], cfg["emb_dim"])

self.dropout = nn.Dropout(cfg["drop_rate"])

  

def forward(self, x):

x = self.fc1(x)

x = self.gelu(x)

x = self.fc2(x)

x = self.dropout(x)

return x

  

# Configuration giống GPT-2 small

cfg = {

"emb_dim": 768,

"ff_dim": 3072,

"drop_rate": 0.1

}

  

ffn = FeedForwardWithDropout(cfg)

  

# Test

x = torch.randn(2, 10, 768)

output = ffn(x)

print(f"Output shape: {output.shape}") # torch.Size([2, 10, 768])

```

  

### Ví dụ 3: Trực quan hóa quá trình transformation

  

```python

import torch

import torch.nn as nn

  

# Tạo một FFN đơn giản để dễ visualize

cfg = {

"emb_dim": 4,

"ff_dim": 8

}

  

ffn = FeedForward(cfg)

  

# Input đơn giản: 1 câu, 3 tokens

x = torch.tensor([

[[1.0, 0.5, -0.3, 0.2], # Token 1

[0.8, -0.2, 0.6, -0.1], # Token 2

[-0.5, 0.9, 0.3, 0.7]] # Token 3

])

  

print("Input shape:", x.shape) # torch.Size([1, 3, 4])

print("Input:\n", x)

  

# Forward pass với intermediate outputs

with torch.no_grad():

# Step 1: Linear expansion (4 -> 8)

after_fc1 = ffn.layers[0](x)

print(f"\nAfter FC1 (4→8): {after_fc1.shape}")

print(after_fc1)

  

# Step 2: GELU activation

after_gelu = ffn.layers[1](after_fc1)

print(f"\nAfter GELU: {after_gelu.shape}")

print(after_gelu)

  

# Step 3: Linear projection back (8 -> 4)

after_fc2 = ffn.layers[2](after_gelu)

print(f"\nAfter FC2 (8→4): {after_fc2.shape}")

print(after_fc2)

```

  

**Output mẫu:**

```

Input shape: torch.Size([1, 3, 4])

After FC1 (4→8): torch.Size([1, 3, 8])

After GELU: torch.Size([1, 3, 8])

After FC2 (8→4): torch.Size([1, 3, 4])

```

  

### Ví dụ 4: FFN trong context của Transformer Block

  

```python

class TransformerBlock(nn.Module):

def __init__(self, cfg):

super().__init__()

# Multi-Head Attention (giả định đã implement)

self.attn = MultiHeadAttention(cfg)

self.ffn = FeedForward(cfg)

  

# Layer Normalization

self.ln1 = nn.LayerNorm(cfg["emb_dim"])

self.ln2 = nn.LayerNorm(cfg["emb_dim"])

  

# Dropout

self.drop = nn.Dropout(cfg["drop_rate"])

  

def forward(self, x):

# Attention block với residual connection

attn_out = self.attn(x)

x = x + self.drop(attn_out) # Residual connection

x = self.ln1(x) # Post-norm

  

# FFN block với residual connection

ffn_out = self.ffn(x)

x = x + self.drop(ffn_out) # Residual connection

x = self.ln2(x) # Post-norm

  

return x

```

  

**Giải thích luồng dữ liệu:**

```

Input → [Attention + Residual + LayerNorm] → [FFN + Residual + LayerNorm] → Output

```

  

## Tại sao FFN dimension lớn hơn embedding dimension?

  

Theo Sebastian Raschka, có 3 lý do chính:

  

1. **Tăng capacity**: Hidden dimension lớn hơn (thường 4×) cho phép mô hình học các transformation phức tạp hơn
2. **Bottleneck architecture**: Mở rộng rồi thu hẹp tạo ra information bottleneck, giúp mô hình học các feature quan trọng nhất
3. **Thực nghiệm cho thấy hiệu quả**: GPT-2, GPT-3 đều dùng tỷ lệ 4:1 và đạt kết quả tốt

  
### Ví dụ so sánh các kích thước khác nhau

```python

def count_parameters(emb_dim, ff_dim):

"""Tính số parameters trong FFN"""

# W1: emb_dim × ff_dim, b1: ff_dim

# W2: ff_dim × emb_dim, b2: emb_dim

params = (emb_dim * ff_dim + ff_dim) + (ff_dim * emb_dim + emb_dim)

return params

  

# So sánh các configurations

configs = [

(768, 768), # 1:1 ratio

(768, 1536), # 2:1 ratio

(768, 3072), # 4:1 ratio (GPT-2 standard)

(768, 4096), # ~5:1 ratio

]

  

print("emb_dim | ff_dim | ratio | parameters")

print("-" * 45)

for emb_dim, ff_dim in configs:

ratio = ff_dim / emb_dim

params = count_parameters(emb_dim, ff_dim)

print(f"{emb_dim:7} | {ff_dim:6} | {ratio:5.1f} | {params:,}")

```

  

**Output:**

```

emb_dim | ff_dim | ratio | parameters

---------------------------------------------

768 | 768 | 1.0 | 1,181,952

768 | 1536 | 2.0 | 2,363,904

768 | 3072 | 4.0 | 4,727,808 ← GPT-2 standard

768 | 4096 | 5.3 | 6,297,600

```

  

## GELU Activation Function


GPT sử dụng GELU thay vì ReLU truyền thống. GELU được định nghĩa:

```

GELU(x) = x · Φ(x)

```
  
Trong đó Φ(x) là cumulative distribution function của Gaussian distribution.
  
### So sánh GELU vs ReLU

```python

import torch

import matplotlib.pyplot as plt

import numpy as np

  

# Tạo input range

x = torch.linspace(-3, 3, 100)

  

# GELU

gelu = nn.GELU()

y_gelu = gelu(x)

  

# ReLU
relu = nn.ReLU()

y_relu = relu(x)

# Visualize

plt.figure(figsize=(10, 5))

plt.plot(x.numpy(), y_gelu.numpy(), label='GELU', linewidth=2)

plt.plot(x.numpy(), y_relu.numpy(), label='ReLU', linewidth=2)

plt.grid(True, alpha=0.3)

plt.xlabel('x')

plt.ylabel('Activation')

plt.title('GELU vs ReLU')

plt.legend()

plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.show()

  

# Test với data thực tế

x_test = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

print("x | ReLU | GELU")

print("-" * 30)

for val in x_test:

print(f"{val:6.1f} | {relu(val):5.2f} | {gelu(val):5.2f}")

```

  

**Ưu điểm của GELU:**

- Smooth gradient, không có hard cutoff tại 0 như ReLU
- Cho phép một số giá trị âm pass through (với weight nhỏ)
- Thực nghiệm cho thấy tốt hơn ReLU cho các mô hình Transformer


## Tổng kết

Feed-Forward Network trong GPT:
- **Cấu trúc**: Two-layer MLP với expansion ratio 4:1
- **Activation**: GELU
- **Vai trò**: Xử lý từng token độc lập sau attention layer
- **Integration**: Sử dụng residual connection và layer normalization

FFN chiếm phần lớn parameters trong Transformer block và là thành phần quan trọng để mô hình học các transformation phi tuyến phức tạp.


## Tham khảo
- Sebastian Raschka, "Build a Large Language Model (From Scratch)", Chapter 4.1: Coding an LLM architecture
- Original Transformer paper: "Attention Is All You Need" (Vaswani et al., 2017)
- GPT-2 paper: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)