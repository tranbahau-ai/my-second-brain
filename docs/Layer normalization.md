
## LayerNorm

Là kĩ thuật chuẩn hóa (normalization) giúp model học ổn định hơn và tránh các giá trị activations "scale-out" hay "scale-in" quá mức khi đi qua nhiều layers trong transformer.

#### In GPT Architecture
Layer normalization is typically applied:
- Before the attention mechanism
- Before the feed-forward network
- This is called "Pre-LN" (Pre-Layer Normalization)

### How it working?
LayerNorm lấy **toàn bộ vector embedding của một token**
1. Tính trung bình
2. Tính độ lệch chuẩn 
3. Chuẩn hóa toàn bộ vector theo công thức
4. Áp dụng hai tham số trainable: **γ (learnable scale)** và **β (learnable shift)**

### Công thức
1- **Calculate the mean** (tính trung bình):
Ta có input là vector $x = [x_1, x_2, \dots, x_d]$ với d là dimesion của vector
 $$ \text{mean} = \mu = \frac{1}{d} \displaystyle\sum_{i=1}^{d} x_i$$

2- **Calculate the variance** (độ lệch chuẩn):
  $$\text{variance} = \sigma^2 =\frac{1}{d} \displaystyle\sum_{i=1}^{d} (x_i - \mu)^2$$

3- **Chuẩn hóa cho vector (Normalization)**

$$LayerNorm(x) = \gamma \hat{x_i} + \beta = \gamma \frac{x_i-\mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$
**γ** và **β** là trainable parameters, giúp mô hình linh hoạt hơn, không bị “ràng buộc cứng” vào phân phối chuẩn hóa, mà vẫn có thể điều chỉnh scale và shift của chuẩn hóa.

## Code Example
```python
import torch  
import torch.nn as nn

class LayerNorm(nn.Module):  
    def __init__(self, dim, eps=1e-4):
        super().__init__()  
        self.eps = eps  
        self.weight = nn.Parameter(torch.ones(dim))  
        self.bias = nn.Parameter(torch.zeros(dim))  
  
    def forward(self, x):  
        mean = x.mean(dim=-1, keepdim=True)  
        var = x.var(dim=-1, keepdim=True, unbiased=False)  
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  
        return x_norm * self.weight + self.bias  
  
# Example usage  
x = torch.tensor([[[2.0, 3.0, 5.0, 6.0]]])  
layer_norm = LayerNorm(4)  
output = layer_norm(x)  
  
print(output)  # tensor([[[-1.2649, -0.6324,  0.6324,  1.2649]]])
```

### In GPT Architecture

Layer normalization is typically applied:
- Before the attention mechanism
- Before the feed-forward network
- This is called "Pre-LN" (Pre-Layer Normalization)