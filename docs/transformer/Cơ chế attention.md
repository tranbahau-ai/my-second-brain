Trong mô hình Transformer, mỗi token trong input không chỉ phụ thuộc vào lân cận mà quan sát **(attention)** toàn bộ sequence để hiểu ngữ cảnh.
### Công thức
$$
Attention(Q,K,V) = \text{softmax} (\frac{Q \cdot K^\top}{\sqrt{d_k}}) * V
$$
### Detail explaination

**1. Dot Product (Tích vô hướng)**
$$
\text{score} = Q \cdot K^\top
$$
Trong đó:
 **Query (Q) :** Token đang đặt câu hỏi
 **Key (K):**    Token chứa thông tin
 **Value (V):**  Thông tin thực sự cần lấy
${\sqrt{d_k}}$: dimension của key (để scaling)
 **softmax:**	 Tính mức độ “nên tập trung vào token nào nhất”
 ![[attent.png]]
#####  Code PyTorch
```python
import torch

Q = torch.tensor([[1.0, 2.0, 3.0]])  # (1 x 3)
K = torch.tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 3.0]])  # (2 x 3)

# compute similarity
score = torch.matmul(Q, K.T)
print(score)  # tensor([[4., 14.]])
```

**note:** Muốn nhân 2 vector → số cột bảng trái **phải bằng** số hàng bảng phải.

**2. Scaling (Stabilization/Tính ổn định)**
$$
\text{scores} = \frac{Q \cdot K^\top}{\sqrt{d_k}}
$$

Lí do scaled cho $\frac{1}{\sqrt{d_k}}$
- Nếu $d_k$ lớn (ví dụ 64, 128, 512), dot product của hai vector có trung bình gần 0 nhưng **phương sai tăng theo $d_k$ (vì tổng $d_k$ phần tử)
- Giá trị dot products lớn hơn dẫn tới vector đầu vào của softmax có **giá trị lớn bất thường**, khiến softmax:
	- phân phối trở nên cực kỳ _peaked_ (chỉ một token có xác suất ~1)
	- và gradient trở nên **cực kỳ nhỏ trong training**, gây khó khăn cho tối ưu (exploding/vanishing gradients). 

(tham khảo bài viết [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) (2017) của **Vaswani**)

Bảng so sánh trường hợp scale và không scale

| $d_k$ | score (no scale) | scaled_score | grad_no_scale | grad_scale |
| ----- | ---------------- | ------------ | ------------- | ---------- |
| 16    | -3.47            | -0.86        | 0.3132        | 0.3040     |
| 64    | -9.32            | -1.16        | 0.3143        | 0.3059     |
| 256   | 4.23             | 0.26         | 0.1189        | 0.3007     |
| 512   | 10.73            | 0.47         | 0.000226      | 0.3025     |
Việc chia do $\sqrt{d_k}$ giúp:
- điều chỉnh phương sai về quy mô kiểm soát được.
- duy trì phân phối đầu vào softmax trong phạm vi ổn định.
- từ đó **giúp mô hình học hiệu quả hơn**.

**3. Final step**

$\text{scores} = \frac{Q \cdot K^\top}{\sqrt{d_k}}$  ->  $\text{weights}=softmax(scores)$  ->  $\text{output}=weights * V$

→ Softmax chuẩn hóa **attention scores** thành **attention weights**. → Multiply với V để lấy **contextual embedding**