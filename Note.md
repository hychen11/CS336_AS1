# Initialization

先定义：`fan_in` = 输入通道/特征数量；`fan_out` = 输出通道/特征数量
 Linear: `fan_in = in_features`, `fan_out = out_features`
 Conv: `fan_in = in_channels * kernel_h * kernel_w`，`fan_out = out_channels * kernel_h * kernel_w`（注意 groups 情况）

### Xavier/Glorot

```
Xavier-normal std = sqrt(2/(fan_in + fan_out))
Xavier-uniform limit = sqrt(6/(fan_in + fan_out))
```

### Kaiming

```
Kaiming-normal std = sqrt(2/fan_in)
Kaiming-uniform limit = sqrt(6/fan_in)
```

uniform 分布的方差 = `limit^2 / 3`，推导上面那些常数

```python
import torch
import torch.nn as nn

W = torch.empty(out_features, in_features)

# Xavier (Glorot)
nn.init.xavier_normal_(W)                     # normal, std = sqrt(2/(fan_in+fan_out))
nn.init.xavier_uniform_(W)                    # uniform, ±sqrt(6/(fan_in+fan_out))

# Kaiming (He) — for ReLU
nn.init.kaiming_normal_(W, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_uniform_(W, mode='fan_in', nonlinearity='relu')

# Orthogonal
nn.init.orthogonal_(W)                        # W needs to be at least 2D

# Truncated normal (PyTorch provides trunc_normal_)
nn.init.trunc_normal_(W, mean=0.0, std=std, a=-2*std, b=2*std)

# Bias
b = torch.empty(out_features)
nn.init.zeros_(b)
```

### In this assignment

Linear weights 

```python
# N(0,2/(d_in+d_out)) truncated at [-3std,3std]
# mean = 0
# std^2 = 2/(d_in+d_out)
self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
# Fill with truncated normal distribution
# Truncate at ±3 sigma
nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
# n.init.trunc_normal_ 会从 正态分布 N(mean, std^2) 采样，但会截断[a,b] 范围内
```

Embeddnig

```python
# N(0,1)
nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3*1, b=3*1)
```

RMSNorm 1

这里 `linear.weight.data = weights` 是**直接用外部给定的权重覆盖初始化的权重**。

PyTorch 中 `.data` 直接操作张量，不会触发梯度计算。

**所以原来的初始化就没有作用了**，因为被新的 `weights` 覆盖了

# Linear

W (d_out,d_in)

X (1,d_in)

$Y=xW^T$

# SwiGLU

`FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) * W3x)`

`SiLU(x) = x * σ(x)`

```python
# element-wise
c = a * b  
# matrix multiplication
d = a @ b  
```

You should set $d_{ff}$ to approximately 8/3 × d_model in your implementation, while ensuring that the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your hardware.

# einops

### einsum

```python
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
# ... means the prefix dimension, such as (batch, seq, )
```

### rearrange

```python
images = torch.randn(64, 128, 128, 3) # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)
## Reshape and multiply
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value
## Or in one go:
dimmed_images = einsum(
images, dim_by,
"batch height width channel, dim_value -> batch dim_value height width channel"
)
```

### aggregate

```python
einops.reduce(tensor, "pattern_in -> pattern_out", reduction="sum|mean|max|...")
```



