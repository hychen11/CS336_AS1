**Unicode codepoint** 可以理解为每一个字符在 Unicode 标准中的编号。
 比如：

- `'A'` → U+0041 → 十进制 65
- `'你'` → U+4F60 → 十进制 20,480

Unicode 总共定义了大约 **154,997** 个 codepoints（虽然很多没用到）。

**UTF-8** 

- UTF-8 会把一个 codepoint 转换为 **1~4 个字节（byte）**。
- 每个字节的范围都是 **0~255**。

任意字符最终都能表示成 **0~255 范围内的整数序列**

[a0] 或者是 [a0 ,a1 a2, a3] 每个a都是0-255返回

BPE原理+ `<|endoftext|>` count, 再**merge**

merge:选择最高的pair,合并pair，并更新pair出现次数 因为每次只合并一个pair，其影响仅仅包含当前pair的单词，更新pair出现的次数不用每次都全部更新。注意这里更新可以通过懒惰更新 + 堆，最基本实现就是每次更新就扫一遍预料库，但是太慢了，GPT2是通过lazy update + heap

用 堆 (heap) 维护所有 pair 的频率，堆顶就是最高频 pair

每次 merge 只更新受影响的相邻 pair → 不用重新扫全语料

这样，top1 pair 查询 O(1)~O(log n)，更新受影响 pair 也很快

注意这里merge终止的条件就是

* len(vocab) >= vocab_size
* 或堆空 / 没有频率大于 1 的 pair


```python
from click import Tuple

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str] = []
) -> Tuple[dict[int, bytes],list[tuple[bytes, bytes]]]:
    pass
```

python不允许返回两个类型不同的,使用Tuple包一下

此外cls vs self

`cls` 的作用就是 **让类方法具备继承的多态性**

cls代表类的本身，用@staticmethod，@staticmethod修饰。@classmethod第一个参数需要cls，@staticmethod不需要cls

self是instance 实例

```python
class MyClass:
    class_variable = "我是类变量"  # 所有实例共享
    
    def __init__(self, name):
        self.name = name  # 每个实例独有的
    
    def instance_method(self):
        print(f"我是实例方法，我叫{self.name}")
        print(f"我可以访问类变量: {self.class_variable}")
    
    @classmethod
    def class_method(cls):
        print(f"我是类方法，我代表{cls.__name__}类")
        print(f"我可以直接访问类变量: {cls.class_variable}")
        # 但不能访问实例变量，因为还没有实例！
    
    @staticmethod
    def static_method():
        print("我是静态方法，我不需要self或cls")
        print("我就像普通函数，但放在类里组织代码")

# 使用方式对比
print("=== 类方法调用 ===")
MyClass.class_method()  # 直接通过类调用，不需要创建实例

print("\n=== 实例方法调用 ===")
obj = MyClass("小明")
obj.instance_method()  # 必须通过实例调用

print("\n=== 静态方法调用 ===")
MyClass.static_method()  # 直接通过类调用
```


```python
    """
    encode:
    indices = list(map(int, string.encode("utf-8")))
    abc 转成 97 98 99
    从params.merges 也就是训练好的词表里找到 pair, new_index, 词表 {97,98}->256
    indices就变成 99 256 ('c', 'ab')
    最后返回 indices [99, 256]
    """
    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)
        return indices

    """
    decode:
    """
    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string
```

```python
special_tokens = ["<|endoftext|>", "<|pad|>"]
GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

```

```python
with multiprocessing.Pool(processes=num_processes) as pool:
    results = pool.imap_unordered(
            functools.partial(process_chunk, input_path=input_path, special_tokens=special_tokens), 
            zip(boundaries[:-1], boundaries[1:]),
        )
    
    for res in results:
        vocab_counter.update(res)
```

`with multiprocessing.Pool(processes=num_processes) as pool:` 
创建包含 num_processes 个工作进程的进程池，with 语句确保进程池在使用后正确关闭和清理

functools.partial：创建一个新函数，固定 process_chunk 的部分参数，相当于：lambda chunk: process_chunk(chunk, input_path, special_tokens)

`pool.imap_unordered(...)`
imap_unordered：异步并行映射，结果按完成顺序返回


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

# Rope

```python
torch.arange(start=0, end, step=1, *, dtype=None, device=None)
```

$\theta_{i,k}=\frac{i}{\theta^{2k/d}}$

```python
freq = (torch.arange(0, d_k, 2, device=device)/d_k) # dimension is d_k/2
position_ids = torch.arange( 0, max_seq_len, 1, dtype=torch.float32, device=device) #seq_len
torch.outer(position_ids,freqs) #torch.outer(input, other, *, out=None) (len(input), len(other))(len(input), len(other))
```

```python
x_pairs = torch.stack([rotated_real, rotated_imag], dim=-1)
# 两个张量沿着 最后一维 (-1) 拼成一个新的维度
```

`dim = 0` 也就是在第1个维度上拼接

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # dimension is d_k/2
        freqs = 1/(theta**(torch.arange(0, d_k, 2, device=device)/d_k))
        position_ids = torch.arange(
            0, max_seq_len, 1, dtype=torch.float32, device=device)  # dimension is seq_len

        freqs = torch.outer(position_ids, freqs)  # {seq_len,d_k/2}

        sin_freqs = torch.sin(freqs)
        cos_freqs = torch.cos(freqs)

        self.register_buffer(
            "cos_freqs", cos_freqs.to(device), persistent=False)
        self.register_buffer(
            "sin_freqs", sin_freqs.to(device), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
        that you should tolerate x with an arbitrary number of batch dimensions. You should assume
        that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
        x along the sequence dimension.
        """
        seq_len, d_k = x.shape[-2:]
        x_pairs = einops.rearrange(
            x, "... seq_len (pairs two) -> ... seq_len pairs two", two=2)

        x_real = x_pairs[..., 0]
        x_imag = x_pairs[..., 1]

        # token position (..., seq_len)
        cos_freqs = self.cos_freqs[token_positions]  # (..., seq_len, pairs)
        sin_freqs = self.sin_freqs[token_positions]
        # insert a dimension at position 1 (for num_heads)
        # (..., num_heads, seq_len, pairs)
        cos_freqs = cos_freqs.unsqueeze(-3)  # (..., 1, seq_len, pairs)
        sin_freqs = sin_freqs.unsqueeze(-3)  # (..., 1, seq_len, pairs)

        rotated_real = x_real * cos_freqs - x_imag * sin_freqs
        rotated_imag = x_real * sin_freqs + x_imag * cos_freqs

        x_pairs = torch.stack([rotated_real, rotated_imag], dim=-1)

        rotated_x = einops.rearrange(
            x_pairs, "... seq_len pairs two -> ... seq_len (pairs two)", two=2)

        return rotated_x
```



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



