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