from typing import List, Dict, Tuple
from collections import Counter

'''
@Author: hychen11
@Date:   2025-09-14 14:00:19
@Description: 
'''


class BPETokenizer:
    # special tokens不参与merge!
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None)
        pass
    
    def encode(self, text: str) -> list[int]:
        pass
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass
    
    def decode(self, ids: list[int]) -> str:
        pass
        