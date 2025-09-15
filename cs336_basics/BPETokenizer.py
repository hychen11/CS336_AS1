from typing import List, Dict, Tuple
from collections import Counter, deque

'''
@Author: hychen11
@Date:   2025-09-14 14:00:19
@Description: 
'''

import regex as re
from collections.abc import Iterable, Iterator
from functools import lru_cache

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    # special tokens不参与merge!
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.inv_vocab = {v: k for k, v in vocab.items()}
        # 长 token 优先匹配
        self.special_tokens_pattern = None
        if special_tokens:
            self.special_tokens_pattern = "|".join(map(re.escape, sorted(special_tokens, key=len, reverse=True)))

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        tokens = []
        start = 0
        # first chunk by special tokens
        if self.special_tokens_pattern:
            for m0 in re.finditer(self.special_tokens_pattern, text):
                # second chunk by tokens
                for match in re.finditer(PAT, text[start:m0.start()]):
                    tokens.extend(self._tokenize_cached(match.group()))
                tokens.append(self.inv_vocab[m0.group().encode("utf-8")])
                start = m0.end()

        # last part
        if start < len(text):
            for match in re.finditer(PAT, text[start:]):
                tokens.extend(self._tokenize(match.group()))
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text_bytes = []
        for id in ids:
            text_bytes.append(self.vocab[id])
        return b''.join(text_bytes).decode('utf-8', errors='replace')

    @lru_cache(maxsize=2 ** 16)
    def _tokenize_cached(self, text: str) -> tuple[int]:
        return tuple(self._tokenize(text))

    def _tokenize(self, text: str) -> list[int]:
        word_list = [bytes([b]) for b in text.encode("utf-8")]
        while True:
            merge_candidates = [(i, b"".join(word_list[i:i + 2])) for i in range(len(word_list) - 1)]
            candidate_ids = [(self.inv_vocab[c], i, c) for i, c in merge_candidates if c in self.inv_vocab]
            if not candidate_ids:
                break
            _, idx, merged_bytes = min(candidate_ids)
            word_list = word_list[:idx] + [merged_bytes] + word_list[idx + 2:]
        return [self.inv_vocab[b] for b in word_list]
