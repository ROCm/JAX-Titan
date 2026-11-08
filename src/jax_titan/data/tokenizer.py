"""Llama 3 tokenizer, matching MaxText's `TikTokenTokenizer`."""

import tiktoken
from tiktoken.load import load_tiktoken_bpe

from jax_titan import paths

_PAT_STR = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_NUM_RESERVED_SPECIAL_TOKENS = 256


class Llama3Tokenizer:

  def __init__(self, path: str | None = None):
    ranks = load_tiktoken_bpe(str(path or paths.llama3_tokenizer()))
    names = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",
    ] + [
        f"<|reserved_special_token_{i}|>"
        for i in range(5, _NUM_RESERVED_SPECIAL_TOKENS - 5)
    ]
    special = {name: len(ranks) + i for i, name in enumerate(names)}
    self._enc = tiktoken.Encoding(
        name="llama3", pat_str=_PAT_STR, mergeable_ranks=ranks,
        special_tokens=special,
    )
    self.bos_id = special["<|begin_of_text|>"]
    self.eos_id = special["<|end_of_text|>"]
    self.vocab_size = len(ranks) + len(special)

  def encode_batch(self, texts: list[str]) -> list[list[int]]:
    return self._enc.encode_ordinary_batch(texts)

  def decode(self, ids: list[int]) -> str:
    return self._enc.decode(ids)
