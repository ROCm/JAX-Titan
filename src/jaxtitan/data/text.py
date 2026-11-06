"""Pretokenized text corpora and a fixed-length window sampler."""

import dataclasses
import pathlib
import urllib.request

import numpy as np

WIKITEXT_103_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-103-raw-v1/{split}-{shard:05d}-of-{num_shards:05d}.parquet"
)
WIKITEXT_103_SHARDS = {"train": 2, "validation": 1}


def prepare_wikitext103(data_dir: str, split: str = "train") -> pathlib.Path:
  """Downloads WikiText-103 (raw) and writes Llama 3 token ids as uint32."""
  import pyarrow.parquet as pq  # pylint: disable=g-import-not-at-top

  from jaxtitan.data.tokenizer import Llama3Tokenizer  # pylint: disable=g-import-not-at-top

  root = pathlib.Path(data_dir)
  root.mkdir(parents=True, exist_ok=True)
  out = root / f"wikitext103_{split}.llama3.bin"
  if out.exists():
    return out
  num_shards = WIKITEXT_103_SHARDS[split]
  lines = []
  for shard in range(num_shards):
    path = root / f"{split}-{shard:05d}.parquet"
    if not path.exists():
      url = WIKITEXT_103_URL.format(split=split, shard=shard, num_shards=num_shards)
      urllib.request.urlretrieve(url, path)
    lines.extend(pq.read_table(path).column("text").to_pylist())
  # WikiText marks article starts with a single " = Title = " line.
  docs, current = [], []
  for line in lines:
    if line.startswith(" = ") and not line.startswith(" = = ") and current:
      docs.append("".join(current))
      current = []
    current.append(line)
  if current:
    docs.append("".join(current))
  tok = Llama3Tokenizer()
  ids = []
  for start in range(0, len(docs), 2048):
    for doc_ids in tok.encode_batch(docs[start:start + 2048]):
      ids.append(np.asarray([tok.bos_id, *doc_ids, tok.eos_id], np.uint32))
  tokens = np.concatenate(ids)
  tmp = out.with_suffix(".tmp")
  tokens.tofile(tmp)
  tmp.rename(out)
  return out


@dataclasses.dataclass
class WindowSampler:
  """Samples [batch, seq_len + 1] windows; `state` restores the cursor."""

  path: str
  seq_len: int
  seed: int = 0
  step: int = 0

  def __post_init__(self):
    self._tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
    self.num_tokens = int(self._tokens.shape[0])

  def next(self, batch: int) -> np.ndarray:
    rng = np.random.default_rng((self.seed, self.step))
    starts = rng.integers(0, self.num_tokens - self.seq_len - 1, size=batch)
    self.step += 1
    return np.stack(
        [self._tokens[s:s + self.seq_len + 1] for s in starts]
    ).astype(np.int32)

  def state(self) -> dict:
    return {"seed": self.seed, "step": self.step}

  def max_token(self) -> int:
    return int(self._tokens.max())


if __name__ == "__main__":
  from jaxtitan import paths  # pylint: disable=g-import-not-at-top

  for _split in WIKITEXT_103_SHARDS:
    _path = prepare_wikitext103(str(paths.work_dir() / "data"), _split)
    _count = np.memmap(_path, dtype=np.uint32, mode="r").shape[0]
    print(f"{_split}: {_path} ({_count / 1e6:.1f}M tokens)")
