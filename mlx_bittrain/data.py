"""data — tokenize a JSONL file once, cache the token ids, and serve padded batches from a resumable position."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _tokenize(path: Path, tokenizer, max_len: int) -> list[np.ndarray]:
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"{path}:{n}: each line must be a JSON object with a 'text' field ({e})") from e
            ids = tokenizer.encode(text)
            if tokenizer.eos_token_id is not None:
                ids = ids + [tokenizer.eos_token_id]
            rows.append(np.asarray(ids[:max_len], dtype=np.int32))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def load_tokens(path: str | Path, tokenizer, model_id: str, max_len: int, cache_dir: Path) -> tuple[list[np.ndarray], str]:
    """Return (rows, data_hash). Token ids are cached per (file, model, max_len)."""
    path = Path(path)
    data_hash = file_hash(path)
    key = hashlib.sha256(f"{data_hash}|{model_id}|{max_len}".encode()).hexdigest()[:16]
    cache = cache_dir / f"tokens-{key}.npz"
    if cache.exists():
        z = np.load(cache)
        flat, offsets = z["flat"], z["offsets"]
        return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)], data_hash
    rows = _tokenize(path, tokenizer, max_len)
    offsets = np.cumsum([0] + [len(r) for r in rows])
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npz")
    np.savez(tmp, flat=np.concatenate(rows), offsets=offsets)
    tmp.rename(cache)
    return rows, data_hash


def split(rows: list[np.ndarray], val_every: int = 20) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Every val_every-th row goes to validation; the split depends only on row order."""
    train = [r for i, r in enumerate(rows) if i % val_every]
    val = [r for i, r in enumerate(rows) if not i % val_every]
    return train, val


class Batches:
    """Padded batches. A position is (epoch, index); the order of an epoch depends only on (seed, epoch)."""

    def __init__(self, rows: list[np.ndarray], batch: int, seed: int):
        if len(rows) < batch:
            raise ValueError(f"{len(rows)} rows is fewer than one batch of {batch}")
        self.rows, self.batch, self.seed = rows, batch, seed
        self.per_epoch = len(rows) // batch
        self._epoch, self._order = -1, None

    def get(self, epoch: int, index: int) -> tuple[mx.array, mx.array]:
        if epoch != self._epoch:
            self._order = np.random.default_rng([self.seed, epoch]).permutation(len(self.rows))
            self._epoch = epoch
        picks = [self.rows[i] for i in self._order[index * self.batch:(index + 1) * self.batch]]
        return pad(picks)


def pad(picks: list[np.ndarray]) -> tuple[mx.array, mx.array]:
    lengths = np.array([len(p) for p in picks], dtype=np.int32)
    out = np.zeros((len(picks), int(lengths.max())), dtype=np.int32)
    for i, p in enumerate(picks):
        out[i, :len(p)] = p
    return mx.array(out), mx.array(lengths)
