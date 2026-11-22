"""ppl — perplexity of an mlx-lm model (full precision or quantized) on fixed, non-overlapping chunks of a text file."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def chunks(path: Path, tokenizer, seq_len: int, max_chunks: int) -> mx.array:
    ids = tokenizer.encode(path.read_text())
    n = min(len(ids) // seq_len, max_chunks)
    if n == 0:
        raise ValueError(f"{path}: fewer than {seq_len} tokens")
    return mx.array(ids[: n * seq_len]).reshape(n, seq_len)


def perplexity(model, data: mx.array, batch: int) -> tuple[float, int]:
    total, toks = 0.0, 0
    for i in range(0, data.shape[0], batch):
        x = data[i:i + batch]
        logits = model(x[:, :-1]).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, x[:, 1:], reduction="sum")
        mx.eval(ce)
        total += ce.item()
        toks += x[:, 1:].size
    return math.exp(total / toks), toks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mlx_bittrain.ppl", description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--text", required=True, type=Path, help="plain-text file to score")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--max-chunks", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args(argv)

    from mlx_lm import load

    model, tokenizer = load(a.model)
    data = chunks(a.text, tokenizer, a.seq_len, a.max_chunks)
    t0 = time.time()
    ppl, toks = perplexity(model, data, a.batch)
    print(json.dumps({"model": a.model, "text": str(a.text), "ppl": ppl, "tokens": toks,
                      "chunks": int(data.shape[0]), "seq_len": a.seq_len, "seconds": round(time.time() - t0, 1),
                      "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
