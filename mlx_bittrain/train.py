"""train — one resumable LoRA training run on an mlx-lm model; saves a checkpoint and exits 0 on SIGTERM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from . import ckpt
from .data import Batches, load_tokens, split

RUN_KEYS = ("model", "data", "lora_rank", "lora_scale", "lora_layers", "lr", "warmup", "batch",
            "max_len", "steps", "seed")


class Run:
    def __init__(self, rundir: Path):
        self.dir = rundir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stop_reason: str | None = None

    def event(self, kind: str, **fields) -> None:
        with open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps({"t": time.time(), "event": kind, **fields}) + "\n")

    def metric(self, **fields) -> None:
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(fields) + "\n")

    def status(self, **fields) -> None:
        tmp = self.dir / ".status.tmp"
        tmp.write_text(json.dumps({**fields, "updated_at": time.time()}, indent=2))
        os.replace(tmp, self.dir / "status.json")


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mlx_bittrain.train", description=__doc__)
    p.add_argument("rundir", type=Path)
    p.add_argument("--model", required=True, help="Hugging Face repo or local path that mlx-lm can load")
    p.add_argument("--data", required=True, help="JSONL file, one {\"text\": ...} object per line")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-scale", type=float, default=20.0)
    p.add_argument("--lora-layers", type=int, default=-1, help="last N blocks get LoRA; -1 = all")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--steps", type=int, required=True, help="total optimizer steps for the run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--keep", type=int, default=2)
    p.add_argument("--stop-after", type=int, default=0,
                   help="pause cleanly after this many steps in this process (simulates a window end)")
    p.add_argument("--resume", action="store_true", help="continue from the newest valid checkpoint")
    p.add_argument("--device", choices=("gpu", "cpu"), default="gpu",
                   help="cpu gives bit-identical runs (Metal kernels are not deterministic); slower")
    return p.parse_args(argv)


def config_of(a: argparse.Namespace) -> dict:
    return {k: getattr(a, k) for k in RUN_KEYS}


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def step_seed(seed: int, step: int) -> int:
    return (seed * 1_000_003 + step) % (2**63)


def loss_fn(model, ids, lengths):
    inputs, targets = ids[:, :-1], ids[:, 1:]
    logits = model(inputs).astype(mx.float32)
    pos = mx.arange(1, targets.shape[1] + 1)[None, :]
    mask = pos < lengths[:, None]
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.sum() / ntoks, ntoks


def lora_param_count(model) -> int:
    from mlx_lm.tuner.lora import LoRALinear
    return sum(m.lora_a.size + m.lora_b.size for _, m in model.named_modules() if isinstance(m, LoRALinear))


def evaluate(model, val: Batches) -> float:
    total, toks = 0.0, 0
    for i in range(val.per_epoch):
        ids, lengths = val.get(0, i)
        loss, n = loss_fn(model, ids, lengths)
        mx.eval(loss, n)
        total += loss.item() * n.item()
        toks += n.item()
    return total / max(toks, 1)


def main(argv=None) -> int:
    a = parse(argv)
    mx.set_default_device(mx.cpu if a.device == "cpu" else mx.gpu)
    run = Run(a.rundir)
    cfg = config_of(a)
    chash = config_hash(cfg)
    has_ckpt = any(run.dir.glob("ckpt-*"))
    if has_ckpt and not a.resume:
        print(f"error: {run.dir} already has checkpoints; pass --resume or use a new run directory", file=sys.stderr)
        return 2

    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tokenizer = load(a.model)
    model.freeze()
    n_layers = len(model.layers) if a.lora_layers < 0 else a.lora_layers
    linear_to_lora_layers(model, n_layers, {"rank": a.lora_rank, "scale": a.lora_scale, "dropout": 0.0})
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    expected = lora_param_count(model)
    if trainable != expected or trainable == 0:
        print(f"error: {trainable} trainable parameters, expected {expected} LoRA parameters", file=sys.stderr)
        return 2

    rows, data_hash = load_tokens(a.data, tokenizer, a.model, a.max_len, run.dir.parent / ".token-cache")
    train_rows, val_rows = split(rows)
    train = Batches(train_rows, a.batch, a.seed)
    val = Batches(val_rows, a.batch, a.seed)

    schedule = optim.join_schedules(
        [optim.linear_schedule(0.0, a.lr, a.warmup), optim.cosine_decay(a.lr, max(1, a.steps - a.warmup))],
        [a.warmup])
    opt = optim.AdamW(learning_rate=schedule, weight_decay=0.0)

    step, best_val, best_step, tokens_done = 0, math.inf, None, 0
    if a.resume and has_ckpt:
        found = ckpt.load_latest(run.dir)
        path, meta, weights, opt_state = found
        if meta["config_hash"] != chash:
            print(f"error: config changed since {path.name} ({meta['config_hash']} -> {chash})", file=sys.stderr)
            return 2
        if meta["data_hash"] != data_hash:
            print(f"error: data file changed since {path.name}", file=sys.stderr)
            return 2
        model.load_weights(list(weights.items()), strict=False)
        opt.init(model.trainable_parameters())
        opt.state = tree_unflatten(list(opt_state.items()))
        step, best_val, best_step = meta["step"], meta["best_val"], meta["best_step"]
        tokens_done = meta["tokens_done"]
        run.event("resumed", step=step, ckpt=path.name)
    else:
        run.event("started", config=cfg, config_hash=chash, data_hash=data_hash, trainable=trainable)

    def on_signal(signum, _frame):
        run.stop_reason = signal.Signals(signum).name
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    value_and_grad = nn.value_and_grad(model, loss_fn)

    def save(reason: str) -> Path:
        epoch, index = divmod(step, train.per_epoch)
        meta = {"config": cfg, "config_hash": chash, "data_hash": data_hash, "epoch": epoch, "index": index,
                "best_val": best_val, "best_step": best_step, "tokens_done": tokens_done, "reason": reason,
                "mlx": mx.__version__}
        weights = dict(tree_flatten(model.trainable_parameters()))
        return ckpt.save(run.dir, step, weights, dict(tree_flatten(opt.state)), meta, keep=a.keep)

    started, steps_here, tokens_here, last_ckpt = time.time(), 0, 0, None
    loss_v = float("nan")
    while step < a.steps:
        epoch, index = divmod(step, train.per_epoch)
        mx.random.seed(step_seed(a.seed, step))
        ids, lengths = train.get(epoch, index)
        (loss, ntoks), grads = value_and_grad(model, ids, lengths)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss, ntoks)
        step += 1
        steps_here += 1
        loss_v, n = loss.item(), ntoks.item()
        tokens_done += n
        tokens_here += n
        elapsed = time.time() - started
        run.metric(step=step, loss=loss_v, tokens=n, lr=float(opt.learning_rate))

        val_loss = None
        if step % a.eval_every == 0 or step == a.steps:
            val_loss = evaluate(model, val)
            if val_loss < best_val:
                best_val, best_step = val_loss, step
            run.metric(step=step, val_loss=val_loss)
        if step % a.ckpt_every == 0 or step == a.steps:
            last_ckpt = save("interval").name
        if a.stop_after and steps_here >= a.stop_after and step < a.steps:
            run.stop_reason = run.stop_reason or "stop-after"
        run.status(state="running", step=step, total=a.steps, epoch=epoch, tokens_done=tokens_done,
                   tokens_per_s=round(tokens_here / max(elapsed, 1e-9), 1), loss=loss_v,
                   val_loss=val_loss, best_val=best_val if best_step else None, last_ckpt=last_ckpt,
                   peak_mem_gb=round(mx.get_peak_memory() / 1e9, 2),
                   eta_s=round(elapsed / steps_here * (a.steps - step), 1))
        if run.stop_reason:
            last_ckpt = save(run.stop_reason).name
            run.event("paused", step=step, reason=run.stop_reason, ckpt=last_ckpt)
            run.status(state="paused", step=step, total=a.steps, reason=run.stop_reason, last_ckpt=last_ckpt,
                       loss=loss_v)
            return 0

    run.event("completed", step=step, best_val=best_val, best_step=best_step)
    run.status(state="completed", step=step, total=a.steps, loss=loss_v, best_val=best_val,
               best_step=best_step, last_ckpt=last_ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
