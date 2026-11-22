"""ckpt — atomic, hash-verified checkpoints: weights, optimizer state and run position, keep the newest K plus the best."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx

FILES = ("weights.safetensors", "optimizer.safetensors")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(rundir: Path, step: int, weights: dict, optimizer: dict, meta: dict, keep: int = 2) -> Path:
    tmp = rundir / f".tmp-ckpt-{step:08d}"
    final = rundir / f"ckpt-{step:08d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    mx.save_safetensors(str(tmp / FILES[0]), weights)
    mx.save_safetensors(str(tmp / FILES[1]), optimizer)
    meta = dict(meta, step=step, sha256={f: _sha(tmp / f) for f in FILES})
    with open(tmp / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    if final.exists():
        shutil.rmtree(final)
    os.rename(tmp, final)
    link = rundir / ".latest.tmp"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(final.name)
    os.replace(link, rundir / "latest")
    _prune(rundir, keep, meta.get("best_step"))
    return final


def _prune(rundir: Path, keep: int, best_step: int | None) -> None:
    ckpts = sorted(rundir.glob("ckpt-*"))
    protect = {c.name for c in ckpts[-keep:]}
    if best_step is not None:
        protect.add(f"ckpt-{best_step:08d}")
    for c in ckpts:
        if c.name not in protect:
            shutil.rmtree(c)


def verify(ckpt: Path) -> dict:
    meta = json.loads((ckpt / "meta.json").read_text())
    for f, want in meta["sha256"].items():
        got = _sha(ckpt / f)
        if got != want:
            raise ValueError(f"{ckpt / f}: sha256 {got[:12]} does not match meta {want[:12]}")
    return meta


def load_latest(rundir: Path) -> tuple[Path, dict, dict, dict] | None:
    """Newest checkpoint that passes verification, newest first. None if the run has no checkpoint."""
    errors = []
    for ckpt in sorted(rundir.glob("ckpt-*"), reverse=True):
        try:
            meta = verify(ckpt)
        except Exception as e:
            errors.append(f"{ckpt.name}: {e}")
            continue
        weights = mx.load(str(ckpt / FILES[0]))
        optimizer = mx.load(str(ckpt / FILES[1]))
        return ckpt, meta, weights, optimizer
    if errors:
        raise RuntimeError("no checkpoint passed verification: " + "; ".join(errors))
    return None
