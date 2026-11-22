# mlx-bittrain

Train small language models at 4, 2 and 1.58 bits (ternary) on an Apple Silicon Mac, overnight, with pause and resume.

**Status: pre-alpha.** LoRA training on mlx-lm models works, with exact pause and resume. Low-bit quantizers, the planner and the supervisor come next.

## Try it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m mlx_bittrain.envcheck
./scripts/resume_check.sh   # trains SmolLM2-135M three ways and checks that resumed runs match exactly
```

A run directory holds `status.json` (live progress), `metrics.jsonl` (loss per step), `events.jsonl` (started, paused, resumed, completed) and checkpoints. Send `SIGTERM` to pause a run cleanly; pass `--resume` to continue it.

## What it does

- **Low-bit training.** Quantization-aware training with a straight-through estimator. The weights stay in 16-bit during training; each forward pass rounds them to the target bit width. Inputs are rounded to 8 bits.
- **Night runs.** A planner measures your Mac on a short pilot run. Then it tells you if a job fits in one night, needs several nights, or is too big for a Mac.
- **Pause and resume.** A checkpoint holds the weights, the optimizer state, the random-number state and the position in the data. A run that stops at 07:00 continues at 23:00 from the same step.
- **Export.** Ternary models export without loss to the native MLX 2-bit format, so `mlx-lm` can run them.

## Design

- Model definitions come from [`mlx-lm`](https://github.com/ml-explore/mlx-lm). The training loop, the checkpoints, distillation and the planner are this repo.
- `train.py` does one run and can always resume. It saves a checkpoint and stops cleanly on `SIGTERM`.
- `supervisor.py` owns the schedule: the night window, memory admission and an event log. On macOS it runs under `launchd`.
- Memory admission is a plugin. The default checks free memory on the Mac. If you run a local memory governor, write an adapter for it. Adapters stay outside the core.

## Planned layout

```text
quant.py        fake-quantizers (k-bit, ternary absmean)
models/zoo.py   mlx-lm models through one interface
kd.py           knowledge distillation (live teacher or top-k logit cache)
data.py         tokenize once, padded batches, resumable data position
train.py        one resumable run
ckpt.py         atomic checkpoints, keep K + best
planner.py      pilot run, calibration table, NIGHT / MULTI-NIGHT / TOO-BIG verdict
export.py       export to MLX 2-bit and BitLinear formats
status.py       machine-readable status JSON
supervisor.py   night window, admission, event log
```

## Requirements

- Apple Silicon Mac (M1 or later). More memory lets you train bigger models.
- Python 3.11+, `mlx`, `mlx-lm`.

## License

MIT
