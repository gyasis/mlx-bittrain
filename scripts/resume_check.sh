#!/usr/bin/env bash
# resume_check.sh — train the same run three ways (straight, paused + resumed, killed + resumed) and compare losses.
set -euo pipefail
MODEL=${MODEL:-HuggingFaceTB/SmolLM2-135M}
STEPS=${STEPS:-16}
MAX_LEN=${MAX_LEN:-128}
STOP_AT=$((STEPS * 3 / 8))
KILL_AT=$((STEPS * 5 / 8))
DEVICE=${DEVICE:-cpu}
ROOT=${ROOT:-runs/resume-check}
PY=${PY:-.venv/bin/python}
DATA=runs/data/shakespeare.jsonl

mkdir -p runs/data
if [[ ! -f $DATA ]]; then
  curl -fsSL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o runs/data/shakespeare.txt
  $PY - <<'PY'
import json
text = open("runs/data/shakespeare.txt").read()
paras, cur = [], ""
for block in text.split("\n\n"):
    cur = f"{cur}\n\n{block}" if cur else block
    if len(cur) > 600:
        paras.append(cur); cur = ""
with open("runs/data/shakespeare.jsonl", "w") as f:
    for p in paras:
        f.write(json.dumps({"text": p}) + "\n")
print(f"{len(paras)} rows")
PY
fi

rm -rf "$ROOT"
COMMON=(--model "$MODEL" --data "$DATA" --steps "$STEPS" --batch 4 --max-len "$MAX_LEN" --ckpt-every 4 --eval-every 8 --seed 7 --device "$DEVICE")

echo "== A: straight through"
$PY -m mlx_bittrain.train "$ROOT/a" "${COMMON[@]}"

echo "== B: pause after $STOP_AT steps, then resume"
$PY -m mlx_bittrain.train "$ROOT/b" "${COMMON[@]}" --stop-after "$STOP_AT"
$PY -m mlx_bittrain.train "$ROOT/b" "${COMMON[@]}" --resume

echo "== C: kill -9 after step $KILL_AT, then resume"
$PY -m mlx_bittrain.train "$ROOT/c" "${COMMON[@]}" &
PID=$!
until [[ -f $ROOT/c/metrics.jsonl ]] && [[ $(grep -c '"loss"' "$ROOT/c/metrics.jsonl") -ge $KILL_AT ]]; do
  kill -0 $PID 2>/dev/null || { echo "run C exited before step $KILL_AT"; exit 1; }
  sleep 0.2
done
kill -9 $PID; wait $PID 2>/dev/null || true
echo "killed at step $(grep -c '"loss"' "$ROOT/c/metrics.jsonl")"
$PY -m mlx_bittrain.train "$ROOT/c" "${COMMON[@]}" --resume

$PY tests/compare_runs.py "$ROOT/a" "$ROOT/b" "$ROOT/c"
