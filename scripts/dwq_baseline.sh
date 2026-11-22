#!/usr/bin/env bash
# dwq_baseline.sh — for each model: perplexity at full precision, plain 4-bit and 4-bit DWQ, on WikiText-2 test.
# Each stage writes its result once; a rerun skips finished stages.
# usage: scripts/dwq_baseline.sh HF_REPO [HF_REPO...]
set -euo pipefail
PY=${PY:-.venv/bin/python}
BITS=${BITS:-4}
GROUP=${GROUP:-64}
DWQ_SAMPLES=${DWQ_SAMPLES:-2048}
DWQ_SEQ=${DWQ_SEQ:-1025}
ROOT=${ROOT:-runs/dwq}
TEXT=runs/data/wikitext2-test.txt

mkdir -p runs/data "$ROOT"
if [[ ! -f $TEXT ]]; then
  curl -fsSL https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet -o runs/data/wikitext2-test.parquet
  $PY -c "import pyarrow.parquet as pq; t=pq.read_table('runs/data/wikitext2-test.parquet'); open('$TEXT','w').write(''.join(t.column('text').to_pylist()))"
fi

stage() {  # stage NAME OUTFILE command...
  local name=$1 out=$2; shift 2
  if [[ -s $out ]]; then echo "-- $name: done ($out)"; return; fi
  echo "-- $name: running"
  local t0=$SECONDS
  "$@"
  echo "{\"stage\":\"$name\",\"seconds\":$((SECONDS - t0))}" >> "$(dirname "$out")/timings.jsonl"
}

for MODEL in "$@"; do
  NAME=$(basename "$MODEL")
  D=$ROOT/$NAME
  mkdir -p "$D"
  echo "== $MODEL"
  stage ppl-full "$D/ppl-full.json" sh -c "$PY -m mlx_bittrain.ppl --model '$MODEL' --text $TEXT > '$D/ppl-full.json.tmp' && mv '$D/ppl-full.json.tmp' '$D/ppl-full.json'"
  stage rtn "$D/rtn$BITS/config.json" $PY -m mlx_lm convert --hf-path "$MODEL" --mlx-path "$D/rtn$BITS" -q --q-bits "$BITS" --q-group-size "$GROUP"
  stage ppl-rtn "$D/ppl-rtn.json" sh -c "$PY -m mlx_bittrain.ppl --model '$D/rtn$BITS' --text $TEXT > '$D/ppl-rtn.json.tmp' && mv '$D/ppl-rtn.json.tmp' '$D/ppl-rtn.json'"
  stage dwq "$D/dwq$BITS/config.json" $PY -m mlx_lm dwq --model "$MODEL" --mlx-path "$D/dwq$BITS" --bits "$BITS" --group-size "$GROUP" \
    --num-samples "$DWQ_SAMPLES" --max-seq-length "$DWQ_SEQ" --grad-checkpoint
  stage ppl-dwq "$D/ppl-dwq.json" sh -c "$PY -m mlx_bittrain.ppl --model '$D/dwq$BITS' --text $TEXT > '$D/ppl-dwq.json.tmp' && mv '$D/ppl-dwq.json.tmp' '$D/ppl-dwq.json'"
done

$PY - "$ROOT" "$@" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print(f"{'model':<16}{'full':>9}{'rtn4':>9}{'dwq4':>9}  dwq recovers")
for m in sys.argv[2:]:
    d = root / Path(m).name
    r = {k: json.loads((d / f"ppl-{k}.json").read_text())["ppl"] for k in ("full", "rtn", "dwq") if (d / f"ppl-{k}.json").exists()}
    rec = (r["rtn"] - r["dwq"]) / (r["rtn"] - r["full"]) if len(r) == 3 and r["rtn"] != r["full"] else float("nan")
    print(f"{Path(m).name:<16}" + "".join(f"{r.get(k, float('nan')):>9.3f}" for k in ("full", "rtn", "dwq")) + f"  {rec:.0%}")
PY
