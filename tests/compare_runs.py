"""compare_runs — check that resumed runs reproduce the per-step losses of a straight run, and that loss falls."""

import json
import sys
from pathlib import Path

TOL = 1e-3


def losses(rundir: Path) -> dict[int, float]:
    out = {}
    for line in (rundir / "metrics.jsonl").read_text().splitlines():
        m = json.loads(line)
        if "loss" in m:
            out[m["step"]] = m["loss"]
    return out


def main() -> int:
    ref_dir, *others = map(Path, sys.argv[1:])
    ref = losses(ref_dir)
    steps = sorted(ref)
    head = sum(ref[s] for s in steps[:10]) / 10
    tail = sum(ref[s] for s in steps[-10:]) / 10
    report = {"steps": len(steps), "first10_mean": round(head, 4), "last10_mean": round(tail, 4),
              "loss_falls": tail < head, "runs": {}}
    ok = tail < head
    for d in others:
        got = losses(d)
        diffs = [abs(got[s] - ref[s]) for s in steps if s in got]
        missing = [s for s in steps if s not in got]
        max_diff = max(diffs) if diffs else float("inf")
        passed = not missing and max_diff <= TOL
        ok &= passed
        report["runs"][d.name] = {"max_abs_diff": max_diff, "missing_steps": missing, "pass": passed}
    report["pass"] = ok
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
