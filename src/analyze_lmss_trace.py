from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _is_fallback(row: dict) -> bool:
    pm = str(row.get("parse_mode", "")).lower()
    gn = str(row.get("gate_notes", "")).lower()
    return ("fallback" in pm) or ("fallback" in gn)


def _is_gate_override(row: dict) -> bool:
    raw = str(row.get("raw_strategy_id", "")).strip()
    sid = str(row.get("strategy_id", "")).strip()
    if raw and sid:
        try:
            return int(raw) != int(sid)
        except Exception:
            pass
    gn = str(row.get("gate_notes", "")).strip().lower()
    return bool(gn) and ("smooth_transition" in gn or "gate:" in gn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    trace_csv = run_dir / "controller_strategy_trace.csv"
    hist_csv = run_dir / "controller_strategy_histogram.csv"

    if not trace_csv.exists():
        print(f"Missing: {trace_csv}")
        print("Trace is saved at end of each round to:")
        print(f"- {trace_csv}")
        print("Histogram is saved to:")
        print(f"- {hist_csv}")
        return

    rows = _read_csv(trace_csv)
    print("round,parse_mode,raw_strategy_id,strategy_id,gate_notes")
    for r in rows:
        print(
            f"{r.get('round','')},{r.get('parse_mode','')},{r.get('raw_strategy_id','')},"
            f"{r.get('strategy_id','')},{r.get('gate_notes','')}"
        )

    n = len(rows)
    fallback_rounds = sum(1 for r in rows if _is_fallback(r))
    gate_override_rounds = sum(1 for r in rows if _is_gate_override(r))
    fallback_rate = (fallback_rounds / n) if n else 0.0
    gate_override_rate = (gate_override_rounds / n) if n else 0.0

    print()
    print(f"fallback_rounds={fallback_rounds}")
    print(f"fallback_rate={fallback_rate:.4f}")
    print(f"gate_override_rounds={gate_override_rounds}")
    print(f"gate_override_rate={gate_override_rate:.4f}")

    if fallback_rate > 0:
        verdict = "LMSS NOT validated (fallback used)"
    elif gate_override_rate > 0.5:
        verdict = "Gate dominates; LMSS effectively neutered"
    else:
        verdict = "LMSS validated; gate OK"
    print(f"verdict={verdict}")

    print()
    if hist_csv.exists():
        print("strategy_id,strategy_name,count,fraction,mean_lr_applied,mean_replay_applied")
        for r in _read_csv(hist_csv):
            print(
                f"{r.get('strategy_id','')},{r.get('strategy_name','')},{r.get('count','')},"
                f"{r.get('fraction','')},{r.get('mean_lr_applied','')},{r.get('mean_replay_applied','')}"
            )
    else:
        print(f"Missing: {hist_csv}")


if __name__ == "__main__":
    main()
