from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


LR_MIN = 5e-5
LR_MAX = 2e-3
REP_MIN = 0.10
REP_MAX = 0.70


def _clamp(x: Any, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except Exception:
        return float(lo)
    return float(max(lo, min(hi, v)))


def validate_action_dwrl(action: Dict[str, Any] | None, policy_source: str = "Unknown") -> Dict[str, Any]:
    """
    Validate/clamp a DWRL action into a safe scalar action schema.
    Returned dict always contains requested + applied values and clamp flags.
    """
    raw = dict(action) if isinstance(action, dict) else {}

    lr_req = float(raw.get("lr_req", raw.get("lr", LR_MIN)))
    rep_req = float(raw.get("replay_req", raw.get("replay_ratio", REP_MIN)))

    lr_applied = _clamp(lr_req, LR_MIN, LR_MAX)
    rep_applied = _clamp(rep_req, REP_MIN, REP_MAX)

    clamped_lr = (lr_applied != lr_req)
    clamped_rep = (rep_applied != rep_req)

    notes = str(raw.get("notes", "")).strip()
    clamp_note = (
        f"req->applied(lr:{lr_req:.6g}->{lr_applied:.6g}, "
        f"rep:{rep_req:.6g}->{rep_applied:.6g})"
    )
    if "req->applied(" in notes:
        pass
    elif notes:
        notes = f"{notes} | {clamp_note}"
    else:
        notes = clamp_note

    return {
        "lr": float(lr_applied),
        "replay_ratio": float(rep_applied),
        "lr_applied": float(lr_applied),
        "replay_applied": float(rep_applied),
        "lr_req": float(lr_req),
        "rep_req": float(rep_req),
        "clamped_lr": bool(clamped_lr),
        "clamped_rep": bool(clamped_rep),
        "notes": notes,
        "policy_source": str(raw.get("policy_source", policy_source)),
    }


def write_state_json(run_dir: str | Path, round_id: int, state: Dict[str, Any]) -> str:
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    out = p / f"state_round_{int(round_id)}.json"
    out.write_text(json.dumps(state, indent=2))
    return str(out)


def write_action_json(
    run_dir: str | Path,
    round_id: int,
    action: Dict[str, Any],
    policy_source: str = "Unknown",
) -> str:
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    out = p / f"action_round_{int(round_id)}.json"
    payload = dict(action)
    payload["policy_source"] = str(payload.get("policy_source", policy_source))
    out.write_text(json.dumps(payload, indent=2))
    return str(out)
