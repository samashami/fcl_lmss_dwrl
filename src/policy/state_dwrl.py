from __future__ import annotations

from typing import Any, Dict, List


def build_controller_state_dwrl(
    round_id: int,
    val_acc_curr: float | None,
    val_acc_prev: float | None,
    val_loss: float | None,
    forget_mean: float,
    divergence: float,
    clients: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "round_id": int(round_id),
        "global": {
            "val_acc_curr": float(val_acc_curr) if val_acc_curr is not None else None,
            "val_acc_prev": float(val_acc_prev) if val_acc_prev is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "forget_mean": float(forget_mean),
            "divergence": float(divergence),
        },
        "clients": clients,
    }


def compact_state_for_lmss_dwrl(state: Dict[str, Any]) -> Dict[str, Any]:
    g = state.get("global", {})
    out = {
        "round_id": int(state.get("round_id", 0)),
        "global": {
            "val_acc_curr": g.get("val_acc_curr"),
            "val_acc_prev": g.get("val_acc_prev"),
            "val_loss": g.get("val_loss"),
            "forget_mean": g.get("forget_mean", 0.0),
            "divergence": g.get("divergence", 0.0),
        },
        "clients": [],
    }
    for c in state.get("clients", []):
        out["clients"].append(
            {
                "id": int(c.get("id", 0)),
                "vloss": c.get("vloss"),
                "vacc": c.get("vacc"),
                "new_batch_size": int(c.get("new_batch_size", 0)),
                "last_lr": c.get("last_lr"),
                "last_replay_ratio": c.get("last_replay_ratio"),
            }
        )
    return out

