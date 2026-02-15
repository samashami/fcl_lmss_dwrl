# src/policy_dwrl.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class V4Config:
    lr_min: float = 5e-5
    lr_max: float = 2e-3
    rep_min: float = 0.10
    rep_max: float = 0.70

    deadband: float = 0.003
    rep_step_high: float = 0.10
    rep_step_low: float  = 0.05

    forget_thr: float = 0.05
    div_thr: float = 0.10

    ema_alpha: float = 0.30
    lr_boost: float = 1.35
    lr_cooldown: float = 1.50

    warmup_rounds: int = 2


class V4ControllerDWRL:
    """
    DWRL V4-A (recommended):
      - single global lr
      - single global replay_ratio
      - no per-client lr_scale
    """

    def __init__(self, base_lr: float, base_rep: float = 0.50, cfg: V4Config | None = None):
        self.cfg = cfg or V4Config()
        self.last_lr = float(base_lr)
        self.last_rep = float(base_rep)
        self.ema_loss = None  # set on first step

    def step(self, round_id: int, acc_curr: float, acc_prev: float | None,
             global_val_loss: float | None,
             forget_mean: float,
             divergence: float) -> dict:
        """
        Returns dict with:
          - applied: lr, replay_ratio
          - requested (pre-clamp): lr_req, replay_req
          - flags: clamped_lr, clamped_rep
          - notes
        acc_curr: current global test acc in [0,1] (fraction)
        acc_prev: previous global test acc in [0,1] (fraction) or None
        forget_mean: mean forgetting in [0,1]
        divergence: scalar (0..~) normalized
        """
        cfg = self.cfg
        dacc = 0.0 if acc_prev is None else float(acc_curr - acc_prev)

        # EMA loss (optional)
        if global_val_loss is not None and np.isfinite(global_val_loss):
            if self.ema_loss is None:
                self.ema_loss = float(global_val_loss)
            else:
                self.ema_loss = cfg.ema_alpha * float(global_val_loss) + (1.0 - cfg.ema_alpha) * float(self.ema_loss)

        L_ema = float(self.ema_loss) if self.ema_loss is not None else 0.0

        # Warmup: fixed defaults
        if round_id < cfg.warmup_rounds:
            lr = self.last_lr
            rep = self.last_rep
            notes = ["warmup(fixed)"]
        else:
            lr = float(self.last_lr)
            rep = float(self.last_rep)
            notes = ["v4"]

            if abs(dacc) < cfg.deadband:
                notes.append(f"deadband(|dacc|<{cfg.deadband})")
            else:
                if (forget_mean > cfg.forget_thr) or (divergence > cfg.div_thr):
                    rep += cfg.rep_step_high
                    notes.append("replay↑(forget/div high)")
                else:
                    rep -= cfg.rep_step_low
                    notes.append("replay↓(forget/div low)")

                if dacc < -cfg.deadband:
                    lr /= cfg.lr_cooldown
                    notes.append("lr↓(dacc<0)")
                elif dacc > cfg.deadband and L_ema > 1.5:
                    lr *= cfg.lr_boost
                    notes.append("lr↑(loss high & improving)")

        # Requested (pre-clamp) values
        lr_req = float(lr)
        replay_req = float(rep)

        # Clamp and track whether requests were clipped
        lr = max(cfg.lr_min, min(cfg.lr_max, lr_req))
        rep = max(cfg.rep_min, min(cfg.rep_max, replay_req))
        clamped_lr = (lr != lr_req)
        clamped_rep = (rep != replay_req)
        notes.append(f"clamp(lr∈[{cfg.lr_min},{cfg.lr_max}], rep∈[{cfg.rep_min},{cfg.rep_max}])")
        notes.append(
            f"req->applied(lr:{lr_req:.6g}->{lr:.6g}, rep:{replay_req:.6g}->{rep:.6g})"
        )

        self.last_lr = float(lr)
        self.last_rep = float(rep)

        return {
            "lr": float(lr),
            "replay_ratio": float(rep),
            "lr_req": float(lr_req),
            "replay_req": float(replay_req),
            "clamped_lr": bool(clamped_lr),
            "clamped_rep": bool(clamped_rep),
            "notes": " | ".join(notes),
        }
