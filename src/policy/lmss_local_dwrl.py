from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


STRATEGY_PALETTE = {
    0: {"name": "Hold", "lr": 1e-4, "replay_ratio": 0.20},
    1: {"name": "Conservative", "lr": 9e-5, "replay_ratio": 0.25},
    2: {"name": "Consolidate", "lr": 8e-5, "replay_ratio": 0.35},
    3: {"name": "Stability", "lr": 7e-5, "replay_ratio": 0.45},
    4: {"name": "Balanced Push", "lr": 1.2e-4, "replay_ratio": 0.20},
    5: {"name": "Aggressive Push", "lr": 1.5e-4, "replay_ratio": 0.15},
}

_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_CACHE: Dict[str, Any] = {"tok": None, "mdl": None, "model_name": None}


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0).replace("\n", " ").strip())
    except Exception:
        return None


def _build_action(strategy_id: int) -> Dict[str, Any]:
    s = STRATEGY_PALETTE.get(int(strategy_id), STRATEGY_PALETTE[0])
    return {
        "lr_req": float(s["lr"]),
        "replay_req": float(s["replay_ratio"]),
        "notes": f"LMSS_LOCAL strategy={int(strategy_id)}:{s['name']}",
        "policy_source": f"LMSS_LOCAL_{int(strategy_id)}",
    }


def lmss_decide_action_local_dwrl(
    state: Dict[str, Any],
    compact_state_fn,
    model_name: str = _DEFAULT_MODEL,
    max_new_tokens: int = 96,
) -> Dict[str, Any]:
    # Deterministic hard guard only for extreme instability.
    g = state.get("global", {})
    val_curr = g.get("val_acc_curr")
    val_prev = g.get("val_acc_prev")
    dval = 0.0 if (val_curr is None or val_prev is None) else float(val_curr - val_prev)
    forget_mean = float(g.get("forget_mean", 0.0))
    div = float(g.get("divergence", 0.0))

    if div > 0.20 or forget_mean > 0.12:
        return _build_action(3)

    # LLM selection path
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return _build_action(0)

    if (
        _CACHE["tok"] is None
        or _CACHE["mdl"] is None
        or _CACHE["model_name"] != model_name
    ):
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mdl = mdl.to(device)
        _CACHE["tok"], _CACHE["mdl"], _CACHE["model_name"] = tok, mdl, model_name

    tok, mdl = _CACHE["tok"], _CACHE["mdl"]
    s_small = compact_state_fn(state)

    palette_text = "\n".join(
        [f"{k}: {v['name']} (lr={v['lr']}, replay={v['replay_ratio']})" for k, v in STRATEGY_PALETTE.items()]
    )
    prompt = f"""You are LMSS, a controller for Federated Continual Learning (FCL) on DWRL (7 classes: PET, PP, PE, TETRA, PS, PVC, Other).
At each round you must select ONE strategy from the palette to set learning rate and replay ratio for the next round.

PRIMARY GOAL (ranked):
1) Increase global validation performance without increasing forgetting.
2) Protect minority/weak classes (especially PS and TETRA). Do NOT sacrifice PVC heavily.
3) Avoid unstable behavior (oscillations). Prefer small changes unless there is clear degradation.

SIGNAL DEFINITIONS:
- dacc = val_acc_curr - val_acc_prev
- forgetting = global.forget_mean
- divergence = global.divergence

RULES:
- Use only the information in STATE.
- Choose the most conservative strategy that addresses the main issue.
- If metrics are stable (|dacc| small, forgetting low, divergence low), prefer hold or small replay decrease.
- If forgetting or divergence is high, increase replay (stability first).
- If PS or TETRA is very low or dropping, prioritize class balance (usually more replay, sometimes lower LR).
- Avoid pushing replay to min/max for many rounds.
- Consider last_2_actions and avoid flipping replay up/down unless metrics justify it.
- If uncertain, choose strategy_id 0 (Hold).

STATE (JSON):
{json.dumps(s_small)}

STRATEGY PALETTE:
{palette_text}

Return ONLY valid JSON on one line with no markdown:
{{"strategy_id": <int>, "reason": "<max 25 words; mention 1-2 key signals>"}}"""

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    txt = tok.decode(out[0], skip_special_tokens=True)
    parsed = _extract_first_json_object(txt)
    if not parsed or "strategy_id" not in parsed:
        return _build_action(0)

    sid = int(parsed.get("strategy_id", 0))
    action = _build_action(sid)
    reasoning = str(parsed.get("reason", parsed.get("reasoning", ""))).strip()
    if reasoning:
        action["notes"] = f"{action['notes']} | {reasoning}"
    return action
