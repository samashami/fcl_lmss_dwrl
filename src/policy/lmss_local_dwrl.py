from __future__ import annotations

import json
import os
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
_DEBUG = os.environ.get("LMSS_DEBUG", "1") == "1"


def _parse_strategy_id(text: str, palette_size: int) -> tuple[Optional[int], str]:
    """
    Parse strategy id with explicit precedence:
      1) strict JSON whole string
      2) extracted JSON object (first '{' ... last '}')
      3) constrained regex 'strategy_id: <int>'
      4) single integer line
    Returns (sid, parse_mode). sid=None means parse failure.
    """
    raw = (text or "").strip()
    if not raw:
        return None, "fallback"

    # 1) strict JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "strategy_id" in obj:
            sid = int(obj["strategy_id"])
            if 0 <= sid < palette_size:
                return sid, "json"
    except Exception:
        pass

    # 2) extracted JSON from first '{' to last '}'
    i = raw.find("{")
    j = raw.rfind("}")
    if i != -1 and j != -1 and j > i:
        blob = raw[i : j + 1]
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict) and "strategy_id" in obj:
                sid = int(obj["strategy_id"])
                if 0 <= sid < palette_size:
                    return sid, "extracted"
        except Exception:
            pass

    # 3) constrained 'strategy_id: <int>' pattern
    m = re.search(r"\bstrategy_id\b\s*[:=]\s*(-?\d+)\b", raw, flags=re.IGNORECASE)
    if m:
        sid = int(m.group(1))
        if 0 <= sid < palette_size:
            return sid, "regex"

    # 4) single integer line only
    if re.fullmatch(r"\s*-?\d+\s*", raw):
        sid = int(raw)
        if 0 <= sid < palette_size:
            return sid, "int_line"

    return None, "fallback"


def _build_action(strategy_id: int) -> Dict[str, Any]:
    s = STRATEGY_PALETTE.get(int(strategy_id), STRATEGY_PALETTE[0])
    return {
        "lr_req": float(s["lr"]),
        "replay_req": float(s["replay_ratio"]),
        "notes": f"LMSS_LOCAL strategy={int(strategy_id)}:{s['name']}",
        "policy_source": f"LMSS_LOCAL_{int(strategy_id)}",
    }

def _fallback_strategy(
    dval: float,
    forget_mean: float,
    div: float,
    dacc_hist: list[float],
    last_2_actions: list[int],
    pvc: float,
) -> int:
    # Plateau: avoid repeating Hold if trend is non-positive and stable but stuck.
    if (
        len(dacc_hist) >= 2
        and dacc_hist[-1] <= 0.0
        and dacc_hist[-2] <= 0.0
        and forget_mean < 0.05
        and div < 0.10
        and len(last_2_actions) >= 2
        and int(last_2_actions[-1]) == 0
        and int(last_2_actions[-2]) == 0
    ):
        return 1 if pvc < 0.70 else 2
    # Mild correction if not improving.
    if dval <= 0.0:
        return 1
    # Instability fallback.
    if div > 0.10 or forget_mean > 0.05:
        return 2
    # Only hold when clearly stable/improving.
    return 0


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
    dacc_hist = [float(x) for x in g.get("dacc_hist", [])[-2:]]
    last_2_actions = state.get("last_2_actions", [])
    pvc = float(g.get("per_class_val_recall", {}).get("PVC", 1.0))

    if div > 0.20 or forget_mean > 0.12:
        return _build_action(3)
    # Plateau guard: avoid repeating Hold when dacc is non-positive for 2 rounds.
    if (
        len(dacc_hist) >= 2
        and dacc_hist[-1] <= 0.0
        and dacc_hist[-2] <= 0.0
        and forget_mean < 0.05
        and div < 0.10
        and len(last_2_actions) >= 2
        and int(last_2_actions[-1]) == 0
        and int(last_2_actions[-2]) == 0
    ):
        return _build_action(1 if pvc < 0.70 else 2)

    # LLM selection path
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        sid = _fallback_strategy(dval, forget_mean, div, dacc_hist, [int(x) for x in last_2_actions], pvc)
        action = _build_action(sid)
        action["notes"] = f"{action['notes']} | fallback(no transformers)"
        return action

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
- dacc = global.dval_acc (= val_acc_curr - val_acc_prev)
- dacc_hist = last two dacc values (most recent at end)
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
- If uncertain, choose the smallest non-hold adjustment consistent with the state (prefer replay +0.05 if dacc <= 0, else Hold).
- PLATEAU: If dacc <= 0 for 2 consecutive rounds AND forgetting < 0.05 AND divergence < 0.10, do NOT repeat Hold; choose a mild change (prefer replay +0.05 unless PVC is dropping).

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
    # Decode only newly generated tokens (exclude prompt text with embedded JSON).
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_ids = out[0][prompt_len:]
    txt = tok.decode(gen_ids, skip_special_tokens=True)
    sid, parse_mode = _parse_strategy_id(txt, palette_size=len(STRATEGY_PALETTE))
    parse_fail = sid is None
    if _DEBUG:
        preview = txt.replace("\n", "\\n")
        print(f"[LMSS_DEBUG] raw_llm_output='{preview[:300]}'", flush=True)
        print(f"[LMSS_DEBUG] parse_mode={parse_mode} parse_fail={parse_fail} sid={sid}", flush=True)
    if parse_fail:
        sid = _fallback_strategy(dval, forget_mean, div, dacc_hist, [int(x) for x in last_2_actions], pvc)
        action = _build_action(sid)
        action["notes"] = f"{action['notes']} | parse_mode=fallback | parse_fail=True"
        return action

    action = _build_action(sid)
    action["notes"] = f"{action['notes']} | parse_mode={parse_mode} | parse_fail=False"
    return action
