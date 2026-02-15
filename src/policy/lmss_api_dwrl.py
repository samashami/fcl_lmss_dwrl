from __future__ import annotations

import json
import os
from typing import Any, Dict


STRATEGY_PALETTE = {
    0: {"name": "Conservative", "lr": 8e-5, "replay_ratio": 0.60},
    1: {"name": "Standard", "lr": 1e-4, "replay_ratio": 0.20},
    2: {"name": "Consolidate", "lr": 1e-4, "replay_ratio": 0.30},
    3: {"name": "Aggressive", "lr": 1.5e-4, "replay_ratio": 0.15},
    4: {"name": "Recover", "lr": 7e-5, "replay_ratio": 0.40},
}


def _fallback_action() -> Dict[str, Any]:
    s = STRATEGY_PALETTE[1]
    return {
        "lr_req": float(s["lr"]),
        "replay_req": float(s["replay_ratio"]),
        "notes": "LMSS_API fallback",
        "policy_source": "LMSS_API_FALLBACK",
    }


def lmss_decide_action_api_dwrl(
    state: Dict[str, Any],
    compact_state_fn,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _fallback_action()

    try:
        from openai import OpenAI
    except Exception:
        return _fallback_action()

    s_small = compact_state_fn(state)
    palette_text = "\n".join(
        [f"{k}: {v['name']} (lr={v['lr']}, replay={v['replay_ratio']})" for k, v in STRATEGY_PALETTE.items()]
    )
    prompt = f"""You are the policy selector for federated continual learning.
Choose one strategy id.

STATE:
{json.dumps(s_small)}

PALETTE:
{palette_text}

Return JSON only:
{{"strategy_id": int, "reasoning": "one sentence"}}
"""
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        res = json.loads(resp.choices[0].message.content)
        sid = int(res.get("strategy_id", 1))
        strat = STRATEGY_PALETTE.get(sid, STRATEGY_PALETTE[1])
        reasoning = str(res.get("reasoning", "")).strip()
        notes = f"LMSS_API strategy={sid}:{strat['name']}"
        if reasoning:
            notes = f"{notes} | {reasoning}"
        return {
            "lr_req": float(strat["lr"]),
            "replay_req": float(strat["replay_ratio"]),
            "notes": notes,
            "policy_source": f"LMSS_API_{model}_S{sid}",
        }
    except Exception:
        return _fallback_action()

