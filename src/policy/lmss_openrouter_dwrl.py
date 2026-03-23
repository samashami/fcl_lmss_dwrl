from __future__ import annotations

import json
import os
from typing import Any, Dict

from .lmss_api_dwrl import STRATEGY_PALETTE


_DEFAULT_MODEL = "openai/gpt-4o-mini"
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _build_action(strategy_id: int, policy_source: str, raw_response: str = "", reasoning: str = "") -> Dict[str, Any]:
    sid = int(strategy_id)
    strat = STRATEGY_PALETTE.get(sid, STRATEGY_PALETTE[1])
    notes = f"LMSS_OPENROUTER strategy={sid}:{strat['name']}"
    if reasoning:
        notes = f"{notes} | {reasoning}"
    return {
        "strategy_id": sid,
        "strategy_name": str(strat["name"]),
        "lr_req": float(strat["lr"]),
        "replay_req": float(strat["replay_ratio"]),
        "notes": notes,
        "policy_source": policy_source,
        "raw_response": raw_response,
        "reasoning": reasoning,
    }


def _fallback_action(policy_source: str, raw_response: str = "", reasoning: str = "") -> Dict[str, Any]:
    return _build_action(1, policy_source=policy_source, raw_response=raw_response, reasoning=reasoning)


def lmss_decide_action_openrouter_dwrl(
    state: Dict[str, Any],
    compact_state_fn,
    model: str = _DEFAULT_MODEL,
) -> Dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        action = _fallback_action(
            policy_source="LMSS_OPENROUTER_NO_KEY_FALLBACK",
            raw_response="<missing OPENROUTER_API_KEY>",
        )
        print(
            "[LMSS_OPENROUTER_DWRL] "
            f"policy_source={action['policy_source']} "
            f"raw_response={action['raw_response']} "
            f"strategy_id={action['strategy_id']} "
            f"applied_lr={action['lr_req']:.6f} "
            f"applied_replay_ratio={action['replay_req']:.2f}",
            flush=True,
        )
        return action

    try:
        from openai import OpenAI
    except Exception as e:
        action = _fallback_action(
            policy_source="LMSS_OPENROUTER_IMPORT_ERROR_FALLBACK",
            raw_response=repr(e),
        )
        print(
            "[LMSS_OPENROUTER_DWRL] "
            f"policy_source={action['policy_source']} "
            f"raw_response={action['raw_response']} "
            f"strategy_id={action['strategy_id']} "
            f"applied_lr={action['lr_req']:.6f} "
            f"applied_replay_ratio={action['replay_req']:.2f}",
            flush=True,
        )
        return action

    s_small = compact_state_fn(state)
    palette_text = "\n".join(
        [
            f"{k}: {v['name']} (lr={v['lr']}, replay={v['replay_ratio']})"
            for k, v in STRATEGY_PALETTE.items()
        ]
    )
    prompt = f"""You are the policy selector for federated continual learning on DWRL.
Choose exactly one strategy id from the palette using the current state.

STATE:
{json.dumps(s_small)}

PALETTE:
{palette_text}

Return JSON only:
{{"strategy_id": int, "reasoning": "one short sentence"}}
"""

    extra_headers = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    title = os.environ.get("OPENROUTER_APP_TITLE")
    if referer:
        extra_headers["HTTP-Referer"] = referer
    if title:
        extra_headers["X-Title"] = title

    raw_response = ""
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL),
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            extra_headers=extra_headers or None,
        )
        raw_response = resp.choices[0].message.content or ""
        parsed = json.loads(raw_response)
        strategy_id = int(parsed.get("strategy_id", 1))
        reasoning = str(parsed.get("reasoning", "")).strip()
        action = _build_action(
            strategy_id=strategy_id,
            policy_source=f"LMSS_OPENROUTER_{model}_S{strategy_id}",
            raw_response=raw_response,
            reasoning=reasoning,
        )
        print(
            "[LMSS_OPENROUTER_DWRL] "
            f"policy_source={action['policy_source']} "
            f"raw_response={raw_response} "
            f"strategy_id={action['strategy_id']} "
            f"applied_lr={action['lr_req']:.6f} "
            f"applied_replay_ratio={action['replay_req']:.2f}",
            flush=True,
        )
        return action
    except Exception as e:
        action = _fallback_action(
            policy_source="LMSS_OPENROUTER_ERROR_FALLBACK",
            raw_response=raw_response or repr(e),
        )
        print(
            "[LMSS_OPENROUTER_DWRL] "
            f"policy_source={action['policy_source']} "
            f"raw_response={action['raw_response']} "
            f"strategy_id={action['strategy_id']} "
            f"applied_lr={action['lr_req']:.6f} "
            f"applied_replay_ratio={action['replay_req']:.2f}",
            flush=True,
        )
        return action
