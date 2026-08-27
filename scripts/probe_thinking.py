#!/usr/bin/env python
"""Probe which thinking-budget controls the e-infra kimi-k3 endpoint honours.

The annotation judge spends ~3200 reasoning tokens to emit a ~150-token answer,
which is ~95% of the wall time. The question this answers is whether that can be
capped *without* losing the answer.

The distinction that matters:

  max_tokens        caps the WHOLE response. On a thinking model the reasoning
                    is generated first, so the cap lands mid-thought and the
                    answer is never emitted -- finish_reason=length, parsed=None.
                    Useless here.

  thinking budget   caps only the reasoning, then forces the model to answer.
                    This is what we want. Whether it exists depends entirely on
                    the serving stack, hence this probe.

Run it against one real annotation prompt so the numbers are comparable to the
real workload. Each variant costs one of the API key's 4 parallel slots, so they
run strictly sequentially.

  python scripts/probe_thinking.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import annotate_language as al  # noqa: E402

# Each entry is (label, extra kwargs passed straight through to the API).
# They are ordered cheapest-hypothesis-first: the vLLM chat-template flag is the
# one the project already uses successfully for the local Qwen user simulator
# (see CZTAU_USER_ARGS in env.sh), so it is the most likely to work here too.
VARIANTS: list[tuple[str, dict]] = [
    ("baseline (no control)", {}),
    ("chat_template_kwargs enable_thinking=false",
     {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}),
    ("chat_template_kwargs thinking=false",
     {"extra_body": {"chat_template_kwargs": {"thinking": False}}}),
    ("reasoning_effort=low", {"reasoning_effort": "low"}),
    ("reasoning_effort=minimal", {"reasoning_effort": "minimal"}),
    ("reasoning_effort=none", {"reasoning_effort": "none"}),
    ("extra_body reasoning_effort=low",
     {"extra_body": {"reasoning_effort": "low"}}),
    ("thinking budget_tokens=256",
     {"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 256}}}),
    ("extra_body max_think_tokens=256",
     {"extra_body": {"max_think_tokens": 256}}),
    # The control: proves the cap lands in the wrong place, i.e. that plain
    # max_tokens really does destroy the answer rather than shortening thought.
    ("max_tokens=600 (expected to truncate)", {"max_tokens": 600}),
]


def main() -> None:
    al.load_env()
    from openai import OpenAI

    client = OpenAI(base_url=os.environ["KIMI_API_BASE"],
                    api_key=os.environ["KIMI_API_KEY"],
                    max_retries=1, timeout=900.0)

    found = al.czech_cells("airline", None)
    data = json.loads(found[0][2].read_text(encoding="utf-8"))
    sim = al.pick_sims(data, 1, 0)[0]
    turns = al.agent_turns(sim)
    prompt = al.build_prompt(turns)
    print(f"probe prompt: task {sim['task_id']}, {len(turns)} turns, "
          f"{sum(map(len, turns))} chars\n")

    hdr = f"{'variant':<44} {'secs':>6} {'reason':>7} {'out':>6} {'finish':>9}  result"
    print(hdr)
    print("-" * len(hdr))
    seen: dict[tuple, str] = {}

    for label, kw in VARIANTS:
        # CACHE BUSTING, and it is not optional. The proxy caches on the
        # messages, and `extra_body` is NOT part of its cache key -- so without a
        # nonce every variant after the first returns the baseline's cached
        # response in ~0 s with byte-identical token counts, and the probe
        # "passes" while measuring nothing at all. An inert trailing comment is
        # enough to make each request unique.
        nonce = uuid.uuid4().hex[:12]
        kw = dict(kw)  # copy: never mutate the VARIANTS table itself
        body = dict(kw.pop("extra_body", {}) or {})
        body.setdefault("cache", {"no-cache": True})  # belt and braces
        t = time.time()
        try:
            resp = client.chat.completions.parse(
                model="kimi-k3",
                messages=[{"role": "user",
                           "content": f"{prompt}\n\n<!-- probe {nonce} -->"}],
                temperature=0.0,
                response_format=al.AnnotationList,
                extra_body=body,
                **kw,
            )
            dt = time.time() - t
            ch = resp.choices[0]
            u = resp.usage
            det = getattr(u, "completion_tokens_details", None)
            reason = None
            if det is not None:
                reason = (det.get("reasoning_tokens") if isinstance(det, dict)
                          else getattr(det, "reasoning_tokens", None))
            ok = ("OK %d ann" % len(ch.message.parsed.annotations)
                  if ch.message.parsed is not None else "NO PARSED OUTPUT")
            # A sub-second reply with token counts identical to something already
            # seen is a cached response, not a fast one. Say so loudly: silently
            # reporting it as a win is how a control that does nothing gets
            # adopted as a 10x speedup.
            sig = (reason, getattr(u, "completion_tokens", None))
            if dt < 5 and sig in seen:
                ok = f"CACHED (identical to {seen[sig]}) -- NOT a real result"
            else:
                seen.setdefault(sig, label)
            print(f"{label:<44} {dt:>6.0f} {str(reason):>7} "
                  f"{getattr(u, 'completion_tokens', '?'):>6} "
                  f"{str(ch.finish_reason):>9}  {ok}", flush=True)
        except Exception as exc:
            dt = time.time() - t
            msg = f"{type(exc).__name__}: {exc}"
            print(f"{label:<44} {dt:>6.0f} {'-':>7} {'-':>6} {'-':>9}  "
                  f"{msg[:80]}", flush=True)

    print("\nWhat to look for: a variant with reasoning tokens well below the "
          "baseline that still says OK. That is a real thinking budget. A "
          "variant that says NO PARSED OUTPUT with finish=length only proves "
          "the cap truncated the answer, which is the failure mode to avoid.")


if __name__ == "__main__":
    main()
