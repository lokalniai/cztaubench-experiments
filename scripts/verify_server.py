#!/usr/bin/env python3
"""Check that a freshly-started vLLM server is actually usable as a tau2 agent.

The failure this exists to catch is silent. A server started without
--enable-auto-tool-choice, or with a tool-call parser that does not match the
model's chat template, does not reject requests -- it returns plain prose and
never emits a tool call. Every tau2 task is driven through function calls, so
the run then scores near zero and reads as model incompetence rather than as a
misconfigured server. By the time that is visible, days of GPU are gone.

Four checks, matching the list in README §3 "vLLM serving":

  1. tools    -- a tools request comes back with tool_calls, not prose
  2. no-args  -- a zero-argument tool yields arguments "{}" and not ""
                 (an empty string is not valid JSON; tau2 fails to parse it)
  3. roundtrip-- system -> tool_call -> tool result is accepted and answered,
                 in the language the system prompt asks for
  4. thinking -- reasoning lands in its own field rather than leaking into
                 `content`, where it would be graded as if the agent had said it

Usage:
    python scripts/verify_server.py http://tdll-8gpu2:8000/v1 Qwen/Qwen3.6-35B-A3B
"""

import json
import sys
import urllib.request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_reservation_details",
            "description": "Look up a reservation by its id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reservation_id": {"type": "string", "description": "The reservation id."}
                },
                "required": ["reservation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_all_airports",
            "description": "List every airport. Takes no arguments.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def chat(base, model, messages, **kw):
    body = {"model": model, "messages": messages, "temperature": 0.0, **kw}
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]


def main(base, model):
    failures = []

    def check(name, ok, detail):
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    print(f"verifying {model} at {base}")

    # 1 + 2. Tool calling, including the zero-argument case. Both tools are
    # offered at once and the model is asked for both, so a single response
    # exercises the parser on an argument-bearing call and an empty one.
    m = chat(
        base, model,
        [{"role": "user",
          "content": "Look up reservation ABC123, and also list all airports. "
                     "Call both tools."}],
        tools=TOOLS, tool_choice="auto",
    )
    calls = m.get("tool_calls") or []
    names = [c["function"]["name"] for c in calls]
    check("tools", bool(calls), f"{len(calls)} tool_calls {names}"
          + ("" if calls else f" -- content was: {(m.get('content') or '')[:120]!r}"))

    noarg = [c for c in calls if c["function"]["name"] == "list_all_airports"]
    if noarg:
        args = noarg[0]["function"]["arguments"]
        ok = args.strip() != ""
        try:
            json.loads(args)
        except Exception:
            ok = False
        check("no-args", ok, f"arguments={args!r}")
    else:
        check("no-args", False, "zero-argument tool was never called")

    # 3. A full round trip. tau2 replays the tool result back as a `tool`
    # message; a server that cannot ingest that shape dies mid-conversation
    # rather than at startup. Czech in the system prompt because every L2
    # Interaction cell puts it there.
    if calls:
        first = calls[0]
        m2 = chat(
            base, model,
            [
                {"role": "system", "content": "Jsi agent letecké společnosti. "
                                              "Odpovídej výhradně česky."},
                {"role": "user", "content": "Jaká je moje rezervace ABC123?"},
                {"role": "assistant", "content": None, "tool_calls": [first]},
                {"role": "tool", "tool_call_id": first["id"],
                 "content": json.dumps({"reservation_id": "ABC123",
                                        "passenger": "Jan Novák",
                                        "flight": "OK123 PRG-LHR",
                                        "status": "confirmed"})},
            ],
            tools=TOOLS,
        )
        text = m2.get("content") or ""
        # Czech-only diacritics; the point is to catch an agent that answers in
        # English despite a Czech system prompt, not to do real language ID.
        czech = any(ch in text for ch in "ěščřžýáíéůúňť")
        check("roundtrip", bool(text.strip()), f"{len(text)} chars")
        check("roundtrip-czech", czech, repr(text[:120]))

    # 4. Thinking. Qwen3.6 thinks by default, so this needs no request kwarg --
    # what matters is that the reasoning is SEPARATED. If it leaked into
    # `content` the judge would grade the agent's scratchpad as its utterance.
    m3 = chat(base, model,
              [{"role": "user", "content": "Kolik je 17 * 23? Odpověz jen číslem."}],
              max_tokens=2048)
    reasoning = m3.get("reasoning_content") or m3.get("reasoning") or ""
    content = m3.get("content") or ""
    check("thinking-separated", len(reasoning) > 0,
          f"reasoning={len(reasoning)} chars, content={content[:60]!r}")
    check("thinking-not-leaked", "<think>" not in content,
          "no <think> in content" if "<think>" not in content else repr(content[:120]))

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
