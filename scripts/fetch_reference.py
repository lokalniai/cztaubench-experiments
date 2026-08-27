#!/usr/bin/env python
"""Vendor the published tau2-bench leaderboard numbers into results/.

The viewer must render with no network (it runs on a cluster node that may not
have egress, and a leaderboard that silently loses its reference bars when a
fetch fails is worse than one that has none). So the numbers are pulled once,
written to results/tau2_reference.json, and read from disk thereafter.

    python scripts/fetch_reference.py

Only per-domain `core` results are kept, and only airline/retail/telecom -- the
domains CzTauBench actually runs. Two deliberate omissions:

* banking_knowledge is skipped. It is a separate, much harder domain (AllTools
  retrieval, 4 trials), and every 2026-08 frontier submission -- Claude Opus 5,
  GPT-5.6-sol, Kimi K3, Qwen 3.8 Max, Gemini 3.1 Pro -- reports ONLY that one,
  with airline/retail/telecom explicitly null. Its pass^k is not on our scale
  (top score there is 0.55 against 0.84 on airline), so those submissions cannot
  appear beside ours no matter how much one would like them to.
* the site's headline core pass^1 is skipped too. It is the unweighted mean over
  the three domains, and telecom -- which we do not run -- is by far the easiest
  (0.85-0.98), so the headline sits well above the airline/retail reality. We
  compare per domain instead.

Even per domain the comparison is indicative, not like-for-like: those runs use
a gpt-5.2 user simulator, ours uses Kimi K3, and the simulator is a large part
of what a tau2 score measures.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "tau2_reference.json"

REPO = "sierra-research/tau2-bench"
SUBS = "web/leaderboard/public/submissions"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/{SUBS}"
API = f"https://api.github.com/repos/{REPO}/contents/{SUBS}"

# Which rows the viewer draws as reference bars. Pinned by directory key, not by
# model name: GPT-5.2 was submitted twice, at reasoning high and none, and only
# the high one belongs next to our thinking-on runs. Everything else with core
# results is still written to the file, so widening the set is a viewer edit
# rather than another fetch.
FEATURED = [
    "qwen3.5-397b-a17b-think_sierra_2026-03-02",
    "claude-opus-4-5_sierra_2026-02-26",
    "gemini-3-pro_sierra_2026-03-02",
    "gpt-5-2_sierra_2026-02-26",
]


def get(url: str) -> dict | list:
    with urllib.request.urlopen(url, timeout=30) as fh:
        return json.load(fh)


def pk(d: dict | None) -> dict | None:
    """Percentages upstream, fractions here -- our own metrics are fractions."""
    if not d:
        return None
    out = {}
    for k in (1, 2, 3, 4):
        v = d.get(f"pass_{k}")
        if v is not None:
            out[f"p{k}"] = v / 100.0
    return out or None


def main() -> None:
    manifest = get(f"{RAW}/manifest.json")
    current = manifest["submissions"]  # excludes legacy + voice tracks

    core = []
    for name in current:
        d = get(f"{RAW}/{name}/submission.json")
        res = d.get("results") or {}
        meth = d.get("methodology") or {}
        air, ret = pk(res.get("airline")), pk(res.get("retail"))
        if not (air or ret):
            continue
        core.append({
            "key": name,
            "model": d.get("model_name"),
            "org": d.get("model_organization"),
            "effort": d.get("reasoning_effort"),
            "tau2_version": meth.get("tau2_bench_version"),
            "user_simulator": meth.get("user_simulator"),
            "date": d.get("submission_date"),
            "featured": name in FEATURED,
            "airline": air,
            "retail": ret,
            "telecom": pk(res.get("telecom")),
        })

    missing = [n for n in FEATURED if n not in {r["key"] for r in core}]
    if missing:
        raise SystemExit(f"featured submissions have no core results: {missing}")

    core.sort(key=lambda r: -(r.get("airline") or {}).get("p1", 0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "fetched": date.today().isoformat(),
        "source": f"https://github.com/{REPO}/tree/main/{SUBS}",
        "leaderboard": "https://taubench.com/leaderboard?benchmark=core",
        "core": core,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {OUT}")
    print(f"  core submissions: {len(core)}  (featured: {sum(r['featured'] for r in core)})")


if __name__ == "__main__":
    main()
