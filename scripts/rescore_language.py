#!/usr/bin/env python3
"""Recompute stored language_correctness for finished runs.

The language metric is computed once, at evaluation time, and frozen into each
simulation's reward_info.info.language_correctness. Changing how it is computed
therefore does NOT change any run that has already finished -- the old numbers
sit on disk and every downstream reader (report.py, the viewer, the analysis
scripts) keeps showing them. This script replays just that one metric over the
existing message logs so completed runs reflect the current definition.

It is safe to re-run: it recomputes from the messages every time rather than
adjusting the previous value, so running it twice gives the same answer as
running it once.

Only reward_info.info.language_correctness is touched. Task reward, pass^k, and
every other field are left exactly as they were -- our runs do not include
LANGUAGE_CORRECTNESS in reward_basis (no run's reward_breakdown contains it), so
this metric multiplies nothing and rewriting it cannot move a task score. The
script asserts that before writing, and refuses the file if it is ever false.

Usage:
    python scripts/rescore_language.py            # report only, writes nothing
    python scripts/rescore_language.py --write    # rewrite results.json in place
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "SEATauBench" / "src"))

# The fastText model path resolves relative to the working directory, so running
# this from the repo root finds no model and every score silently becomes None.
# run_cell.sh already chdirs here; do the same so the answer does not depend on
# where the script was invoked from.
os.chdir(ROOT / "SEATauBench")

from seatau.metrics.language_use import (  # noqa: E402
    compute_role_language_correctness,
    infer_expected_language,
    load_fasttext_model,
)


def expected_lang_for(results: dict) -> str:
    """Read the run's target language out of its own stored config."""
    info = results.get("info") or {}
    # lang_id / lang_components sit directly on info (an English cell simply has
    # lang_id=None). infer_expected_language returns "en" unless agent_system is
    # among the translated components, which is the same rule the evaluator used
    # when it wrote the stored value -- so a cell whose score does not move is
    # genuine agreement, not both sides defaulting to English.
    return infer_expected_language(
        role="assistant",
        lang_id=info.get("lang_id"),
        lang_components=info.get("lang_components") or [],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="rewrite files in place")
    ap.add_argument("--results-dir", type=Path,
                    default=ROOT / "SEATauBench" / "data" / "simulations")
    args = ap.parse_args()

    model, err = load_fasttext_model()
    if model is None:
        print(f"fastText unavailable: {err}", file=sys.stderr)
        return 1

    files = sorted(args.results_dir.glob("**/results.json"))
    if not files:
        print(f"no results.json under {args.results_dir}", file=sys.stderr)
        return 1

    print(f"{'cell':58s} {'stored':>8s} {'recomputed':>11s} {'delta':>7s} {'sims':>6s}")
    for path in files:
        results = json.loads(path.read_text())
        sims = results.get("simulations") or []
        if not sims:
            continue

        expected = expected_lang_for(results)
        old_total = new_total = 0.0
        counted = changed = 0

        for sim in sims:
            ri = sim.get("reward_info")
            if not ri:
                continue
            # Guard: if a run ever DOES fold language into the reward, rewriting
            # the metric silently would desync it from the stored reward.
            if "LANGUAGE_CORRECTNESS" in (ri.get("reward_breakdown") or {}):
                print(f"  SKIP {path}: language is part of reward_basis here",
                      file=sys.stderr)
                break

            fresh = compute_role_language_correctness(
                messages=sim.get("messages") or [],
                role="assistant",
                expected_language=expected,
                detector_model=model,
            )
            info = ri.setdefault("info", {})
            prev = (info.get("language_correctness") or {}).get("score")
            if prev is not None:
                old_total += prev
                counted += 1
                if fresh.get("score") is not None:
                    new_total += fresh["score"]
                    if abs(fresh["score"] - prev) > 1e-9:
                        changed += 1
            info["language_correctness"] = fresh
        else:
            if counted:
                o, n = old_total / counted, new_total / counted
                label = "/".join(path.parts[-3:-1])
                print(f"{label:58s} {o:8.4f} {n:11.4f} {n - o:+7.4f} {changed:4d}/{counted}")
            if args.write:
                # Keep one backup per file so a bad definition is recoverable
                # without re-running the simulations themselves.
                backup = path.with_suffix(".json.prelang")
                if not backup.exists():
                    shutil.copy2(path, backup)
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False))
                tmp.replace(path)

    if not args.write:
        print("\n(dry run -- nothing written; pass --write to apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
