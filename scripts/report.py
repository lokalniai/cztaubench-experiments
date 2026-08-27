#!/usr/bin/env python
"""Progress and results for CzTauBench runs.

Two things this guards against, both of which have already bitten us:

* Partial runs read as finished. pass^k is computed only over tasks that have
  at least k completed trials, so a run that is 3% done reports a confident
  pass^1 over a handful of easy tasks. Rows that are not complete are marked
  PARTIAL and their metrics should be treated as noise.
* Infrastructure errors hide. tau2 EXCLUDES them from every metric, so a run
  dying on rate limits looks empty rather than broken. The infra column and the
  FAILING marker surface that.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from collections import Counter
from pathlib import Path

# Silence tau2's import-time registry chatter before importing it.
os.environ.setdefault("LOGURU_LEVEL", "CRITICAL")

from tau2.data_model.simulation import Results, TerminationReason  # noqa: E402
from tau2.metrics.agent_metrics import compute_metrics  # noqa: E402

INFRA_ERROR = TerminationReason.INFRASTRUCTURE_ERROR

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "SEATauBench" / "data" / "simulations"
REFERENCE = ROOT / "SEATauBench" / "data" / "seatau" / "experiments.csv"

REF_MODEL = "qwen-3-235b-it"  # the paper's open-weights agent row
# Throughput probes and smoke tests live alongside real runs; they are not results.
SCRATCH = re.compile(r"smoke|probe|verify|scratch|_test", re.I)


def load_reference() -> dict[tuple[str, str, str], dict]:
    if not REFERENCE.exists():
        return {}
    out = {}
    with REFERENCE.open() as fh:
        for row in csv.DictReader(fh):
            if row["normalized_agent_llm"] == REF_MODEL:
                out[(row["scenario"], row["domain"], row["language_senario"])] = row
    return out


def summarise(path: Path) -> dict:
    results = Results.load(str(path))
    sims = results.simulations
    n_tasks = len(results.tasks)
    n_trials = getattr(results.info, "num_trials", None) or 1
    expected = n_tasks * n_trials

    # Match the enum, not its str(): str(TerminationReason.INFRASTRUCTURE_ERROR)
    # is the qualified UPPERCASE name, so the obvious .endswith("error") test is
    # always False and this column would sit at 0 even for a wholly failed run --
    # exactly the silence the module docstring claims to break.
    good = [s for s in sims if s.termination_reason != INFRA_ERROR]
    infra = len(sims) - len(good)

    # How many trials each task has actually finished -> which pass^k are real.
    per_task = Counter(s.task_id for s in good)
    covered = {
        k: sum(1 for t in results.tasks if per_task.get(t.id, 0) >= k)
        for k in range(1, n_trials + 1)
    }
    complete = len(good) >= expected

    metrics = compute_metrics(results)
    pk = dict(metrics.pass_hat_ks or {})
    p1 = pk.get(1)

    durations = [s.duration for s in good if getattr(s, "duration", 0)]
    return {
        "done": len(good),
        "expected": expected,
        "remaining": max(expected - len(good), 0),
        "infra": infra,
        "n_trials": n_trials,
        "covered": covered,
        "complete": complete,
        "avg_reward": metrics.avg_reward,
        "p1": p1,
        "p2": pk.get(2),
        "p3": pk.get(3),
        "rho3": (pk[3] / p1) if (p1 and pk.get(3) is not None) else None,
        "mean_dur": (sum(durations) / len(durations)) if durations else None,
        "mtime": path.stat().st_mtime,
    }


def fmt(v, w: int = 6) -> str:
    return ("-" if v is None else f"{v:.3f}").rjust(w)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument("--all", action="store_true", help="include probe/smoke runs")
    args = ap.parse_args()

    reference = load_reference()
    files = sorted(args.results_dir.glob("**/results.json"))
    if not args.all:
        files = [f for f in files if not SCRATCH.search(str(f.relative_to(args.results_dir)))]
    if not files:
        print(f"No runs found under {args.results_dir}")
        return

    hdr = (
        f"{'cell':<26} {'progress':>13} {'infra':>5} "
        f"{'pass^1':>6} {'pass^2':>6} {'pass^3':>6} {'rho_3':>6}  "
        f"{'paper p^1':>9} {'rho3':>6}  status"
    )
    print(hdr)
    print("-" * len(hdr))

    partial_notes = []
    for path in files:
        cell, run = path.parent.name, path.parent.parent.name
        try:
            r = summarise(path)
        except Exception as exc:
            print(f"{cell:<26} !! {type(exc).__name__}: {exc}")
            continue

        parts = cell.split("_")
        key = None
        if cell.startswith("english"):
            key = ("english", parts[1], "english")
        elif cell.startswith("l2_interaction"):
            key = ("l2_interaction", parts[2], None)

        ref_p1 = ref_rho = None
        if key and key[2]:
            ref = reference.get(key)
            if ref:
                ref_p1, ref_rho = float(ref["pass_hat_1"]), float(ref["rho_hat_3"])
        elif key:
            sea = [v for (sc, dm, _l), v in reference.items()
                   if sc == key[0] and dm == key[1]]
            if sea:
                ref_p1 = sum(float(v["pass_hat_1"]) for v in sea) / len(sea)
                ref_rho = sum(float(v["rho_hat_3"]) for v in sea) / len(sea)

        pct = 100.0 * r["done"] / r["expected"] if r["expected"] else 0.0
        progress = f"{r['done']}/{r['expected']} {pct:5.1f}%"

        if r["infra"] and r["infra"] >= max(r["done"], 1) / 2:
            status = "FAILING"
        elif r["complete"]:
            status = "complete"
        else:
            status = "PARTIAL"
            idle = (time.time() - r["mtime"]) / 60
            note = (
                f"  {cell}: {r['remaining']} sims remaining; "
                f"trial coverage " +
                ", ".join(f"{k}x:{v}/{r['expected'] // r['n_trials']}"
                          for k, v in r["covered"].items())
            )
            if r["mean_dur"]:
                note += f"; mean {r['mean_dur']:.0f}s/sim"
            note += f"; last write {idle:.0f} min ago"
            partial_notes.append(note)

        print(
            f"{cell:<26} {progress:>13} {r['infra']:>5} "
            f"{fmt(r['p1'])} {fmt(r['p2'])} {fmt(r['p3'])} {fmt(r['rho3'])}  "
            f"{fmt(ref_p1, 9)} {fmt(ref_rho, 6)}  {status}  [{run}]"
        )

    if partial_notes:
        print("\nIn progress:")
        for n in partial_notes:
            print(n)

    print(
        "\nPARTIAL rows: pass^k is computed only over tasks having >=k finished\n"
        "trials, so early numbers are unrepresentative -- ignore until complete.\n"
        "infra: sims killed by rate limits / connection errors. Excluded from all\n"
        "other columns, so a high count invalidates the row (auto-resume retries them).\n"
        "paper: Qwen3-235B-A22B-Inst from experiments.csv; for l2_interaction it is\n"
        "the mean over the five SEA languages (there is no Czech row)."
    )


if __name__ == "__main__":
    main()
