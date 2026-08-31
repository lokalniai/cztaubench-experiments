#!/usr/bin/env python3
"""Assemble the public repository: static site, raw results, scripts, docs.

    python scripts/publish_repo.py [--out experiments] [--skip-build]

Three things this does that a `cp -r` would not:

**It redacts the API key out of the raw results.** Every `results.json` tau2
wrote carries the live e-infra key in `info.user_info.llm_args.api_key` (and in
the agent and judge blocks beside it), because tau2 records the LLM arguments it
was called with and the key is one of them. The rendered pages never show it --
the viewer prints policies and transcripts, not call arguments -- so this is the
one artifact that cannot be published as it stands. Every `api_key` is replaced
with a placeholder, and the copy is then re-scanned for the literal secret; a hit
aborts the publish rather than warning about it.

**It refuses to run on a dirty secret set.** The scan looks for the values in
`SEATauBench/.env`, so a key that was rotated after the runs finished is still
caught in the old files: the value that matters is the one on disk in the data,
not the one currently in use.

**It excludes what has its own home.** The SEATauBench checkout is upstream code
under its own licence and is not vendored; `logs/`, `discarded/` and the venv are
noise for a reader. `docs/DEPLOYMENT.md` -- the protocol and this layout -- stays
local by choice, so it is not copied out either.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

SIMS = ROOT / "SEATauBench" / "data" / "simulations"
ENV_FILE = ROOT / "SEATauBench" / ".env"

PLACEHOLDER = "REDACTED-see-README"

# Field names whose value is a credential wherever it appears. Matched exactly,
# not by substring: a task or a policy may legitimately contain a field called
# "key", and rewriting those would corrupt the data this repository exists to
# publish.
SECRET_FIELDS = {"api_key", "apiKey", "api-key", "authorization", "Authorization"}

SCRIPTS = [
    "env.sh", "run_cell.sh", "submit_chain.sh", "run_annotate.sh",
    "vllm_server.sh", "download_model.sh", "report.py", "viewer.py",
    "build_site.py", "publish_repo.py", "annotate_language.py",
    "rescore_language.py", "fetch_reference.py", "verify_server.py",
    "probe_thinking.py",
]

# DEPLOYMENT.md is ignored rather than merely not copied: it used to be published
# here, and an ignore entry is what stops a stale copy from being re-added by a
# `git add -A` in the checkout.
GITIGNORE = """\
__pycache__/
*.pyc
.env
.DS_Store
/DEPLOYMENT.md
"""


def secrets() -> list[str]:
    """The credential values to hunt for, read from the untracked .env."""
    out = []
    if not ENV_FILE.exists():
        return out
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip("\"'")
        if ("KEY" in k.upper() or "TOKEN" in k.upper()) and len(v) > 8:
            out.append(v)
    return out


def scrub(node):
    """Replace every credential field in a parsed JSON tree, in place."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k in SECRET_FIELDS and isinstance(v, str):
                node[k] = PLACEHOLDER
            else:
                scrub(v)
    elif isinstance(node, list):
        for v in node:
            scrub(v)
    return node


RAW_DATA_NOTE = {
    "full": "verbatim",
    "reasoning": ("provider response envelopes dropped; the reasoning traces "
                  "inside them kept at their original path"),
    "none": "dropped entirely",
}


def slim(doc: dict, mode: str) -> dict:
    """Thin the per-message provider envelopes out of one results document.

    `raw_data` is the whole response object the provider returned, and it is
    two thirds of the file: 333 MB of the 610 MB these twenty files weigh. Most
    of that is redundant by construction -- the assistant text is already the
    message's own `content`, and the ids, fingerprints and service tiers beside
    it describe the HTTP call rather than the experiment. Token counts are not
    lost with it: `usage` sits on the message itself, outside `raw_data`.

    The one thing in there that exists nowhere else is the **user simulator's**
    reasoning, 82 MB of it. No page shows it -- the viewer looks for
    `raw_data.reasoning_content` while the data carries
    `raw_data.choices[].message.reasoning_content`, so the thinking panel has
    never rendered against these runs -- but it is the only record of how the
    simulated customer decided what to say, which docs/DEPLOYMENT.md §7 names as
    a threat to validity. `reasoning` keeps exactly that and drops the rest.
    """
    if mode == "full":
        return doc
    for sim in doc.get("simulations") or []:
        for m in sim.get("messages") or []:
            rd = m.get("raw_data")
            if not isinstance(rd, dict):
                continue
            kept = None
            if mode == "reasoning":
                for ch in rd.get("choices") or []:
                    reasoning = ((ch or {}).get("message") or {}).get(
                        "reasoning_content")
                    if reasoning:
                        # Original path, so a reader parsing the real response
                        # shape still finds it where it always was.
                        kept = {"choices": [{"message":
                                             {"reasoning_content": reasoning}}]}
                        break
            m["raw_data"] = kept
    return doc


def copy_results(out_root: Path, known: list[str], raw_mode: str) -> int:
    """Copy every cell's results.json, redacted and thinned, verifying the copy."""
    total = 0
    dest_root = out_root / "results" / "simulations"
    shutil.rmtree(dest_root, ignore_errors=True)
    for src in sorted(SIMS.rglob("results.json")):
        rel = src.relative_to(SIMS)
        if "smoke" in str(rel) or "probe" in str(rel):
            continue
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Reformatted rather than byte-copied: redaction means re-serialising,
        # and indent=2 is what tau2 itself wrote, so a diff against a local run
        # shows the redacted fields and nothing else.
        doc = slim(scrub(json.loads(src.read_text(encoding="utf-8"))), raw_mode)
        # The file says what was done to it, so a reader comparing against a
        # local run does not have to guess which differences are deliberate.
        # Not named `api_key`: the check below is a blunt regex over the whole
        # file, and a stamp saying the keys are redacted would trip it.
        doc.setdefault("info", {})["published"] = {
            "tool": "scripts/publish_repo.py",
            "credentials": "api_key fields redacted",
            "raw_data": RAW_DATA_NOTE[raw_mode],
        }
        dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        text = dest.read_text(encoding="utf-8")
        for s in known:
            if s in text:
                raise SystemExit(f"ABORT: secret still present in {dest}")
        if re.search(r'"api_key"\s*:\s*"(?!REDACTED|dummy)', text):
            raise SystemExit(f"ABORT: unredacted api_key in {dest}")
        total += dest.stat().st_size
        print(f"  {rel}  {dest.stat().st_size / 1e6:.1f} MB")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments")
    ap.add_argument("--skip-build", action="store_true",
                    help="reuse the site already in --out")
    ap.add_argument("--raw-data", choices=("reasoning", "none", "full"),
                    default="reasoning",
                    help="how much of each message's provider envelope to "
                         "publish: 'reasoning' keeps only the traces inside it "
                         "(default, 358 MB), 'none' drops it (276 MB), 'full' "
                         "publishes it verbatim (610 MB). The local runs are "
                         "never modified whichever is chosen.")
    args = ap.parse_args()
    out = args.out.resolve()
    t0 = time.time()

    known = secrets()
    print(f"{len(known)} credential value(s) to scan for")

    if not args.skip_build:
        import build_site
        build_site.build(out, None)

    print(f"\nraw results (raw_data: {RAW_DATA_NOTE[args.raw_data]}):")
    size = copy_results(out, known, args.raw_data)
    print(f"  {size / 1e6:.0f} MB of raw results")

    # Annotations and the vendored reference numbers: no credentials, copied as
    # they are so the site and the data behind it stay byte-comparable.
    for name in ("language_annotations.json", "tau2_reference.json"):
        src = ROOT / "results" / name
        if src.exists():
            shutil.copy2(src, out / "results" / name)

    sdir = out / "scripts"
    sdir.mkdir(parents=True, exist_ok=True)
    for name in SCRIPTS:
        src = HERE / name
        if src.exists():
            shutil.copy2(src, sdir / name)

    # The README is copied verbatim: it is the front page of the published
    # repository and nothing else, so there is no header to splice in. The
    # protocol it used to carry now lives in docs/DEPLOYMENT.md, which stays
    # local -- it is not copied here and is ignored in the checkout.
    shutil.copy2(ROOT / "README.md", out / "README.md")

    # Pages runs Jekyll by default; with 5,764 pages there is nothing for it to
    # do and a build step that can only fail or time out.
    (out / ".nojekyll").write_text("")
    (out / ".gitignore").write_text(GITIGNORE)

    files = sum(1 for p in out.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\n{files} files, {total / 1e6:.0f} MB in {time.time() - t0:.0f}s")
    print(f"repository ready at {out}")


if __name__ == "__main__":
    main()
