# Repository layout and deployment

How the CzTauBench round is split across a private working root, one public
repository of results, and a static site served from GitHub Pages. The same
three-layer split as the BenchMAX-cs round, with three differences forced by
what this benchmark produces; each is noted where it comes up.

## The three layers

**Private working root** (`~/projects/cztaubench/`, never published). Holds the
SEATauBench checkout, `logs/` (Slurm output, ~40 MB), `discarded/` (the FP8 35B
run, §3.1), the papers, and the `.venv` name file. The public repository is
checked out inside it as `experiments/`.

**Public repository** (`experiments/` ->
[github.com/lokalniai/cztaubench-experiments](https://github.com/lokalniai/cztaubench-experiments)).
Carries `results/` (raw tau2 output), `scripts/` (how it was produced),
`README.md` (the protocol), and the built site. `results/` is the single source
of truth: every number on the site and every table in `README.md` §9 reduces from
those files.

**Site** (the repository root). Static pages that read nothing at run time —
they are the reduction, already applied. Deleting `index.html`, `sim/`,
`compare/`, `cell/`, `annotations.html` and `static/` loses no information,
because `python scripts/build_site.py` regenerates all of it from `results/`.

**Difference 1: the site is at the repository root, not in `viewer/`.** The URL
was fixed in advance (`lokalniai.github.io/cztaubench-experiments`), and Pages
maps a repository path to a URL path, so a `viewer/` directory would have put the
page one level down from where it was meant to be.

## What the public repository excludes

```
SEATauBench/     upstream checkout; clone it and apply README.md §4
logs/            Slurm output, one file per cell
discarded/       the FP8 35B-A3B run, kept locally for provenance (§3.1)
*.prelang        pre-§6.1 backups of results.json, superseded by the file beside them
```

The upstream code has its own home and its own MIT licence; §4 of `README.md`
records the six files changed, so the checkout stays reproducible without being
vendored.

## Layout

```
experiments/
  index.html                     leaderboard, pass^1 (index-p2, index-p3 for the others)
  annotations.html               every flagged span, filtered in the browser
  sim/<run>/<cell>/<id>.html     4,920 trajectories
  compare/<run>/<domain>/<t>.html  820 EN/CS side-by-sides
  cell/<run>/<cell>.html         redirect into a cell, as the live viewer does
  static/style.css               one stylesheet for 5,764 pages
    picker/<run>__<cell>.js      the trajectory chip strip, once per cell
    brief/<sha>.js               task briefing panels, content-addressed
  results/
    simulations/<run>/<cell>/results.json   raw tau2 output, redacted and thinned
    language_annotations.json    the Czech judge's spans
    tau2_reference.json          vendored published tau2-bench numbers
  scripts/                       the pipeline, including build_site.py
  README.md                      the protocol: §1-§10
```

## One reduction, two front ends

`scripts/viewer.py` runs a live server that re-reads `results/` on every request,
which is what you want while a run is still landing. `scripts/build_site.py`
imports it and calls the same view functions once, ahead of time. The static site
cannot drift from the live view, because there is only one implementation of what
a score is, which turns are flagged, and where a span sits in a trace.

**Difference 2: the pages are prerendered, not a JavaScript app over JSON.** The
BenchMAX viewer ships data files and renders in the browser. Here the renderer
already existed as 1,800 lines of server-side Python, and rewriting it in JS
would have created exactly the second implementation the split above exists to
avoid. Three things move to the browser anyway, each because a file server cannot
do them:

* **Query strings become paths.** Pages has no routing, so
  `/sim?run=X&cell=Y&id=Z` is rewritten to `sim/X/Y/Z.html`, and rewritten
  *relative* to the page holding the link — the site is served from a repository
  subpath, where a root-relative `/sim/...` resolves against the domain and 404s.
* **Repeated fragments are hoisted.** The live viewer inlines the stylesheet, the
  sim picker and the briefing panels into every page. The picker alone is 63 KB
  on each of 3,420 retail pages; inlined, it would cost more than 200 MB to say
  the same thing 342 times per cell. Both are pulled in from `static/` instead,
  and the briefings are content-addressed, which collapses 4,920 renderings into
  328 files. Sim pages fall from ~120 KB to ~25 KB.
* **Filtering happens in place.** `/annotations?run=X&cat=MAJOR` is 18
  combinations over the same 3,661 rows; prerendering them all writes ~36 MB to
  say the same thing 18 times. The rows are emitted once and the filter bar is
  buttons that hide rows.

## The redaction step, which is not optional

**Difference 3: the raw results cannot be published as they stand.** tau2 records
the arguments it called each LLM with, and the API key is one of them, so every
`results.json` carries the live e-infra key in plaintext at
`info.user_info.llm_args.api_key` and in the agent and judge blocks beside it.
The rendered pages never expose it — the viewer prints policies and transcripts,
not call arguments — so `results/` is the only artifact that needs handling.

`scripts/publish_repo.py` replaces every `api_key` field with a placeholder,
re-reads the copy, and aborts the publish if the literal secret from
`SEATauBench/.env` survives anywhere in it.

## What else the published results drop, and why

The same script thins one field, `raw_data`, which is the entire provider
response stored per message. It is **333 MB of the 610 MB** these twenty files
weigh, and almost all of it is redundant by construction: the assistant's text is
already the message's own `content`, and the ids, fingerprints and service tiers
beside it describe the HTTP call rather than the experiment. Token counts survive
the cut — `usage` sits on the message, outside `raw_data`.

One thing inside those envelopes exists nowhere else: the **user simulator's**
reasoning, 82 MB of it. No page has ever shown it — the viewer looks for
`raw_data.reasoning_content` while the data carries
`raw_data.choices[].message.reasoning_content`, so that panel is dead code
against these runs — but it is the only record of how the simulated customer
decided what to say, which `README.md` §7 names as a threat to validity. It is
kept, at its original path.

| `--raw-data` | published size | what is lost |
|---|---|---|
| `reasoning` (default) | 358 MB | response envelopes; nothing unique |
| `none` | 276 MB | the simulator's reasoning traces as well |
| `full` | 610 MB | nothing |

Three properties hold whichever is chosen:

* **The local runs are never modified.** The thinning happens on the way out; a
  full-fidelity copy stays in `SEATauBench/data/simulations/`.
* **Every published number is still reproducible from the published data.**
  Rewards, transcripts, `reward_info`, timings and token counts are untouched, so
  `scripts/report.py` over `results/simulations/` returns §9's tables exactly.
* **Each file records what was done to it**, at `info.published`, so a reader
  diffing against their own run can tell a deliberate change from a discrepancy.

Two consequences worth stating plainly:

* **The key that was used for these runs should be treated as spent.** It sat in
  plaintext in 20 files on a shared filesystem for the length of the project.
  Rotating it costs one line in `.env`; the published data does not depend on it.
* **The scan reads `.env`, so it catches the old value too.** If the key is
  rotated before publishing, the files still contain the *previous* one — which
  is exactly why the check hunts for the values on disk rather than trusting the
  field names alone. Keep the retired value in `.env` (commented out is not
  enough — it is parsed) until a publish has confirmed the data is clean.

## Deployment

GitHub Pages, serving branch `main` from `/` (Settings -> Pages -> Source:
Deploy from a branch). No Actions workflow, no `gh-pages` branch, no build step
on GitHub's side. `.nojekyll` sits at the root: with 5,764 prerendered pages
there is nothing for Jekyll to do and a build step that can only fail or time
out.

```bash
python scripts/build_site.py          # ~7 min, writes into experiments/
python scripts/publish_repo.py        # rebuilds, then syncs data + docs
cd experiments && git add -A && git commit -m "Rebuild site" && git push
```

Pages then serves the tree verbatim:

    https://lokalniai.github.io/cztaubench-experiments/

`results/` is served too, at `.../results/simulations/<run>/<cell>/results.json`,
which is the point: anyone doubting a number on the site can open the file it
came from and rerun `scripts/report.py` over it.

## Setting this up for another round

1. **Create the public repository and check it out inside the working root.**
   Keep the working root private. The site's URL is its path in the repository,
   so put `index.html` where you want the URL to point.

2. **Write the reduction once, and have the build import it.** Resolve paths
   from the module's own location, never from the working directory.

3. **Look for credentials in the data before the first push, not after.** Any
   harness that records the arguments it called a model with has probably
   recorded the key. Scan the *values* from your `.env`, not field names.

4. **Hoist whatever repeats across pages** — stylesheet, navigation, anything
   rendered per page but identical within a group. Content-address it if you
   cannot tell in advance which copies are equal.

5. **Rewrite links relative to the page.** A site served from a repository
   subpath breaks on every root-relative href, and the failure is invisible
   locally, where `file://` and a dev server both resolve them fine.

6. **Rebuild in the same commit as the data changes.** A build that lags its
   data is worse than no build: the page looks authoritative and shows last
   week's numbers.
