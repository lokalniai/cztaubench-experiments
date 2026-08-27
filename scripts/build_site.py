#!/usr/bin/env python3
"""Build the static results site from the same reduction the live viewer uses.

    python scripts/build_site.py --out experiments [--limit N]

`viewer.py` answers routes out of a running process. This calls the same view
functions once, ahead of time, and writes their output as files GitHub Pages can
serve. The reduction is imported, never restated, so the published page cannot
drift from the live one -- the only differences are the three below, each forced
by the absence of a server.

**Query strings become paths.** Pages has no routing: `/sim?run=X&cell=Y&id=Z`
is a request for a file literally named `sim`. Every internal link is therefore
rewritten to a real path (`sim/X/Y/Z.html`), and rewritten *relative* to the page
that carries it, because the site is served from a repository subpath
(`/cztaubench-experiments/`) where a root-relative `/sim/...` would resolve
against the domain and 404.

**Two fragments are shared instead of inlined.** The live viewer inlines the
stylesheet, the sim picker and the briefing panels into every page, which is free
when one process serves one reader. Committed to a repository it is not: the
picker alone is 63 KB on each of 3,420 retail pages, so inlining it would cost
more than 200 MB to say the same thing 342 times per cell. Both are hoisted into
files that a page pulls in, and the stylesheet with them. The transcript itself
stays inline -- it is what the page is for, and it is different every time.

**Filtering moves to the browser.** `/annotations?run=X&cat=MAJOR` is 18 filter
combinations over the same 4,174 rows. Prerendering all of them writes ~36 MB to
say the same thing 18 times, so the rows are emitted once and the filter bar is
replaced by buttons that hide rows in place.

Everything else -- what a score is, which turns are flagged, where a span sits in
a trace -- comes from `viewer.py` untouched.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import posixpath
import re
import shutil
import sys
import time
from hashlib import sha1
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import viewer as V  # noqa: E402

# Where each kind of page lands. Sim pages of one cell share a directory, which
# is what lets their shared chip strip use bare `<id>.html` links.
SIM_DIR = "sim/{run}/{cell}"
CMP_DIR = "compare/{run}/{domain}"

# Shared fragments, keyed by content where they repeat across cells.
pickers: dict[str, str] = {}   # "<run>__<cell>" -> chip strip HTML
briefs: dict[str, str] = {}    # sha1 -> briefing panels HTML

# Filled during discovery; needed to resolve a `/compare` link that names no
# task, which the live viewer answers by falling through to the first shared one.
first_task: dict[tuple[str, str], str] = {}


# ── URL mapping ──────────────────────────────────────────────────────────────

def site_path(url: str) -> tuple[str, str] | None:
    """Map one viewer URL to (site-relative path, fragment).

    Returns None for anything that is not an internal link, which the caller
    leaves exactly as it found it -- external hrefs, `mailto:`, and the empty
    href the reference table emits when a leaderboard row has no upstream row.
    """
    u = urlparse(html.unescape(url))
    if u.scheme or u.netloc or not u.path.startswith("/"):
        return None
    q = parse_qs(u.query)
    one = lambda k: (q.get(k) or [""])[0]  # noqa: E731
    p, frag = u.path, u.fragment

    if p == "/":
        m = one("m") or "p1"
        return ("index.html" if m == "p1" else f"index-{m}.html", frag)
    if p == "/annotations":
        # Filters are applied in the browser, so every filtered URL is the same
        # document. Dropping the query here is what makes that true.
        return ("annotations.html", frag)
    if p == "/sim":
        run, cell, sid = one("run"), one("cell"), one("id")
        return (f"{SIM_DIR.format(run=run, cell=cell)}/{sid}.html", frag)
    if p == "/compare":
        run, cell = one("run"), one("cell")
        domain = V.parse_cell(cell)[1] if cell else "airline"
        task = one("task") or first_task.get((run, domain), "")
        return (f"{CMP_DIR.format(run=run, domain=domain)}/{task}.html", frag)
    if p == "/cell":
        return (f"cell/{one('run')}/{one('cell')}.html", frag)
    if p == "/favicon.ico":
        return ("static/favicon.ico", frag)
    if p.startswith("/static/"):
        return (p[1:], frag)
    return None


LINK_RE = re.compile(r"""\b(href|src)=(['"])(/[^'"]*)\2""")


def rewrite(text: str, page_dir: str) -> str:
    """Make every internal link relative to the directory holding this page."""
    def sub(m: re.Match) -> str:
        attr, quote, url = m.groups()
        got = site_path(url)
        if got is None:
            return m.group(0)
        path, frag = got
        rel = posixpath.relpath(path, page_dir or ".")
        return f"{attr}={quote}{rel}{'#' + frag if frag else ''}{quote}"
    return LINK_RE.sub(sub, text)


INLINE_CSS = f"<style>{V.CSS}</style>"


def finish(raw: bytes, out: Path, out_root: Path) -> None:
    """Rewrite one rendered page and write it to disk."""
    page_dir = posixpath.dirname(str(out.relative_to(out_root)).replace(os.sep, "/"))
    text = raw.decode()
    css_rel = posixpath.relpath("static/style.css", page_dir or ".")
    text = text.replace(INLINE_CSS, f"<link rel=stylesheet href='{css_rel}'>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rewrite(text, page_dir), encoding="utf-8")


# ── shared fragments ─────────────────────────────────────────────────────────

ORIG_PICKER = V.sim_picker
ORIG_BRIEFING = V.briefing


def picker_stub(run: str, cell: str, rows: list, current: str) -> str:
    """Keep the picker's header inline, hoist its chip strip into one file.

    The header is ~300 bytes and genuinely per-page: it carries prev/next and
    the position counter. The chips are 25-63 KB and identical for every page of
    a cell apart from which one is marked current, which the loader re-marks in
    the browser from the page's own filename.
    """
    full = ORIG_PICKER(run, cell, rows, current)
    head, chips = full.split("<div class=chips>", 1)
    chips = chips[: -len("</div></div>")]
    key = f"{run}__{cell}"
    if key not in pickers:
        # Rendered with no current sim, so the stored copy carries no `on`.
        plain = ORIG_PICKER(run, cell, rows, "")
        pickers[key] = plain.split("<div class=chips>", 1)[1][: -len("</div></div>")]
    return (f"{head}<div class=chips data-chips data-cur='{V.esc(current)}'></div>"
            f"</div><script src='/static/picker/{key}.js'></script>")


def brief_stub(data: dict, task_id) -> str:
    """Hoist the briefing panels, content-addressed.

    The same task briefing is rendered on all three trials of a task, and again
    for every model that ran it, so five agents x three trials is fifteen copies
    of one policy. Hashing the rendered panels collapses those without having to
    prove in advance which of them are equal.
    """
    full = ORIG_BRIEFING(data, task_id)
    if not full:
        return full
    key = sha1(full.encode()).hexdigest()[:16]
    briefs.setdefault(key, full)
    return (f"<div class=briefslot data-brief='{key}'></div>"
            f"<script src='/static/brief/{key}.js'></script>")


# The stored chips are one strip per cell, so the current one is marked here
# rather than baked in. Matching is on the tail of the href: the strip is
# written once against a directory it does not itself sit in, so every chip
# carries a `../../run/cell/` prefix that the page's own id does not.
LOADER = """\
(function(){var b=document.querySelector('[data-chips]');if(!b)return;
b.innerHTML=%s;var c=b.getAttribute('data-cur');if(!c)return;
var a=b.querySelector('a[href$="/'+c+'.html"]');if(!a)return;
a.className+=' on';
if(a.scrollIntoView)a.scrollIntoView({block:'nearest',inline:'center'});})();
"""

BRIEF_LOADER = """\
(function(){var s=document.querySelector('[data-brief="%s"]');
if(s)s.outerHTML=%s;})();
"""

# Client-side replacement for the annotations filter bar. The rows themselves
# are the viewer's own; only which of them are visible is decided here.
ANN_FILTER_JS = """\
<script>
(function(){
 var rows=[].slice.call(document.querySelectorAll('a.arow'));
 var cur={run:'',cat:''};
 function apply(){
  var n=0;
  rows.forEach(function(r){
   var ok=(!cur.run||r.dataset.run===cur.run)&&(!cur.cat||r.dataset.cat===cur.cat);
   r.style.display=ok?'':'none'; if(ok)n++;
  });
  var c=document.getElementById('anncount'); if(c)c.textContent=n;
  [].slice.call(document.querySelectorAll('[data-f]')).forEach(function(b){
   var k=b.getAttribute('data-f'),v=b.getAttribute('data-v');
   b.className=cur[k]===v?'on':'';
  });
 }
 [].slice.call(document.querySelectorAll('[data-f]')).forEach(function(b){
  b.addEventListener('click',function(){
   cur[b.getAttribute('data-f')]=b.getAttribute('data-v'); apply();
  });
 });
 apply();
})();
</script>
"""


def static_filters(runs: list[str]) -> str:
    """The filter bar, as buttons over rows already on the page."""
    def btn(field: str, value: str, label: str) -> str:
        return (f"<button type=button data-f='{field}' data-v='{V.esc(value)}'>"
                f"{V.esc(label)}</button>")
    models = " &middot; ".join(
        [btn("run", "", "all")]
        + [btn("run", r, r.replace("-think-on", "")) for r in runs])
    sev = " &middot; ".join([btn("cat", "", "all"), btn("cat", "MAJOR", "major"),
                             btn("cat", "MINOR", "minor")])
    return (f"<div class=filters><span>model: {models}</span>"
            f"<span>severity: {sev}</span></div>")


# The bar is `<div class=filters><span>…</span><span>…</span></div>` with no
# nested div, so the first close tag is its own.
FILTER_BAR_RE = re.compile(r"<div class=filters>.*?</div>", re.S)

BUTTON_CSS = """
/* ── static build only ───────────────────────────────────────────────────── */
/* The annotations filter bar is buttons here, not links: there is no server to
   re-render the page, so filtering hides rows in place. They are styled to read
   as the links they replace. */
.filters button{font:inherit;color:var(--link);background:none;border:0;
padding:0;cursor:pointer}
.filters button:hover{text-decoration:underline}
.filters button.on{font-weight:700}
"""
SHOWN_RE = re.compile(r"<span class=sub2>(\d+) shown")


# ── the build ────────────────────────────────────────────────────────────────

def build(out_root: Path, limit: int | None) -> None:
    t0 = time.time()
    if out_root.exists():
        for sub in ("sim", "compare", "cell", "static"):
            shutil.rmtree(out_root / sub, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    cells = V.cells()
    print(f"{len(cells)} cells")

    # Discovery pass: the first shared task per (run, domain), so a `/compare`
    # link that names no task resolves to the page the live viewer would show.
    for run, cell, path in cells:
        domain = V.parse_cell(cell)[1]
        if (run, domain) in first_task:
            continue
        rows = {}
        for c in cells:
            if c[0] != run:
                continue
            if c[1] == f"english_{domain}":
                rows["en"] = V.sim_rows(V.load(c[2]))
            elif c[1] == f"l2_interaction_{domain}_cs":
                rows["cs"] = V.sim_rows(V.load(c[2]))
        if len(rows) == 2:
            def tasks(rs):
                return {str(r["task"]) for r in rs
                        if r["dur"] > 0 and r["trial"] in (0, None)}
            shared = sorted(tasks(rows["en"]) & tasks(rows["cs"]), key=str)
            if shared:
                first_task[(run, domain)] = shared[0]

    V.sim_picker = picker_stub
    V.briefing = brief_stub

    # 1. Leaderboard, one file per metric.
    for m in ("p1", "p2", "p3"):
        name = "index.html" if m == "p1" else f"index-{m}.html"
        finish(V.view_index({"m": [m]}), out_root / name, out_root)
    print(f"  index x3                        {time.time() - t0:6.1f}s")

    # 2. Annotations: the viewer's rows, a browser-side filter bar.
    raw = V.view_annotations({}).decode()
    runs = sorted({r["item"]["run"] for r in V.ann_rows()})
    raw = SHOWN_RE.sub(r"<span class=sub2><span id=anncount>\1</span> shown", raw)
    # Lambda replacement: the bar is built from run tags, and a `\1` or a
    # backslash inside one would otherwise be read as a group reference.
    bar = static_filters(runs)
    raw = FILTER_BAR_RE.sub(lambda _m: bar, raw, count=1)
    if bar not in raw:
        raise SystemExit("filter bar not replaced -- viewer.py markup changed")
    raw = raw.replace("</body>", ANN_FILTER_JS + "</body>")
    finish(raw.encode(), out_root / "annotations.html", out_root)
    print(f"  annotations                     {time.time() - t0:6.1f}s")

    # 3. Cell links redirect to a cell's first sim in the live viewer; keep that
    #    behaviour as a real page rather than dropping the link.
    for run, cell, _ in cells:
        try:
            V.view_cell({"run": [run], "cell": [cell]})
            continue
        except V.Redirect as r:
            got = site_path(r.location)
        if not got:
            continue
        dest = posixpath.relpath(got[0], f"cell/{run}")
        finish(
            (f"<!doctype html><meta charset=utf-8>"
             f"<meta http-equiv=refresh content='0;url={dest}'>"
             f"<title>{V.esc(cell)}</title>"
             f"<p><a href='{dest}'>{V.esc(cell)}</a></p>").encode(),
            out_root / "cell" / run / f"{cell}.html", out_root)

    # 4. Every trajectory.
    n = 0
    for run, cell, path in cells:
        d = V.load(path)
        sims = d.get("simulations") or []
        if limit:
            sims = sims[:limit]
        for s in sims:
            sid = str(s["id"])
            finish(V.view_sim({"run": [run], "cell": [cell], "id": [sid]}),
                   out_root / SIM_DIR.format(run=run, cell=cell) / f"{sid}.html",
                   out_root)
            n += 1
        print(f"  {run:32} {cell:26} {len(sims):4} sims "
              f"{time.time() - t0:6.1f}s")
    print(f"  {n} sim pages                   {time.time() - t0:6.1f}s")

    # 5. EN/CS comparisons. Both cell names of a domain produce the same page,
    #    so it is keyed by domain and built once.
    m = 0
    for (run, domain), _first in sorted(first_task.items()):
        cell = f"l2_interaction_{domain}_cs"
        rows = {}
        for c in cells:
            if c[0] != run:
                continue
            if c[1] == f"english_{domain}":
                rows["en"] = V.sim_rows(V.load(c[2]))
            elif c[1] == cell:
                rows["cs"] = V.sim_rows(V.load(c[2]))
        def tasks(rs):
            return {str(r["task"]) for r in rs
                    if r["dur"] > 0 and r["trial"] in (0, None)}
        shared = sorted(tasks(rows["en"]) & tasks(rows["cs"]), key=str)
        if limit:
            shared = shared[:limit]
        for task in shared:
            finish(V.view_compare({"run": [run], "cell": [cell], "task": [task]}),
                   out_root / CMP_DIR.format(run=run, domain=domain) / f"{task}.html",
                   out_root)
            m += 1
        print(f"  {run:32} compare {domain:20} {len(shared):4} "
              f"{time.time() - t0:6.1f}s")

    # 6. Shared fragments. Sim pages all sit three levels down, so one relative
    #    rewrite is correct for every page that loads these.
    sim_dir_depth = "sim/run/cell"
    pdir = out_root / "static" / "picker"
    pdir.mkdir(parents=True, exist_ok=True)
    for key, chips in pickers.items():
        (pdir / f"{key}.js").write_text(
            LOADER % json.dumps(rewrite(chips, sim_dir_depth)), encoding="utf-8")
    bdir = out_root / "static" / "brief"
    bdir.mkdir(parents=True, exist_ok=True)
    for key, frag in briefs.items():
        (bdir / f"{key}.js").write_text(
            BRIEF_LOADER % (key, json.dumps(rewrite(frag, sim_dir_depth))),
            encoding="utf-8")

    # 7. Stylesheet and brand assets. The extra rules exist only here: the live
    #    viewer's filter bar is anchors, because it filters by re-rendering.
    (out_root / "static" / "style.css").write_text(V.CSS + BUTTON_CSS,
                                                   encoding="utf-8")
    for name in ("favicon.ico", "logo_light.png", "logo_dark.png"):
        src = V.ASSETS / name
        if src.exists():
            shutil.copy2(src, out_root / "static" / name)

    print(f"\n{n} sim pages, {m} compare pages, {len(pickers)} pickers, "
          f"{len(briefs)} briefings in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments",
                    help="repository root to write the site into")
    ap.add_argument("--limit", type=int, default=None,
                    help="only N sims per cell and N tasks per comparison "
                         "(smoke test; the result is a broken-linked site)")
    args = ap.parse_args()
    build(args.out.resolve(), args.limit)


if __name__ == "__main__":
    main()
