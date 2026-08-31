#!/usr/bin/env python
"""Trajectory viewer for CzTauBench runs.

Serves the simulations under data/simulations/ as browsable HTML: an overview of
every cell, a task list per cell, full conversations, and -- the point of the
exercise -- an English/Czech side-by-side for the same task.

Reads from disk on every request, so it tracks running jobs live.

  python scripts/viewer.py --port 8765
  http://sol4.ufal.hide.ms.mff.cuni.cz:8765/

Standard library only; no dependency on the tau2 package or its heavy imports.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

ROOT = Path(__file__).resolve().parent.parent
SIMS = ROOT / "SEATauBench" / "data" / "simulations"
REF_FILE = ROOT / "results" / "tau2_reference.json"
ANN_FILE = ROOT / "results" / "language_annotations.json"
ASSETS = Path(__file__).resolve().parent / "assets"
LANG_FILE = ROOT / "SEATauBench" / "data" / "seatau" / "languages.json"
SCRATCH = re.compile(r"smoke|probe|verify|scratch|_test", re.I)

SCENARIOS = ("l2_interaction", "l2_tools", "l2_domain", "english")
DOMAIN_ORDER = ("airline", "retail", "telecom")

CSS = """
/* lokalni.AI house style: Monokai Vivid, Barlow for text, JetBrains Mono for
   anything that is literally code or a number.

   Theming follows the site's mechanism -- `data-theme` on <html>, an explicit
   choice persisted in localStorage, NOT `prefers-color-scheme` -- but inverts
   its default: the site is dark-first, this viewer is light-first, because it is
   read as a document in daylight rather than browsed as a landing page.

   The neon accents cannot cross over. Monokai's cyan/green/purple/orange are
   tuned for dark terminal chrome and fail contrast as text on a light page
   (DESIGN.md is explicit about this), so each one has a darkened light-theme
   counterpart in the same hue family, all measured >=4.5:1 on both --bg and
   --surface. Accent use stays narrow: cyan/teal is interactive and the agent,
   purple is the Czech side, red marks what is wrong. */
:root{
/* Light (default). --link is NOT the brand's --accent-crimson (#869a12): that
   token measures 2.88:1 on #f4f5f0 and fails AA at any size, which DESIGN.md
   itself flags as a known issue. This is its suggested replacement, #5c6b0c at
   5.4:1 -- same olive, dark enough to read at table sizes. */
--bg:#f4f5f0;--fg:#1f211c;--muted:#585a54;--line:#d5d7d2;--card:#eceee9;
--link:#5c6b0c;
--agent:#0e6e7a;--user:#8b3fa8;--tool:#8a5a00;
--bad:#c2003f;--good:#5c6b0c;--emph:#5c6b0c;
--maj-line:#c2003f;--maj-bg:rgba(194,0,63,.15);
--min-line:#8a5a00;--min-bg:rgba(138,90,0,.14);
--track:rgba(31,33,28,.09);
--sans:'Barlow',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
--cond:'Barlow Condensed','Barlow Semi Condensed',var(--sans);
--mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
}
:root[data-theme='dark']{
--bg:#1f211c;--fg:#f8f8f2;--muted:#c2b889;--line:#3e3d32;--card:#252820;
--link:#dcff00;
--agent:#41ffff;--user:#f082ff;--tool:#ffb800;
--bad:#ff005e;--good:#dcff00;--emph:#dcff00;
--maj-line:#ff005e;--maj-bg:rgba(255,0,94,.24);
--min-line:#ffb800;--min-bg:rgba(255,184,0,.16);
--track:rgba(248,248,242,.07);
}
*{box-sizing:border-box}
body{margin:0;padding:1.4rem clamp(1rem,4vw,3rem) 3rem;background:var(--bg);
color:var(--fg);font:15px/1.65 var(--sans);
-webkit-font-smoothing:antialiased}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}
h1{font-family:var(--cond);font-weight:700;font-size:2rem;line-height:1.1;
letter-spacing:-.02em;margin:0 0 .3rem;text-wrap:balance}
h2{font-family:var(--cond);font-weight:600;font-size:1.3rem;letter-spacing:-.01em;
margin:1.6rem 0 .5rem;text-transform:lowercase}
.sub{color:var(--muted);margin-bottom:1.2rem}

/* ── masthead ────────────────────────────────────────────────────────────── */
/* The logomark reads two ways at once: a terminal window (rule under the title
   bar) and a railway wagon (wheels below the frame). Line art on a grid, drawn
   inline so it inherits colour and needs no asset pipeline. */
.mast{display:flex;align-items:center;gap:.7rem;padding-bottom:.9rem;
margin-bottom:1.1rem;border-bottom:1px solid var(--line)}
.brand{display:block;line-height:0}
.brand img{height:2rem;width:auto;display:block}
/* Two files, not one filtered image: the wordmark is a raster asset and the
   site ships a light-on-dark and a dark-on-light cut of it. Both are always in
   the DOM and the theme picks one, so switching cannot flash a missing image. */
:root .logo-dark{display:none}
:root[data-theme='dark'] .logo-light{display:none}
:root[data-theme='dark'] .logo-dark{display:block}
.mastsep{color:var(--line)}
.spacer{flex:1}
/* Icon only, and square: the label said nothing the sun/moon did not, and a
   text button next to a sentence-length title read as a second heading. */
.tgl{display:flex;align-items:center;justify-content:center;flex:none;
width:1.9rem;height:1.9rem;border:1px solid var(--line);background:var(--card);
color:var(--muted);border-radius:4px;cursor:pointer;padding:0}
.tgl:hover{color:var(--fg);border-color:var(--muted)}
.tgl svg{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:2;
stroke-linecap:round;stroke-linejoin:round}
:root .tgl .moon{display:none}
:root[data-theme='dark'] .tgl .sun{display:none}
:root[data-theme='dark'] .tgl .moon{display:block}
/* Not uppercased and not letter-spaced, unlike the other Condensed labels: this
   is a sentence-length Czech title, and uppercase tracking at that length turns
   a masthead into a banner and drops the diacritics into the row above. */
.proj{color:var(--muted);font-family:var(--cond);font-weight:600;
font-size:1.15rem;letter-spacing:0;text-wrap:balance}
.proj:hover{color:var(--fg);text-decoration:none}

/* ── tabs ────────────────────────────────────────────────────────────────── */
/* Centred, underlined, sized like navigation rather than like a control. The
   active rule is accent-red and sits ON the shared bottom border, so the two
   read as one object instead of a tab floating above a line. */
.tabs{display:flex;justify-content:center;gap:clamp(1.4rem,5vw,3.5rem);
border-bottom:1px solid var(--line);margin:0 0 2rem}
.tabs a{position:relative;padding:.5rem .3rem .8rem;color:var(--muted);
font-family:var(--cond);font-weight:600;font-size:1.25rem;
text-transform:uppercase;letter-spacing:.07em}
.tabs a:hover{color:var(--fg);text-decoration:none}
.tabs a.on{color:var(--fg)}
.tabs a.on::after{content:"";position:absolute;left:0;right:0;bottom:-1px;
height:3px;background:var(--bad)}
table{border-collapse:collapse;width:100%;margin-bottom:1rem;font-size:.92rem}
th,td{text-align:left;padding:.4rem .6rem;border-bottom:1px solid var(--line);
white-space:nowrap}
th{color:var(--muted);font-family:var(--cond);font-weight:600;font-size:.9rem;
text-transform:uppercase;letter-spacing:.06em}
tr:hover td{background:var(--card)}
.num{text-align:right;font-variant-numeric:tabular-nums;font-family:var(--mono);
font-size:.85rem}
.pill{display:inline-block;padding:.05rem .45rem;border-radius:10px;font-size:.78rem;
border:1px solid var(--line);font-family:var(--mono)}
.ok{color:var(--good);border-color:var(--good)}
.no{color:var(--bad);border-color:var(--bad)}
.msg{border-left:3px solid var(--line);padding:.5rem .8rem;margin:.55rem 0;
background:var(--card);border-radius:0 6px 6px 0;white-space:pre-wrap;
overflow-wrap:anywhere}
.msg.assistant{border-left-color:var(--agent)}
.msg.user{border-left-color:var(--user)}
.msg.tool{border-left-color:var(--tool);font-family:var(--mono);
font-size:.85rem}
.role{font-family:var(--cond);font-size:.85rem;text-transform:uppercase;
letter-spacing:.08em;color:var(--muted);margin-bottom:.25rem}
.msg.assistant .role{color:var(--agent)}.msg.user .role{color:var(--user)}
.msg.tool .role{color:var(--tool)}
.flag{outline:2px solid var(--bad);outline-offset:2px}
.tc{font-family:var(--mono);font-size:.84rem;color:var(--tool);
margin-top:.4rem;padding-left:.6rem;border-left:2px dotted var(--tool)}
details{margin:.3rem 0}summary{cursor:pointer;color:var(--muted);font-size:.85rem}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
.col h3{margin:.2rem 0 .5rem;font-family:var(--cond);font-size:1.15rem;
text-transform:uppercase;letter-spacing:.06em}
.wrap{overflow-x:auto}
.note{color:var(--muted);font-size:.85rem;margin-top:1.4rem;
border-top:1px solid var(--line);padding-top:.7rem;max-width:78ch;
text-wrap:pretty}
code{background:var(--bg);border:1px solid var(--line);padding:.02rem .3rem;
border-radius:3px;font-family:var(--mono);font-size:.86em;color:var(--tool)}

/* ── leaderboard ─────────────────────────────────────────────────────────── */
.chart{margin:.2rem 0 1.1rem}
.crow{display:grid;grid-template-columns:minmax(9rem,15rem) 1fr;gap:.9rem;
align-items:center;padding:.42rem 0;border-bottom:1px solid var(--line)}
.crow:last-child{border-bottom:0}
.cname{font-size:.88rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cname .org{color:var(--muted);font-size:.76rem}
.bars{display:flex;flex-direction:column;gap:3px;min-width:0}
/* The track is the 0..1 axis; every bar is a percentage of it, so EN and CS
   bars in different rows stay comparable by eye. */
.track{position:relative;height:15px;background:var(--track);border-radius:2px}
.bar{position:absolute;left:0;top:0;height:100%;border-radius:3px}
.bar.en{background:var(--agent)}
.bar.cs{background:var(--user)}
.bar.ref{background:var(--muted);opacity:.55}
/* The gap is drawn as its own span starting where CS ends and finishing where
   EN ends -- the point of the page is that distance, so it gets ink rather
   than being left as the difference between two bars the eye must subtract. */
.gap{position:absolute;top:0;height:100%;border-radius:0 3px 3px 0;
background:repeating-linear-gradient(135deg,var(--bad) 0 4px,transparent 4px 8px);
border-right:2px solid var(--bad)}
/* Values sit in their own column rather than floating at the bar tip: a bar at
   88% would push its label off the end of the track, and a fixed column also
   lets the numbers be read down the page independently of bar length. */
.tag{font-family:var(--cond);font-size:.8rem;text-transform:uppercase;
letter-spacing:.06em;color:var(--muted);width:1.6rem;flex:none;text-align:right}
.brow{display:flex;align-items:center;gap:.45rem}
.brow .track{flex:1;min-width:0}
.val{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:.76rem;
flex:none;width:4.2rem;text-align:right;white-space:nowrap}
.val .pt{color:var(--muted);font-size:.65rem}
.dval{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:.74rem;
color:var(--bad);white-space:nowrap;flex:none;width:4.4rem}
.dval.up{color:var(--good)}
.legend{display:flex;flex-wrap:wrap;gap:1rem;color:var(--muted);font-size:.8rem;
margin:.1rem 0 .8rem}
/* Airline and retail sit side by side: they are the same measurement on two
   task sets, so the comparison a reader wants is across the pair, and stacking
   them put a screen of scrolling between the two halves of one question. */
.domcols{display:grid;grid-template-columns:1fr 1fr;gap:0 2.4rem;
align-items:start}
@media(max-width:1200px){.domcols{grid-template-columns:1fr}}
.domcols .crow{grid-template-columns:minmax(5rem,8.5rem) 1fr;gap:.6rem}
.domcols .val{width:3.5rem}.domcols .dval{width:3.8rem}
.domcols .hd{margin-top:.2rem}
.legend i{display:inline-block;width:.7rem;height:.7rem;border-radius:2px;
margin-right:.3rem;vertical-align:middle}
.partial{opacity:.45}
.hd{display:flex;align-items:baseline;gap:.6rem;margin:1.7rem 0 .3rem}
.hd h2{margin:0}
.hd .sub2{color:var(--muted);font-size:.82rem}
/* Same header row inside a panel, where the surrounding margins are already
   set by the panel's padding. */
.phd{display:flex;align-items:baseline;gap:.6rem;margin:0 0 .6rem}
.phd h2{margin:0}
/* The SVG export button. A caption-weight control, not a chip: boxed and
   uppercased it read as loud as the heading it sits beside, which is backwards
   for a utility most readers will never press. Borderless, muted, in the same
   line art as the theme toggle, and pushed to the right so it never comes
   between the chart's title and its caption. */
.dl{margin-left:auto;flex:none;display:inline-flex;align-items:center;
gap:.28rem;font:500 .78rem/1 var(--sans);color:var(--muted);background:none;
border:0;padding:.1rem 0;cursor:pointer;white-space:nowrap}
.dl svg{width:13px;height:13px;stroke:currentColor;fill:none;stroke-width:2;
stroke-linecap:round;stroke-linejoin:round}
.dl:hover{color:var(--link)}
.dl:hover span{text-decoration:underline}
.dl:focus-visible{outline:2px solid var(--link);outline-offset:3px;
border-radius:2px}
.caveat{background:var(--card);border-left:3px solid var(--muted);
padding:.55rem .8rem;border-radius:0 5px 5px 0;color:var(--muted);
font-size:.83rem;margin:.5rem 0 1rem;text-wrap:pretty}
/* A legend, not prose: every term/clause pair runs inline and the whole thing
   reflows across the full page width. A one-per-line grid was correct
   typography for a narrow measure and wrong here -- five short definitions in a
   tall stack left most of the box empty and pushed the page down for nothing. */
.leg{margin:0;display:block}
.leg dt{display:inline;font-weight:600;color:var(--fg);white-space:nowrap}
.leg dd{display:inline;margin:0}
/* Separator before every pair but the first. Sits on the dt so it cannot be
   orphaned onto a line of its own by the wrap. */
.leg dt:not(:first-child)::before{content:"\2022";color:var(--line);
margin:0 .5rem 0 .35rem;font-weight:400}

/* ── briefing panels ─────────────────────────────────────────────────────── */
/* The conversation alone is unreadable without the instructions behind it, but
   those instructions are not all visible to the same party -- so each panel is
   labelled with WHO SAW IT. Conflating the user simulator's brief with the
   agent's policy is how a reader concludes the agent "should have known" a
   fact it was never told. */
.brief{border:1px solid var(--line);border-radius:6px;margin:.6rem 0 1rem;
background:var(--card)}
.brief>summary{padding:.5rem .8rem;font-size:.86rem;color:var(--fg);
list-style:none;display:flex;align-items:baseline;gap:.5rem;flex-wrap:wrap}
.brief>summary::-webkit-details-marker{display:none}
.brief>summary::before{content:"\25b8";color:var(--muted);font-size:.8rem}
.brief[open]>summary::before{content:"\25be"}
.brief>summary:hover{background:var(--bg)}
.brief .who{font-size:.7rem;text-transform:uppercase;letter-spacing:.04em;
border:1px solid var(--line);border-radius:9px;padding:.02rem .45rem;
color:var(--muted);white-space:nowrap}
.brief .who.agent{color:var(--agent);border-color:var(--agent)}
.brief .who.user{color:var(--user);border-color:var(--user)}
.brief .who.none{color:var(--bad);border-color:var(--bad)}
.bbody{padding:.1rem .9rem .8rem;border-top:1px solid var(--line)}
.field{margin:.7rem 0}
.field>.k{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;
color:var(--muted);margin-bottom:.15rem}
.field>.v{white-space:pre-wrap;overflow-wrap:anywhere}
.doc{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.87rem;
max-height:26rem;overflow-y:auto;padding:.6rem .75rem;background:var(--bg);
border:1px solid var(--line);border-radius:5px}
/* The appended L2 component is the entire difference between an english cell
   and its l2_interaction twin, so it is shown as its own block rather than
   left for the reader to spot at the bottom of an 8 KB policy. */
.l2add{border-left:3px solid var(--user);background:var(--bg);
padding:.5rem .75rem;border-radius:0 5px 5px 0;white-space:pre-wrap;
overflow-wrap:anywhere;font-size:.87rem;margin:.4rem 0}
.recon{color:var(--muted);font-size:.78rem;margin:.35rem 0 .1rem}
ul.crit{margin:.2rem 0;padding-left:1.1rem}
ul.crit li{margin:.2rem 0}

/* ── judge annotations ───────────────────────────────────────────────────── */
/* Severity by hue, not by weight of one hue: MAJOR (would fail a grammar
   checker) takes accent-red, MINOR (legal but unidiomatic) takes accent-orange.
   Red/orange reads as error/warning without either wash being loud enough that
   a page dense with MINORs looks like a page full of broken Czech. The washes
   are translucent so the body text keeps its own colour and contrast. The four
   tokens are defined per theme at the top of this sheet, not here. */
mark.ann{padding:.02rem .12rem;border-radius:2px;border-bottom:2px solid;
color:inherit;cursor:help;scroll-margin-top:2.5rem}
mark.ann:target{outline:2px solid var(--fg);outline-offset:2px}
mark.ann.MAJOR{background:var(--maj-bg);border-bottom-color:var(--maj-line)}
mark.ann.MINOR{background:var(--min-bg);border-bottom-color:var(--min-line)}
.sev{font-family:var(--cond);font-size:.8rem;text-transform:uppercase;
letter-spacing:.06em;border-radius:9px;padding:.02rem .45rem;border:1px solid;
white-space:nowrap}
.sev.MAJOR{color:var(--maj-line);border-color:var(--maj-line)}
.sev.MINOR{color:var(--min-line);border-color:var(--min-line)}
.bar.maj{background:var(--maj-line)}
.bar.min{background:var(--min-line)}

/* The explanation popup. The reason a span was flagged is reference material,
   not something to read down the page: shown inline it tripled the height of
   every row and buried the Czech under English commentary, which is the wrong
   thing to be scanning. Now the highlight is the content and the explanation
   is one hover away.

   Positioned BELOW the span: above would clip out of the viewport on the first
   row of the list, which is exactly where a reader starts. pointer-events:none
   keeps it from swallowing the click on the row link underneath. */
.hitwrap{position:relative;display:inline}
.tip{position:absolute;top:calc(100% + .45rem);left:0;z-index:30;
width:max-content;max-width:min(34rem,70vw);white-space:normal;text-align:left;
background:var(--card);border:1px solid var(--line);border-left:3px solid;
border-radius:0 4px 4px 0;padding:.5rem .75rem;
font:400 .86rem/1.5 var(--sans);color:var(--fg);
box-shadow:0 8px 26px rgba(0,0,0,.55);
opacity:0;visibility:hidden;transition:opacity .12s ease;pointer-events:none}
.hitwrap:hover .tip,mark.ann:target+.tip{opacity:1;visibility:visible}
.tip.MAJOR{border-left-color:var(--maj-line)}
.tip.MINOR{border-left-color:var(--min-line)}
.tip b{font-family:var(--cond);font-weight:600;text-transform:uppercase;
letter-spacing:.06em;font-size:.82rem;display:block;margin-bottom:.15rem}
.tip.MAJOR b{color:var(--maj-line)}.tip.MINOR b{color:var(--min-line)}

/* Language-quality header: chart left, numbers right. Same data twice on
   purpose -- the bars carry the ranking at a glance, the table carries the
   counts you cannot read off a bar. */
.lqtop{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);
gap:1.4rem 2.2rem;align-items:start;margin-bottom:1.6rem}
@media(max-width:1000px){.lqtop{grid-template-columns:1fr}}
.panel{border:1px solid var(--line);border-radius:5px;padding:.9rem 1.1rem;
background:var(--card)}
.panel h2{margin:0 0 .6rem;font-size:1.15rem}
.panel table{margin-bottom:0}
.panel th,.panel td{padding:.3rem .45rem}
/* One row per flagged span. The context is set dim so the eye lands on the
   highlight first -- the row exists to be scanned, not read. */
.arow{display:block;padding:.6rem .8rem;border-bottom:1px solid var(--line);
color:inherit}
.arow:hover{background:var(--card);text-decoration:none}
.ameta{display:flex;gap:.5rem;align-items:baseline;flex-wrap:wrap;
font-size:.78rem;color:var(--muted);margin-bottom:.25rem}
.actx{overflow-wrap:anywhere;line-height:1.7}
.actx .dim{color:var(--muted)}
.unloc{font-size:.72rem;color:var(--muted);border:1px dashed var(--line);
border-radius:9px;padding:.02rem .4rem}
.filters{display:flex;gap:1.4rem;flex-wrap:wrap;align-items:baseline;
margin:.2rem 0 1rem;font-size:.88rem;color:var(--muted)}

/* ── trajectory picker ───────────────────────────────────────────────────── */
/* This replaced the cell listing outright. The old table was a stop on the way
   to a conversation; the chips are the same index rendered small enough to sit
   permanently above the conversation itself, so moving between examples costs
   one click and never a trip back to a list page. */
.picker{border:1px solid var(--line);border-radius:5px;background:var(--card);
padding:.55rem .7rem;margin-bottom:1.1rem}
.pickhd{display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap;
margin-bottom:.5rem}
.pickcell{font-family:var(--cond);font-weight:600;font-size:1.05rem;
text-transform:uppercase;letter-spacing:.05em}
.sub2{color:var(--muted);font-size:.82rem}
.nav{font-family:var(--cond);font-weight:600;font-size:.9rem;
text-transform:uppercase;letter-spacing:.06em;white-space:nowrap}
.nav.off{color:var(--line)}
/* Capped and scrollable: a retail cell is 114 chips and an uncapped strip
   pushed the conversation itself below the fold, which defeats the point. */
.chips{display:flex;flex-wrap:wrap;gap:3px;max-height:7.2rem;overflow-y:auto}
.chip{position:relative;min-width:2.1rem;padding:.12rem .35rem;text-align:center;
border:1px solid var(--line);border-radius:3px;font-family:var(--mono);
font-size:.72rem;color:var(--muted);background:var(--bg);line-height:1.4}
.chip:hover{text-decoration:none;border-color:var(--muted);color:var(--fg)}
/* Pass/fail rides the left edge rather than the whole chip: a wall of 114
   saturated blocks is a texture, not information, and the eye needs the
   current-chip fill to win against it. */
.chip.ok{border-left:3px solid var(--good)}
.chip.no{border-left:3px solid var(--bad)}
.chip.on{background:var(--fg);color:var(--bg);border-color:var(--fg);
font-weight:500}
/* A judged-and-flagged marker, because "has the judge found anything here" is
   the reason to open most of these and it was nowhere in the old table. */
.chip .fl{position:absolute;top:-2px;right:-2px;width:6px;height:6px;
border-radius:50%;border:1px solid var(--card)}
.chip .fl.maj{background:var(--maj-line)}
.chip .fl.min{background:var(--min-line)}
"""


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def esc_url(x) -> str:
    """Escape a value for use inside a query string in an HTML attribute.

    Both encodings are needed and in this order: percent-encoding so a run tag
    containing `&` or `+` cannot invent a new query parameter, then HTML
    escaping so the result is a legal attribute value.
    """
    return esc(quote(str(x or ""), safe=""))


# latin-ext is not optional: every string on these pages is Czech, and without
# it the diacritics fall back mid-word and the type visibly breaks apart.
FONTS = (
    "<link rel=preconnect href='https://fonts.googleapis.com'>"
    "<link rel=preconnect href='https://fonts.gstatic.com' crossorigin>"
    "<link rel=stylesheet href='https://fonts.googleapis.com/css2?"
    "family=Barlow:wght@300;400;500;600&"
    "family=Barlow+Condensed:wght@600;700&"
    "family=JetBrains+Mono:wght@400;500&display=swap'>"
)

# Applied before first paint, in <head>, deliberately not at the end of <body>:
# read late, the page renders light and then snaps to dark, which is the flash
# the site's own Layout.astro takes the same precaution against. Light is the
# default here when nothing is stored -- note the site defaults the other way.
THEME_BOOT = (
    "<script>try{var t=localStorage.getItem('theme');"
    "if(t==='dark')document.documentElement.setAttribute('data-theme','dark');}"
    "catch(e){}</script>"
)

# Storage can throw outright (Safari private mode, site data blocked), not just
# return null, so both the read above and the write here are guarded. A toggle
# that forgets is a minor annoyance; one that throws takes the page down.
THEME_JS = (
    "<script>document.addEventListener('click',function(e){"
    "var b=e.target.closest&&e.target.closest('[data-theme-toggle]');if(!b)return;"
    "var d=document.documentElement,n=d.getAttribute('data-theme')==='dark'"
    "?'light':'dark';"
    "if(n==='dark')d.setAttribute('data-theme','dark');"
    "else d.removeAttribute('data-theme');"
    "try{localStorage.setItem('theme',n);}catch(err){}});</script>"
)

SUN = ("<svg class=sun viewBox='0 0 24 24' aria-hidden=true><circle cx=12 cy=12 "
       "r=4 /><path d='M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2"
       "M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4' /></svg>")
MOON = ("<svg class=moon viewBox='0 0 24 24' aria-hidden=true>"
        "<path d='M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z' /></svg>")
# Arrow into a tray, drawn on the same 24-grid and with the same line weight as
# the pair above, so the two controls on a page look drawn by one hand.
DOWNLOAD = ("<svg viewBox='0 0 24 24' aria-hidden=true>"
            "<path d='M12 3.5v10.5M7.5 10l4.5 4.5 4.5-4.5M4.5 20h15' /></svg>")

MAST = (
    "<header class=mast>"
    "<a class=brand href='https://lokalni.ai' aria-label='lokalni.ai'>"
    "<img class=logo-light src='/static/logo_dark.png' alt='lokalni.ai' "
    "width=102 height=32>"
    "<img class=logo-dark src='/static/logo_light.png' alt='lokalni.ai' "
    "width=102 height=32></a>"
    "<span class=mastsep>&frasl;</span>"
    "<a class=proj href='https://lokalni.ai/blog/umi-agenti-cesky-3/'>"
    "Um&iacute; agenti &#269;esky? &#268;&aacute;st 3: Z&aacute;kaznick&aacute; podpora"
    "</a>"
    "<span class=spacer></span>"
    f"<button class=tgl type=button data-theme-toggle "
    f"aria-label='Switch theme' title='Switch theme'>{SUN}{MOON}"
    "</button></header>"
)


def page(title: str, body: str, tab: str = "") -> bytes:
    """Wrap page content in the shared shell.

    `tab` names the active top-level tab; pages below that level (one
    conversation, an EN/CS comparison) pass nothing and get no tab bar, since
    highlighting a tab there would claim they are one.
    """
    return (
        f"<!doctype html><html lang=en><head><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{esc(title)} &middot; CzTauBench</title>"
        f"<link rel=icon href='/static/favicon.ico' sizes=any>"
        f"{THEME_BOOT}{FONTS}"
        f"<style>{CSS}</style></head>"
        f"<body>{MAST}{tabs(tab) if tab else ''}{body}{THEME_JS}</body></html>"
    ).encode()


@lru_cache(maxsize=64)
def _load(path_str: str, mtime: float) -> dict:
    """Cached on (path, mtime) so live runs re-read only when they change."""
    with open(path_str, encoding="utf-8") as fh:
        return json.load(fh)


def load(path: Path) -> dict:
    return _load(str(path), path.stat().st_mtime)


def cells(include_scratch: bool = False) -> list[tuple[str, str, Path]]:
    """Return (run_tag, cell_name, results_path), newest run first."""
    out = []
    for p in sorted(SIMS.glob("**/results.json")):
        rel = p.relative_to(SIMS)
        if not include_scratch and SCRATCH.search(str(rel)):
            continue
        parts = rel.parts
        run = parts[0] if len(parts) > 2 else "(root)"
        cell = parts[-2]
        out.append((run, cell, p))
    return out


def sim_rows(data: dict) -> list[dict]:
    rows = []
    for s in data.get("simulations", []):
        ri = s.get("reward_info") or {}
        lc = ((ri.get("info") or {}).get("language_correctness")) or {}
        rows.append(
            {
                "id": s.get("id"),
                "task": s.get("task_id"),
                "trial": s.get("trial"),
                "reward": ri.get("reward"),
                "dur": s.get("duration") or 0,
                "turns": len(s.get("messages") or []),
                "term": str(s.get("termination_reason") or ""),
                "lang": lc.get("score"),
                "lang_bad": lc.get("incorrect_turn_indices") or [],
                "sim": s,
            }
        )
    return rows


def fmt(v, nd=3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


# ── judge annotations ────────────────────────────────────────────────────────
# scripts/annotate_language.py hands a concatenation of one simulation's agent
# turns to Kimi K3 and gets back flagged spans. The spans come back as text, not
# as offsets -- the judge saw a joined string, not the message array -- so the
# viewer has to find each one again in the conversation it is rendering.


@lru_cache(maxsize=1)
def _anns(path_str: str, mtime: float) -> dict:
    with open(path_str, encoding="utf-8") as fh:
        return json.load(fh)


def annotations() -> dict:
    """The annotation document, or an empty one if the judge has not run."""
    try:
        return _anns(str(ANN_FILE), ANN_FILE.stat().st_mtime)
    except (OSError, ValueError):
        # Absent (the judge has not run), unreadable, or half-written while a
        # run is rewriting it. All three mean the same thing to the page.
        return {"meta": {}, "items": []}


def ann_index() -> dict:
    """(run, cell, sim_id) -> annotation item."""
    return {(i["run"], i["cell"], str(i["sim_id"])): i
            for i in annotations().get("items", [])}


@lru_cache(maxsize=1)
def _greetings() -> frozenset[str]:
    try:
        cfg = json.loads(LANG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return frozenset()
    return frozenset(
        str(v.get("greeting") or "").strip() for v in cfg.values() if v.get("greeting")
    )


def agent_texts(sim: dict) -> list[tuple[int, str]]:
    """(message index, text) for exactly the turns the judge was shown.

    Must stay in step with `agent_turns` in annotate_language.py: assistant
    messages with text, minus the scripted opening greeting. If the two drift,
    spans still resolve -- the search is over text, not positions -- but a span
    the judge never saw would silently fail to highlight.
    """
    fixed, out = _greetings(), []
    for idx, m in enumerate(sim.get("messages") or []):
        if m.get("role") != "assistant":
            continue
        content = str(m.get("content") or "")
        if not content.strip():
            continue
        if not out and content.strip() in fixed:
            continue
        # Raw, not stripped: the offsets returned here index the same string
        # render_messages escapes, so the highlight lands where it should.
        out.append((idx, content))
    return out


def _span_re(span: str) -> re.Pattern | None:
    """A pattern matching `span` with any run of whitespace standing in for any
    other. The judge is told to copy verbatim and mostly does, but a model that
    collapses a newline inside a quoted span is not wrong about the Czech, and
    losing the highlight over it would be a poor trade."""
    parts = [re.escape(p) for p in span.strip().split()]
    return re.compile(r"\s+".join(parts)) if parts else None


def locate(sim: dict, anns: list[dict]) -> tuple[dict, list[int]]:
    """Place each annotation in the conversation.

    Returns ({message index: [(start, end, ann_index)]}, [unplaced indices]).

    First match wins, scanning turns in order. Placements already taken are
    skipped rather than overwritten, so two annotations that quote the same
    words land on the first two occurrences instead of stacking on one -- which
    is what a reader counting highlights would expect.
    """
    texts = agent_texts(sim)
    placed: dict[int, list[tuple[int, int, int]]] = {}
    missing: list[int] = []
    for ai, ann in enumerate(anns):
        pat = _span_re(str(ann.get("span") or ""))
        if pat is None:
            missing.append(ai)
            continue
        for msg_idx, text in texts:
            hit = next(
                (m for m in pat.finditer(text)
                 if not any(m.start() < e and s < m.end()
                            for s, e, _ in placed.get(msg_idx, []))),
                None,
            )
            if hit:
                placed.setdefault(msg_idx, []).append((hit.start(), hit.end(), ai))
                break
        else:
            missing.append(ai)
    return placed, missing


def flagged(ann: dict, text: str, mark_id: str = "") -> str:
    """A highlighted span with its explanation in a hover popup.

    The explanation is deliberately not rendered in the flow. It is English
    commentary about Czech, and inline it competes with the very text it is
    describing -- so it lives in a popup and the highlight carries the page.

    A styled popup rather than the native `title` attribute: `title` waits about
    a second, cannot be styled, cannot show the severity, and is invisible on
    touch. The `:target` rule also opens it automatically when a reader arrives
    via a deep link, so the answer is already on screen rather than requiring a
    hover they have no reason to guess at.
    """
    cat = "MAJOR" if ann.get("category") == "MAJOR" else "MINOR"
    idq = f" id='{esc(mark_id)}'" if mark_id else ""
    return (f"<span class=hitwrap><mark{idq} class='ann {cat}'>{esc(text)}</mark>"
            f"<span class='tip {cat}'><b>{cat}</b>"
            f"{esc(ann.get('explanation'))}</span></span>")


def mark_text(text: str, spans: list[tuple[int, int, int]], anns: list[dict]) -> str:
    """Escape `text` and wrap each located span in a highlight plus popup."""
    out, cursor = [], 0
    for start, end, ai in sorted(spans):
        if start < cursor:  # defensive; locate() already rejects overlaps
            continue
        out.append(esc(text[cursor:start]))
        # id so /annotations rows can deep-link straight to the span rather than
        # dropping the reader at the top of a long conversation to hunt for it.
        out.append(flagged(anns[ai], text[start:end], f"a{ai}"))
        cursor = end
    out.append(esc(text[cursor:]))
    return "".join(out)


# ── metrics ──────────────────────────────────────────────────────────────────

def parse_cell(cell: str) -> tuple[str, str, str]:
    """'l2_interaction_airline_cs' -> ('l2_interaction', 'airline', 'cs')."""
    for scen in SCENARIOS:
        if cell.startswith(scen + "_"):
            rest = cell[len(scen) + 1:].split("_")
            if scen == "english":
                return scen, "_".join(rest), "en"
            return scen, "_".join(rest[:-1]) or rest[0], rest[-1]
    return "?", cell, "?"


def is_success(reward) -> bool:
    return reward is not None and (1 - 1e-6) <= reward <= (1 + 1e-6)


def cell_metrics(data: dict) -> dict:
    """pass^k, rho_3 and coverage for one cell.

    Infrastructure errors are dropped before anything is counted. They are
    simulations that never ran, so scoring them as failures would silently
    deflate pass^k -- and because they leave a task with fewer trials than the
    rest, k is capped at the thinnest task rather than at num_trials. A cell
    missing even one trial therefore reports pass^3 as None instead of quietly
    computing it over a task that only has two.
    """
    rows = sim_rows(data)
    infra = sum(1 for r in rows if r["term"].rsplit(".", 1)[-1].lower()
                == "infrastructure_error")
    good = [r for r in rows if r["term"].rsplit(".", 1)[-1].lower()
            != "infrastructure_error"]

    n_tasks = len(data.get("tasks") or [])
    trials = (data.get("info") or {}).get("num_trials") or 1
    expected = n_tasks * trials

    by_task: dict = {}
    for r in good:
        by_task.setdefault(r["task"], []).append(r)

    pk: dict[int, float] = {}
    # full[k] is True when EVERY task has at least k trials, i.e. pass^k covers
    # the whole task set rather than whichever tasks finished first. That, not
    # "the cell is 100% done", is what makes a number comparable: a cell at
    # 149/150 still has all 50 tasks covered at k=1, while one at 6/150 has 6.
    full: dict[int, bool] = {}
    if by_task:
        max_k = min(trials, min(len(v) for v in by_task.values()))
        for k in range(1, max_k + 1):
            vals = [
                math.comb(sum(1 for x in v if is_success(x["reward"])), k)
                / math.comb(len(v), k)
                for v in by_task.values()
            ]
            pk[k] = sum(vals) / len(vals)
            full[k] = len(by_task) >= n_tasks

    rewards = [r["reward"] for r in good if r["reward"] is not None]
    langs = [r["lang"] for r in good if r["lang"] is not None]
    return {
        "done": len(good),
        "expected": expected,
        "infra": infra,
        "complete": len(good) >= expected and expected > 0,
        "trials": trials,
        "p1": pk.get(1), "p2": pk.get(2), "p3": pk.get(3),
        "full": full,
        "rho3": (pk[3] / pk[1]) if pk.get(3) and pk.get(1) else None,
        "avg_reward": (sum(rewards) / len(rewards)) if rewards else None,
        "lang": (sum(langs) / len(langs)) if langs else None,
    }


@lru_cache(maxsize=1)
def _reference(mtime: float) -> dict:
    with open(REF_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def reference() -> dict:
    """Published tau2-bench numbers, or empty if nobody has fetched them yet."""
    try:
        return _reference(REF_FILE.stat().st_mtime)
    except OSError:
        return {}


def collect() -> tuple[dict, list]:
    """(model, domain) -> {lang: metrics}, plus the raw per-cell listing."""
    grid: dict = {}
    listing = []
    for run, cell, p in cells():
        m = cell_metrics(load(p))
        scen, domain, lang = parse_cell(cell)
        listing.append((run, cell, domain, lang, p, m))
        if scen in ("english", "l2_interaction"):
            grid.setdefault((run, domain), {})[lang] = (cell, m)
    return grid, listing


def ann_stats() -> tuple[list[dict], dict]:
    """Per-run annotation totals, plus the overall roll-up.

    Rates are per *simulation*, not per annotation: a model that writes longer
    replies gives the judge more to flag, so the raw count rewards terseness.
    Per-simulation at least holds the unit of work fixed. `clean` -- the share
    of conversations with nothing flagged at all -- is the more robust reading
    of the two, since it does not care how errors cluster.
    """
    rows: dict[str, dict] = {}
    for item in annotations().get("items", []):
        r = rows.setdefault(item["run"], {
            "run": item["run"], "n": 0, "maj": 0, "min": 0, "clean": 0, "err": 0})
        if item.get("error"):
            r["err"] += 1
            continue
        r["n"] += 1
        anns = item.get("annotations") or []
        r["maj"] += sum(1 for a in anns if a.get("category") == "MAJOR")
        r["min"] += sum(1 for a in anns if a.get("category") != "MAJOR")
        r["clean"] += not anns
    out = []
    for r in rows.values():
        n = r["n"] or 1
        out.append(r | {"maj_per": r["maj"] / n, "min_per": r["min"] / n,
                        "clean_frac": r["clean"] / n})
    out.sort(key=lambda r: (r["maj_per"], r["min_per"]))
    total = {k: sum(r[k] for r in out) for k in ("n", "maj", "min", "clean", "err")}
    return out, total


def ann_overview() -> str:
    """The header of the Language quality tab: chart left, numbers right.

    Both halves show the same five models. That is not redundancy -- the bars
    rank them at a glance and make the major/minor split a proportion you can
    see, while the table carries the absolute counts and the clean-conversation
    share, neither of which can be read off a bar.
    """
    doc = annotations()
    meta = doc.get("meta") or {}
    if not doc.get("items"):
        return ("<div class=sub>No judge output yet. Run "
                "<code>python scripts/annotate_language.py</code> to produce "
                "<code>results/language_annotations.json</code>.</div>")

    rows, total = ann_stats()
    # One shared axis so the bars are comparable down the column. Anchored on
    # the worst model rather than a round number: the interesting quantity is
    # the spread between models, and a fixed ceiling flattens it to nothing.
    top = max([r["maj_per"] + r["min_per"] for r in rows] + [0.5])
    bars = []
    for r in rows:
        maj_w, min_w = r["maj_per"] / top * 100, r["min_per"] / top * 100
        bars.append(
            f"<div class=crow style='grid-template-columns:minmax(6rem,11rem) 1fr'>"
            f"<div class=cname><a href='/annotations?run={esc(r['run'])}'>"
            f"{esc(r['run'].replace('-think-on', ''))}</a></div>"
            f"<div class=brow><div class=track>"
            f"<div class='bar maj' style='width:{maj_w:.1f}%'></div>"
            f"<div class='bar min' style='left:{maj_w:.1f}%;width:{min_w:.1f}%'></div>"
            f"</div><span class=val>{r['maj_per'] + r['min_per']:.2f}"
            f"<span class=pt>/sim</span></span></div></div>")

    tbl = ["<table><tr><th>model</th><th class=num>major</th>"
           "<th class=num>minor</th><th class=num>clean</th></tr>"]
    for r in rows:
        tbl.append(
            f"<tr><td><a href='/annotations?run={esc(r['run'])}'>"
            f"{esc(r['run'].replace('-think-on', ''))}</a></td>"
            f"<td class=num>{r['maj']}</td><td class=num>{r['min']}</td>"
            f"<td class=num>{r['clean']}/{r['n']}</td></tr>")
    tbl.append("</table>")

    err = (f" &middot; <span class='pill no'>{total['err']} judge errors</span>"
           if total["err"] else "")
    dl = svg_download("lang-quality", "cztaubench-language-quality.svg",
                      ann_svg(rows, top, total))
    return (
        f"<div class=sub>"
        f"<b>{total['maj'] + total['min']}</b> flagged spans over "
        f"<b>{total['n']}</b> judged conversations "
        f"&middot; {esc(', '.join(meta.get('domains') or []) or meta.get('domain', '?'))}"
        f"{esc('' if meta.get('complete', True) else ' · run in progress')}"
        f"{err}</div>"
        "<div class=lqtop>"
        "<div class=panel>"
        f"<div class=phd><h2>spans per conversation</h2>{dl}</div>"
        "<div class=legend style='margin:0 0 .4rem'>"
        "<span><i style='background:var(--maj-line)'></i>major</span>"
        "<span><i style='background:var(--min-line)'></i>minor</span></div>"
        f"<div class=chart>{''.join(bars)}</div></div>"
        f"<div class=panel><h2>totals</h2><div class=wrap>{''.join(tbl)}</div>"
        "<div class=note style='margin-top:.7rem;border-top:0;padding-top:0'>"
        "<b>major</b> = a grammar or syntax checker would flag it; "
        "<b>minor</b> = legal Czech a native speaker would not write. Bars are "
        "spans per conversation, so a verbose model is not punished for length "
        "alone; <b>clean</b> counts conversations with nothing flagged at all, "
        "which is the more robust of the two since it does not care how errors "
        f"cluster. Judged by <code>{esc(meta.get('judge', 'the judge'))}</code>."
        "</div></div></div>")


# ── views ────────────────────────────────────────────────────────────────────

def tabs(active: str) -> str:
    items = [("/", "Benchmark", "index"),
             ("/annotations", "Language quality", "ann")]
    return "<nav class=tabs>" + "".join(
        f"<a href='{h}'{' class=on' if k == active else ''}>{esc(t)}</a>"
        for h, t, k in items) + "</nav>"


def bar_row(name: str, sub: str, series: list, note: str = "") -> str:
    """One chart row: a label plus one track per series entry.

    series := [(css_class, tag, value, partial, gap_from)] where gap_from, if
    set, hatches the span between that value and this one.
    """
    bars = []
    for cls, tag, val, partial, gap_from in series:
        if val is None:
            bars.append(f"<div class=brow><span class=tag>{esc(tag)}</span>"
                        f"<div class=track></div><span class=val>&ndash;</span>"
                        f"<span class=dval></span></div>")
            continue
        pct = max(0.0, min(1.0, val)) * 100
        gap = ""
        if gap_from is not None:
            lo, hi = sorted((val, gap_from))
            gap = (f"<div class=gap style='left:{lo * 100:.2f}%;"
                   f"width:{(hi - lo) * 100:.2f}%'></div>")
            d = val - gap_from
            cl = "dval up" if d > 0 else "dval"
            dval = f"<span class='{cl}'>&Delta; {d:+.3f}</span>"
        else:
            dval = "<span class=dval></span>"
        pc = " partial" if partial else ""
        mark = "<span class=pt> part</span>" if partial else ""
        bars.append(
            f"<div class=brow><span class=tag>{esc(tag)}</span>"
            f"<div class='track{pc}'><div class='bar {cls}' "
            f"style='width:{pct:.2f}%'></div>{gap}</div>"
            f"<span class=val>{val:.3f}{mark}</span>{dval}</div>"
        )
    # `sub` and `note` are caller-built markup; `name` is data, so only it is
    # escaped here -- running esc() over the whole thing turned &middot; into
    # a literal &amp;middot; on the page.
    return (f"<div class=crow><div class=cname>{esc(name)}"
            f"{f'<br><span class=org>{sub}</span>' if sub else ''}"
            f"{note}</div><div class=bars>{''.join(bars)}</div></div>")


# ── SVG export ───────────────────────────────────────────────────────────────
# The charts above are HTML boxes, which is the right thing for a page -- they
# reflow, they carry links, they follow the theme -- and the wrong thing the
# moment somebody wants one in a slide or a post. So each of the three overview
# charts is rendered a second time as a standalone SVG and offered as a
# download. There is no second reduction behind it: both renderers are handed
# the same row list the view already built, so the file cannot disagree with the
# picture it was exported from.
#
# The exported file is deliberately self-contained: literal colours (no CSS
# variables, which vector editors do not resolve), a `prefers-color-scheme`
# block so it still reads on a dark page, and font *stacks* rather than embedded
# faces -- the chart is numbers and short labels, and it degrades to the
# reader's sans without losing anything.

SVG_STYLE = (
    "text{font-family:'Barlow',-apple-system,'Segoe UI',Roboto,sans-serif;"
    "fill:#1f211c}"
    ".cond{font-family:'Barlow Condensed','Barlow Semi Condensed','Barlow',"
    "sans-serif}"
    ".mono{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace}"
    ".pg{fill:#f4f5f0}.mut{fill:#585a54}.sep{stroke:#d5d7d2;stroke-width:1}"
    ".tk{fill:#1f211c;fill-opacity:.09}"
    ".en{fill:#0e6e7a}.cs{fill:#8b3fa8}.ref{fill:#585a54;fill-opacity:.55}"
    ".mj{fill:#c2003f}.mn{fill:#8a5a00}"
    ".bad{fill:#c2003f}.good{fill:#5c6b0c}.gp{stroke:#c2003f}"
    "@media(prefers-color-scheme:dark){"
    "text{fill:#f8f8f2}.pg{fill:#1f211c}.mut{fill:#c2b889}"
    ".sep{stroke:#3e3d32}.tk{fill:#f8f8f2;fill-opacity:.07}"
    ".en{fill:#41ffff}.cs{fill:#f082ff}.ref{fill:#c2b889;fill-opacity:.55}"
    ".mj{fill:#ff005e}.mn{fill:#ffb800}"
    ".bad{fill:#ff005e}.good{fill:#dcff00}.gp{stroke:#ff005e}}"
)

# The hatch the EN->CS gap is filled with, the pattern equivalent of the page's
# repeating-linear-gradient. Its stroke is a class, so it flips with the rest.
SVG_HATCH = ("<defs><pattern id='hatch' width='8' height='8' "
             "patternUnits='userSpaceOnUse' patternTransform='rotate(45)'>"
             "<line class='gp' x1='2' y1='0' x2='2' y2='8' stroke-width='3'/>"
             "</pattern></defs>")

BAR_H = 15      # matches .track in the stylesheet, so the two read as one chart
BAR_GAP = 4
ROW_PAD = 7


def sv_text(x: float, y: float, s, cls: str = "", size: float = 12.5,
            anchor: str = "start", weight: str = "") -> str:
    a = f" text-anchor='{anchor}'" if anchor != "start" else ""
    w = f" font-weight='{weight}'" if weight else ""
    c = f" class='{cls}'" if cls else ""
    return (f"<text x='{x:.1f}' y='{y:.1f}'{c} font-size='{size:g}'{a}{w}>"
            f"{esc(s)}</text>")


def sv_rect(x: float, y: float, w: float, h: float, cls: str = "",
            rx: float = 2, fill: str = "") -> str:
    c = f" class='{cls}'" if cls else ""
    f = f" fill='{fill}'" if fill else ""
    return (f"<rect x='{x:.1f}' y='{y:.1f}' width='{max(w, 0):.1f}' "
            f"height='{h:.1f}' rx='{rx:g}'{c}{f}/>")


def sv_clip(s, n: int) -> str:
    """Truncate a label to what fits its column. SVG has no text-overflow."""
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


def sv_legend(x: float, y: float, width: float, items: list) -> tuple[str, float]:
    """A wrapping legend row. Returns (markup, height).

    Item widths are estimated from character count rather than measured -- there
    is no text metric without a font engine -- which is why the labels are kept
    short and the row is allowed to wrap.
    """
    out, cx, cy = [], x, y
    for cls, label, fill in items:
        w = 13 + len(label) * 5.4 + 16
        if cx > x and cx + w > x + width:
            cx, cy = x, cy + 16
        out.append(sv_rect(cx, cy, 9, 9, cls, rx=2, fill=fill))
        out.append(sv_text(cx + 13, cy + 8.5, label, "mut", 10.5))
        cx += w
    return "".join(out), cy - y + 9


def svg_doc(title: str, width: float, height: float, body: str) -> str:
    return (f"<?xml version='1.0' encoding='UTF-8'?>"
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width:.0f}' "
            f"height='{height:.0f}' viewBox='0 0 {width:.0f} {height:.0f}'>"
            f"<title>{esc(title)}</title><style>{SVG_STYLE}</style>{SVG_HATCH}"
            f"{sv_rect(0, 0, width, height, 'pg', rx=0)}{body}</svg>")


def svg_download(key: str, filename: str, svg: str) -> str:
    """A button that hands the reader the SVG, plus the payload it downloads.

    The file is carried in the page rather than written beside it: the static
    build is 5,764 files already, and three more that only ever leave the site
    through a click are better off inside the page that offers them. The click
    handler is installed once however many buttons a page has -- three listeners
    on the leaderboard would mean three downloads per click.
    """
    payload = json.dumps(svg).replace("</", "<\\/")
    return (
        f"<button class=dl type=button data-svg='{esc(key)}' "
        # Label before icon, and not only because it reads that way: the button
        # is a flex box baseline-aligned against the heading beside it, and a
        # flex box takes its baseline from its first item -- an <svg>, which has
        # none, would align the whole control by its bottom edge instead.
        f"title='Download this chart as SVG'><span>SVG</span>{DOWNLOAD}</button>"
        f"<script>(function(){{var S=window.BMSVG=window.BMSVG||{{}};"
        f"S[{json.dumps(key)}]={{n:{json.dumps(filename)},s:{payload}}};"
        f"if(window.BMSVGH)return;window.BMSVGH=1;"
        f"document.addEventListener('click',function(e){{"
        f"var b=e.target.closest&&e.target.closest('[data-svg]');if(!b)return;"
        f"var it=window.BMSVG[b.getAttribute('data-svg')];if(!it)return;"
        f"var u=URL.createObjectURL(new Blob([it.s],"
        f"{{type:'image/svg+xml;charset=utf-8'}}));"
        f"var a=document.createElement('a');a.href=u;a.download=it.n;"
        f"document.body.appendChild(a);a.click();"
        f"setTimeout(function(){{URL.revokeObjectURL(u);a.remove();}},0);"
        f"}});}})();</script>")


def domain_svg(domain: str, rows: list, metric: str,
               n_ours: int, n_ref: int) -> str:
    """The leaderboard chart of one domain, as a standalone figure.

    `rows` is what `domain_chart` drew: dicts of name, sub and the same
    `(class, tag, value, partial, gap_from)` series `bar_row` takes.
    """
    W, PAD, LABEL_W = 800, 18, 200
    x_tag = PAD + LABEL_W + 24          # tags are right-aligned to here
    x_track = x_tag + 8
    x_dval = W - PAD                    # delta, right-aligned to the margin
    x_val = x_dval - 60
    track_w = x_val - 46 - x_track

    body = [sv_text(PAD, 25, domain, "cond", 18, weight="700"),
            sv_text(PAD, 42, f"{metric.replace('p', 'pass^')} · {n_ours} of "
                    f"ours, {n_ref} published", "mut", 11)]
    legend, leg_h = sv_legend(PAD, 53, W - 2 * PAD, [
        ("en", "English", ""),
        ("cs", "Czech (L2 interaction)", ""),
        ("", "EN→CS gap", "url(#hatch)"),
        ("ref", "published τ²-bench (English)", "")])
    body.append(legend)

    y = 53 + leg_h + 12
    for i, r in enumerate(rows):
        series = r["series"]
        sub = r.get("subtxt") or ""
        h = (len(series) * BAR_H + (len(series) - 1) * BAR_GAP + 2 * ROW_PAD)
        mid = y + h / 2
        if sub:
            body.append(sv_text(PAD, mid - 1, sv_clip(r["name"], 32), size=12.5))
            body.append(sv_text(PAD, mid + 11, sv_clip(sub, 34), "mut", 10.5))
        else:
            body.append(sv_text(PAD, mid + 4.5, sv_clip(r["name"], 32), size=12.5))

        for j, (cls, tag, val, partial, gap_from) in enumerate(series):
            by = y + ROW_PAD + j * (BAR_H + BAR_GAP)
            g = ["<g opacity='.45'>"] if partial else ["<g>"]
            if tag:
                body.append(sv_text(x_tag, by + 11, tag, "mut cond", 10.5, "end"))
            g.append(sv_rect(x_track, by, track_w, BAR_H, "tk"))
            if val is None:
                g.append("</g>")
                body.append("".join(g))
                body.append(sv_text(x_val, by + 11, "–", "mono mut", 10.5,
                                    "end"))
                continue
            frac = max(0.0, min(1.0, val))
            g.append(sv_rect(x_track, by, track_w * frac, BAR_H, cls, rx=3))
            if gap_from is not None:
                lo, hi = sorted((val, gap_from))
                gx, gw = x_track + track_w * lo, track_w * (hi - lo)
                g.append(sv_rect(gx, by, gw, BAR_H, rx=0, fill="url(#hatch)"))
                g.append(f"<line class='gp' x1='{gx + gw:.1f}' y1='{by}' "
                         f"x2='{gx + gw:.1f}' y2='{by + BAR_H}' "
                         f"stroke-width='2'/>")
            g.append("</g>")
            body.append("".join(g))
            body.append(sv_text(x_val, by + 11,
                                f"{val:.3f}{' part' if partial else ''}",
                                "mono", 10.5, "end"))
            if gap_from is not None:
                d = val - gap_from
                body.append(sv_text(x_dval, by + 11, f"Δ {d:+.3f}",
                                    "mono " + ("good" if d > 0 else "bad"),
                                    10.5, "end"))
        y += h
        if i < len(rows) - 1:
            body.append(f"<line class='sep' x1='{PAD}' y1='{y:.1f}' "
                        f"x2='{W - PAD}' y2='{y:.1f}'/>")
    return svg_doc(f"{domain} · {metric.replace('p', 'pass^')}",
                   W, y + PAD, "".join(body))


def ann_svg(rows: list, top: float, total: dict) -> str:
    """The language-quality chart: flagged spans per conversation, per model."""
    W, PAD, LABEL_W = 760, 18, 190
    x_track = PAD + LABEL_W + 14
    x_val = W - PAD
    track_w = x_val - 66 - x_track

    body = [sv_text(PAD, 25, "spans per conversation", "cond", 18, weight="700"),
            sv_text(PAD, 42, f"{total['maj'] + total['min']} flagged spans over "
                    f"{total['n']} judged conversations", "mut", 11)]
    legend, leg_h = sv_legend(PAD, 53, W - 2 * PAD,
                              [("mj", "major", ""), ("mn", "minor", "")])
    body.append(legend)

    y = 53 + leg_h + 12
    h = BAR_H + 2 * ROW_PAD
    for i, r in enumerate(rows):
        by = y + ROW_PAD
        body.append(sv_text(PAD, by + 11,
                            sv_clip(r["run"].replace("-think-on", ""), 28)))
        body.append(sv_rect(x_track, by, track_w, BAR_H, "tk"))
        maj_w = track_w * (r["maj_per"] / top)
        body.append(sv_rect(x_track, by, maj_w, BAR_H, "mj", rx=3))
        body.append(sv_rect(x_track + maj_w, by, track_w * (r["min_per"] / top),
                            BAR_H, "mn", rx=3))
        body.append(sv_text(x_val, by + 11,
                            f"{r['maj_per'] + r['min_per']:.2f} /sim",
                            "mono", 10.5, "end"))
        y += h
        if i < len(rows) - 1:
            body.append(f"<line class='sep' x1='{PAD}' y1='{y:.1f}' "
                        f"x2='{W - PAD}' y2='{y:.1f}'/>")
    return svg_doc("Language quality · spans per conversation",
                   W, y + PAD, "".join(body))


def domain_chart(domain: str, grid: dict, ref: dict, metric: str) -> str:
    """One domain: our models as EN/CS pairs, then the published bars."""
    ours = sorted(
        ((run, d) for (run, dom), d in grid.items() if dom == domain),
        key=lambda rd: -((rd[1].get("en") or rd[1].get("cs"))[1].get(metric) or 0),
    )
    if not ours:
        return ""

    # Collected as data first, then rendered twice -- once as the page's HTML
    # rows, once as the SVG the export button hands over -- so the two cannot
    # disagree about what is in the chart.
    data = []
    for run, langs in ours:
        k = int(metric[1])
        en_m = (langs.get("en") or (None, {}))[1]
        cs_m = (langs.get("cs") or (None, {}))[1]
        en, cs = en_m.get(metric), cs_m.get(metric)
        en_part = not en_m.get("full", {}).get(k, True)
        cs_part = not cs_m.get("full", {}).get(k, True)
        # The EN->CS gap is the headline of this page, so it is only drawn once
        # both sides cover the whole task set at this k. A partial cell's pass^k
        # is computed over whichever tasks happened to finish first, and
        # subtracting that from a finished cell produces a confident-looking
        # delta out of noise.
        gap_from = en if (en is not None and not en_part and not cs_part) else None
        series = [("en", "EN", en, en_part, None),
                  ("cs", "CS", cs, cs_part, gap_from)]
        link = ""
        for lang in ("cs", "en"):
            if lang in langs:
                link = (f"<br><a class=org href='/cell?run={esc(run)}"
                        f"&cell={esc(langs[lang][0])}'>trajectories &rsaquo;</a>")
                break
        data.append({"name": run, "sub": "", "series": series, "note": link})

    ref_rows = []
    for r in ref.get("core", []):
        if not r.get("featured"):
            continue
        val = (r.get(domain) or {}).get(metric)
        if val is None:
            continue
        ref_rows.append((val, r))
    ref_rows.sort(key=lambda t: -t[0])
    for val, r in ref_rows:
        # Two spellings of the same subtitle: the page wants the entity, the
        # SVG wants the character.
        org, eff = r.get("org") or "", r.get("effort") or "?"
        data.append({"name": r["model"],
                     "sub": f"{esc(org)} &middot; reasoning {esc(eff)}",
                     "subtxt": f"{org} · reasoning {eff}",
                     "series": [("ref", "", val, False, None)], "note": ""})

    rows = [bar_row(d["name"], d["sub"], d["series"], d["note"]) for d in data]
    n_ours, n_ref = len(ours), len(ref_rows)
    dl = svg_download(
        f"dom-{domain}-{metric}",
        f"cztaubench-{domain}-{metric.replace('p', 'pass')}.svg",
        domain_svg(domain, data, metric, n_ours, n_ref))
    return (
        f"<div class=hd><h2>{esc(domain)}</h2>"
        f"<span class=sub2>{metric.replace('p', 'pass^')} &middot; "
        f"{n_ours} of ours, {n_ref} published</span>{dl}</div>"
        f"<div class=legend>"
        f"<span><i style='background:var(--agent)'></i>English</span>"
        f"<span><i style='background:var(--user)'></i>Czech (L2 interaction)</span>"
        f"<span><i style='background:repeating-linear-gradient(135deg,"
        f"var(--bad) 0 3px,transparent 3px 6px)'></i>EN&rarr;CS gap</span>"
        f"<span><i style='background:var(--muted);opacity:.55'></i>"
        f"published tau2-bench (English)</span></div>"
        f"<div class=chart>{''.join(rows)}</div>"
    )


def view_index(q) -> bytes:
    metric = q.get("m", ["p1"])[0]
    if metric not in ("p1", "p2", "p3"):
        metric = "p1"

    grid, listing = collect()
    ref = reference()

    body = ["<h1>Leaderboard</h1>",
            "<div class=sub>Reads from disk on each request; safe to refresh "
            "while jobs run. Metric: "]
    body += [" ".join(
        f"<a href='/?m={k}'>{'<b>' if k == metric else ''}pass^{k[1]}"
        f"{'</b>' if k == metric else ''}</a>" for k in ("p1", "p2", "p3"))]
    body.append("</div>")

    if not listing:
        body.append("<p>No runs found under <code>data/simulations/</code>.</p>")
        return page("Benchmark", "".join(body), tab="index")

    # One domain per column. Each chart is self-contained -- same models, same
    # axis, different task set -- so they sit as siblings rather than stacked,
    # and the airline/retail comparison is a glance across instead of a scroll.
    domains = [d for d in DOMAIN_ORDER if any(dom == d for _, dom in grid)]
    charts = "".join(f"<div>{domain_chart(d, grid, ref, metric)}</div>"
                     for d in domains)
    body.append(f"<div class=domcols>{charts}</div>")

    if ref.get("core"):
        body.append(
            "<div class=caveat><b>On the published bars.</b> Taken from "
            f"<a href='{esc(ref.get('leaderboard', ''))}'>taubench.com</a> "
            f"(fetched {esc(ref.get('fetched'))}), English only &mdash; there is "
            "no Czech reference to compare against. They are <i>indicative, not "
            "like-for-like</i>: those runs drive the user simulator with "
            "<code>gpt-5.2</code> while ours uses <code>kimi-k3</code>, and the "
            "simulator is a large part of what a &tau;&sup2; score measures. "
            "The site's headline <code>core</code> number is also not shown here "
            "&mdash; it averages airline, retail and telecom, and telecom (which "
            "we do not run) is the easiest of the three, so it sits well above "
            "the per-domain values.</div>")

    body.append("<div class=hd><h2>all cells</h2></div>")
    body.append("<div class=wrap><table><tr><th>run</th><th>cell</th>"
                "<th class=num>done</th><th class=num>pass^1</th>"
                "<th class=num>pass^2</th><th class=num>pass^3</th>"
                "<th class=num>&rho;<sub>3</sub></th><th class=num>avg reward</th>"
                "<th class=num>lang ok</th><th class=num>infra</th>"
                "<th></th></tr>")
    for run, cell, _domain, _lang, _p, m in listing:
        state = "" if m["complete"] else " class=partial"
        body.append(
            f"<tr{state}><td>{esc(run)}</td>"
            f"<td><a href='/cell?run={esc(run)}&cell={esc(cell)}'>{esc(cell)}</a></td>"
            f"<td class=num>{m['done']}/{m['expected']}</td>"
            f"<td class=num>{fmt(m['p1'])}</td><td class=num>{fmt(m['p2'])}</td>"
            f"<td class=num>{fmt(m['p3'])}</td><td class=num>{fmt(m['rho3'])}</td>"
            f"<td class=num>{fmt(m['avg_reward'])}</td>"
            f"<td class=num>{fmt(m['lang'])}</td>"
            f"<td class=num>{m['infra'] or ''}</td>"
            f"<td><a href='/compare?run={esc(run)}&cell={esc(cell)}'>"
            f"compare EN/CS</a></td></tr>")
    body.append("</table></div>")

    body.append(
        "<div class=caveat><dl class=leg>"
        "<dt>pass^k</dt> <dd>all k trials of a task succeed, averaged over "
        "tasks</dd>"
        "<dt>&rho;<sub>3</sub></dt> <dd>pass^3 / pass^1 &mdash; how much of the "
        "single-shot score survives three attempts</dd>"
        "<dt>lang ok</dt> <dd>share of agent turns fastText detected as the "
        "target language &mdash; not whether the Czech is any "
        "<a href='/annotations'>good</a></dd>"
        "<dt>dimmed, <i>part</i></dt> <dd>cell unfinished; pass^k covers only "
        "the tasks done so far and is not yet representative</dd>"
        "<dt>infra</dt> <dd>simulations that never ran &mdash; excluded "
        "throughout, and they cap k at the thinnest task</dd>"
        "</dl></div>")
    return page("Benchmark", "".join(body), tab="index")


def cell_sims(run: str, cell: str) -> tuple[list[dict], str]:
    """Every completed simulation in a cell, in task order, plus the real run tag.

    The run tag is returned because callers may arrive without one (an old
    bookmark, a hand-typed URL) and everything downstream -- annotation lookup,
    prev/next links -- needs the tag that actually matched, not the empty string.
    """
    match = [(r, c, p) for r, c, p in cells(True)
             if c == cell and (not run or r == run)]
    if not match:
        return [], run
    real_run, _, path = match[0]
    rows = sorted((r for r in sim_rows(load(path)) if r["dur"] > 0),
                  key=lambda r: (str(r["task"]).zfill(6), r["trial"] or 0))
    return rows, real_run


def view_cell(q) -> bytes:
    """No page of its own any more -- hand straight to the first conversation.

    A list of trajectories was a stop on the way to the thing people actually
    wanted, and everything it showed is either on the conversation page already
    or in the picker strip at the top of it.
    """
    run, cell = q.get("run", [""])[0], q.get("cell", [""])[0]
    rows, real_run = cell_sims(run, cell)
    if not rows:
        return page("not found",
                    f"<h1>Nothing to show</h1><div class=sub>No completed "
                    f"simulations in <code>{esc(cell)}</code>"
                    f"{f' for run <code>{esc(run)}</code>' if run else ''}. "
                    f"<a href='/'>back to the leaderboard</a></div>")
    raise Redirect(f"/sim?run={esc_url(real_run)}&cell={esc_url(cell)}"
                   f"&id={esc_url(str(rows[0]['id']))}")


def sim_picker(run: str, cell: str, rows: list[dict], current: str) -> str:
    """The trajectory switcher: one chip per simulation, current one filled.

    This replaces the old cell listing rather than supplementing it. A chip
    carries the two things that made the table worth scanning -- which task, and
    whether it passed -- and adds the one the table never had: whether the
    language judge flagged anything, which is the reason to open most of these.
    """
    idx = ann_index()
    multi = len({r["trial"] for r in rows}) > 1
    chips = []
    pos = 0
    for i, r in enumerate(rows):
        sid = str(r["id"])
        if sid == current:
            pos = i
        item = idx.get((run, cell, sid))
        anns = (item or {}).get("annotations") or []
        maj = sum(1 for a in anns if a.get("category") == "MAJOR")
        cls = "chip " + ("ok" if is_success(r["reward"]) else "no")
        if sid == current:
            cls += " on"
        label = esc(r["task"]) + (f".{esc(r['trial'])}" if multi else "")
        dot = ""
        if anns:
            dot = (f"<i class='fl {'maj' if maj else 'min'}'></i>")
        title = (f"task {r['task']} · reward {fmt(r['reward'], 2)}"
                 + (f" · {len(anns)} flagged" if anns else
                    " · judged clean" if item and not item.get("error") else ""))
        chips.append(
            f"<a class='{cls}' title='{esc(title)}' href='/sim?run={esc_url(run)}"
            f"&cell={esc_url(cell)}&id={esc_url(sid)}'>{label}{dot}</a>")

    def step(delta, label, way):
        j = pos + delta
        if not 0 <= j < len(rows):
            return f"<span class='nav off'>{label}</span>"
        return (f"<a class='nav {way}' href='/sim?run={esc_url(run)}"
                f"&cell={esc_url(cell)}"
                f"&id={esc_url(str(rows[j]['id']))}'>{label}</a>")

    return (
        "<div class=picker>"
        f"<div class=pickhd><span class=pickcell>{esc(cell)}</span>"
        f"<span class=sub2>{esc(run)}</span><span class=spacer></span>"
        f"{step(-1, '&larr; prev', 'prev')}"
        f"<span class=sub2>{pos + 1} / {len(rows)}</span>"
        f"{step(1, 'next &rarr;', 'next')}</div>"
        f"<div class=chips>{''.join(chips)}</div></div>"
    )


# ── task briefing ────────────────────────────────────────────────────────────
# A conversation on its own does not say what the customer was trying to do,
# what the agent was allowed to do, or what the grader was looking for. All
# three are already in results.json; these helpers surface them, keeping strict
# track of which party actually saw each one.


MISSING_LANG_NOTE = (
    '<div class=recon>The <code>user_system</code> component was enabled for this cell, but its text could not be reconstructed &mdash; <code>seatau</code> is not importable. Start the viewer under the <code>cztaubench</code> venv to see it.</div>'
)


@lru_cache(maxsize=8)
def lang_instructions(lang_id: str) -> tuple[str, str]:
    """(user_system, agent_system) instruction text for a language.

    The agent_system component is recoverable from results.json -- it is
    appended to the stored policy. The user_system one is NOT: tau2 prepends it
    to the per-task user instructions at prompt-build time
    (`_prepend_user_system_instruction`) and stores neither the result nor the
    template. So it is reconstructed from the same source the runner uses, and
    labelled as reconstructed in the UI rather than presented as a record of
    what was sent.

    Imported lazily and defensively: the rest of this file is stdlib-only by
    design, and a viewer that dies because a package moved is worse than one
    that renders every panel but this.
    """
    if not lang_id or lang_id == "en":
        return "", ""
    try:
        from seatau.translation.language import get_language_config

        cfg = get_language_config(lang_id)
        return cfg.user_system_instruction, cfg.agent_system_instruction
    except Exception:
        return "", ""


def task_of(data: dict, task_id) -> dict:
    for t in data.get("tasks") or []:
        if str(t.get("id")) == str(task_id):
            return t
    return {}


def _field(label: str, value) -> str:
    if value in (None, "", [], {}):
        return ""
    return (f"<div class=field><div class=k>{esc(label)}</div>"
            f"<div class=v>{esc(value)}</div></div>")


def _panel(title: str, who: str, who_cls: str, body: str, open_: bool = False) -> str:
    if not body.strip():
        return ""
    return (f"<details class=brief{' open' if open_ else ''}>"
            f"<summary>{title}<span class='who {who_cls}'>{esc(who)}</span></summary>"
            f"<div class=bbody>{body}</div></details>")


def user_brief(task: dict) -> str:
    """What the customer was told to want, and how to behave while wanting it."""
    sc = task.get("user_scenario") or {}
    ins = sc.get("instructions") or {}
    body = "".join([
        _field("persona", sc.get("persona")),
        _field("reason for call", ins.get("reason_for_call")),
        _field("known to the user", ins.get("known_info")),
        _field("deliberately unknown", ins.get("unknown_info")),
        _field("how to behave", ins.get("task_instructions")),
    ])
    return _panel("Customer brief", "user simulator only", "user", body, open_=True)


def agent_brief(data: dict) -> str:
    """The agent's system prompt, exactly as stored, with the L2 tail split out."""
    env = data.get("info", {}).get("environment_info") or {}
    policy = env.get("policy") or ""
    if not policy:
        return ""
    info = data.get("info", {})
    _, agent_add = lang_instructions(info.get("lang_id") or "")
    tail = ""
    if agent_add and "agent_system" in (info.get("lang_components") or []):
        marker = "\n\n" + agent_add
        if policy.endswith(marker):
            policy, tail = policy[: -len(marker)], agent_add
    body = f"<div class=doc>{esc(policy)}</div>"
    if tail:
        body += ("<div class=recon>appended by <code>--lang-components "
                 "agent_system</code> &mdash; present in this cell and absent "
                 "from its english twin:</div>"
                 f"<div class=l2add>{esc(tail)}</div>")
    return _panel("Agent policy (system prompt)", "agent only", "agent", body)


def sim_user_brief(data: dict) -> str:
    """The user simulator's standing rules, plus the L2 language instruction."""
    ui = data.get("info", {}).get("user_info") or {}
    guide = ui.get("global_simulation_guidelines") or ""
    info = data.get("info", {})
    user_add, _ = lang_instructions(info.get("lang_id") or "")
    body = f"<div class=doc>{esc(guide)}</div>" if guide else ""
    if "user_system" in (info.get("lang_components") or []) and not user_add:
        body += MISSING_LANG_NOTE
    if user_add and "user_system" in (info.get("lang_components") or []):
        body += ("<div class=recon>prepended to the customer brief by "
                 "<code>--lang-components user_system</code>. Reconstructed from "
                 "<code>seatau.translation.language</code>: unlike the agent "
                 "policy, this text is not recorded in results.json.</div>"
                 f"<div class=l2add>{esc(user_add)}</div>")
    return _panel("User-simulator guidelines", "user simulator only", "user", body)


def grading_brief(task: dict) -> str:
    """What success meant. Shown last, and marked as seen by neither party."""
    desc = task.get("description") or {}
    ev = task.get("evaluation_criteria") or {}
    body = "".join([
        _field("what this task probes", desc.get("purpose")),
        _field("relevant policies", desc.get("relevant_policies")),
        _field("notes", desc.get("notes")),
    ])
    nl = ev.get("nl_assertions") or []
    if nl:
        items = "".join(f"<li>{esc(a)}</li>" for a in nl)
        body += ("<div class=field><div class=k>judged by the NL-assertion "
                 f"judge</div><ul class=crit>{items}</ul></div>")
    acts = ev.get("actions") or []
    if acts:
        items = ""
        for a in acts:
            args = a.get("arguments")
            if isinstance(args, (dict, list)):
                args = json.dumps(args, ensure_ascii=False)
            items += (f"<li><code>{esc(a.get('name'))}"
                      f"({esc(args or '')})</code></li>")
        body += ("<div class=field><div class=k>database actions required</div>"
                 f"<ul class=crit>{items}</ul></div>")
    comm = ev.get("communicate_info") or []
    if comm:
        items = "".join(f"<li>{esc(c)}</li>" for c in comm)
        body += ("<div class=field><div class=k>must be communicated to the "
                 f"user</div><ul class=crit>{items}</ul></div>")
    body += _field("reward basis", ", ".join(ev.get("reward_basis") or []) or None)
    return _panel("How this was graded", "neither party", "none", body)


def briefing(data: dict, task_id) -> str:
    """All four panels for one task, ordered by how often they are wanted."""
    task = task_of(data, task_id)
    if not task and not (data.get("info", {}).get("environment_info")):
        return ""
    return (user_brief(task) + agent_brief(data) + sim_user_brief(data)
            + grading_brief(task))



def render_messages(sim: dict, flag: set[int] | None = None,
                    anns: list[dict] | None = None) -> str:
    """Render one conversation.

    `flag` marks assistant turns fastText detected as off-language; `anns` are
    the judge's flagged spans, highlighted in place. The two are independent
    signals: fastText asks "is this Czech at all", the judge asks "is this Czech
    any good", and a turn can fail either without the other.
    """
    flag = flag or set()
    anns = anns or []
    placed, _ = locate(sim, anns) if anns else ({}, [])
    out, a_idx = [], 0
    for m_idx, m in enumerate(sim.get("messages") or []):
        role = m.get("role") or "?"
        content = m.get("content")
        klass = role if role in ("assistant", "user", "tool") else "tool"
        extra = ""
        if role == "assistant":
            if a_idx in flag:
                extra = " flag"
            a_idx += 1
        label = {"assistant": "agent", "user": "user", "tool": "tool result"}.get(
            role, role
        )
        if role == "tool" and m.get("error"):
            label += " (error)"
        chunk = [f"<div class='msg {klass}{extra}'><div class=role>{esc(label)}</div>"]
        if content:
            spans = placed.get(m_idx)
            chunk.append(mark_text(str(content), spans, anns) if spans
                         else esc(content))
        # Reasoning traces, when the server's reasoning parser split them out.
        raw = m.get("raw_data") or {}
        reasoning = raw.get("reasoning_content") if isinstance(raw, dict) else None
        if reasoning:
            chunk.append(
                f"<details><summary>thinking "
                f"({len(str(reasoning))} chars)</summary>{esc(reasoning)}</details>"
            )
        for tc in m.get("tool_calls") or []:
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            args = tc.get("arguments") or (tc.get("function") or {}).get("arguments")
            if isinstance(args, (dict, list)):
                args = json.dumps(args, ensure_ascii=False)
            chunk.append(f"<div class=tc>&rarr; {esc(name)}({esc(args)})</div>")
        if not content and not (m.get("tool_calls")):
            chunk.append("<span style='color:var(--muted)'>(empty)</span>")
        chunk.append("</div>")
        out.append("".join(chunk))
    return "".join(out)


def reward_panel(sim: dict) -> str:
    ri = sim.get("reward_info") or {}
    lc = ((ri.get("info") or {}).get("language_correctness")) or {}
    bits = [f"reward <b>{fmt(ri.get('reward'),2)}</b>"]
    for k, v in (ri.get("reward_breakdown") or {}).items():
        bits.append(f"{esc(k)} {fmt(v,2)}")
    if lc:
        bits.append(
            f"language {fmt(lc.get('score'),2)} "
            f"({lc.get('correct_turn_count')}/{lc.get('assistant_turn_count')} turns "
            f"in {esc(lc.get('expected_language'))})"
        )
    bits.append(f"{sim.get('duration',0):.0f}s")
    bits.append(esc(sim.get("termination_reason")))
    nl = ri.get("nl_assertions") or []
    extra = ""
    if nl:
        items = "".join(
            f"<div class=msg><div class=role>"
            f"{'met' if a.get('met') else 'NOT met'}</div>"
            f"{esc(a.get('nl_assertion'))}"
            f"<div style='color:var(--muted);margin-top:.3rem'>"
            f"{esc(a.get('justification'))}</div></div>"
            for a in nl
        )
        extra = f"<details open><summary>judge assertions ({len(nl)})</summary>{items}</details>"
    return f"<div class=sub>{' &middot; '.join(bits)}</div>{extra}"


def find_sim(cell: str, run: str, sim_id: str) -> tuple[dict | None, dict]:
    """Return (simulation, whole results document) -- the briefing panels need
    `tasks` and `info`, which sit beside the simulation rather than inside it."""
    for r, c, p in cells(True):
        if c != cell or (run and r != run):
            continue
        d = load(p)
        for s in d.get("simulations", []):
            if str(s.get("id")) == str(sim_id):
                return s, d
    return None, {}


def view_sim(q) -> bytes:
    run, cell, sid = (q.get(k, [""])[0] for k in ("run", "cell", "id"))
    sim, data = find_sim(cell, run, sid)
    if sim is None:
        return page("not found", "<p>No such simulation. <a href='/'>back</a></p>")
    ri = sim.get("reward_info") or {}
    lc = ((ri.get("info") or {}).get("language_correctness")) or {}
    item = ann_index().get((run, cell, str(sid)))
    anns = (item or {}).get("annotations") or []
    rows, real_run = cell_sims(run, cell)
    body = [
        sim_picker(real_run, cell, rows, str(sid)) if rows else "",
        f"<h1>task {esc(sim.get('task_id'))} &middot; trial {esc(sim.get('trial'))}</h1>",
        f"<div class=sub><a href='/'>&larr; leaderboard</a> &middot; "
        f"<a href='/compare?run={esc_url(real_run)}&cell={esc_url(cell)}"
        f"&task={esc_url(sim.get('task_id'))}'>compare EN/CS</a></div>",
        reward_panel(sim),
        ann_summary(sim, item),
        briefing(data, sim.get("task_id")),
        render_messages(sim, set(lc.get("incorrect_turn_indices") or []), anns),
        "<div class=note>Agent turns outlined in red were detected as off-target "
        "language by fastText. Highlighted phrases were flagged by the Czech "
        "language judge &mdash; hover for the reason. The panels above are the "
        "instructions behind the conversation; each is tagged with which party "
        "could actually see it, because the agent is not told what the customer "
        "was asked to do and is graded on criteria neither side is shown. "
        "Use &larr; and &rarr; to step through the cell.</div>",
        # Arrow keys, because reading these in sequence is the whole job and
        # returning to the mouse for every one of 114 conversations is not.
        # Guarded so it never steals a keystroke from a text field or a
        # modifier combination the browser owns.
        "<script>document.addEventListener('keydown',function(e){"
        "if(e.metaKey||e.ctrlKey||e.altKey)return;"
        "var t=e.target.tagName;"
        "if(t==='INPUT'||t==='TEXTAREA'||e.target.isContentEditable)return;"
        "var s=e.key==='ArrowLeft'?'.nav.prev':e.key==='ArrowRight'?'.nav.next':0;"
        "if(!s)return;var a=document.querySelector('a'+s);"
        "if(a){e.preventDefault();location.href=a.href;}});</script>",
    ]
    return page(f"task {sim.get('task_id')}", "".join(body))


def ann_summary(sim: dict, item: dict | None) -> str:
    """One line above the conversation: what the judge said about this run.

    Distinguishes 'judged and clean' from 'not judged'. Only a sample of
    simulations has been through the judge, and a page that showed nothing in
    both cases would let a reader read the absence of highlights as a verdict.
    """
    if item is None:
        return ""
    anns = item.get("annotations") or []
    if item.get("error"):
        return (f"<div class=caveat>Language judge failed on this simulation: "
                f"<code>{esc(str(item['error'])[:200])}</code></div>")
    if not anns:
        return ("<div class=sub>Czech language judge: <b>no problems found</b> "
                "in the agent's turns.</div>")
    maj = sum(1 for a in anns if a.get("category") == "MAJOR")
    _, missing = locate(sim, anns)
    note = (f" &middot; <span class=unloc>{len(missing)} span"
            f"{'s' if len(missing) != 1 else ''} not found in the text</span>"
            if missing else "")
    return (f"<div class=sub>Czech language judge: "
            f"<span class='sev MAJOR'>{maj} major</span> "
            f"<span class='sev MINOR'>{len(anns) - maj} minor</span>{note}</div>")


def l2_delta(docs: dict, cols_first_msg: dict) -> str:
    """The complete inventory of what l2_interaction changed for this pair.

    Under `asset_mode: original` the tasks, tools, policy body and database are
    byte-identical across the two cells -- the scenario adds three components
    and nothing else. Listing them is more useful than asserting it, because
    the agent_system component is not purely a language instruction: it also
    carries an identifiers clause the english cell never sees, which is a
    difference in the condition rather than in the language.
    """
    cs = docs.get("Czech") or {}
    info = cs.get("info") or {}
    comps = list(info.get("lang_components") or [])
    user_add, agent_add = lang_instructions(info.get("lang_id") or "")
    body = []
    if "agent_system" in comps and agent_add:
        body.append("<div class=recon>appended to the agent's policy "
                    "(<code>agent_system</code>):</div>"
                    f"<div class=l2add>{esc(agent_add)}</div>")
    if "user_system" in comps and user_add:
        body.append("<div class=recon>prepended to the customer brief "
                    "(<code>user_system</code>):</div>"
                    f"<div class=l2add>{esc(user_add)}</div>")
    if "greeting" in comps:
        pairs = "".join(
            f"<div class=recon>{esc(lang)}:</div><div class=l2add>{esc(msg)}</div>"
            for lang, msg in cols_first_msg.items() if msg
        )
        if pairs:
            body.append("<div class=recon>opening turn "
                        "(<code>greeting</code>):</div>" + pairs)
    if not body:
        return ""
    body.append("<div class=recon>Everything else &mdash; task, tools, policy "
                "body, database, seed &mdash; is identical between the two "
                "columns.</div>")
    return _panel("What L2 Interaction changed", "both, differently", "agent",
                  "".join(body), open_=True)


def view_compare(q) -> bytes:
    """English vs Czech for the same task -- the core comparison of the study."""
    cell = q.get("cell", [""])[0]
    # parse_cell, not cell.split("_")[1]. The naive index reads the domain of
    # `l2_interaction_airline_cs` as "interaction", so arriving here from a
    # CZECH cell looked for an `english_interaction` cell, never found one, and
    # showed "Need both ... currently have: Czech" for every task in the run.
    # Arriving from the english cell happened to work, which is why the page
    # looked merely incomplete rather than broken. Same underlying mistake the
    # cell-name parser elsewhere already fixed for `banking_knowledge`.
    domain = parse_cell(cell)[1] if cell else "airline"
    task = q.get("task", [""])[0]
    # The run must be pinned. Matching on domain alone would happily pair one
    # model's English cell with a different model's Czech cell, which is a
    # model comparison wearing a language comparison's label.
    want_run = q.get("run", [""])[0]

    found = {}
    for run, c, p in cells():
        if want_run and run != want_run:
            continue
        # Exact names on both sides. `domain in c` also matched a cell for a
        # different target language in the same domain, silently picking
        # whichever the glob happened to list first.
        if c == f"english_{domain}":
            found["English"] = (run, c, p)
        elif c == f"l2_interaction_{domain}_cs":
            found["Czech"] = (run, c, p)
    if len(found) < 2:
        have = ", ".join(found) or "none"
        return page(
            "compare",
            f"<h1>Compare EN/CS &middot; {esc(domain)}</h1>"
            f"<div class=sub><a href='/'>&larr; all cells</a></div>"
            f"<p>Need both an <code>english_{esc(domain)}</code> and an "
            f"<code>l2_interaction_{esc(domain)}_cs</code> cell"
            + (f" in run <code>{esc(want_run)}</code>" if want_run else "")
            + f"; currently have: <b>{esc(have)}</b>.</p>"
            f"<p>The Czech run is queued behind the English one, so this becomes "
            f"available once it starts producing results.</p>",
        )

    by_lang, docs = {}, {}
    for lang, (_run, _c, p) in found.items():
        d = load(p)
        docs[lang] = d
        by_lang[lang] = {
            str(r["task"]): r for r in sim_rows(d)
            if r["dur"] > 0 and (r["trial"] in (0, None))
        }
    shared = sorted(set(by_lang["English"]) & set(by_lang["Czech"]), key=str)
    if not shared:
        return page("compare", "<h1>Compare</h1><p>No tasks completed in both yet.</p>"
                               "<div class=sub><a href='/'>&larr; back</a></div>")
    if not task or task not in shared:
        task = shared[0]

    nav = " ".join(
        f"<a href='/compare?run={esc(want_run)}&cell={esc(cell)}&task={esc(t)}'>"
        f"{'<b>' if t == task else ''}{esc(t)}{'</b>' if t == task else ''}</a>"
        for t in shared
    )
    idx = ann_index()
    cols, first_msg = [], {}
    for lang in ("English", "Czech"):
        r = by_lang[lang][task]
        lc_bad = set(r["lang_bad"])
        msgs = r["sim"].get("messages") or []
        first_msg[lang] = (msgs[0].get("content") if msgs else "") or ""
        # Judge annotations exist only for the Czech side -- the English cell is
        # the control and its language was never in question.
        run_c, cell_c, _ = found[lang]
        item = idx.get((run_c, cell_c, str(r["id"])))
        cols.append(
            f"<div class=col><h3>{lang}</h3>{reward_panel(r['sim'])}"
            f"{ann_summary(r['sim'], item)}"
            f"{render_messages(r['sim'], lc_bad, (item or {}).get('annotations'))}"
            f"</div>"
        )
    # The brief and the grading criteria are shared: asset_mode is `original`,
    # so both columns ran the same task object. Rendering them once says that,
    # where rendering them twice would imply they might differ.
    cs_doc = docs.get("Czech") or {}
    shared_task = task_of(cs_doc, task)
    briefs = (user_brief(shared_task)
              + l2_delta(docs, first_msg)
              + agent_brief(docs.get("English") or cs_doc)
              + sim_user_brief(cs_doc)
              + grading_brief(shared_task))
    body = [
        f"<h1>English vs Czech &middot; task {esc(task)}</h1>",
        f"<div class=sub><a href='/'>&larr; all cells</a></div>",
        f"<div class=sub>tasks in both (trial 1): {nav}</div>",
        briefs,
        f"<div class=cols>{''.join(cols)}</div>",
        "<div class=note>Same task, same tools, same policy, same database &mdash; "
        "the only difference is the conversation language. Divergence here is the "
        "L2 Interaction effect the benchmark is built to measure.</div>",
    ]
    return page(f"compare task {task}", "".join(body))


CTX = 110  # characters of surrounding text shown either side of a flagged span


def ann_rows() -> list[dict]:
    """Every flagged span, resolved back to its place in the conversation.

    Resolution needs the simulation, not just the annotation file: the span
    arrives as bare text and the context around it -- the reason this list is
    scannable at all -- only exists in the trace. Spans that cannot be found are
    kept and marked, because a systematically unfindable span is a bug in the
    judge's verbatim-copying, and hiding those would hide the bug.
    """
    items = annotations().get("items", [])
    wanted = {(i["run"], i["cell"]) for i in items}
    sims = {}
    for run, cell, p in cells(True):
        if (run, cell) not in wanted:  # never load the English cells for this
            continue
        for s in load(p).get("simulations", []):
            sims[(run, cell, str(s.get("id")))] = s

    out = []
    for item in items:
        anns = item.get("annotations") or []
        if not anns:
            continue
        key = (item["run"], item["cell"], str(item["sim_id"]))
        sim = sims.get(key)
        placed, _ = locate(sim, anns) if sim else ({}, list(range(len(anns))))
        where = {ai: (m, s, e) for m, sp in placed.items() for s, e, ai in sp}
        msgs = (sim or {}).get("messages") or []
        for ai, ann in enumerate(anns):
            row = {"item": item, "ann": ann, "ai": ai, "found": ai in where,
                   "left": "", "hit": str(ann.get("span") or ""), "right": ""}
            if ai in where:
                m_idx, s, e = where[ai]
                text = str(msgs[m_idx].get("content") or "")
                row["left"] = ("…" if s > CTX else "") + text[max(0, s - CTX):s]
                row["hit"] = text[s:e]
                row["right"] = text[e:e + CTX] + ("…" if len(text) > e + CTX else "")
            out.append(row)
    # Major first: the list is read top-down and the majors are the ones that
    # would fail a checker, so they should not be buried under stylistic notes.
    out.sort(key=lambda r: (r["ann"].get("category") != "MAJOR",
                            r["item"]["run"], str(r["item"]["task_id"])))
    return out


def view_annotations(q) -> bytes:
    want_run = q.get("run", [""])[0]
    want_cat = q.get("cat", [""])[0].upper()

    rows = ann_rows()
    runs = sorted({r["item"]["run"] for r in rows})
    shown = [r for r in rows
             if (not want_run or r["item"]["run"] == want_run)
             and (want_cat not in ("MAJOR", "MINOR")
                  or r["ann"].get("category") == want_cat)]

    cur = {"run": want_run, "cat": want_cat if want_cat in ("MAJOR", "MINOR") else ""}

    def link(label, **kw):
        qs = "&".join(f"{k}={esc(v)}" for k, v in (cur | kw).items() if v)
        # Bold when the filter is ALREADY what this link would set -- compared
        # against `cur`, not against the merged args, which would always match.
        on = all(cur.get(k) == v for k, v in kw.items())
        return (f"<a href='/annotations{'?' + qs if qs else ''}'>"
                f"{'<b>' if on else ''}{esc(label)}{'</b>' if on else ''}</a>")

    body = [
        "<h1>Language quality</h1>",
        ann_overview(),
        "<div class=hd><h2>flagged spans</h2>"
        f"<span class=sub2>{len(shown)} shown &middot; hover a highlight for "
        "why it was flagged, click the row for its conversation</span></div>",
        "<div class=filters><span>model: " + " &middot; ".join(
            [link("all", run="")] + [link(r.replace("-think-on", ""), run=r)
                                     for r in runs]) + "</span>",
        "<span>severity: " + " &middot; ".join(
            [link("all", cat=""), link("major", cat="MAJOR"),
             link("minor", cat="MINOR")]) + "</span></div>",
    ]

    if not rows:
        body.append(
            "<p>Nothing to show. Run <code>python scripts/annotate_language.py"
            "</code> to produce <code>results/language_annotations.json</code>."
            "</p>")
        return page("Language quality", "".join(body), tab="ann")

    for r in shown:
        it, a = r["item"], r["ann"]
        cat = "MAJOR" if a.get("category") == "MAJOR" else "MINOR"
        href = (f"/sim?run={esc(it['run'])}&cell={esc(it['cell'])}"
                f"&id={esc(it['sim_id'])}"
                + (f"#a{r['ai']}" if r["found"] else ""))
        rw = it.get("reward")
        body.append(
            # The data attributes are inert here -- this page filters by
            # re-rendering. They exist so the static build can filter the same
            # rows in the browser instead of prerendering one page per
            # combination of the two filters. See scripts/build_site.py.
            f"<a class=arow data-run='{esc(it['run'])}' data-cat='{cat}' "
            f"href='{href}'>"
            f"<div class=ameta><span class='sev {cat}'>{cat}</span>"
            f"<span>{esc(it['run'].replace('-think-on', ''))}</span>"
            f"<span>task {esc(it['task_id'])} &middot; trial {esc(it['trial'])}</span>"
            + (f"<span>reward {fmt(rw, 2)}</span>" if rw is not None else "")
            + ("" if r["found"] else
               "<span class=unloc>span not found in the trace</span>")
            + "</div>"
            f"<div class=actx><span class=dim>{esc(r['left'])}</span>"
            f"{flagged(a, r['hit'])}"
            f"<span class=dim>{esc(r['right'])}</span></div></a>")

    body.append(
        "<div class=caveat><dl class=leg>"
        "<dt><span class='sev MAJOR'>major</span></dt> <dd>a grammar or syntax "
        "checker would catch it</dd>"
        "<dt><span class='sev MINOR'>minor</span></dt> <dd>grammatical Czech "
        "that reads as translated</dd>"
        "<dt>agent only</dt> <dd>the customer is the same fixed simulator in "
        "every cell, so its Czech says nothing about the model</dd>"
        "<dt>excluded</dt> <dd>pronoun, person and gender inconsistency &mdash; "
        "the agent was never instructed on it</dd>"
        "<dt><i>span not found</i></dt> <dd>not copied verbatim, so it could "
        "not be located in the trace; the flag may still be right</dd>"
        "</dl></div>")
    return page("Language quality", "".join(body), tab="ann")


ROUTES = {"/": view_index, "/cell": view_cell, "/sim": view_sim,
          "/compare": view_compare, "/annotations": view_annotations}


class Redirect(Exception):
    """Raised by a view to send the browser somewhere else.

    Used by /cell, which no longer has a page of its own: it resolves the cell
    to its first conversation and hands over, so the address bar ends up on the
    canonical /sim URL rather than on a path that only ever bounces.
    """

    def __init__(self, location: str):
        super().__init__(location)
        self.location = location


def static_file(path: str) -> tuple[bytes, str] | None:
    """Serve one file out of scripts/assets. Only the brand logos live there.

    The name is matched against a fixed whitelist rather than joined onto the
    directory: this server is reachable on the lab network, and a path that
    reaches the filesystem from a URL is a directory traversal waiting to be
    found. `..` never gets the chance to mean anything here.
    """
    allowed = {"logo_light.png": "image/png", "logo_dark.png": "image/png",
               "favicon.ico": "image/x-icon"}
    ctype = allowed.get(path)
    if ctype is None:
        return None
    try:
        return (ASSETS / path).read_bytes(), ctype
    except OSError:
        return None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        # Browsers ask for /favicon.ico at the root on their own, regardless of
        # what the <link> says, so the root path is served as well as /static/.
        if u.path.startswith("/static/") or u.path == "/favicon.ico":
            got = static_file(u.path.rsplit("/", 1)[-1])
            if got is not None:
                blob, ctype = got
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(blob)))
                # Immutable brand assets; without this every page view refetches
                # 35 KB of PNG that has not changed since the file was written.
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(blob)
                return

        fn = ROUTES.get(u.path)
        try:
            out = fn(parse_qs(u.query)) if fn else page(
                "404", "<h1>404</h1><p><a href='/'>home</a></p>")
            code = 200 if fn else 404
        except Redirect as r:
            self.send_response(302)
            self.send_header("Location", r.location)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        except Exception as exc:  # keep the server alive on malformed data
            out = page("error", f"<h1>error</h1><pre>{esc(repr(exc))}</pre>"
                                f"<p><a href='/'>home</a></p>")
            code = 500
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):  # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    import socket

    print(f"serving {SIMS}")
    print(f"  http://{socket.getfqdn()}:{args.port}/")
    srv.serve_forever()


if __name__ == "__main__":
    main()
