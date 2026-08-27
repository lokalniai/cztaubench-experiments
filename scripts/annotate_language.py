#!/usr/bin/env python
"""LLM-as-a-judge annotation of Czech language quality in agent turns.

For every simulation in a Czech cell, the agent's text turns are concatenated
(separated by `---`) and handed to Kimi K3 in ONE request, which returns a list
of spans that are ungrammatical, syntactically wrong, or merely awkward.

This measures the *language*, not the task. A simulation that failed its task
can be flawless Czech and a simulation that scored 1.0 can be full of errors --
that dissociation is the point, and the paper's own finding (R^2=0.014 between
language correctness and task success) is why it is worth measuring separately.

Only AGENT turns are annotated. The user simulator is Kimi K3 in every cell, so
its Czech is a property of the fixed apparatus, not of the model under test.

  python scripts/annotate_language.py --print-prompt      # show prompt, exit
  python scripts/annotate_language.py --limit 10 --domain airline
  python scripts/annotate_language.py --limit 10 --domain airline --dry-run

Writes results/language_annotations.json, which scripts/viewer.py reads.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from threading import Lock

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
SIMS = ROOT / "SEATauBench" / "data" / "simulations"
OUT = ROOT / "results" / "language_annotations.json"
ENV_FILE = ROOT / "SEATauBench" / ".env"
LANGS = ROOT / "SEATauBench" / "data" / "seatau" / "languages.json"

TURN_SEP = "\n---\n"

# Same phrase language_use.py excludes from language correctness: the domain
# policy orders the agent to emit it verbatim, in English, so it is a protocol
# token rather than a language choice. The judge is told to ignore it too.
TRANSFER_PHRASE = re.compile(
    r"(?i)you\s+are\s+being\s+transferred\s+to\s+a\s+human\s+agent[.!]?"
    r"\s*(?:please\s+hold\s+on[.!]?)?"
)


# ── the schema the judge is constrained to ───────────────────────────────────


class Category(str, Enum):
    MAJOR = "MAJOR"
    MINOR = "MINOR"


class Annotation(BaseModel):
    span: str = Field(
        description="The exact text span from the input, copied verbatim, long "
        "enough to be unambiguous."
    )
    explanation: str = Field(
        description="Why the span is wrong, and what it should have been."
    )
    category: Category = Field(description="MAJOR or MINOR.")


class AnnotationList(BaseModel):
    annotations: list[Annotation] = Field(
        description="All flagged spans. Empty if the text has no problems."
    )


PROMPT = """\
You are a native Czech proofreader assessing the language quality of a customer-\
service agent that was instructed to speak Czech.

You are judging the CZECH LANGUAGE ONLY. Whether the agent solved the customer's
problem, followed policy, or called the right tools is irrelevant here -- a
perfectly helpful reply full of broken Czech must be flagged, and a useless
reply in flawless Czech must not be.

The content below is the concatenation of one agent's turns from a single
conversation, separated by lines containing only `---`. Read all of it and
report every span whose Czech is wrong or unnatural.

# Categories

- MAJOR -- something a grammar or syntax checker would flag: a non-word, a wrong
  case ending, broken agreement, wrong word order that breaks the sentence,
  negation written apart from its verb, a missing or doubled word.
- MINOR -- disfluency: the Czech is grammatically and syntactically legal, but a
  native speaker would not phrase it that way. Calques from English, wrong
  register, an odd collocation, a stiff or machine-translated feel.

Examples (`->` shows what a native speaker would have written):

- "Mujní se omlouvám, ale na základě detailů vaší rezervace EHGLP3..."
  flag "Mujní" -- not a Czech word at all (probably meant "Moc"). MAJOR.
- "Rezervace OI5L9G je jednosměrná cesta z MCO do CLT a ne odpovídá vaší popisu."
  two separate errors: "ne odpovídá" (-> "neodpovídá", negation must be written
  as one word with the verb) and "vaší popisu" (-> "vašemu popisu", "popis" is
  masculine and takes the dative). Both MAJOR.
- "musím najít i druhý úsek: z Chicagoa (ORD) do Filadelfie (PHL)"
  flag "z Chicagoa" -- awkward declension, a Czech speaker writes "z Chicaga".
  MINOR.
- "Na jaký nový variantní produkt ji chcete vyměnit"
  flag "Na jaký nový variantní produkt" -- literal translation of "variant
  product"; naturally "Za jakou jinou variantu produktu". MINOR.
- "Abych mohla zpracovat vrácení, prosím, uveďte číslo objednávky"
  flag "Abych mohla zpracovat vrácení" -- "zpracovat vrácení" is a calque;
  naturally "Abych mohla produkt vrátit". MINOR.

# What NOT to flag

- **Anything that is deliberately English.** Identifiers and codes (`EHGLP3`,
  `raj_sanchez_734`, `MCO`, `basic_economy`), tool and function names, tool-call
  syntax, proper names, and the mandated hand-off sentence "YOU ARE BEING
  TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON." The agent was ordered by its
  policy to emit that sentence verbatim in English. None of this is a Czech
  error.
- **Inconsistent pronouns, person, or grammatical gender** -- both what the
  agent uses for itself ("mohu" vs "mohla bych", masculine vs feminine self-
  reference) and how it addresses the customer (vykání vs tykání, switching
  between them). The agent was never given instructions about this, so it cannot
  be held to a standard that was not set. Ignore it entirely, including gender
  agreement on the agent's own participles.
- **Anything that is merely factually wrong or unhelpful.** Not your job.
- **Formatting**: markdown, bullet lists, bold, line breaks, emoji.
- The `---` separators themselves, and the fact that turns repeat information.

# Output

Return JSON matching the schema: an object with one key `annotations`, a list of
objects with `span`, `explanation`, `category`.

- `span` must be copied **verbatim** from the content below, character for
  character, so it can be found again by exact string search. Do not correct it,
  do not normalise whitespace, do not add ellipses.
- Make the span **long enough to be unambiguous**. If the wrong expression
  occurs more than once in the text, include enough surrounding words that the
  span identifies one specific place. Keep it to one sentence or less otherwise.
- `explanation` says what is wrong and gives the correct form. Write it in
  English.
- One object per distinct problem. If a sentence has two unrelated errors,
  return two objects, as in the "Rezervace OI5L9G" example above.

**It is entirely possible that the text contains no errors at all.** Many of
these agents write good Czech. If you find nothing genuinely wrong, return
`{"annotations": []}`. Do not invent marginal findings to fill the list -- a
false MAJOR is worse than a missed MINOR.

Content to be annotated (agent turns, separated by ---):
\"\"\"
{content}
\"\"\"
"""


# ── data ─────────────────────────────────────────────────────────────────────


def load_env() -> None:
    """Read KIMI_* out of SEATauBench/.env without needing python-dotenv."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


@lru_cache(maxsize=1)
def greetings() -> frozenset[str]:
    """Every configured opening line, so the fixed one can be recognised.

    The `greeting` lang component makes the agent open with a string taken
    verbatim from languages.json. It is a template, not the model's Czech:
    annotating it would credit or blame a model for text it did not write, and
    since it is byte-identical in all 750 Czech simulations it cannot separate
    one model from another either.
    """
    try:
        cfg = json.loads(LANGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return frozenset()
    return frozenset(
        str(v.get("greeting") or "").strip() for v in cfg.values() if v.get("greeting")
    )


def agent_turns(sim: dict) -> list[str]:
    """The agent's text turns, in order.

    Turns that carry only tool calls are dropped: they contain no Czech to
    judge, and feeding the judge a wall of JSON invites it to annotate argument
    names. A turn with both text and a tool call keeps its text.
    """
    fixed = greetings()
    out = []
    for m in sim.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        if not out and content in fixed:  # the scripted opening turn
            continue
        out.append(content)
    return out


def cell_domain(cell: str) -> str:
    """'l2_interaction_banking_knowledge_cs' -> 'banking_knowledge'.

    Strips from both ends rather than splitting on '_', because a domain name
    can itself contain one -- the same trap parse_cell in viewer.py documents.
    """
    return cell.removeprefix("l2_interaction_").removesuffix("_cs")


def czech_cells(domain: str | None, run: str | None) -> list[tuple[str, str, Path]]:
    """(run_tag, cell, results.json) for every Czech l2_interaction cell."""
    out = []
    for p in sorted(SIMS.glob("**/results.json")):
        rel = p.relative_to(SIMS)
        if len(rel.parts) < 3:
            continue
        run_tag, cell = rel.parts[0], rel.parts[-2]
        if re.search(r"smoke|probe|verify|scratch|_test", str(rel), re.I):
            continue
        if not (cell.startswith("l2_interaction_") and cell.endswith("_cs")):
            continue
        if domain and f"_{domain}_" not in cell:
            continue
        if run and run != run_tag:
            continue
        out.append((run_tag, cell, p))
    return out


def pick_sims(data: dict, limit: int, trial: int | None) -> list[dict]:
    """First `limit` simulations, ordered by task id then trial.

    Restricted to one trial by default: three trials of the same task are three
    samples of near-identical text, so spending the budget on ten distinct tasks
    says more about the model's Czech than on three tasks seen three times.
    """
    sims = [s for s in data.get("simulations", []) if (s.get("duration") or 0) > 0]
    if trial is not None:
        sims = [s for s in sims if s.get("trial") == trial]

    def key(s):
        t = str(s.get("task_id"))
        return (0, int(t), s.get("trial") or 0) if t.isdigit() else (1, 0, t)

    ordered = sorted(sims, key=key)
    return ordered if limit <= 0 else ordered[:limit]


# ── the judge ────────────────────────────────────────────────────────────────


def build_prompt(turns: list[str]) -> str:
    return PROMPT.replace("{content}", TURN_SEP.join(turns))


def _parse_with_429_retry(client, _tries: int = 40, _wait: float = 30.0, **kw):
    """`client.chat.completions.parse`, but actually waiting out a 429.

    The SDK's own `max_retries` does NOT cover these. A parallel-limit 429 from
    the proxy comes back in 0.1 s with no retry attempted at all, because
    LiteLLM marks it non-retryable -- sensible for a per-minute quota you cannot
    hurry, wrong for `max_parallel_requests`, where a slot frees the moment any
    in-flight request finishes. One full pass at 4 workers lost 218 of 668
    requests this way (33%), every one of them instantly and for free.

    So the retry is ours, and it sleeps: a slot opens on the timescale of one
    request (~400 s), not one backoff. 40 x 30 s = 20 min of patience, which is
    ~3 request lifetimes -- long enough to outlast any transient pile-up,
    bounded enough that a genuinely revoked key still surfaces as an error.
    """
    from openai import RateLimitError

    for attempt in range(_tries):
        try:
            return client.chat.completions.parse(**kw)
        except RateLimitError:
            if attempt == _tries - 1:
                raise
            time.sleep(_wait)
    raise AssertionError("unreachable")


# Escapes for a response that hit --max-tokens.
#
# The observed truncations are not a cap set too low. Across 467 successful
# requests the largest completion is 18,118 tokens against a 24,576 cap (p99 is
# 16,140), and one airline simulation -- deepseek task 33 -- truncated at a
# 16,384 cap and then again at 24,576, burning 1,133 s and 1,532 s to do it.
# That shape is a reasoning loop, not a long answer, so raising the cap again
# only buys a longer wait before the same failure and eventually collides with
# --timeout. Two escapes instead, tried in order:
#
#  1. Re-ask at a non-zero temperature. The primary call is at temperature 0, so
#     a plain retry reproduces the loop token for token; sampling is the only
#     thing that has to change for the judge to take a different path.
#  2. Judge the conversation in halves and merge the annotation lists. Spans are
#     copied verbatim out of the content and the halves are disjoint, so nothing
#     is duplicated; what a segment loses is the rest of the conversation as
#     context, which is why this is the second resort and not the first. Each
#     half goes through the same ladder, so a stubborn half splits again.
#
# The floor is one turn: a single turn that truncates twice cannot be split any
# further and is recorded as an error rather than chopped mid-sentence, which
# would break the verbatim-span contract the viewer relies on.
#
# Any item that took an escape says so in its `fallback` field, so the count is
# recoverable from the output file and not only from the log.
RETRY_TEMPERATURE = 0.7

# The ladder is exponential -- every level doubles the number of segments, and
# each segment pays the same two attempts before splitting again -- so a
# conversation that truncates at every granularity would issue 2^depth requests
# of up to --timeout seconds each, unattended, while holding one of the four
# parallel slots. A 17-turn simulation could reach ~60 requests that way.
#
# 12 is where the ladder stops. It is one more than the 10 requests it costs to
# take a simulation all the way down to quarters (2 whole + 2x(2 half + 2x1
# quarter)), and a conversation whose quarters still truncate is not failing for
# lack of a shorter prompt -- it is failing on something in the text, which
# splitting further will not fix. Better to bank the error and the segments that
# did work than to spend a night proving it.
MAX_CALLS_PER_SIM = 12


def _usage_of(resp) -> dict:
    """The token counts worth keeping off one response.

    Reasoning tokens are the cost driver here and are invisible in the returned
    JSON, so they are recorded: a run that is slow because the judge is thinking
    for 4000 tokens is a different problem from one that is slow because the
    endpoint is queueing.
    """
    u = getattr(resp, "usage", None)
    finish = resp.choices[0].finish_reason if getattr(resp, "choices", None) else None
    if not u:
        return {"finish": finish}
    details = getattr(u, "completion_tokens_details", None)
    return {
        "in": getattr(u, "prompt_tokens", None),
        "out": getattr(u, "completion_tokens", None),
        "reason": (details if isinstance(details, dict)
                   else getattr(details, "__dict__", {})).get("reasoning_tokens"),
        "finish": finish,
    }


def _merge_usage(*usages: dict) -> dict:
    """Sum the token counts of the calls one simulation actually cost.

    A retried or split simulation is several requests, and reporting only the
    last one would make the expensive items look like the cheap ones. `calls`
    keeps the request count visible; `finish` is the last call's, since that is
    the one that produced the annotations.
    """
    out: dict = {"calls": 0}
    for u in usages:
        if not u:
            continue
        out["calls"] += u.get("calls", 1)
        for k in ("in", "out", "reason"):
            if u.get(k) is not None:
                out[k] = (out.get(k) or 0) + u[k]
        if u.get("finish") is not None:
            out["finish"] = u["finish"]
    return out


def _judge_call(client, model: str, turns: list[str], max_tokens: int | None,
                temperature: float) -> tuple[list[dict], str | None, dict, bool]:
    """One request. Returns (annotations, error, usage, truncated)."""
    from openai import LengthFinishReasonError

    try:
        resp = _parse_with_429_retry(
            client,
            model=model,
            messages=[{"role": "user", "content": build_prompt(turns)}],
            temperature=temperature,
            response_format=AnnotationList,
            **({"max_tokens": max_tokens} if max_tokens else {}),
        )
        msg = resp.choices[0].message
        usage = _usage_of(resp)
        if msg.parsed is None:  # refusal, truncation, or schema ignored
            truncated = usage.get("finish") == "length"
            err = (f"no parsed output ({usage.get('finish')}): "
                   f"{str(msg.content)[:200]!r}")
            return [], err, usage, truncated
        return [a.model_dump(mode="json") for a in msg.parsed.annotations], None, usage, False
    except LengthFinishReasonError as exc:
        # The SDK raises rather than returning when finish_reason == "length",
        # but it hands the completion over, so the reasoning tokens the truncated
        # call burned are still recorded instead of showing up as `?`.
        return [], f"{type(exc).__name__}: {exc}", _usage_of(exc.completion), True
    except Exception as exc:  # one bad simulation must not sink the batch
        return [], f"{type(exc).__name__}: {exc}", {}, False


def judge(client, model: str, turns: list[str], max_tokens: int | None,
          _budget: list[int] | None = None
          ) -> tuple[list[dict], str | None, dict, str | None]:
    """Annotate one simulation. Returns (annotations, error, usage, fallback).

    `_budget` is the shared remaining-call count for one simulation, spent by
    every segment of it; callers pass nothing and get MAX_CALLS_PER_SIM.
    """
    budget = [MAX_CALLS_PER_SIM] if _budget is None else _budget

    if budget[0] <= 0:
        return [], "call budget exhausted before this segment", {}, "budget"
    budget[0] -= 1
    anns, err, usage, truncated = _judge_call(client, model, turns, max_tokens, 0.0)
    if not truncated:
        return anns, err, usage, None

    if budget[0] <= 0:
        return anns, err, usage, "budget"
    budget[0] -= 1
    anns, err, retry_usage, truncated = _judge_call(
        client, model, turns, max_tokens, RETRY_TEMPERATURE)
    usage = _merge_usage(usage, retry_usage)
    if not truncated:
        return anns, err, usage, f"temperature={RETRY_TEMPERATURE}"

    if len(turns) < 2:  # nothing left to split
        return [], err, usage, "unsplittable"

    half = len(turns) // 2
    merged: list[dict] = []
    errors: list[str] = []
    segments = 0
    for part in (turns[:half], turns[half:]):
        p_anns, p_err, p_usage, p_fallback = judge(
            client, model, part, max_tokens, budget)
        merged += p_anns
        usage = _merge_usage(usage, p_usage)
        # A half that split again contributes its own leaves, so the final
        # `split:N` is how many pieces the conversation ended up in, not how
        # many times this frame split.
        segments += (int(p_fallback.removeprefix("split:"))
                     if (p_fallback or "").startswith("split:") else 1)
        if p_err:
            errors.append(p_err)
    return merged, "; ".join(errors) or None, usage, f"split:{segments}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=10,
                    help="simulations per cell; 0 for every task (default 10)")
    ap.add_argument("--domain", default="airline", help="airline|retail|... ")
    ap.add_argument("--run", default=None, help="restrict to one run tag")
    ap.add_argument("--trial", type=int, default=0,
                    help="only this trial; -1 for all trials")
    ap.add_argument("--model", default="kimi-k3")
    # The e-infra key is capped at max_parallel_requests=4 and returns 429 the
    # instant a fifth call opens, so the ceiling is a hard one, not a rate to
    # ride. Three leaves a slot free for anything else pointed at the same key.
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--resume", action="store_true",
                    help="skip simulations already annotated in --out")
    # A backstop against runaway reasoning, not a speed knob. The endpoint
    # exposes no reliable handle on thinking (see scripts/probe_thinking.py:
    # `enable_thinking` and `reasoning_effort=minimal` are silently ignored), and
    # reasoning has been observed anywhere from 3.2k to 9.2k tokens on prompts of
    # the same size. This caps the worst case so a pathological request fails
    # fast and visibly instead of running the 1800 s timeout down and then being
    # retried five times.
    #
    # Sized against the timeout, not guessed, and RE-sized once real data
    # existed. 16384 was set from airline, where the largest response was 9862.
    # Retail is longer and messier, and its true distribution turned out to be
    # median 7.7k / p90 13.3k / p99 16.1k -- i.e. the old cap sat inside the fat
    # part of the tail, and it truncated 39 requests (6% of retail), one of them
    # landing on exactly 16383. A cap that clips 6% of the corpus is not a
    # backstop, it is a sampling bias, and a biased one: the worse the agent's
    # Czech, the longer the judge thinks, so the cap preferentially eats the
    # most-annotated conversations (25 of the 39 were a single weak model).
    #
    # 24576 clears p99 by 50% while still fitting the timeout at the SLOWEST
    # observed generation rate: 13 tok/s p10 -> ~1890 s, inside --timeout 2700.
    # Check both numbers together if you change either; the cap is only a
    # backstop while it is reached before the timeout is.
    ap.add_argument("--max-tokens", type=int, default=24576,
                    help="cap the response, thinking included (default 24576; "
                         "0 for uncapped)")
    ap.add_argument("--timeout", type=float, default=2700.0,
                    help="per-request timeout in seconds (default 2700)")
    ap.add_argument("--print-prompt", action="store_true",
                    help="print the prompt for one real simulation and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be sent; make no API calls")
    args = ap.parse_args()

    trial = None if args.trial < 0 else args.trial
    found = czech_cells(args.domain, args.run)
    if not found:
        sys.exit(f"no Czech cells for domain={args.domain!r} run={args.run!r}")

    jobs = []
    for run_tag, cell, path in found:
        data = json.loads(path.read_text(encoding="utf-8"))
        for sim in pick_sims(data, args.limit, trial):
            turns = agent_turns(sim)
            if not turns:
                continue
            jobs.append({
                "run": run_tag, "cell": cell, "sim_id": sim.get("id"),
                "task_id": sim.get("task_id"), "trial": sim.get("trial"),
                "reward": (sim.get("reward_info") or {}).get("reward"),
                "n_turns": len(turns), "n_chars": sum(map(len, turns)),
                "turns": turns,
            })

    if not jobs:
        sys.exit("no simulations with agent text turns matched those filters")

    # Resume skips work already banked in the output file. Successes only --
    # a previous run's rate-limit failures are exactly what a resume is for.
    prior: dict[tuple, dict] = {}
    # Failures from a domain this invocation is NOT retrying have to be carried
    # across too. run_annotate.sh runs one domain per process against a shared
    # --out, so without this the retail pass rewrites the file minus airline's
    # failures: they are neither re-queued (out of --domain) nor recorded, and
    # the error count silently under-reports the gap. 124 airline failures went
    # missing exactly this way.
    carried: dict[tuple, dict] = {}
    if args.resume and args.out.exists():
        try:
            for it in json.loads(args.out.read_text(encoding="utf-8"))["items"]:
                key = (it["run"], it["cell"], str(it["sim_id"]))
                (prior if not it.get("error") else carried)[key] = it
        except (OSError, ValueError, KeyError) as exc:
            print(f"could not read {args.out} to resume: {exc}")
    if prior:
        before = len(jobs)
        jobs = [j for j in jobs
                if (j["run"], j["cell"], str(j["sim_id"])) not in prior]
        print(f"resuming: {len(prior)} already done, {before - len(jobs)} skipped")
        if not jobs:
            sys.exit("nothing left to do")
    # Anything this pass is about to redo must not also survive as a stale
    # failure record, or the same simulation appears twice in the output.
    for j in jobs:
        carried.pop((j["run"], j["cell"], str(j["sim_id"])), None)

    if args.print_prompt:
        j = jobs[0]
        print(build_prompt(j["turns"]))
        print(f"\n{'=' * 70}\n[{j['run']}/{j['cell']} task {j['task_id']} "
              f"trial {j['trial']} -- {j['n_turns']} turns, {j['n_chars']} chars]",
              file=sys.stderr)
        return

    total_chars = sum(j["n_chars"] for j in jobs)
    print(f"{len(jobs)} requests over {len(found)} cells, "
          f"{total_chars:,} chars of Czech (~{total_chars // 3:,} tokens in)")
    for run_tag, cell, _ in found:
        n = sum(1 for j in jobs if j["run"] == run_tag and j["cell"] == cell)
        print(f"  {n:>4}  {run_tag}/{cell}")
    if args.dry_run:
        return

    load_env()
    from openai import OpenAI

    # Timeout must clear the real request time by a wide margin. Measured cost
    # is wildly variable and trending up -- 230 s / 3200 reasoning tokens at the
    # low end, 885 s / 9150 at the high end, on prompts of comparable size. The
    # judge burns thousands of reasoning tokens to emit ~150 tokens of JSON, so
    # generation dominates and the spread is the model's, not the network's.
    # 1800 s is 2x the worst observed; anything near the observed maximum would
    # convert ordinary slow requests into the silent retry loop described below.
    #
    # A too-short timeout here is silently catastrophic rather than merely slow,
    # because the SDK RETRIES timeouts. At timeout=300 every request over that
    # threshold looped 13 x 300 s = 65 minutes before surfacing a single error,
    # which presents as a job that is running, holding connections, and
    # producing neither results nor failures. Do not tighten this to "fail
    # fast"; it fails slow and invisibly.
    #
    # max_retries is correspondingly modest, and note it does NOT cover 429s at
    # all -- the proxy returns those as non-retryable and the SDK obeys, which is
    # why _parse_with_429_retry exists. What this covers is timeouts and 5xx,
    # where each retry costs a full timeout period; 12 of those is an hour spent
    # on one doomed request.
    client = OpenAI(base_url=os.environ["KIMI_API_BASE"],
                    api_key=os.environ["KIMI_API_KEY"],
                    max_retries=5, timeout=args.timeout)

    # Results are written as they land, not batched to the end. At several
    # minutes per request a 50-job batch runs for over an hour, and a single
    # final write means an interruption anywhere in that hour destroys all of
    # it -- which is exactly what happened once. It also lets the viewer show
    # partial results while the run is still going.
    done, t0 = 0, time.monotonic()
    lock = Lock()
    items: list[dict] = list(prior.values()) + list(carried.values())

    def flush() -> None:
        """Atomically replace the output file. Caller holds `lock`."""
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "judge": args.model,
                # Derived from what is actually in the file, not from this
                # invocation's --domain: the file accumulates across runs, so a
                # single scalar here would be whichever batch happened to finish
                # last, silently mislabelling everything else in it.
                "domains": sorted({cell_domain(i["cell"]) for i in items}),
                "limit": args.limit,
                "trial": trial, "n_simulations": len(items),
                "complete": done >= len(jobs),
            },
            "items": items,
        }
        tmp = args.out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        # os.replace is atomic, so a viewer reading concurrently sees either the
        # old file or the new one, never a half-written one.
        os.replace(tmp, args.out)

    def run_one(j: dict) -> None:
        nonlocal done
        t = time.monotonic()
        anns, err, usage, fallback = judge(
            client, args.model, j["turns"], args.max_tokens)
        dt = time.monotonic() - t
        rec = {k: v for k, v in j.items() if k != "turns"} | {
            "annotations": anns, "error": err, "seconds": round(dt, 1),
            "usage": usage, "fallback": fallback}
        with lock:
            items.append(rec)
            done += 1
            n, elapsed = done, time.monotonic() - t0
            eta = (elapsed / n) * (len(jobs) - n)
            flush()
        flag = f"ERR {err[:70]}" if err else f"{len(anns):>2} ann"
        # An item that only survived because it was retried or split is worth
        # seeing in the log as it happens, not just in the final tally.
        if fallback:
            flag += f" [{fallback}]"
        # The run tag, not the cell: every cell in a domain batch has the same
        # name, so a log keyed on it cannot say which model is failing.
        print(f"  [{n:>3}/{len(jobs)}] {j['run'][:30]:<30} "
              f"task {str(j['task_id']):>4} {dt:>6.1f}s "
              f"r={usage.get('reason') or '?':>5} -> {flag}"
              f"   | eta {eta / 60:.0f}m", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run_one, jobs))

    n_err = sum(1 for i in items if i["error"])
    n_ann = sum(len(i["annotations"]) for i in items)
    maj = sum(1 for i in items for a in i["annotations"] if a["category"] == "MAJOR")
    clean = sum(1 for i in items if not i["error"] and not i["annotations"])
    secs = [i["seconds"] for i in items if i.get("seconds")]
    # Items that only landed because of an escape are reported separately: they
    # were judged under different conditions from the rest (a non-zero
    # temperature, or with only half the conversation in view), so a reader
    # deciding how much to trust the corpus needs the number, not just the fact
    # that nothing failed.
    n_fb = collections.Counter(
        i["fallback"].split(":")[0] for i in items if i.get("fallback"))
    print(f"\n{n_ann} annotations ({maj} MAJOR, {n_ann - maj} MINOR) over "
          f"{len(items)} simulations; {clean} clean, {n_err} failed")
    if n_fb:
        print("recovered from truncation: "
              + ", ".join(f"{v} by {k}" for k, v in sorted(n_fb.items())))
    if secs:
        secs.sort()
        print(f"per request: median {secs[len(secs) // 2]:.0f}s, "
              f"max {secs[-1]:.0f}s, total {(time.monotonic() - t0) / 60:.0f}m")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
