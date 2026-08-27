# CzTauBench — a Czech replication of SEATauBench (L2 Interaction)

> **This repository is the published record of the CzTauBench run.** The
> browsable results are at
> <https://lokalniai.github.io/cztaubench-experiments/> — leaderboard, every
> trajectory, and the Czech language-quality annotations. The document below is
> the protocol: what was run, how, and every decision that shapes the numbers.
>
> | | |
> |---|---|
> | `index.html`, `sim/`, `compare/`, `annotations.html` | the static site, built by `scripts/build_site.py` |
> | `results/simulations/<run>/<cell>/results.json` | raw tau2 output — every score on the site reduces from these |
> | `results/language_annotations.json` | the Czech judge's flagged spans |
> | `scripts/` | the pipeline that produced all of it |
>
> **Not included, and why.** `SEATauBench/` is an upstream checkout under its own
> MIT licence — clone it and apply the six changes in §4 to reproduce. `logs/`
> and `discarded/` are noise. Two changes are made to the raw results and no
> others, both recorded inside each file at `info.published`: `api_key` fields
> are redacted, and the per-message provider response envelopes are dropped —
> they duplicate the message's own `content` and weigh more than the rest of the
> data put together. The user simulator's reasoning traces, the one thing inside
> those envelopes that exists nowhere else, are kept. Rewards, transcripts,
> token counts and timings are untouched, so `scripts/report.py` reproduces
> every number in §9 from this copy. See `DEPLOYMENT.md`.
>
> Task data, policies and databases are derived from
> [τ²-bench](https://github.com/sierra-research/tau2-bench) and
> [SEATauBench](https://github.com/SEACrowd/SEATauBench), both MIT licensed.


This repository adapts [SEATauBench](SEATauBench/) (Nguyen et al., 2026), itself an
adaptation of τ²-Bench, to **Czech**, and evaluates locally-served open-weight models
against it.

The purpose of this document is to make every experimental decision explicit — both
so the results can be judged, and so the run can be reproduced or contested.

**Status: complete as of 2026-08-23.** All 20 benchmark cells (5 agents × airline
and retail × English and Czech, 4,920 simulations) finished with zero
infrastructure errors, and the Czech language-quality annotation covers all 820
Czech tasks. Results are in [§9](#9-results); the deliberate exclusions — telecom
and τ³ `banking_knowledge` — are in §1 and §10.

**Published at <https://lokalniai.github.io/cztaubench-experiments/>** — the
leaderboard, all 4,920 trajectories, the EN/CS comparisons and the flagged spans,
alongside the raw results they reduce from. Built by
[scripts/build_site.py](scripts/build_site.py) from the same code as the live
viewer ([§8.1](#81-the-leaderboard-page)); see `docs/DEPLOYMENT.md` for how it is
assembled and what is redacted on the way out.

---

## 1. Scope

**Only the L2 Interaction scenario is run.** L2 Tools and L2 Domain are deliberately
deferred.

SEATauBench defines four scenarios, distinguished by an `asset_mode` and a set of
`lang_components` (see `src/seatau/experiment_matrix.py`):

| scenario | `asset_mode` | `lang_components` |
|---|---|---|
| `english` | `original` | — |
| `l2_interaction` | `original` | `user_system`, `agent_system`, `greeting` |
| `l2_tools` | `original` | `tool_mix` |
| `l2_domain` | **`translated`** | + `tools`, `policy`, `db`, `tasks` |

`l2_interaction` changes only the conversation language. `l2_tools` keeps the
environment in English but presents tools drawn from a *mix* of languages (the
paper's `tool_mix_2` … `tool_mix_5` = how many languages are in the mix).
`l2_domain` is the only scenario that translates the environment itself.

L2 Interaction is the mildest condition: only the conversation changes language.
This matters practically, because it means **no translated assets are required** —
the scenario runs with `asset_mode: original` and only three language components
(`user_system`, `agent_system`, `greeting`). Adding Czech therefore required
*only* a new entry in `data/seatau/languages.json`, with no translation of tasks,
tool schemas, policies, or databases:

```json
"cs": {
  "code": "cs",
  "display_name": "Czech",
  "instruction_label": "Czech (čeština)",
  "greeting": "Dobrý den! Jak vám mohu dnes pomoci?"
}
```

Extending to **L2 Domain** would require translating tools, policy, database and
tasks into Czech, and therefore a translation-quality story this setup does not
currently have. **L2 Tools** does not need translated Czech assets (it runs at
`asset_mode: original`), but it does need a Czech-inclusive `tool_mix`
configuration, which does not exist yet.

**Every Czech cell is paired with an English cell for the same model.** The English
run is the control: the quantity of interest is the English↔Czech delta for one
model, not the absolute score, which is sensitive to the user simulator, the judge,
and the serving stack (see §7).

### Domains

| domain | tasks | trials | simulations per cell |
|---|---|---|---|
| `airline` | 50 | 3 | 150 — **run, ×5 agents ×2 languages** |
| `retail` | 114 | 3 | 342 — **run, ×5 agents ×2 languages** |
| `telecom` (`base` split) | 114 | 3 | 342 — **not run** |
| `banking_knowledge` (τ³) | 97 | 3 | 291 — **not run**; see §10 |

Telecom was dropped from the queue: its dual-control tasks run far longer than the
other domains, and the GPU time was better spent on a second model. That trade was
taken and the extra models are in §9; telecom stays unrun.

---

## 2. The three LLM roles

τ²-Bench simulations involve three distinct models. Conflating them is the single
easiest way to produce a meaningless number, so they are configured separately in
[scripts/env.sh](scripts/env.sh):

| role | model | why |
|---|---|---|
| **agent** | the model under test | varies per experiment (`CZTAU_PROFILE`) |
| **user simulator** | **Kimi K3**, fixed | plays the customer |
| **NL-assertion judge** | **Kimi K3**, fixed | grades free-text expectations |

**The user simulator and judge are held fixed across every agent.** If they varied
with the agent, cross-model scores would not be comparable.

**Kimi K3 is never evaluated as an agent.** Using the model under test as its own
simulated user is self-play: the agent converses with something that shares its
failure modes and its idea of what a reasonable request looks like. Note that the
SEATauBench paper itself has this flaw — it pairs a Qwen agent with a Qwen user
simulator — which is one reason absolute numbers here differ from the paper's.

Kimi K3 was selected for the user role after testing three candidates on Czech
roleplay. It was the only one that **honoured progressive disclosure** (it did not
volunteer the reservation ID before being asked) and that got Czech grammatical
gender right. The paper flags simulated-user error as a major confound — it reports
a ~20% critical user-error rate — so this choice materially affects results and
should be scrutinised.

All three roles run at **temperature 0.0**. The agent runs with **thinking enabled**
— see [§3.2](#32-thinking-mode-is-not-a-uniform-default) for why that takes a
different flag on each model family.

---

## 3. Models under evaluation

| profile | model | precision | served on |
|---|---|---|---|
| `local` | `Qwen/Qwen3.6-27B` | bf16 | vLLM, `dll-4gpu4` |
| `local35` | `Qwen/Qwen3.6-35B-A3B` | bf16 | vLLM, `tdll-8gpu2`, **4-way TP** |
| `gemma` | `google/gemma-4-E4B-it` | bf16 | vLLM, `dll-3gpu1` |
| `gemma31` | `google/gemma-4-31B-it` | bf16 | vLLM, `dll-4gpu3:8001` |
| `deepseek` | `deepseek-v4-flash` | **fp4** | e-infra API |

Gemma 4 E4B is far smaller than the two Qwen agents, so low absolute scores are
expected. It is included as a capability point on the same axis, not as a peer;
the quantity that stays comparable across models is the **within-model
English↔Czech delta**, not the cross-model ranking.

DeepSeek V4 Flash is the only agent that is **not** locally served, which changes
the shape of a run rather than just the model — see
[§3.3](#33-the-api-served-agent-shares-the-quota-with-the-user-and-judge).

### 3.1 Precision is a confound — read this before comparing models

Every locally-served agent now runs at its **released precision**, so a
"35B-A3B vs 27B" gap is a model difference and not a quantisation one. That was
not true until 2026-08-20: the 35B was served from the `-FP8` checkpoint,
because it fit a single 95 GB card and the bf16 weights (72 GB) do not.

**That run was discarded and redone in bf16.** The immediate reason was
availability — the 95 GB card stopped being obtainable — but the FP8 numbers
were not worth keeping even so: the model's only cross-model comparison would
have carried a precision difference the other rows do not have. The bf16
weights are served four-way tensor-parallel instead (see
[vllm_server.sh](scripts/vllm_server.sh)). The discarded results are parked in
`discarded/qwen3.6-35b-a3b-fp8-think-on/` rather than deleted.

Run tags still record precision where it applies, so this cannot be lost
downstream: `deepseek-v4-flash-fp4-think-on` is quantised;
`qwen3.6-35b-a3b-think-on` and `qwen3.6-27b-think-on` are not. The *absence* of
a precision suffix is what marks released weights, which only works if the
convention is never broken — a bf16 run must not inherit an `-fp8-` tag, and
that is also what stops `--auto-resume` from reading the discarded
simulations back in.

**The KV cache is bf16 everywhere, and always has been.** Weight precision is
not the only precision a serving stack picks: `--kv-cache-dtype` is a separate
axis, and a quantised *checkpoint* can carry a `kv_cache_scheme` that vLLM
honours without being asked. Neither happened here. The flag has never been
passed by [vllm_server.sh](scripts/vllm_server.sh), so every server has run at
vLLM's default `auto` — "use the model data type", i.e. bf16 — and the FP8 35B
checkpoint's `quantization_config` turns out to be weight-only
(`quant_method: fp8`, `activation_scheme: dynamic`, no `kv_cache_scheme`). Its
own startup log confirms the combination:

```
dtype=torch.bfloat16, ... quantization=fp8, ... kv_cache_dtype=auto
```

So the FP8-vs-bf16 confound was confined to the weights; attention state was the
same precision in both runs. Caveat on the verification: that is the only vLLM
log this repo kept — the 27B and Gemma servers were launched outside it — so for
those it rests on the default and on the script, not on an artifact.

For the 35B-A3B there are **two** caches, since 30 of its 40 layers are linear
attention. The full-attention KV cache is bf16 as above; the recurrent state
follows the model config's own `mamba_ssm_dtype: float32` (conv state bf16), and
vLLM's `mamba_cache_dtype` is likewise left at `auto`. Nothing quantised,
nothing overridden — which is also why this model wants more cache memory than
its parameter count suggests (§3, "Multi-GPU").

The same applies to the **fixed roles**: the e-infra endpoint is a LiteLLM proxy,
and `GET /v1/model/info` reports that its Kimi K3 is served at **int4**:

```bash
curl -s https://llm.ai.e-infra.cz/v1/model/info -H "Authorization: Bearer $KIMI_API_KEY"
```

So the user simulator and the judge are both quantized. This does not break
comparability — the same quantized simulator and judge face every agent — but a
quantized judge is a plausible source of grading noise that cannot be ruled out
without a full-precision reference run.

### 3.2 Thinking mode is not a uniform default

"Thinking enabled" means a different request for each family, and getting this
wrong silently produces a **different experimental condition** rather than an
error — the model just answers without reasoning and the run looks fine.

| family | default | what `CZTAU_THINKING=on` sends |
|---|---|---|
| Qwen3.6 | **on** | nothing (the default already thinks); `off` sends `enable_thinking: false` |
| Gemma 4 | **off** | `extra_body: {chat_template_kwargs: {enable_thinking: true}}` |
| DeepSeek V4 Flash | **off** | nothing — a *different model id*, `deepseek-v4-flash-thinking` |

DeepSeek is the exception to the per-request rule below, and not by choice: the
e-infra proxy publishes two aliases over one backend, and the `-thinking` one
carries `chat_template_kwargs: {thinking: true}` in its own server-side config
(visible in `litellm_params` from `/v1/model/info`). That is exactly the
server-side arrangement argued against here — but because the condition is
encoded in the *model id*, and the model id is what lands in the run tag, it
stays recoverable from the results, which is the property that actually matters.

For Gemma, `enable_thinking` is the **only** key that works. `thinking`,
`reasoning` and `include_thoughts` are all accepted and then silently ignored,
yielding an empty reasoning field. Verified on this server:

| request kwargs | reasoning chars | completion tokens |
|---|---|---|
| *(none)* | 0 | 166 |
| `enable_thinking: true` | 1,069 | 517 |

Two deliberate choices here:

* **Per request, not per server.** The flag is sent by the client, not via the
  server's `--default-chat-template-kwargs`. A server flag would make "thinking
  was on" a property of how that vLLM job happened to be launched — invisible in
  this repo and unrecoverable from the results afterwards.
* **The tag records it.** `gemma-4-e4b-it-think-on`, so a non-thinking run can
  never be mistaken for a thinking one.

`extra_body` was previously unexercised in this repo (the Qwen agents run
think-on, which sends no kwargs at all), so it was confirmed end to end that
LiteLLM forwards it to vLLM rather than dropping it, and that thinking and tool
calling coexist — the reasoning is separated into `reasoning`, leaving `content`
clean, with `tool_calls` still populated.

### 3.3 The API-served agent shares the quota with the user and judge

Every other agent runs on a local vLLM, so a simulation alternates between an
**unmetered** GPU (agent) and the **metered** e-infra key (user, judge). The
`deepseek` profile breaks that: all three roles land on the same key, which
enforces `max_parallel_requests: 4` across every model on it.

Two things follow, both handled in [scripts/env.sh](scripts/env.sh):

* **The semaphore already covers it.** The limiter matches on the call's
  `api_base` substring, so agent calls to `e-infra.cz` are counted like any
  other — no change was needed to stay under the cap.
* **Concurrency must be sized differently.** The `12` used for local profiles
  comes from wanting GPU work to overlap API waits; with no unmetered side left,
  extra concurrency only lengthens the queue on the same semaphore. This profile
  uses `2 × slots` instead. The semaphore is acquired *before* the LiteLLM call
  starts, so queueing time does not count against the request timeout.

Two defects were found and fixed when this profile was first exercised, both of
which had been latent since it was written but only bite a non-local agent:

* `CZTAU_CONCURRENCY` set inside the profile was silently overwritten by the
  user-simulator block further down, so the intended `3` resolved to `12`.
* The environment-interface LLM paired the *agent's* model id with
  `CZTAU_VLLM_BASE`. Those coincide for local profiles; for `deepseek` it aimed
  an API model id at a local server. It is unused by these domains, so the
  breakage was invisible — it now follows `CZTAU_AGENT_API_BASE`.

The preflight in [scripts/run_cell.sh](scripts/run_cell.sh) likewise now probes
the agent's *own* endpoint rather than `CZTAU_VLLM_BASE`, which would otherwise
have gated an API run on an unrelated local server still being up.

Verified against the endpoint before queueing: tool calls are returned,
zero-argument tools yield `arguments: "{}"` rather than `""`, a
`system` → `tool_call` → `tool` result round trip is accepted and answered in
Czech, and reasoning arrives in a separate field (`merge_reasoning_content_in_choices`
is false on the proxy) rather than leaking into `content`.

### vLLM serving

Tool calling requires explicit flags; without them vLLM rejects `tool_choice: auto`.
Qwen3.6 emits XML-format tool calls:

```bash
vllm serve "$MODEL" --served-model-name "$MODEL" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --max-model-len 65536 --gpu-memory-utilization 0.90 --max-num-seqs 64
```

See [scripts/vllm_server.sh](scripts/vllm_server.sh). Gemma 4 needs its own
parsers (`--reasoning-parser gemma4`); note that the reasoning parser alone
changes nothing until a request actually asks for thinking (§3.2).

**Multi-GPU.** Every model here fit one card except the bf16 35B-A3B (72 GB of
weights). That one is served four-way tensor-parallel; the script derives
`--tensor-parallel-size` from `CUDA_VISIBLE_DEVICES` rather than taking it as a
second setting, so the sbatch `-G` flag is the only place the number lives:

```bash
sbatch scripts/download_model.sh Qwen/Qwen3.6-35B-A3B          # 72 GB, ~30 min
sbatch --dependency=afterok:$DL_JOBID \
    -p gpu-troja -C gpuram40G -G 4 -N 1 --cpus-per-task=8 --mem=128G \
    --export=ALL,CZTAU_MODEL=Qwen/Qwen3.6-35B-A3B scripts/vllm_server.sh
```

Two details that are not cosmetic:

* **`-N 1`.** TP shards one model across GPUs over the intra-node interconnect,
  so a `-G 4` that Slurm satisfies from two nodes fails outright.
* **The weights are fetched by a separate CPU job, and the server waits on it.**
  `vllm serve` downloads happily by itself, but it does so *holding the GPU
  allocation* — half an hour of four idle A100s. `afterok` claims them only
  once the weights are on disk.

`--gpu-memory-utilization 0.90` leaves ~72 GB for cache across the four 40 GB
cards, which is comfortable here; on a 2×48 GB allocation the same model leaves
only ~14 GB, and this model spends it fast — the hybrid layer stack keeps a
float32 recurrent state per sequence (~63 MB × `--max-num-seqs`) *on top of* the
KV cache for its ten full-attention layers. 4×40 GB was chosen over 2×48 GB for
that headroom and for twice the aggregate memory bandwidth, which is what an
MoE at batch size >1 is actually limited by.

Whatever the flags, the serving side is worth verifying rather than assuming —
a server started without `--enable-auto-tool-choice` does not fail loudly, it
just produces an agent that never calls a tool, which reads as model
incompetence. The checks worth running against a new server are: a tool call is
returned for a tools request; zero-argument tools yield `arguments: "{}"` and
not `""`; a `system` → `tool_call` → `tool` result round trip is accepted; and
reasoning, when enabled, lands in `reasoning` rather than leaking into
`content`.

Those four had been run by hand each time a server moved, which is exactly the
kind of check that gets skipped on the run where it matters. They are now
[scripts/verify_server.py](scripts/verify_server.py):

```bash
python scripts/verify_server.py http://tdll-8gpu2:8000/v1 Qwen/Qwen3.6-35B-A3B
```

Run it before queueing cells against a new server. `run_cell.sh`'s preflight is
not a substitute — it only asks whether the model id is *served*, which a
tool-blind server answers just as happily as a working one.

---

## 4. Modifications to SEATauBench

`SEATauBench/` is an upstream checkout at commit `1d6f92a` with **six** files
changed (`git -C SEATauBench diff --stat` is the authoritative list; nothing else
in that tree is modified, and the τ³ additions once described in §10.2 are **not**
in this checkout). In summary:

**`data/seatau/languages.json`** — adds the Czech entry. This is the entire
linguistic content of the adaptation (see §1).

**`src/tau2/config.py`** — the judge and environment-interface models were
hardcoded to `azure/gpt-4.1-mini` with no CLI flag, which fails with no Azure
credentials and would silently make an external call. Now configurable via
`TAU2_LLM_NL_ASSERTIONS*` / `TAU2_LLM_ENV_INTERFACE*` environment variables, and
the LLM cache is env-gated.

**`src/tau2/utils/llm_utils.py`** — adds a **per-host concurrency limiter**.
`--max-concurrency` counts whole *simulations*, and each simulation alternates
between the local vLLM and the metered API. Using it to respect the API quota
would therefore throttle the GPU to the same rate. Measured: ~23 s per agent call
versus ~3 s per user call, so only ~10% of wall time touches the API. The limiter
caps in-flight calls to `TAU2_RATE_LIMITED_HOST` and leaves vLLM unthrottled,
letting simulation concurrency sit where the GPU saturates.

**`src/tau2/evaluator/evaluator_nl_assertions.py`** — **bug fix, affects results.**
The NL-assertion judge parsed its verdict with a bare `json.loads(content)`. Kimi K3
sometimes wraps its JSON in a ` ```json ` fence, which raises
`JSONDecodeError: Expecting value: line 1 column 1 (char 0)` and marks the whole
simulation `infrastructure_error` — *after* the conversation has completed. Because
`run_with_retry` re-runs the entire simulation and everything is at temperature 0,
the retry failed identically, so each affected task burned four full conversations
before being dropped.

tau2 already ships `extract_json_from_llm_response` (handles fences and bare
`{...}`), and every *other* LLM-judge call site uses it; the NL-assertions
evaluator did not. The fix routes through the same helper.

> **Impact:** ~10% of retail simulations were being silently discarded before this
> was found. **Airline was unaffected** — 0 infrastructure errors across all 150
> simulations in both languages — so the completed airline results predate and are
> unaffected by the fix. Retail was restarted with the fix in place.

**`src/seatau/metrics/language_use.py`** — excludes the mandated transfer sentence
from language correctness, in the one function every consumer of the metric routes
through. Fully described in [§6.1](#61-the-mandated-transfer-sentence-is-excluded-from-language-correctness).

**`src/tau2/metrics/agent_metrics.py`** — **bug fix, affects results.** The
infrastructure-error exclusion compared a DataFrame column against the
`TerminationReason` enum. Under pandas ≥3 that column lands in the new `str`
dtype, whose elements stringify to the *qualified* name
(`TerminationReason.INFRASTRUCTURE_ERROR`) while the str-mixin enum equals its
bare value (`infrastructure_error`) — so the comparison matches **zero** rows and
the filter becomes a no-op. It does not merely skip the exclusion: infra-errored
simulations carry `reward = NaN`, which `is_successful()` reads as a failure, so
they are scored as failed trials and `max_k` is not capped either. Both sides are
now normalised before comparing. Every number in §9 comes from the fixed version;
the runs themselves are unaffected, since this is read-side only.

---

## 5. Fixed experimental parameters

| parameter | value |
|---|---|
| trials per task (`--num-trials`) | 3 |
| seed (`--seed`) | 300 |
| agent / user / judge temperature | 0.0 |
| agent thinking | enabled |
| language components | `user_system`, `agent_system`, `greeting` |
| asset mode | `original` (no translated assets) |
| telecom split | `base` (114 tasks, not the 2285 generated ones) |
| LLM response cache | **disabled** (see §7) |

---

## 6. Metrics

Computed by [scripts/report.py](scripts/report.py) via tau2's own
`compute_metrics`, so metric definitions are upstream's, not ours.

- **pass^k** — fraction of tasks solved in *all* k trials. `pass^1` is the ordinary
  success rate; higher k measures consistency.
- **ρ_q = pass^q / pass^1** (paper eq. 1, q=3) — **robustness**. How much success
  survives repetition.
- **language correctness** — fraction of agent turns whose fastText-detected
  language (`lid.176.bin`) matched the target. Should be 1.0 in `english` cells.
  The paper found this correlates only weakly with task success (R²=0.014), so a
  low language score does not by itself explain failures.
  **We exclude the mandated transfer sentence from this metric — see §6.1.**

### 6.1 The mandated transfer sentence is excluded from language correctness

Every domain policy (airline, retail, telecom) ends its transfer rule with:

> To transfer, first make a tool call to `transfer_to_human_agents`, and then send
> the message `'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'` to the user.

An agent that emits that sentence verbatim in a Czech run is **obeying the policy**,
not failing to speak Czech, so scoring it as a Czech-language turn measures the
wrong thing. Three facts make this the right reading rather than a convenient one:

- We run `asset_mode: original`, so the policy reaches the agent **in English**, in
  quotes, as a literal.
- Upstream agrees. Their own translated Vietnamese policy
  (`data/tau2/domains/telecom/vi/main_policy.md`) translates the surrounding
  sentence and **leaves the quoted phrase in English**.
- Nothing string-matches it. No grader or task assertion checks for the phrase, so
  translating it would not have failed a task either — the instruction is genuinely
  ambiguous, which is exactly why it should not be scored as language evidence.

It was not a rounding error. The phrase appears in **16.1% of Czech simulations**
(395/2,460 — recounted over the finished matrix) and accounted for very nearly
*all* apparent Czech language failure:

| cell (Czech airline) | before | after |
|---|---|---|
| `deepseek-v4-flash-fp4` | 0.962 | **1.000** |
| `gemma-4-31b-it`        | 0.907 | **0.952** |
| `gemma-4-e4b-it`        | 0.900 | **0.998** |
| `qwen3.6-27b`           | 0.905 | **0.993** |
| `qwen3.6-35b-a3b`       | 0.920 | **0.998** |

This **reverses a ranking**: `gemma-4-e4b` looked like the second-worst Czech
speaker (0.900) and is in fact near-perfect (0.998), while `gemma-4-31b-it` — which
looked mid-pack — is the genuine worst (0.952). The old numbers were largely
measuring how often each model chose to transfer.

Implementation is one filter in `text_turns`
([language_use.py](SEATauBench/src/seatau/metrics/language_use.py)), which every
language consumer routes through — the evaluator, the batched run scorer, and the
three analysis scripts — so the definition cannot drift between them. Two details:

- The phrase is **stripped from** the turn, not the turn dropped. A turn mixing
  real Czech with the mandated sentence is still judged on its Czech. This matters
  because fastText labels *"Czech sentence + the phrase"* as **Spanish** — such
  turns were noise in both directions, not merely false negatives.
- It is stripped **in every language, including English**, so the exclusion cannot
  bias the EN/CS comparison the benchmark exists to make. Measured effect on the
  English cells is nil (worst case −0.0004).

Because the metric is frozen into `results.json` at evaluation time, changing the
definition does **not** update finished runs. Replay it over the stored messages:

```bash
python scripts/rescore_language.py            # dry run, writes nothing
python scripts/rescore_language.py --write    # rewrite in place, keeps .prelang backups
```

It is idempotent and touches only `reward_info.info.language_correctness`. **No task
score moved**: `LANGUAGE_CORRECTNESS` is absent from every run's `reward_basis`, so
it multiplies nothing here — the script asserts that per file and refuses to write
if it is ever false. `pass^k` and ρ₃ are unchanged.

`report.py` guards two failure modes that have already occurred here:

- **Partial runs reading as finished.** `pass^k` is computed only over tasks with
  at least k completed trials, so a 3%-complete run reports a confident `pass^1`
  over a handful of easy tasks. Incomplete rows are marked `PARTIAL`.
- **Hidden infrastructure errors.** tau2 *excludes* them from every metric, so a
  run dying on rate limits looks empty rather than broken. The `infra` column and
  the `FAILING` marker surface this. Anything above zero should be investigated
  before the row is trusted, and `--auto-resume` retries them.

The `paper p^1` / `rho3` columns compare against
`data/seatau/experiments.csv` for the paper's open-weights agent
(`qwen-3-235b-it` = Qwen3-235B-A22B-Instruct). **This is a different, much larger
model** — the columns are a sanity check on the shape of the effect, not a
like-for-like baseline. There is no Czech row in the paper, so for
`l2_interaction` the reference is the mean over the five SEA languages.

### 6.2 Language *correctness* is not language *quality* — the judge that fills the gap

`language_correctness` asks fastText one question: **is this Czech at all?** After
§6.1 the Czech cells answer "yes" 95–100% of the time, and the metric is
consequently finished as a discriminator. It has nothing to say about the thing a
Czech reader notices immediately, which is that an agent can stay in Czech for
every turn and still write Czech that is wrong (`ne odpovídá`, `vaší popisu`) or
merely translated-sounding (`z Chicagoa`, `zpracovat vrácení`).

[scripts/annotate_language.py](scripts/annotate_language.py) measures that second
thing with an LLM judge. It is **not** part of any score: nothing here feeds
`reward`, `pass^k`, or `reward_basis`. It is a descriptive layer over finished
runs.

**Coverage: every Czech task, one trial each.** The pass runs at `--limit 0
--trial 0` over airline and retail for all five agents — **820 simulations**
(5 × (50 + 114)). It is exhaustive in *tasks* and sampled in *trials*: three
trials of one task are three samples of near-identical text, so the second and
third cost the same and buy far less. Telecom and `banking_knowledge` are not
annotated because they were not run (§1, §10).

**One request per conversation, not per turn.** Every agent text turn in a
simulation is concatenated with `---` separators and sent to Kimi K3 in a single
call. Per-turn calls would multiply the request count by ~6 for no gain: the judge
reads the whole conversation either way, and the parallel-request cap (below) is
the binding constraint. The judge returns a list of `{span, explanation,
category}`.

Decisions that shape what the number means:

* **Agent turns only.** The customer is played by Kimi K3 in every single cell, so
  its Czech is a property of the fixed apparatus and identical across models — it
  cannot discriminate and would only add noise.
* **Turns that carry only tool calls are dropped**, and the scripted opening
  greeting is dropped. The greeting comes verbatim from `languages.json` and is
  byte-identical in all 2,460 Czech simulations; annotating it would credit the
  model for text it did not write.
* **MAJOR vs MINOR.** MAJOR is what a grammar or syntax checker would flag — a
  non-word, a wrong case ending, broken agreement, negation split from its verb.
  MINOR is legal Czech a native speaker would not write. The split matters because
  the two have different causes and the counts move independently.
* **Pronoun, person, and gender consistency is explicitly excluded**, both what the
  agent uses for itself and how it addresses the customer. The agent was never
  instructed on any of it, so flagging it would score models against a standard the
  experiment did not set. English identifiers, tool names, and the §6.1 transfer
  sentence are excluded for the same reason — the policy ordered them.
* **The judge is told an empty list is a normal answer.** Left unsaid, an
  annotation prompt will always find something.

**Structured outputs, not prompt-and-parse.** The e-infra endpoint accepts
`response_format` with a strict JSON schema, so the script uses
`client.chat.completions.parse()` against a Pydantic model and the fenced-block
parsing fallback is not needed.

**Truncation is recovered from, in a bounded ladder.** A judge that thinks past
the token limit returns `finish_reason == "length"` and no parsed output — the
annotations are simply lost. Two escapes are tried in order: re-ask at
`temperature=0.7` (a plain retry at temperature 0 reproduces the loop token for
token, so sampling is the only thing that can change), then judge the
conversation in halves and merge, each half re-entering the same ladder. Halves
are disjoint and spans are copied verbatim, so nothing is double-counted; what a
segment loses is the rest of the conversation as context, which is why splitting
is the *second* resort. The floor is one turn — a single turn that truncates
twice is banked as an error rather than chopped mid-sentence, which would break
the verbatim-span contract the viewer depends on to re-find spans (§8.1.1).

The ladder is capped at `MAX_CALLS_PER_SIM = 12`, because it is exponential:
every level doubles the segments and each segment pays two attempts before
splitting again, so an unattended pathological conversation could issue ~60
requests of up to 900 s while holding one of the four parallel slots. A
conversation whose quarters still truncate is not failing for want of a shorter
prompt. Each item records which escape it took in its `fallback` field, so the
rate is recoverable from the output and not just the log. Over the finished pass
it was needed **16 times in 820 conversations (2.0%): 10 recovered by re-asking
at temperature 0.7, 6 by splitting** (5 in halves, 1 in quarters). No
conversation exhausted the ladder — the final file contains **zero** error
records.

### 6.2.1 Three failure modes this cost a morning to find

The judge is slow for a real reason — it spends **a median of ~8,200 reasoning
tokens to emit ~450 tokens of JSON**, 95% of its output, at a median ~450 s per
conversation (p90 ~890 s), measured over all 820 conversations of the finished
pass. Those figures are roughly double the first estimate taken from a
50-conversation airline sample; retail conversations are longer, and the tail is
heavier than a small sample shows. Everything below is about not mistaking an
operational fault for that expected slowness.

**1. The client timeout must clear the request time by a wide margin, because the
SDK retries timeouts.** This was set to `timeout=300.0` against requests that take
230–350 s. Every request over the threshold raised `APITimeoutError`, which the SDK
retries, so each one looped `13 × 300 s ≈ 65 minutes` before surfacing a single
error. The symptom is a job that is running, holding live connections, consuming no
CPU, and producing **neither results nor failures** — indistinguishable from a hang.
Now `timeout=900.0, max_retries=5`. Do not tighten it to "fail fast": it fails slow
and invisibly. 429s are cheap to retry (they return instantly); *timeouts* are not.

**That margin closed over the run, and a repeat pass should widen it.** 900 s was
chosen against requests believed to take 230–350 s. Measured over all 820
conversations the p90 is ~890 s — the setting ended up *at* the 90th percentile
rather than clear of it, and the clearance came from `max_retries` absorbing the
overshoot, not from the timeout being generous. Anything longer-winded than
retail (τ³ banking, §10) should raise the timeout before queueing rather than let
the `13 × timeout` retry loop come back. (Per-*request*, note: a split simulation
is several requests, which is why `seconds` in the output file runs to 6,911 s at
the maximum without any single call approaching the limit.)

**2. The parallel-request cap is a hard ceiling, and killed clients leak it.** The
key allows `max_parallel_requests: 4` and returns 429 the instant a fifth call
opens. The first attempt at 8 workers lost 46 of 50 requests. Worse, the proxy does
**not** release a slot when a client vanishes: `kill -9` on three in-flight workers
consumed the whole cap until the window reset (the `Limit resets at` timestamp in
the 429 body), which again looks exactly like a hung job. **Use `scancel`/SIGTERM
and let clients drain.** And nothing else may touch the key while a batch runs — a
benchmark cell (§3.3), a probe script, or a stray `curl` does not "share" the cap,
it takes a slot the batch then spends its time being 429'd out of.

**3. A 200 from that endpoint does not prove capacity.** The proxy caches: three
probes returned the *identical* response id and HTTP 200 while the key was in fact
at `Remaining: 0`. Cached responses bypass the parallel-request counter. **Always
probe with a unique nonce**, or the health check will cheerfully confirm a
non-existent slot.

### 6.2.2 The endpoint can die mid-pass, and a clean exit hides it

Found the hard way on 2026-08-22 (job `6764101`). The airline pass finished
normally; the retail pass lost the backend at request **89 of 272** and every
request after it failed, to the end of the queue. **184 of 820 simulations ended
with an `error` record instead of annotations** — spread across four of the five
retail cells (`deepseek-v4-flash-fp4` had already finished when the backend went;
`gemma-4-e4b-it` was worst hit at 67). The errors are HTTP 500s from the proxy,
and the proxy is not the thing that broke:

```
litellm.InternalServerError: Hosted_vllmException - Server disconnected.       (159)
… - Cannot connect to host frps-proxy-tunnel.vllm-ns.svc.cluster.local:8003    ( 17)
… - Can not write request body for …/v1/chat/completions                       (  8)
```

Three things make this worth its own section rather than a line in §6.2.1:

* **It fails *fast*, which is why it is easy to miss.** The §6.2.1 timeout hang
  produced neither results nor failures for 65 minutes at a stretch. This does the
  opposite: ~40 s per request, `r=?`, no usage recorded, so the job *races* through
  the remaining queue and exits `COMPLETED`. Same lesson as the resubmit in
  [§9](#9-results) — **a clean Slurm exit is not proof the work is whole.**
* **`meta.complete` does not mean what the name suggests.** It is
  `done >= len(jobs)` — every job *attempted*, not every job *succeeded* — so the
  viewer's header reads a run that failed 184 times as finished. The number to
  trust is the `N judge errors` pill beside it, which counts items with an
  `error`. Either read the pill or fix the field; do not read the word.
* **A resume erases the evidence as it works.** `--resume` sorts prior successes
  into `prior` (skipped) and prior *failures* into `carried`, then pops from
  `carried` exactly the items it is about to redo — so the moment the retry writes
  its first result, the error count in the file drops to **zero** and stays there.
  It does not tick down as the retry progresses. While a retry runs, the honest
  progress signal is *judged conversations climbing toward 820*, not errors
  falling.

The remedy is just the resume, naming only the domain that has gaps:

```bash
sbatch scripts/run_annotate.sh retail    # re-queues the failures, skips the rest
```

That is what closed it: job `6765856` re-judged the 184 in 8 h 18 m and the file
now holds **820 of 820 with zero error records** (§9.1).

One trap this narrowly avoided: `run_annotate.sh` runs one domain per process
against a shared `--out`, so without the `carried` set a retail-only pass would
rewrite the file *minus airline's failures* — neither re-queued (out of
`--domain`) nor recorded, with the error count silently under-reporting the gap.
124 airline failures went missing exactly that way once. The seed is
`list(prior.values()) + list(carried.values())`, and it is load-bearing.

Results are written **atomically after every completed request** (`os.replace`), so
an interruption costs one request rather than the batch, the viewer can serve
partial results safely, and `--resume` skips what is already banked — retrying
prior failures but never re-paying for successes.

```bash
python scripts/annotate_language.py --print-prompt      # inspect the prompt, exit
python scripts/annotate_language.py --dry-run           # count requests, no calls
python scripts/annotate_language.py --limit 10 --domain airline

# the full single-trial queue (airline then retail) as a Slurm job -- a pure API
# client like run_cell.sh, so it belongs on a CPU partition, not the submit node
sbatch scripts/run_annotate.sh

# one domain only. every invocation passes --resume, so this is also the retry
# path after an endpoint drop (§6.2.2): it re-queues that domain's failures and
# skips its successes, and carries the other domain's records through untouched
sbatch scripts/run_annotate.sh retail

# which thinking controls this endpoint actually honours (run only when idle)
python scripts/probe_thinking.py
```

`--trial 0` is the default, so `--limit 10` means **ten distinct tasks**, not three
tasks seen three times — three trials of one task are three samples of near-identical
text. Output goes to `results/language_annotations.json`, which the viewer reads.

**Known limitation: the judge grades its own dialect partner.** Kimi K3 is the user
simulator in every cell and the judge here. It is not judging its own output — only
agent turns are annotated, and no agent is Kimi K3 (§2) — but it is judging Czech
produced in conversation with itself, and a shared idea of what natural Czech looks
like is exactly the kind of correlation that inflates agreement. Treat the absolute
counts as soft and the between-model ordering as the usable signal.

---

## 7. Known issues and threats to validity

**Absolute scores are not comparable to the paper.** Different agent, different
user simulator, different judge (paper used GPT-4.1; we use int4 Kimi K3), and a
different serving stack. The English↔Czech delta within one model is the
defensible quantity.

**Precision is confounded with model identity.** See §3.

**The user simulator is a confound.** ~20% critical user-error rate is reported by
the paper for this class of setup. Holding it fixed removes it from *cross-model*
comparisons but not from absolute numbers.

**The judge is quantized and occasionally malformed.** See §4. The fence bug is
fixed, but a quantized judge grading Czech free-text assertions has not been
validated against human annotation — there is no manual annotation in this project.

**LLM response caching is disabled.** LiteLLM's disk cache is SQLite via
`diskcache`, and this project lives on Lustre, where SQLite's WAL locking is
unreliable: it survives single-threaded use but dies with
`sqlite3.OperationalError: locking protocol` under concurrent worker threads.
LiteLLM hardcodes `dc.Cache(dir)` with no way to pass a network-safe journal mode.
Reuse across runs is instead handled by `--auto-resume`, which skips completed
simulations and retries only infrastructure failures. To re-enable anyway, point
the cache at node-local disk (`TAU2_LLM_DISK_CACHE_DIR="$TMPDIR/..."`), which
survives only within a single job.

**API quota.** The e-infra key enforces `max_parallel_requests: 4` across *all*
models on that key. The limiter (§4) is set per job via `CZTAU_API_SLOTS`; when two
chains run concurrently each gets 2, so their combined in-flight calls never exceed
the cap. For the `deepseek` profile the agent draws on those same slots — see
[§3.3](#33-the-api-served-agent-shares-the-quota-with-the-user-and-judge).

---

## 8. Running it

```bash
# one cell = (scenario, domain, language)
sbatch --export=ALL,CZTAU_PROFILE=local scripts/run_cell.sh english airline
sbatch --export=ALL,CZTAU_PROFILE=local scripts/run_cell.sh l2_interaction airline cs

# a dependency chain of cells; -s splits the API quota between concurrent chains
scripts/submit_chain.sh -p local   -s 2 english:retail l2_interaction:retail:cs
scripts/submit_chain.sh -p local35 -s 2 english:airline l2_interaction:airline:cs

# EN and CS in parallel against ONE server: split by LANGUAGE, chain by domain.
# 2 + 2 = the key's whole parallel-request budget, so do not run a third chain.
scripts/submit_chain.sh -p local35 -s 2 english:airline        english:retail
scripts/submit_chain.sh -p local35 -s 2 l2_interaction:airline:cs l2_interaction:retail:cs

# -a chains behind an existing job, so a third model waits for a slot to free up
scripts/submit_chain.sh -a "$JOBID" -p gemma -s 2 \
    english:airline l2_interaction:airline:cs english:retail l2_interaction:retail:cs

# the API-served agent (§3.3). Still -s 2 while a local chain holds the other 2 —
# the difference is that here the agent's own calls count against those slots too.
scripts/submit_chain.sh -a "$JOBID" -p deepseek -s 2 \
    english:airline l2_interaction:airline:cs

# progress + metrics
python scripts/report.py

# Czech language-quality judge (§6.2); writes results/language_annotations.json
python scripts/annotate_language.py --limit 10 --domain airline

# leaderboard + browsable trajectories, incl. EN/CS side-by-side
python scripts/viewer.py --port 8765

# refresh the published tau2-bench reference bars (needs network; run rarely)
python scripts/fetch_reference.py
```

Cells are pure API clients — all compute is on the vLLM server — so they run on
**CPU** partitions. They use `--auto-resume`, so a timed-out or requeued job picks
up where it left off; `run_cell.sh` preflights the vLLM endpoint and fails fast if
the server it targets has since expired.

### Files

| path | purpose |
|---|---|
| [scripts/env.sh](scripts/env.sh) | all role/model/quota configuration; source, don't execute |
| [scripts/run_cell.sh](scripts/run_cell.sh) | sbatch script for one cell |
| [scripts/submit_chain.sh](scripts/submit_chain.sh) | queue cells as a dependency chain |
| [scripts/report.py](scripts/report.py) | progress and metrics |
| [scripts/viewer.py](scripts/viewer.py) | HTML leaderboard + trajectory viewer (stdlib only) |
| [scripts/fetch_reference.py](scripts/fetch_reference.py) | vendor published tau2-bench numbers |
| [scripts/vllm_server.sh](scripts/vllm_server.sh) | vLLM launch with the required tool-calling flags |
| [scripts/download_model.sh](scripts/download_model.sh) | pre-fetch weights on a CPU node, so no GPU idles behind a download |
| [scripts/verify_server.py](scripts/verify_server.py) | the four tool-calling / thinking checks a new vLLM server must pass |
| [scripts/rescore_language.py](scripts/rescore_language.py) | replay language correctness over finished runs after a definition change (§6.1) |
| [scripts/annotate_language.py](scripts/annotate_language.py) | LLM-as-a-judge annotation of Czech language *quality* (§6.2) |
| [scripts/run_annotate.sh](scripts/run_annotate.sh) | sbatch wrapper queueing the annotation run across domains |
| [scripts/probe_thinking.py](scripts/probe_thinking.py) | which thinking-budget controls the endpoint honours (§6.2.1) |
| `scripts/assets/` | the viewer's logo and favicon; the only static files it will serve |
| `results/tau2_reference.json` | the vendored reference numbers |
| `results/language_annotations.json` | the judge's flagged spans, read by the viewer |
| `SEATauBench/data/simulations/<run_tag>/<cell>/` | results |
| `discarded/` | runs deliberately excluded from reporting (see §3.1) |
| `logs/` | Slurm job output, one file per cell / annotation / vLLM server |

Three leftovers are still on disk and are **not** part of the pipeline:
`scripts/submit_airline.sh` (the original two-cell submitter, superseded by
`submit_chain.sh`), `results/probe_queries_cs.json` (the Czech query set of the
lost τ³ retrieval probe, §10), and a copy of `favicon.ico` in the repo root — the
viewer serves `scripts/assets/favicon.ico` and never looks at it.

Secrets (`KIMI_API_BASE`, `KIMI_API_KEY`) live in `SEATauBench/.env`, loaded by
`env.sh`.

The viewer's EN/CS comparison is **pinned to a single run tag**. Matching on domain
alone would happily pair one model's English cell with another model's Czech cell —
a model comparison wearing a language comparison's label.

### 8.1 The leaderboard page

The viewer's index is a per-domain leaderboard: one chart per domain, one row per
model, English and Czech bars stacked with the **EN→CS gap hatched between them** —
the distance is the thing the project is measuring, so it gets its own ink rather
than being left as a subtraction for the eye. `?m=p1|p2|p3` switches the metric.

Two guards, both of which exist because the naive version was wrong:

* **Gaps are drawn only when both sides cover the whole task set at that k.** The
  test is task coverage, not "the cell is 100% done" — a cell at 149/150 still has
  every task covered at k=1, whereas one at 9/150 has nine tasks of fifty and would
  otherwise have contributed a confident-looking Δ made of noise.
* **Infrastructure errors are excluded and cap k at the thinnest task**, so a cell
  one trial short reports pass³ as `–` rather than computing it over a task that
  only has two trials. See §6.

The grey bars are published tau2-bench submissions, vendored by
`fetch_reference.py` so the page never depends on the network. They are
**indicative, not like-for-like**, and the page says so: those runs drive the user
simulator with `gpt-5.2` while ours uses Kimi K3, and the simulator is a large part
of what a τ² score measures.

Two things deliberately absent from that comparison:

* **The site's headline `core` number.** It is the unweighted mean of airline,
  retail and telecom; telecom is the easiest of the three (0.85–0.98) and we do not
  run it, so the headline sits well above per-domain reality. Compare per domain.
* **The 2026-08 frontier submissions** — Claude Opus 5, GPT-5.6-sol, Kimi K3,
  Qwen 3.8 Max, Gemini 3.1 Pro. Every one of them reports **only**
  `banking_knowledge`, with `airline`/`retail`/`telecom` explicitly `null`. That is
  a different and much harder domain (AllTools retrieval, 4 trials; top score 0.55
  against 0.84 on airline), so putting them on our axis would have read as "the 27B
  beats Claude Opus 5". The comparable frontier rows are the 2026-02/03 ones:
  Claude Opus 4.5, GPT-5.2, Qwen3.5-397B-A17B, Gemini 3 Pro.

The index is **two columns**: leaderboard charts on the left, the §6.2 language-quality
summary on the right. They answer different questions about the same runs — *did the
agent succeed* and *did it write decent Czech* — and the project's whole premise is
that those can come apart, which is easier to see with both on screen. The right
column reports **spans per conversation** rather than raw counts (a verbose model
gives the judge more to flag, so the raw count rewards terseness) alongside
`clean`, the share of conversations with nothing flagged at all — the more robust
of the two, since it does not care how errors cluster.

### 8.1.1 The annotations tab

`/annotations` is a flat, filterable list of every flagged span: one row per error,
severity chip, model, task, the span highlighted in dimmed left/right context, and
the judge's explanation. Rows deep-link into the conversation and scroll to the
span itself, rather than dropping the reader at the top of a long transcript to
hunt for it. MAJOR sorts first, because
the list is read top-down and the checker-catchable errors should not sit under
stylistic notes.

The same spans are highlighted **in place** on the sim and compare pages — light
red for MAJOR, light pink for MINOR, explanation on hover. Two details in how they
get there:

* **Spans are re-found by text search, not by offset.** The judge saw a joined
  string of turns, not the message array, so it has no offsets to give. Each span
  is searched turn by turn, first match wins, and a match overlapping an
  already-placed span is skipped so two annotations quoting the same words land on
  two occurrences instead of stacking on one. Whitespace is matched loosely (`\s+`
  for any run), since a model that collapses a newline inside a quoted span is not
  wrong about the Czech.
* **Spans that cannot be found are kept and marked**, not dropped. A span the judge
  failed to copy verbatim is evidence about the judge, and silently hiding those
  would hide the bug.

The sim page distinguishes **judged-and-clean** from **not judged**. The judge runs
on a sample, so a page that showed nothing in both cases would let an absence of
highlights read as a verdict.

### 8.2 The trajectory pages show the instructions, not just the transcript

A τ² transcript is close to unreadable on its own. The customer asks for
something the reader cannot verify was ever the goal, the agent refuses on
grounds the reader cannot see, and the reward says `0.0` for reasons stated
nowhere on the page. All of it is already in `results.json`; the sim and compare
pages surface it as four panels, each tagged with **who could actually see it**:

| panel | source | visible to |
|---|---|---|
| Customer brief | `tasks[].user_scenario.instructions` | user simulator only |
| Agent policy (system prompt) | `info.environment_info.policy` | agent only |
| User-simulator guidelines | `info.user_info.global_simulation_guidelines` | user simulator only |
| How this was graded | `tasks[].description` + `evaluation_criteria` | neither party |

The tagging is the point, not decoration. The agent is never told what the
customer was instructed to do, and neither side is shown the grading criteria —
so a reader who sees all four in one place will otherwise conclude the agent
"should have known" something it was never given. The last panel in particular
is the judge's brief, not the agent's.

**What L2 Interaction actually changed** gets its own panel on the compare page,
listing the three components verbatim: the text appended to the agent's policy,
the text prepended to the customer brief, and the greeting. This is worth
reading once, because `agent_system` is **not purely a language instruction** —
upstream's template bundles it with an identifiers clause:

> Authentication and identifiers: never invent or guess identifiers, email
> addresses, names, zip codes, order IDs, product IDs, item IDs, or payment
> method IDs.

The Czech agent gets that sentence and the English agent does not. It is
inherited from SEATauBench (`DEFAULT_AGENT_SYSTEM_INSTRUCTION_TEMPLATE` in
`src/seatau/translation/language.py`), not something added here, but it means
the EN↔CS delta contains a small prompt difference beyond language. It pushes
against the measured effect rather than inflating it — the Czech side gets an
extra anti-hallucination instruction and still scores lower — so it does not
explain the gap, but it should be stated rather than discovered later.

One asymmetry in what can be shown: the `agent_system` text is recoverable from
the stored policy, because it was appended to it. The `user_system` text is
**not stored anywhere** — tau2 prepends it to the user's instructions at
prompt-build time and records neither. The viewer reconstructs it from
`seatau.translation.language`, the same source the runner uses, and labels it as
reconstructed. That import is lazy and optional: run the viewer under the
`cztaubench` venv and the panel appears, run it anywhere else and it says so
rather than silently omitting the instruction.

**Fixed at the same time:** `/compare` derived the domain with
`cell.split("_")[1]`, which reads `l2_interaction_airline_cs` as domain
"interaction". Arriving from a **Czech** cell therefore looked for an
`english_interaction` cell, never found one, and showed *"Need both … currently
have: Czech"* for the whole run; arriving from the English cell happened to work,
so the page looked incomplete rather than broken. It now uses `parse_cell`, which
already handled this (it is the same mistake §10.2 records for
`banking_knowledge`), and matches both cell names exactly instead of by
substring.

---

## 9. Results

**The experiment is finished.** All 20 cells — 5 agents × 2 domains × {English,
Czech} — are complete at 100%, with **0 infrastructure errors** anywhere:
`deepseek-v4-flash-fp4`, `qwen3.6-35b-a3b` (bf16), `gemma-4-31b-it`,
`qwen3.6-27b`, `gemma-4-e4b-it`. That is 984 simulations per agent, **4,920 in
total**, plus 820 judged conversations of language annotation (§9.1). Nothing is
pending. `python scripts/report.py` is authoritative for the task metrics and the
two tables below are its output plus the language column; §9.1's tables come from
`results/language_annotations.json`.

**Deliberately not run:** telecom (§1), and `banking_knowledge` (§10).

### The full matrix

`pass^1` and `pass^3` per cell, with the EN→CS delta that is the quantity of
interest, and Czech language correctness after the §6.1 correction. Rows ordered
by English `pass^1`.

**Airline** (50 tasks × 3 trials = 150 simulations per cell):

| agent | EN p^1 | CS p^1 | **Δ p^1** | EN p^3 | CS p^3 | Δ p^3 | EN ρ₃ | CS ρ₃ | CS lang |
|---|---|---|---|---|---|---|---|---|---|
| `deepseek-v4-flash-fp4` | 0.880 | 0.807 | **−0.073** | 0.800 | 0.760 | −0.040 | 0.909 | 0.942 | 1.000 |
| `qwen3.6-35b-a3b` | 0.880 | 0.780 | **−0.100** | 0.780 | 0.620 | −0.160 | 0.886 | 0.795 | 0.998 |
| `gemma-4-31b-it` | 0.847 | 0.800 | **−0.047** | 0.720 | 0.740 | +0.020 | 0.850 | 0.925 | 0.952 |
| `qwen3.6-27b` | 0.807 | 0.753 | **−0.054** | 0.680 | 0.640 | −0.040 | 0.843 | 0.850 | 0.993 |
| `gemma-4-e4b-it` | 0.607 | 0.567 | **−0.040** | 0.500 | 0.440 | −0.060 | 0.824 | 0.776 | 0.998 |
| *mean* | | | **−0.063** | | | −0.056 | | | |

**Retail** (114 tasks × 3 trials = 342 simulations per cell):

| agent | EN p^1 | CS p^1 | **Δ p^1** | EN p^3 | CS p^3 | Δ p^3 | EN ρ₃ | CS ρ₃ | CS lang |
|---|---|---|---|---|---|---|---|---|---|
| `deepseek-v4-flash-fp4` | 0.895 | 0.865 | **−0.030** | 0.816 | 0.746 | −0.070 | 0.912 | 0.861 | 1.000 |
| `qwen3.6-35b-a3b` | 0.851 | 0.845 | **−0.006** | 0.754 | 0.675 | −0.079 | 0.887 | 0.799 | 0.997 |
| `qwen3.6-27b` | 0.842 | 0.810 | **−0.032** | 0.746 | 0.605 | −0.141 | 0.885 | 0.747 | 1.000 |
| `gemma-4-31b-it` | 0.822 | 0.833 | **+0.011** | 0.737 | 0.737 | 0.000 | 0.897 | 0.884 | 0.979 |
| `gemma-4-e4b-it` | 0.711 | 0.553 | **−0.158** | 0.491 | 0.360 | −0.131 | 0.691 | 0.651 | 0.999 |
| *mean* | | | **−0.043** | | | −0.084 | | | |

### What the matrix says

**1. Czech costs something, but the cost is small and unevenly distributed.**
`pass^1` falls in 9 of 10 cells; the mean penalty is 6.3 points on airline and 4.3
on retail — a few percent relative, which is the expected shape for L2
Interaction, the mildest of the three L2 conditions. It is not uniform: the
per-model delta ranges from −0.158 to **+0.011**, so *one model's* delta is not an
estimate of "the Czech penalty". Treat individual cell deltas cautiously — at 150
simulations a `pass^1` near 0.8 carries a standard error around 0.03 (0.02 at
342), so most airline deltas here are one to two standard errors. **The
direction's consistency across five independent models is the load-bearing
evidence, not any single row.**

**2. The retail robustness collapse is real and general.** In retail, ρ₃ drops in
Czech for **all five agents** (mean −0.066) and Δ`pass^3` is roughly twice
Δ`pass^1`: the gap *widens with every additional trial*. Airline shows no such
pattern — ρ₃ moves either way and averages to nothing (−0.005). So in retail Czech
does not mainly make an agent worse on a given attempt; it makes it **less
consistent across attempts**, which only a `pass^k`-style metric exposes at all.
Retail's tasks are longer and more multi-step, which is the obvious candidate
explanation and one this experiment does not test.

**3. Language correctness does not explain any of it.** After the §6.1 correction
the Czech cells sit at 0.95–1.00, and the model with *perfect* Czech in retail
(`qwen3.6-27b`, 1.000) is the one whose robustness collapses hardest (ρ₃ 0.885 →
0.747), while the *worst* Czech speaker by this metric (`gemma-4-31b-it`, 0.979)
is the only agent that does not lose `pass^1` at all. Drift and task failure move
independently, which is the paper's finding too, and it means language correctness
cannot be used as a proxy for the L2 penalty. §6.1 sharpened this rather than
softening it: the correction pushed every cell to near-1.0 without moving a single
`pass^k`, so the Czech penalty is visibly **not** a language-drift effect.

**4. The cross-model ranking is not the point, and is partly a precision
artifact.** DeepSeek V4 Flash leads both domains, but it is quantized (fp4), API
served, and the only agent sharing an endpoint with the user simulator and judge
(§3.1, §3.3). Gemma 4 E4B is far smaller than the others and its scores read as a
capability floor, not a peer comparison. The within-model EN↔CS delta is what
survives these differences.

### Two operational notes on the runs themselves

The bf16 35B-A3B run replaces the FP8 run of the same model, which is discarded —
see [§3.1](#31-precision-is-a-confound--read-this-before-comparing-models) for why
those numbers were not kept even though English airline had finished.

Its Czech airline cell needed one resubmit: the cell exited `COMPLETED` at 147/150
having burned all three retries on three simulations whose *first* agent call timed
out (`duration 0.0s`, `messages 0` — the same 600 s client stall documented in
[§7](#7-known-issues-and-threats-to-validity)). `--auto-resume` drops exactly the
`infrastructure_error` simulations and keeps the rest, so the fix was one
`submit_chain.sh` call that reported `Resuming run from 147 runs. 3 runs remaining.`
and finished in 9 minutes. **A cell exiting `COMPLETED` is not proof it is whole —
check the `infra` column in `report.py`, not the Slurm exit code.**

### 9.1 Language quality separates the models far more sharply than `pass^k` does

**The annotation pass is complete: 820 of 820 Czech conversations judged, zero
error records** — all five agents, both domains, one trial per task (§6.2). It
flagged **4,174 spans (1,144 MAJOR, 3,030 MINOR)**; 64 conversations were clean.

Spans per conversation, ordered by MAJOR. `MAJ/1k` is the same count normalized
by agent characters written, which matters because the models are not equally
verbose (1.7k–3.4k chars per conversation):

**Airline** (50 conversations per agent):

| model | MAJOR/conv | MINOR/conv | clean | MAJ/1k chars |
|---|---|---|---|---|
| `deepseek-v4-flash-fp4` | **0.20** | 2.20 | 7/50 | **0.07** |
| `qwen3.6-27b` | 0.78 | 2.98 | 8/50 | 0.46 |
| `gemma-4-31b-it` | 1.82 | 2.38 | 4/50 | 1.05 |
| `qwen3.6-35b-a3b` | 2.18 | 7.30 | **0/50** | 0.66 |
| `gemma-4-e4b-it` | 3.38 | 5.90 | 1/50 | 1.28 |

**Retail** (114 conversations per agent):

| model | MAJOR/conv | MINOR/conv | clean | MAJ/1k chars |
|---|---|---|---|---|
| `deepseek-v4-flash-fp4` | **0.21** | 2.10 | 21/114 | **0.07** |
| `qwen3.6-35b-a3b` | 0.93 | 4.55 | 2/114 | 0.28 |
| `qwen3.6-27b` | 0.98 | 2.95 | 10/114 | 0.39 |
| `gemma-4-31b-it` | 1.04 | 2.30 | 9/114 | 0.50 |
| `gemma-4-e4b-it` | 3.21 | 5.58 | 2/114 | 1.02 |

**The spread is an order of magnitude wider than the task metric's.** On Czech
`pass^1` these five sit inside 0.567–0.807 (airline) and 0.553–0.865 (retail), a
factor of 1.4–1.6. Their MAJOR rates span a factor of **15–17** in both domains.
Two models a few points apart on the task can be nowhere near each other on
whether their Czech is well-formed, which is the entire reason this layer exists.

**Only the two endpoints are stable; the middle reorders.**
`deepseek-v4-flash-fp4` is best and `gemma-4-e4b-it` worst on every cut — both
domains, both normalizations. The other three change places depending on the
domain and on whether you divide by conversations or by characters, and the gaps
between them (0.93 / 0.98 / 1.04 in retail) are smaller than that instability.
**Read the endpoints as findings and the middle as a tie.**

`qwen3.6-35b-a3b` is the clearest case, and it corrects an earlier
airline-only reading of these numbers. On airline it looked like the sharpest
illustration of task-and-language coming apart: near the top on Czech `pass^1`
(0.780) and second-*worst* on MAJOR spans, with **not one of its 50 conversations
clean**. Retail does not reproduce that — 0.93 MAJOR/conv there puts it second
*best*. Much of the airline gap was verbosity: it writes the longest turns of any
agent (3.3k chars/conv against 1.7k for the 27B), and per 1k characters it sits
mid-table in airline and near the front in retail. What does survive every
normalization is its **clean rate: 0/50 and 2/114, the worst in both domains**.
Its errors are thinly spread across nearly every conversation rather than
concentrated in a few bad ones, which is a real and separate property — but "it
completes the task while writing Czech no native speaker would sign" was an
airline artifact and should not be repeated.

**None of this is recoverable from `language_correctness`.** It spans only
0.952–1.000 across these ten cells and does not rank the models the same way.
`gemma-4-e4b-it` reads 0.998/0.999 — essentially perfect Czech, by that metric —
while carrying the worst MAJOR rate in both domains by a factor of three. The
model with the *lowest* correctness score, `gemma-4-31b-it` (0.952 airline, 0.979
retail), is mid-table on MAJOR. fastText answers "is this Czech at all?"; it was
never going to answer "is this Czech any good?"
([§6.1](#61-the-mandated-transfer-sentence-is-excluded-from-language-correctness),
[§6.2](#62-language-correctness-is-not-language-quality--the-judge-that-fills-the-gap)).

**MAJOR and MINOR move independently**, which is what the split was for (§6.2).
`gemma-4-31b-it` has 1.3× the MAJOR rate of `qwen3.6-27b` in airline but a
*lower* MINOR rate; `qwen3.6-35b-a3b` has the worst MINOR rate in both domains
(7.30 and 4.55, against 5.90 and 5.58 for the next) while being nowhere near
worst on MAJOR in retail. Grammatical breakage and translated-sounding-but-legal
Czech are different failures with different causes.

**What this does not license.** Task success and language quality agree at the
endpoints here — the best agent on `pass^1` is also the best on MAJOR, and the
worst is worst — and disagree in the middle. With five models, one judge, and no
human annotation, neither the agreement nor the disagreement is estimable as a
correlation; the honest reading is that the two are separate measurements of
which only one is scored, and §6.2's caveat about the judge sharing a dialect
with the user simulator applies to every number above.

---

## 10. τ³ / `banking_knowledge` — assessed, not run, and the wiring is gone

> **Read this before using the rest of §10.** The code described in
> [§10.2](#102-what-was-added) is **no longer in this checkout**. `git -C
> SEATauBench diff --stat` shows six modified files (§4) and none of the τ³
> additions: there is no `einfra_embedder.py`, no `einfra_embeddings*` retrieval
> variant, no `TAU2_EMBEDDINGS_CACHE_DIR` handling, no retrieval block in
> `env.sh`, and `scripts/probe_retrieval.py` does not exist. `report.py` still
> parses cell names by fixed field index, so it still mis-reads a domain
> containing an underscore. What survives is the *assessment* — the measurements
> below were really taken, and the smoke runs are still on disk under
> `SEATauBench/data/simulations/smoke_banking/` — but **anything here that reads
> as an instruction is a rebuild, not a re-run.** §10.2 is kept as the
> specification of what would have to be written again.

[τ-Knowledge](https://taubench.com/blog/tau-knowledge.html) (τ³) adds one domain,
`banking_knowledge`: a fintech support setting where the agent must reason over a
698-document knowledge base (~195K tokens) instead of a fixed policy, discover
tools it is not given up front, and is graded on database state rather than on
what it says.

**It is already in this checkout.** The SEATauBench fork sits on a tau2 version
that ships the domain, its data and its retrieval machinery — nothing needs to be
vendored:

| | |
|---|---|
| tasks | **97** (`data/tau2/domains/banking_knowledge/tasks/`) |
| documents | **698** (18 MB) |
| reward basis | `DB` — database state, so the quantized judge matters far less here |
| missing dependency | `rank-bm25` (installed) |
| upstream support for `--lang-id` | **unchanged** — the L2 Interaction components are domain-agnostic |

Verified end to end before writing any of this: a DeepSeek V4 Flash agent, Kimi K3
user and judge, `--lang-id cs --lang-components user_system agent_system greeting`,
and the domain runs, scores a DB-based reward, and reports language correctness
(1.0 over 16 Czech agent turns) exactly like airline and retail do.

### 10.1 The retrieval variant is an experimental condition, not a setting

τ³ is retrieval-agnostic by design: `--retrieval-config` selects how the agent
reaches the knowledge base (`bm25`, `grep_only`, embeddings, `full_kb`,
`golden_retrieval`, agentic `terminal_use`, …). For a *Czech* run this stops being
a tuning knob, because **the knowledge base stays English**. `asset_mode:
original` translates nothing but the conversation, so an L2 Interaction cell asks
the agent to retrieve English documents from Czech-shaped queries.

Whether that is even possible is a property of the retriever. Measured with a
probe script (`scripts/probe_retrieval.py`, **since lost** — see the note opening
§10) — 40 KB documents, each
document's own title as the English query, the same title translated to Czech by
Kimi K3, scoring whether the source document comes back:

| variant | EN recall@10 | CS recall@10 | EN MRR | CS MRR | distinct top-1 (EN / CS) |
|---|---|---|---|---|---|
| `bm25` (upstream default) | 0.600 | **0.100** | 0.284 | 0.047 | 33 / **22** |
| `einfra_embeddings` (`qwen3-embedding-4b`) | 0.700 | **0.600** | 0.400 | 0.326 | 36 / 37 |

BM25 is lexical, so a Czech query matches almost no English term and the ranking
collapses toward a constant — two *unrelated* Czech queries return the same
documents, which is why the distinct-top-1 count drops to 22 of 40. Under `bm25`
the Czech agent is working with `KB_search` effectively disabled, and an EN→CS
gap would be measuring the retriever rather than the model. The first Czech smoke
run shows the agent noticing this and hedging — it issued every query twice, once
in Czech and once in English:

```
KB_search {"query": "osobní kreditní karta cashback roční poplatek"}
KB_search {"query": "credit card cashback annual fee personal"}
```

A multilingual dense retriever is therefore the default here
(`CZTAU_RETRIEVAL=einfra_embeddings_grep`). `bm25*` remains available for
reproducing upstream's out-of-the-box condition, but should not be used for a
Czech cell.

`grep_only` is excluded from the table rather than reported as a failure: `grep`
takes a regex, not a natural-language question, so a title-as-query probe scores
it zero by construction. It carries the same English-lexical problem as BM25 —
the agent must produce English patterns to match English documents.

### 10.2 What was added — and is no longer present

**None of the following is in the checkout any more** (see the note opening §10).
It was five files modified in `SEATauBench/` beyond those in §4, plus three new
ones (the embedder and two prompt templates), all additive — nothing that airline,
retail or telecom touches changed behaviour, which is also why its disappearance
cost none of the §9 results. Read the rest of this subsection as the build
specification for a second attempt, including the traps it already paid for:

**`src/tau2/knowledge/embedders/einfra_embedder.py`** (new) — upstream ships an
OpenAI embedder (`text-embedding-3-large`) and an OpenRouter one
(`qwen3-embedding-8b`); we have neither key. The e-infra gateway does serve
embedding models (`qwen3-embedding-4b`, `multilingual-e5-large-instruct`,
`nomic-embed-text-v2-moe`) on the key already used for the user simulator and the
judge. The class adds endpoint/model from environment, internal batching (the
indexer otherwise embeds all 698 documents in a single request), and the
Qwen-style instruction prefix on queries only.

**`embedding_indexer.py` / `embedding_encoder.py`** — register the embedder, and
generalise the "do not prefix documents" rule from a hardcoded `openrouter` check
to a list. Getting that wrong does not raise; it silently costs recall.

**`banking_knowledge/retrieval.py`** — two new variants, `einfra_embeddings` and
`einfra_embeddings_grep`, mirroring the `qwen_embeddings*` pair. The embedding
model is resolved into the variant *at import*, not left `None` for the embedder
to fill in later: the document-embedding cache is keyed on the embedder params, so
a `None` there would make two different embedding models share one cache entry and
serve each other's vectors.

**`embeddings_cache.py`** — the cache directory was CWD-relative
(`data/.embeddings_cache`), so a job started from anywhere else re-embeds all 698
documents against the metered key. Now honours `TAU2_EMBEDDINGS_CACHE_DIR`, which
`env.sh` pins to an absolute path. Same reasoning as the judge-model change in §4.

On our side: `env.sh` gained the retrieval block, `run_cell.sh` accepts the domain
(this one **survives** — it takes any domain as `$2`), and `report.py` /
`viewer.py` learned that a domain name can contain an underscore — both parsed the
cell name as `<scenario>_<domain>_<lang>` with a fixed field index, which silently
mis-reads `banking_knowledge`. Only `viewer.py` still has that fix, in `parse_cell`;
`report.py` is back to `cell.split("_")` with a fixed index and would mis-read the
domain again.

The retrieval variant goes into the **run tag**, not the cell name
(`deepseek-v4-flash-fp4-think-on-einfraembeddingsgrep-qwen3-embedding-4b`), for
the reason precision and thinking mode are already there: it changes what the
number means, and the cell-name grammar is what the reporting parses.

### 10.3 Cost, and what a run would buy

97 tasks × 3 trials = **291 simulations per cell**, and banking conversations are
long — the smoke runs took 124–171 s each with 7–18 tool calls, against airline's
much shorter exchanges. On the metered key that is roughly 4 hours per cell at 3
API slots, so an EN+CS pair for one model is most of a day; a locally-served agent
only pays for the user and judge calls and should be faster.

Two things this would **not** buy:

* **Comparability with the τ³ leaderboard.** Every 2026-08 frontier submission
  reports `banking_knowledge` at 4 trials with its own retrieval configuration and
  a `gpt-5.2` user simulator; top score is 0.55. Ours would be 3 trials, a
  different retriever, and a Kimi K3 simulator. `fetch_reference.py` deliberately
  skips banking for this reason, and that stays true.
* **A clean EN→CS delta**, unless §10.1 is respected. With a multilingual
  retriever the delta is interpretable but now contains a *cross-lingual
  retrieval* component that airline and retail do not have. That is arguably the
  interesting part — it is the first condition here where Czech changes what the
  agent can *find*, not just how it talks — but it is a different quantity from
  the existing airline/retail deltas and should not be put on the same axis
  without saying so.

### 10.4 How it would be run

**These commands do not work as written today** — `CZTAU_RETRIEVAL`,
`einfra_embeddings_grep` and `probe_retrieval.py` all belong to the §10.2 code
that is no longer here. They record the intended shape of a run, and the second
one (`bm25`, English only) is the only one that would need nothing rebuilt beyond
`report.py`'s cell parsing:

```bash
# defaults: einfra_embeddings_grep, qwen3-embedding-4b, 3 trials
scripts/submit_chain.sh -p deepseek -s 2 \
    english:banking_knowledge l2_interaction:banking_knowledge:cs

# upstream's out-of-the-box retriever, English only (see §10.1)
CZTAU_RETRIEVAL=bm25 scripts/submit_chain.sh -p deepseek -s 2 \
    english:banking_knowledge

# re-check the retrieval floor before trusting any Czech banking number
python scripts/probe_retrieval.py
```

The document embeddings are computed once (~20 s, 14 MB) and cached at
`SEATauBench/data/.embeddings_cache`. Two banking cells starting at the same
instant would race to write it; cells within a chain run sequentially, so this
only matters if two chains are launched together — warm the cache first if so.

**Not attempted:** `full_kb` inlines the entire knowledge base into the system
prompt — 866K characters, roughly 217K tokens, against local vLLM servers started
at `--max-model-len 65536`. `terminal_use*` needs `sandbox-runtime` plus
`bubblewrap`/`socat`, which is a cluster-permissions question, not a code one.
`golden_retrieval` (required documents inlined, no retrieval at all) *is* offline
and cheap, and is the obvious control if we want the reasoning half of τ³ without
the retrieval confound.
