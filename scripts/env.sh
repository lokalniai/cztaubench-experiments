#!/bin/bash
# Shared environment for CzTauBench runs. Source this, don't execute it.
#
# Three roles, deliberately decoupled:
#   agent  -- the model under test; varies per experiment (CZTAU_PROFILE)
#   user   -- the simulated user; FIXED across all agents, or scores are not
#             comparable between them (the paper pins Qwen3-235B-A22B-Inst)
#   judge  -- NL-assertion grader; FIXED for the same reason
#
# Keeping the user simulator on the local vLLM also matters practically: the
# e-infra key allows only 4 parallel requests, so putting the user sim there
# would halve the agent's share of them.

export CZTAU_ROOT="/lnet/work/people/kasner/projects/cztaubench"
export CZTAU_REPO="${CZTAU_ROOT}/SEATauBench"

# Remember whether the caller pinned concurrency explicitly. The profile blocks
# below each want to supply a default, and without this an earlier block's
# default would look like a user-supplied value to a later one.
_CONC_EXPLICIT="${CZTAU_CONCURRENCY:-}"

# Load API keys (KIMI_API_BASE / KIMI_API_KEY) from one place.
if [ -f "${CZTAU_REPO}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "${CZTAU_REPO}/.env"
    set +a
fi

# ── Local vLLM ────────────────────────────────────────────────────────────
# Defaults for the `local` profile; `local35` overrides them below. Update when
# a server job moves to a different node.
#
# NB: as of 2026-08-08 this host:port serves the `gemma` profile's model instead
# -- the 27B server is gone and all `local` cells are complete. A `local` cell
# submitted now fails the run_cell.sh preflight (it greps /v1/models for the
# agent's own model id), which is the intended loud failure, not a silent run
# against the wrong model. Repoint this line before running `local` again.
export CZTAU_VLLM_BASE="${CZTAU_VLLM_BASE:-http://dll-4gpu4:8000/v1}"
export CZTAU_VLLM_MODEL="${CZTAU_VLLM_MODEL:-Qwen/Qwen3.6-27B}"

# ── Thinking mode ─────────────────────────────────────────────────────────
# Applies to the agent only. Qwen exposes it via chat_template_kwargs;
# DeepSeek exposes it as a separate model id (…-thinking).
export CZTAU_THINKING="${CZTAU_THINKING:-on}"
if [ "${CZTAU_THINKING}" = "off" ]; then
    _QWEN_THINK=',"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}'
else
    _QWEN_THINK=''
fi

# Gemma 4 is the mirror image of Qwen: thinking defaults to OFF, so `on` is the
# case that must be requested explicitly. Only `enable_thinking` has any effect
# -- `thinking`, `reasoning` and `include_thoughts` are all accepted and then
# silently ignored, yielding a plain answer and an empty reasoning field, which
# is a DIFFERENT experimental condition than the Qwen models run under.
#
# Sent per request rather than via the server's --default-chat-template-kwargs
# on purpose: a server flag would make "thinking was on" a property of how that
# vLLM job happened to be launched -- invisible in this repo and unrecoverable
# from the results afterwards.
if [ "${CZTAU_THINKING}" = "off" ]; then
    _GEMMA_THINK=''
else
    _GEMMA_THINK=',"extra_body":{"chat_template_kwargs":{"enable_thinking":true}}'
fi

# ── Agent profile ─────────────────────────────────────────────────────────
# Set by a profile whose agent lives on the metered e-infra key rather than on
# a local vLLM; the user block below sizes concurrency differently in that case.
_AGENT_ON_METERED_API=""
export CZTAU_PROFILE="${CZTAU_PROFILE:-local}"
case "${CZTAU_PROFILE}" in
    local)
        export CZTAU_AGENT_LLM="openai/${CZTAU_VLLM_MODEL}"
        export CZTAU_AGENT_API_BASE="${CZTAU_VLLM_BASE}"
        export CZTAU_AGENT_API_KEY="dummy"
        export CZTAU_AGENT_ARGS="{\"temperature\":0.0,\"api_base\":\"${CZTAU_VLLM_BASE}\",\"api_key\":\"dummy\"${_QWEN_THINK}}"
        # Local server: bounded by GPU, not by a quota.
        export CZTAU_CONCURRENCY="${CZTAU_CONCURRENCY:-16}"
        export CZTAU_RUN_TAG="qwen3.6-27b-think-${CZTAU_THINKING}"
        ;;
    local35)
        # Second local server: the MoE. Only ~3B params are active per token, so
        # it should run faster than the dense 27B despite the larger total.
        #
        # bf16, NOT the FP8 checkpoint this profile originally pointed at
        # (2026-08-20). The switch was forced by GPU availability -- the single
        # 95 GB card the FP8 weights fit on stayed occupied -- but it also
        # removes the precision confound §3.1 flags: every other agent here is
        # served at its released precision, and the FP8 35B was the one row
        # whose gap against the 27B mixed a model difference with a
        # quantisation difference. The FP8 run (english_airline complete,
        # l2_interaction_airline_cs 149/150) was discarded rather than kept
        # alongside: two rows for one model, differing only in precision and
        # with one of them partial, is a comparison nobody would want to make.
        # It is parked in discarded/ if it is ever wanted back.
        #
        # 72 GB of bf16 weights does not fit one card, so this needs a
        # tensor-parallel server: 4 GPUs, launched as documented in
        # vllm_server.sh. Repoint the host below when that job moves.
        export CZTAU_VLLM_BASE="http://tdll-8gpu2:8000/v1"
        export CZTAU_VLLM_MODEL="Qwen/Qwen3.6-35B-A3B"
        export CZTAU_AGENT_LLM="openai/${CZTAU_VLLM_MODEL}"
        export CZTAU_AGENT_API_BASE="${CZTAU_VLLM_BASE}"
        export CZTAU_AGENT_API_KEY="dummy"
        export CZTAU_AGENT_ARGS="{\"temperature\":0.0,\"api_base\":\"${CZTAU_VLLM_BASE}\",\"api_key\":\"dummy\"${_QWEN_THINK}}"
        export CZTAU_CONCURRENCY="${CZTAU_CONCURRENCY:-16}"
        # No precision suffix in the tag, matching qwen3.6-27b-think-on: the
        # absence marks released weights, exactly as -fp8/-fp4 marks quantised
        # ones. Deliberately NOT the old -fp8- tag, so nothing resumes into or
        # reports over the discarded run.
        export CZTAU_RUN_TAG="qwen3.6-35b-a3b-think-${CZTAU_THINKING}"
        ;;
    gemma)
        # Third local server. Thinking is opt-in here (see _GEMMA_THINK above),
        # so that the model is compared to the Qwen agents under the same
        # condition rather than an accidentally non-thinking one.
        # Moved dll-3gpu1 -> dll-4gpu4 when the server was restarted (2026-08-08).
        # Re-verified against the new job before use, since the launch line is not
        # in this repo: tool calls returned, zero-arg tools give "{}", a
        # system -> tool_call -> tool round trip is accepted and answered in Czech,
        # and enable_thinking populates `reasoning` (1125 chars vs 0 without) while
        # leaving `content` clean -- with thinking off the probe question was also
        # answered WRONG, so a silently non-thinking server is not a cosmetic
        # difference here.
        export CZTAU_VLLM_BASE="http://dll-4gpu4:8000/v1"
        export CZTAU_VLLM_MODEL="google/gemma-4-E4B-it"
        export CZTAU_AGENT_LLM="openai/${CZTAU_VLLM_MODEL}"
        export CZTAU_AGENT_API_BASE="${CZTAU_VLLM_BASE}"
        export CZTAU_AGENT_API_KEY="dummy"
        export CZTAU_AGENT_ARGS="{\"temperature\":0.0,\"api_base\":\"${CZTAU_VLLM_BASE}\",\"api_key\":\"dummy\"${_GEMMA_THINK}}"
        export CZTAU_CONCURRENCY="${CZTAU_CONCURRENCY:-16}"
        export CZTAU_RUN_TAG="gemma-4-e4b-it-think-${CZTAU_THINKING}"
        ;;
    gemma31)
        # The dense 31B sibling of the `gemma` profile above -- same family, same
        # opt-in thinking, so the two are directly comparable and both sit under
        # the same condition as the Qwen agents.
        #
        # Port 8001, not the 8000 every other profile uses: this server runs in a
        # hand-started interactive job rather than via vllm_server.sh. Verified
        # before first use that it parses tool calls and honours enable_thinking
        # (343 completion tokens vs 14 with it off, and only the thinking answer
        # was correct) -- a server started without --enable-auto-tool-choice
        # would score a flat zero on every task while looking merely bad at the
        # benchmark, so that check is not optional when the launch line is not
        # in this repo.
        export CZTAU_VLLM_BASE="http://dll-4gpu3:8001/v1"
        export CZTAU_VLLM_MODEL="google/gemma-4-31B-it"
        export CZTAU_AGENT_LLM="openai/${CZTAU_VLLM_MODEL}"
        export CZTAU_AGENT_API_BASE="${CZTAU_VLLM_BASE}"
        export CZTAU_AGENT_API_KEY="dummy"
        export CZTAU_AGENT_ARGS="{\"temperature\":0.0,\"api_base\":\"${CZTAU_VLLM_BASE}\",\"api_key\":\"dummy\"${_GEMMA_THINK}}"
        export CZTAU_CONCURRENCY="${CZTAU_CONCURRENCY:-16}"
        export CZTAU_RUN_TAG="gemma-4-31b-it-think-${CZTAU_THINKING}"
        ;;
    deepseek)
        # The only API-served agent. Thinking is a separate model id rather than
        # a request kwarg: the proxy exposes two aliases over one backend, and
        # the -thinking one carries chat_template_kwargs={"thinking":true} in
        # its server-side config. That is the arrangement §3.2 argues against
        # for our own servers, but here it is at least self-documenting -- the
        # condition is encoded in the model id, which lands in the run tag.
        if [ "${CZTAU_THINKING}" = "off" ]; then
            _DS_MODEL="deepseek-v4-flash"
        else
            _DS_MODEL="deepseek-v4-flash-thinking"
        fi
        export CZTAU_AGENT_LLM="openai/${_DS_MODEL}"
        export CZTAU_AGENT_API_BASE="${KIMI_API_BASE}"
        export CZTAU_AGENT_API_KEY="${KIMI_API_KEY}"
        export CZTAU_AGENT_ARGS="{\"temperature\":0.0,\"api_base\":\"${KIMI_API_BASE}\",\"api_key\":\"${KIMI_API_KEY}\",\"num_retries\":12,\"retry_strategy\":\"exponential_backoff_retry\"}"
        # This profile is the one case where the AGENT also sits on the metered
        # key, so all three roles contend for the same 4 parallel slots. The
        # per-host semaphore below already caps in-flight calls correctly (it
        # matches on api_base, so agent calls are now counted too); what changes
        # is that --max-concurrency no longer buys throughput, because there is
        # no unmetered GPU left to overlap with. See the user block below.
        _AGENT_ON_METERED_API=1
        # fp4 in the tag for the same reason FP8 is in the 35B one: this is a
        # quantised deployment, not the released weights.
        export CZTAU_RUN_TAG="deepseek-v4-flash-fp4-think-${CZTAU_THINKING}"
        ;;
    # No kimi agent profile on purpose: Kimi K3 is reserved for the user
    # simulator, the judge, and translation. Evaluating it as an agent would
    # make it grade and converse with itself.
    *)
        echo "unknown CZTAU_PROFILE: ${CZTAU_PROFILE} (local|local35|gemma|gemma31|deepseek)" >&2
        return 1 2>/dev/null || exit 1 ;;
esac

# ── Simulated user ────────────────────────────────────────────────────────
# FIXED across every agent, or scores are not comparable between them.
#
# Default is Kimi K3 rather than the local model, for two reasons:
#  * neutrality -- it is none of the models under test, so no agent gets the
#    advantage of conversing with itself;
#  * fidelity -- on the Czech roleplay probe it was the only candidate that
#    honoured progressive disclosure instead of volunteering the reservation
#    id unprompted, which is exactly the simulated-user noise the paper flags.
#
# Set CZTAU_USER_PROFILE=local to trade that for speed: the local server has no
# quota, whereas Kimi shares the key's 4 parallel slots with the judge.
export CZTAU_USER_PROFILE="${CZTAU_USER_PROFILE:-kimi}"
if [ "${CZTAU_USER_PROFILE}" = "local" ]; then
    export CZTAU_USER_LLM="openai/${CZTAU_VLLM_MODEL}"
    export CZTAU_USER_ARGS="{\"temperature\":0.0,\"api_base\":\"${CZTAU_VLLM_BASE}\",\"api_key\":\"dummy\",\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
else
    export CZTAU_USER_LLM="openai/kimi-k3"
    # num_retries well above the tau2 default of 3: under contention for the
    # key's 4 slots, 3 retries get exhausted and the simulation dies as an
    # infrastructure_error rather than merely being slowed down. LiteLLM
    # already selects exponential backoff for RateLimitError; naming the
    # strategy makes that explicit rather than implicit.
    export CZTAU_USER_ARGS="{\"temperature\":0.0,\"api_base\":\"${KIMI_API_BASE}\",\"api_key\":\"${KIMI_API_KEY}\",\"num_retries\":12,\"retry_strategy\":\"exponential_backoff_retry\"}"
    # Throttle the metered host directly rather than throttling everything.
    # tau2's --max-concurrency counts whole simulations, and each alternates
    # between the local vLLM and this API, so using it to respect the API quota
    # would idle the GPU too -- measured at ~23s per agent call vs ~3s per user
    # call, i.e. only ~10% of wall time actually touches the API. The limiter in
    # llm_utils.py caps in-flight calls to this host and leaves vLLM alone.
    export TAU2_RATE_LIMITED_HOST="e-infra.cz"
    # The key allows 4 in flight; 3 leaves one spare. The semaphore is
    # per-process, so if two cells run CONCURRENTLY (e.g. one per vLLM server)
    # each job must be given half: CZTAU_API_SLOTS=2 on both.
    export TAU2_RATE_LIMITED_MAX_PARALLEL="${CZTAU_API_SLOTS:-3}"
    # With the API bounded independently, simulation concurrency is free to sit
    # where the GPU saturates. Measured on 4xA40: 12 gives ~3x the throughput of
    # 2 with zero rate-limit errors, at the cost of higher per-sim latency.
    #
    # That reasoning assumes the agent is on the unmetered GPU, so raising
    # concurrency overlaps GPU work with API waits. When the agent is on the
    # metered key too, every call in a simulation queues on the same semaphore
    # and extra concurrency only lengthens the queue. Keep a small multiple of
    # the slot count -- enough to keep every slot busy across the non-LLM gaps,
    # not so much that a simulation's calls are interleaved with a dozen others.
    if [ -n "${_AGENT_ON_METERED_API}" ]; then
        export CZTAU_CONCURRENCY="${_CONC_EXPLICIT:-$(( ${TAU2_RATE_LIMITED_MAX_PARALLEL} * 2 ))}"
    else
        export CZTAU_CONCURRENCY="${_CONC_EXPLICIT:-12}"
    fi
fi

# ── NL-assertion judge: Kimi K3 (paper used GPT-4.1) ──────────────────────
export TAU2_LLM_NL_ASSERTIONS="${CZTAU_JUDGE_MODEL:-openai/kimi-k3}"
export TAU2_LLM_NL_ASSERTIONS_API_BASE="${KIMI_API_BASE}"
export TAU2_LLM_NL_ASSERTIONS_API_KEY="${KIMI_API_KEY}"
export TAU2_LLM_NL_ASSERTIONS_NUM_RETRIES=12

# Environment-interface LLM. Unused by airline/retail/telecom as configured, but
# it defaults to azure/gpt-4.1-mini upstream, which would fail with no Azure
# credentials. Point it at the local server so nothing can silently reach out.
# Follows the agent's own endpoint, not CZTAU_VLLM_BASE: for the deepseek
# profile those differ, and pairing an API model id with the local server's base
# would turn an unused default into a guaranteed failure if it ever were used.
export TAU2_LLM_ENV_INTERFACE="${CZTAU_AGENT_LLM}"
export TAU2_LLM_ENV_INTERFACE_API_BASE="${CZTAU_AGENT_API_BASE}"
export TAU2_LLM_ENV_INTERFACE_API_KEY="${CZTAU_AGENT_API_KEY}"

# ── Persistent LLM cache: OFF ─────────────────────────────────────────────
# LiteLLM's disk cache is SQLite via diskcache, and the project lives on Lustre,
# where SQLite's WAL locking is unreliable: it survives a single-threaded run but
# dies with "sqlite3.OperationalError: locking protocol" once several worker
# threads write concurrently. LiteLLM hardcodes dc.Cache(dir) with no way to pass
# a network-safe journal mode, so there is no clean fix here.
#
# Reuse across runs is instead handled by --auto-resume, which skips completed
# simulations and retries only infrastructure failures. That covers the case
# that actually matters; the cache only ever saved re-work *within* a partially
# completed simulation.
#
# To re-enable anyway, point the cache at node-local disk (survives only within
# one job, since jobs land on different nodes):
#   export TAU2_LLM_CACHE_ENABLED=true TAU2_LLM_CACHE_TYPE=disk
#   export TAU2_LLM_DISK_CACHE_DIR="${TMPDIR:-/tmp}/cztau_llm_cache"
export TAU2_LLM_CACHE_ENABLED="${TAU2_LLM_CACHE_ENABLED:-false}"

# ── Misc ──────────────────────────────────────────────────────────────────
export TAU2_FASTTEXT_LID_MODEL_PATH="${CZTAU_REPO}/data/models/lid.176.bin"
export CZTAU_TRIALS="${CZTAU_TRIALS:-3}"

# shellcheck disable=SC1091
source "${WD_VIRTUALENV_DIR}/cztaubench/bin/activate"
