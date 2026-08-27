#!/bin/bash
#SBATCH -J cztau
#SBATCH -p cpu-ms,cpu-troja
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH -o /lnet/work/people/kasner/projects/cztaubench/logs/cell_%j.out
#SBATCH -e /lnet/work/people/kasner/projects/cztaubench/logs/cell_%j.out
#
# Run one experiment cell = (scenario, domain, language).
#
# These jobs are pure API clients — all compute happens on the vLLM server — so
# they belong on a CPU partition, not a GPU one.
#
# Usage: run_cell.sh <scenario> <domain> [lang]
#   scenario: english | l2_interaction
#   domain:   airline | retail | telecom
#   lang:     required for l2_interaction (e.g. cs)

set -euo pipefail

SCENARIO="${1:?usage: run_cell.sh <scenario> <domain> [lang]}"
DOMAIN="${2:?usage: run_cell.sh <scenario> <domain> [lang]}"
LANG_ID="${3:-}"

# shellcheck disable=SC1091
source /lnet/work/people/kasner/projects/cztaubench/scripts/env.sh

# telecom ships 2285 generated tasks; the paper (and tau2-bench proper) use the
# 114-task "base" split.
SPLIT_ARGS=()
if [ "${DOMAIN}" = "telecom" ]; then
    SPLIT_ARGS=(--task-split-name base)
fi

LANG_ARGS=()
TAG="${SCENARIO}_${DOMAIN}"
case "${SCENARIO}" in
    english)
        ;;
    l2_interaction)
        : "${LANG_ID:?l2_interaction requires a language code}"
        # The scenario flag only records metadata + scores language correctness;
        # the runtime components must be listed explicitly.
        LANG_ARGS=(--lang-id "${LANG_ID}"
                   --lang-components user_system agent_system greeting)
        TAG="${SCENARIO}_${DOMAIN}_${LANG_ID}"
        ;;
    *)
        echo "unknown scenario: ${SCENARIO}" >&2; exit 2 ;;
esac

# tau2 resolves --save-to under data/simulations/, and `tau2 view`, auto-resume
# and the checkpointing all assume that layout, so keep the name relative.
OUT="${CZTAU_RUN_TAG}/${TAG}"
mkdir -p "${CZTAU_ROOT}/logs"

# Preflight: a queued cell can start hours later, by which time the vLLM job it
# targets may have hit its time limit. Fail loudly here instead of grinding
# through per-call retries and writing a results file full of infra errors --
# --auto-resume then makes a resubmit pick up exactly where this left off.
#
# Checks the AGENT's own endpoint rather than CZTAU_VLLM_BASE: those coincide
# for the local profiles but not for deepseek, where the old form would have
# gated an API run on an unrelated local server still being up. The key is
# passed as a header and never echoed.
AGENT_MODEL_ID="${CZTAU_AGENT_LLM#openai/}"
if ! curl -sf --max-time 20 \
        -H "Authorization: Bearer ${CZTAU_AGENT_API_KEY}" \
        "${CZTAU_AGENT_API_BASE}/models" \
        | grep -qF "${AGENT_MODEL_ID}"; then
    echo "PREFLIGHT FAILED: ${AGENT_MODEL_ID} not served at ${CZTAU_AGENT_API_BASE}" >&2
    echo "  (restart the vLLM job, then resubmit this cell; it will auto-resume)" >&2
    exit 3
fi

echo "=== ${TAG} [${CZTAU_RUN_TAG}] ==="
echo "agent=${CZTAU_AGENT_LLM} thinking=${CZTAU_THINKING} trials=${CZTAU_TRIALS}"
echo "user =${CZTAU_USER_LLM}"
echo "judge=${TAU2_LLM_NL_ASSERTIONS}  concurrency=${CZTAU_CONCURRENCY}"
echo "out=${OUT}"
date

# --auto-resume makes the cell restartable: a requeued or timed-out job picks up
# the simulations already written instead of redoing them.
tau2 run \
    --domain "${DOMAIN}" \
    "${SPLIT_ARGS[@]}" \
    "${LANG_ARGS[@]}" \
    --seatau-scenario "${SCENARIO}" \
    --agent-llm "${CZTAU_AGENT_LLM}" \
    --agent-llm-args "${CZTAU_AGENT_ARGS}" \
    --user-llm "${CZTAU_USER_LLM}" \
    --user-llm-args "${CZTAU_USER_ARGS}" \
    --num-trials "${CZTAU_TRIALS}" \
    --max-concurrency "${CZTAU_CONCURRENCY}" \
    --seed 300 \
    --auto-resume \
    --save-to "${OUT}"

date
echo "=== done ${TAG} ==="
