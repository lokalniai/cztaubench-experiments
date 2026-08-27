#!/bin/bash
# Queue a set of CzTauBench cells as one dependency chain.
#
# Cells run STRICTLY SEQUENTIALLY within a chain. Cells sharing one vLLM server
# would only split the same throughput if parallelised, while adding contention
# -- the total finishes no sooner and the failure modes get worse.
#
# That holds while ONE cell already saturates the server. It stops holding on a
# multi-GPU server, where a single cell's --max-concurrency of 12 leaves the
# batch half empty: there two chains on the same server do finish sooner,
# because the second one is filling idle capacity rather than taking a share of
# busy capacity. Split them by LANGUAGE (one chain english:*, one chain
# l2_interaction:*:cs) so both halves of the EN/CS comparison land together --
# a partial result is then a partial comparison rather than one language only.
# The API budget is unchanged either way: -s 2 on both, never a third chain.
#
# Two chains against two DIFFERENT vLLM servers can overlap, but they still
# share one rate-limited API key for the user simulator and the judge. Give each
# job half the slots in that case: CZTAU_API_SLOTS=2 (see env.sh).
#
# afterany (not afterok) so one bad cell does not strand the whole queue; a
# failed cell can be resubmitted on its own and will --auto-resume.
#
# Usage:
#   submit_chain.sh [-a after_jobid] [-p profile] [-s api_slots] cell [cell ...]
#
#   cell := scenario:domain[:lang]     e.g. english:retail  l2_interaction:retail:cs
#
# Example -- two chains overlapping on two vLLM servers, splitting the key's
# 4 parallel slots evenly so their combined in-flight calls never exceed it:
#   scripts/submit_chain.sh -p local   -s 2 english:retail l2_interaction:retail:cs
#   scripts/submit_chain.sh -p local35 -s 2 english:airline l2_interaction:airline:cs

set -euo pipefail

AFTER=""
PROFILE="local"
SLOTS=""
while getopts "a:p:s:" opt; do
    case "${opt}" in
        a) AFTER="${OPTARG}" ;;
        p) PROFILE="${OPTARG}" ;;
        s) SLOTS="${OPTARG}" ;;
        *) echo "usage: submit_chain.sh [-a after_jobid] [-p profile] [-s slots] cell..." >&2
           exit 2 ;;
    esac
done
shift $((OPTIND - 1))

[ "$#" -gt 0 ] || { echo "no cells given" >&2; exit 2; }

ROOT="/lnet/work/people/kasner/projects/cztaubench"
cd "${ROOT}"
mkdir -p logs

# Short label for the job name, so squeue stays readable across models.
case "${PROFILE}" in
    local)    pfx="27b"      ;;
    local35)  pfx="35b"      ;;
    *)        pfx="${PROFILE}" ;;
esac

dep="${AFTER}"
for spec in "$@"; do
    IFS=':' read -r scenario domain lang <<<"${spec}"
    [ -n "${scenario}" ] && [ -n "${domain}" ] || {
        echo "bad cell spec: ${spec}" >&2; exit 2; }

    args=("${scenario}" "${domain}")
    tag="${scenario%%_*}"
    if [ -n "${lang:-}" ]; then
        args+=("${lang}")
        tag="${lang}"
    fi

    export_spec="ALL,CZTAU_PROFILE=${PROFILE}"
    [ -n "${SLOTS}" ] && export_spec="${export_spec},CZTAU_API_SLOTS=${SLOTS}"

    sbatch_args=(--parsable
                 --job-name "cztau-${pfx}-${tag}-${domain}"
                 --export="${export_spec}")
    [ -n "${dep}" ] && sbatch_args+=(--dependency=afterany:"${dep}")

    jid=$(sbatch "${sbatch_args[@]}" scripts/run_cell.sh "${args[@]}")
    printf '  %-14s %-8s %-3s -> %s%s\n' \
        "${scenario}" "${domain}" "${lang:--}" "${jid}" \
        "${dep:+  (after ${dep})}"
    dep="${jid}"
done

echo
echo "queue tail: ${dep}"
echo "watch:  squeue --me"
echo "report: python ${ROOT}/scripts/report.py"
