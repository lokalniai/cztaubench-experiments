#!/bin/bash
#SBATCH -J cztau-ann
#SBATCH -p cpu-ms,cpu-troja
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH -o /lnet/work/people/kasner/projects/cztaubench/logs/annotate_%j.out
#SBATCH -e /lnet/work/people/kasner/projects/cztaubench/logs/annotate_%j.out
#
# Czech language-quality annotation over finished runs (README §6.2).
#
# Like run_cell.sh this is a pure API client — every token is generated on the
# e-infra endpoint — so it belongs on a CPU partition. It is slow for a reason
# that is not our CPU: Kimi K3 spends ~3200 reasoning tokens per conversation to
# emit a ~150-token answer, which is ~230 s per request at the endpoint's rate.
#
# Domains run SEQUENTIALLY and inside one process, because the binding
# constraint is the API key's global cap of 4 parallel requests. Two concurrent
# batches would not go faster; they would just spend the difference on 429s.
#
# Workers default to 4 = the whole cap. Two consequences worth knowing:
#
#  * Nothing else may touch KIMI_API_KEY while this runs -- not a benchmark cell
#    (§3.3), not a probe script, not a stray curl. Anything that does is not
#    "sharing"; it takes a slot this job then spends its time being 429'd out of.
#  * NEVER kill -9 a client with requests in flight. The proxy's parallel-request
#    counter is not released when the client vanishes, so the slots stay consumed
#    until the window resets (the "Limit resets at" timestamp in the 429 body).
#    Killing three in-flight workers once cost the whole cap for ~30 minutes and
#    looked exactly like a hung job. Use scancel / SIGTERM and let it drain.
#
# Every invocation passes --resume, so a requeued or timed-out job picks up
# where it left off instead of re-paying for work already banked. The output
# file is rewritten atomically after each completed request, so it is safe to
# read (and safe for the viewer to serve) while this is running.
#
# Usage: sbatch scripts/run_annotate.sh [domain ...]      (default: airline retail)

set -euo pipefail

ROOT=/lnet/work/people/kasner/projects/cztaubench
cd "${ROOT}"

# shellcheck disable=SC1091
source /lnet/work/people/kasner/virtualenv/cztaubench/bin/activate

DOMAINS=("$@")
if [ ${#DOMAINS[@]} -eq 0 ]; then
    DOMAINS=(airline retail)
fi

echo "=== annotation queue: ${DOMAINS[*]} ==="
echo "started $(date -Is) on $(hostname)"

for domain in "${DOMAINS[@]}"; do
    echo
    echo "=== ${domain} ($(date -Is)) ==="
    # --limit 0 = every task in the cell; --trial 0 = one trial per task, since
    # three trials of one task are three samples of near-identical text.
    python -u scripts/annotate_language.py \
        --domain "${domain}" \
        --limit 0 \
        --trial 0 \
        --resume \
        --workers "${CZTAU_ANN_WORKERS:-4}"
done

echo
echo "=== done $(date -Is) ==="
