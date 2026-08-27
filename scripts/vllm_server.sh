#!/bin/bash
#SBATCH -J cztau-vllm
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --constraint="gpuram95G|gpuram48G"
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# 7 days, not 3. A server outliving its cells is free -- it is cancelled when
# they finish -- whereas a server that expires mid-cell costs a restart, a
# resubmit, and however long it takes someone to notice. gpu-troja and gpu-ms
# both have MaxTime=UNLIMITED and DefaultTime=7-00:00:00, so this asks for
# nothing unusual. Note that the limit can only be LOWERED after submission
# (scontrol update TimeLimit is admin-only upwards), so it has to be right here.
#SBATCH --time=7-00:00:00
#SBATCH -o /lnet/work/people/kasner/projects/cztaubench/logs/vllm_%j.out
#SBATCH -e /lnet/work/people/kasner/projects/cztaubench/logs/vllm_%j.out
#
# vLLM server for CzTauBench.
#
# The critical difference from a plain `vllm serve` is tool calling: tau2-bench
# drives every task through function calls, so the server must be started with
# --enable-auto-tool-choice and a parser matching the model's chat template.
# Qwen3.6's template emits <tool_call><function=...><parameter=...> XML, which
# is the qwen3_xml parser (NOT hermes, which expects JSON in <tool_call>).
#
# Thinking mode is left ENABLED at the server level; it is switched per request
# by the client via chat_template_kwargs.enable_thinking, so both modes are
# available without a restart.
#
# The #SBATCH lines above size a SINGLE-GPU allocation, which fits every model
# run so far except the bf16 35B-A3B (72 GB of weights). Override on the command
# line for a multi-GPU one -- sbatch flags win over the directives, and
# --tensor-parallel-size is derived from the allocation below rather than being
# a second place to keep the number in sync:
#
#   sbatch -G 4 -N 1 -C gpuram40G -p gpu-troja \
#       --export=ALL,CZTAU_MODEL=Qwen/Qwen3.6-35B-A3B scripts/vllm_server.sh
#
# -N 1 is not decorative: TP shards one model across GPUs over NVLink/PCIe, so
# `-G 4` spread over two nodes would fail rather than run slowly.

set -euo pipefail

MODEL="${CZTAU_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
PORT="${CZTAU_PORT:-8000}"
MAX_LEN="${CZTAU_MAX_LEN:-65536}"

# Tensor-parallel degree = however many GPUs Slurm actually gave us. Reading it
# from the allocation means the sbatch -G flag is the single source of truth; a
# hardcoded value silently mismatches the moment the job is submitted with a
# different one, and vllm's failure for that is a CUDA OOM deep into loading.
NUM_GPUS="$(echo -n "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | grep -c .)"
TP="${CZTAU_TP:-${NUM_GPUS}}"

source /lnet/work/people/kasner/virtualenv/vllm/bin/activate

echo "*** CzTauBench vLLM server ***"
echo "NODE:  $(hostname -f)"
echo "PORT:  ${PORT}"
echo "MODEL: ${MODEL}"
echo "GPUS:  ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}), TP=${TP}"
echo "Set in scripts/env.sh:  CZTAU_VLLM_BASE=http://$(hostname -s):${PORT}/v1"
echo "******************************"

exec vllm serve "${MODEL}" \
    --served-model-name "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64
