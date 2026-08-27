#!/bin/bash
#SBATCH -J cztau-dl
#SBATCH -p cpu-ms,cpu-troja
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -o /lnet/work/people/kasner/projects/cztaubench/logs/download_%j.out
#SBATCH -e /lnet/work/people/kasner/projects/cztaubench/logs/download_%j.out
#
# Pre-fetch a model into the shared HF cache, on a CPU node.
#
# `vllm serve` would download it too, but it holds the GPU allocation while it
# does -- 72 GB of bf16 weights is the better part of an hour with a GPU idling
# behind it. Splitting the fetch out lets the server job be queued with
# --dependency=afterok on this one, so the GPU is claimed only once the weights
# are on disk.
#
# Usage: sbatch scripts/download_model.sh Qwen/Qwen3.6-35B-A3B

set -euo pipefail

MODEL="${1:?usage: download_model.sh <hf-model-id>}"

# Pinned explicitly rather than inherited: a job submitted with a different
# --export would otherwise download into a per-user cache the GPU node's job
# cannot see, and vllm would silently re-fetch the whole thing.
export HF_HOME="${HF_HOME:-/lnet/work/people/kasner/storage/huggingface}"

source /lnet/work/people/kasner/virtualenv/vllm/bin/activate

echo "MODEL:   ${MODEL}"
echo "HF_HOME: ${HF_HOME}"
date

hf download "${MODEL}" --max-workers 8

date
echo "=== downloaded ${MODEL} ==="
