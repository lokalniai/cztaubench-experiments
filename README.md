# CzTauBench — Czech results

Simulation traces, scores and an interactive viewer for the experiments with
open LLMs on a Czech adaptation of
[τ²-bench](https://github.com/sierra-research/tau2-bench), following
[SEATauBench](https://github.com/SEACrowd/SEATauBench) (Nguyen et al., 2026).

- [💻 **Interactive viewer of the runs**](https://lokalniai.github.io/cztaubench-experiments/)
- [🖋 **Blogpost (in Czech)**](https://lokalni.ai/blog/umi-agenti-cesky-3/)

---

## What was run

Only the **L2 Interaction** scenario: the conversation moves to Czech. The
environment (tools, policy, database, tasks) stays English. Every Czech cell is
paired with an English cell for the same model.

| agent | id | served by |
|---|---|---|
| Qwen3.6 27B | `Qwen/Qwen3.6-27B` | local vLLM, bf16 |
| Qwen3.6 35B-A3B | `Qwen/Qwen3.6-35B-A3B` | local vLLM, bf16, 4-way TP |
| Gemma 4 31B | `google/gemma-4-31B-it` | local vLLM, bf16 |
| Gemma 4 E4B | `google/gemma-4-E4B-it` | local vLLM, bf16 |
| DeepSeek V4 Flash | `deepseek-v4-flash-0731` | e-INFRA API |

Every agent runs with reasoning enabled, at temperature 0. 

The **user simulator** and the **NL-assertion judge** are both Kimi K3 (Kimi K3 is never evaluated as an agent).

| domain | tasks | trials | simulations per cell |
|---|---|---|---|
| `airline` | 50 | 3 | 150 |
| `retail` | 114 | 3 | 342 |


On top of the task metrics (`pass^k`, robustness ρ₃, fastText language
correctness), all **820 Czech conversations** are annotated for language
*quality* by an LLM judge, one trial per task.


## Data

`results/simulations/<run>/<cell>/results.json` is the single source of truth:
every number on the site reduces from those files, and `python
scripts/report.py` over them reproduces the tables. 

Tasks, policies and databases derive from
[τ²-bench](https://github.com/sierra-research/tau2-bench) and
[SEATauBench](https://github.com/SEACrowd/SEATauBench), both MIT licensed.
