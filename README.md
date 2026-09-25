# ComoRAG × Gemini 3.1 Flash-Lite

**An end-to-end Gemini adaptation of ComoRAG for InfiniteBench EN.QA and EN.MC**

> This repository is a research fork/adaptation of the official [ComoRAG](https://github.com/EternityJune25/ComoRAG) implementation accompanying the paper [*ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning*](https://arxiv.org/abs/2508.10419).  
> It is **not** the official ComoRAG repository.

Basic pipeline:
<img width="1122" height="1402" alt="ChatGPT Image Sep 25, 2026, 06_05_49 PM" src="https://github.com/user-attachments/assets/c893de51-f4f9-427e-9f36-44fafd2d94d4" />


The main experiment lives on the branch:

```text
experiment/gemini31-infinitebench
```

This branch replaces the original GPT-4o-mini backbone with **Gemini 3.1 Flash-Lite**, adapts the code for the Gemini OpenAI-compatible API and native Windows execution, and evaluates the resulting end-to-end system on a frozen budget-constrained subset of **InfiniteBench EN.QA** and **EN.MC**.

---

## Research question

> **How does an end-to-end Gemini 3.1 Flash-Lite implementation of ComoRAG perform on InfiniteBench EN.QA and EN.MC relative to the published GPT-4o-mini ComoRAG results?**

This is intentionally treated as a **replication/extension**, not as a perfectly controlled backbone swap. In ComoRAG, the LLM is used not only for final answering but also for model-dependent intermediate stages such as summarization and OpenIE. Replacing the LLM therefore changes parts of the indexing pipeline as well as the final reasoning stage.

---

## Main changes in this branch

Compared with the upstream implementation, this branch adds or modifies:

- **Gemini 3.1 Flash-Lite** through Google's OpenAI-compatible endpoint.
- **BAAI/bge-m3** embeddings.
- Native **Windows / Git Bash** compatibility.
- Platform markers for Linux-only dependencies such as `vllm`, `triton`, NCCL, and `uvloop`.
- A **6,000-token reconstructed V:S:E:H evidence budget** using the paper's `8:2:2:1` ratio.
- Historical-memory token budgeting.
- Separate **retrieval queries** and **final QA queries** for EN.MC so that answer options are hidden during retrieval but shown to the final answering stage.
- Shared index reuse between EN.QA and EN.MC when the underlying context is identical.
- Index-completion markers and artifact validation for safer resume behavior.
- Robust handling of singleton retrieval/normalization cases.
- Gemini response validation for missing messages/content and content-filter responses.
- Timeline summarization robustness, including deterministic extractive fallback for Gemini `PROHIBITED_CONTENT` windows.
- A deterministic, budget-aware InfiniteBench subset-selection workflow.
- EN.MC evaluation using both the official InfiniteBench-style scorer and a stricter explicit-choice audit.
- Recovery scripts documenting and correcting a detected EN.MC implementation bug before final evaluation.

---

## Experimental configuration

The frozen experiment uses:

| Setting | Value |
|---|---|
| LLM | `gemini-3.1-flash-lite` |
| API | Gemini OpenAI-compatible endpoint |
| Embedding model | `BAAI/bge-m3` |
| OpenIE | Online / API-based |
| Chunk size | 512 tokens |
| Embedding batch size | 32 |
| Semantic clustering | Enabled |
| Temperature | 0 |
| Experiment seed | 0 |
| Maximum metacognitive iterations | 5 |
| Maximum probing queries | 3 |
| Veridical budget | 3692 tokens |
| Semantic budget | 923 tokens |
| Episodic budget | 923 tokens |
| Historical budget | 462 tokens |
| Total QA evidence budget | 6000 tokens |
| V:S:E:H allocation | 8:2:2:1 |

The exact frozen configuration is stored in:

```text
experiment_manifests/reproducibility_2026-09-25/experiment_config.txt
```

### Seed note

The experiment configuration records seed `0` for reproducibility. The Gemini OpenAI-compatible API path used here does not send the unsupported `seed` parameter to Gemini, so this should not be interpreted as deterministic Gemini sampling at the API level.

---

## Frozen InfiniteBench subset

Running the full benchmark end-to-end was cost-prohibitive for this seminar experiment, so a fixed subset was selected **before final evaluation** using a deterministic budget-constrained procedure.

The selection prioritizes:

- reuse of already-indexed contexts,
- contexts shared between EN.QA and EN.MC,
- representation across small, medium, and large documents,
- a fixed random seed,
- a fixed API-budget ceiling.

Final frozen subset:

| Task | Contexts | Questions |
|---|---:|---:|
| EN.QA | 28 | 143 |
| EN.MC | 25 | 99 |

Frozen manifest:

```text
experiment_manifests/infinitebench_subset_seed0.json
```

Manifest SHA-256:

```text
ae7642814f94d51facbe9523f92672732dbba685ee67399c44d80aca70b72631
```

The selection script is available at:

```text
scripts/select_infinitebench_subset.py
```

For reproducing the reported experiment, use the **committed frozen manifest** rather than generating a new subset.

---

## Results

### Final Gemini 3.1 Flash-Lite + ComoRAG results

| Task | Metric | Result |
|---|---|---:|
| EN.QA | Exact Match | **16.78%** |
| EN.QA | F1 | **24.46%** |
| EN.MC | Official InfiniteBench Accuracy | **31.31%** (31/99) |
| EN.MC | Strict explicit-choice Accuracy | **30.30%** (30/99) |
| EN.MC | Explicit-choice rate | **39.39%** (39/99) |
| EN.MC | Final genuine `<NO_OUTPUT>` responses | **2.02%** (2/99) |

Machine-readable summaries:

```text
evaluation_results/enqa_subset/evaluation_summary.json
evaluation_results/enmc_subset/evaluation_summary.json
```

### Published ComoRAG reference

The original ComoRAG paper reports the following full-benchmark results with GPT-4o-mini:

| Task | Metric | Published ComoRAG + GPT-4o-mini |
|---|---|---:|
| EN.QA | F1 | 34.52 |
| EN.QA | Exact Match | 25.07 |
| EN.MC | Accuracy | 72.93 |

These numbers are included only as a **descriptive reference**. The original paper evaluates the complete InfiniteBench tasks, while this experiment evaluates the frozen budget-constrained subset above. The differences therefore must **not** be interpreted as controlled estimates of the effect of replacing GPT-4o-mini with Gemini 3.1 Flash-Lite.

---

## EN.MC diagnostic observation

The final official EN.MC accuracy is **31.31%**. Only 39 of the 99 responses contained an explicit A/B/C/D choice.

Among those 39 explicit-choice responses, 30 were correct:

```text
30 / 39 = 76.92%
```

This value is **diagnostic only** and is not a benchmark accuracy measure, because the 39 cases are a self-selected subset in which the model chose to commit to an option.

The official InfiniteBench-style scorer and the stricter explicit-choice audit differ by only one correct prediction (31 vs. 30), so permissive answer extraction is not the primary explanation for the low aggregate EN.MC score.

---

## Evaluation integrity and EN.MC recovery

During validation, an indentation error in the Gemini response-handling patch was detected. It incorrectly converted normal `finish_reason=STOP` responses into `<NO_OUTPUT>`.

The issue affected **54 final EN.MC predictions**. The recovery procedure was deliberately restricted to those affected questions:

1. identify the 54 affected final predictions,
2. remove only invalid `STOP + <NO_OUTPUT>` cache entries,
3. preserve genuine content-filter responses,
4. rerun only the affected 54 questions,
5. validate all 54 recovered outputs,
6. merge them into the frozen 99-question EN.MC result,
7. recompute the final evaluation.

The final reported **31.31%** EN.MC accuracy is the **post-recovery** result. The earlier pre-recovery evaluation is not a valid experimental result.

Relevant recovery files:

```text
experiment_manifests/enmc_stop_bug_recovery.json
scripts/find_bad_enmc_stop_outputs.py
scripts/remove_bad_stop_cache.py
scripts/recover_enmc_stop_bug.py
scripts/apply_enmc_stop_bug_recovery.py
```

---

## Installation

### 1. Clone the repository and switch to the experiment branch

```bash
git clone https://github.com/KritikBansal/ComoRAG.git
cd ComoRAG
git checkout experiment/gemini31-infinitebench
```

### 2. Create a Python environment

The frozen experiment was run with Python 3.12 on native Windows.

```bash
python -m venv .venv
```

Git Bash on Windows:

```bash
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The exact environment used for the reported experiment is preserved in:

```text
experiment_manifests/reproducibility_2026-09-25/pip_freeze.txt
```

### 4. Configure Gemini

Create a `.env` file in the repository root:

```env
GEMINI_API_KEY=your_api_key_here
```

Do not commit `.env` or your API key.

---

## Prepare InfiniteBench

The preparation script downloads InfiniteBench through Hugging Face, groups questions by context, chunks each context into 512-token BGE-M3 chunks, and creates the EN.QA and EN.MC directory structure expected by the runner.

```bash
python scripts/prepare_infinitebench.py
```

Prepared data are stored under:

```text
dataset/infinitebench/enqa/
dataset/infinitebench/enmc/
```

For EN.MC, each prepared example stores both:

- `question`: question + A/B/C/D options for final answering,
- `retrieval_question`: question without options for retrieval.

---

## Run the frozen experiment

### EN.QA

```bash
python run_gemini_infinitebench.py --task enqa
```

### EN.MC

```bash
python run_gemini_infinitebench.py --task enmc
```

### Both tasks

```bash
python run_gemini_infinitebench.py --task all
```

By default, the runner uses:

```text
experiment_manifests/infinitebench_subset_seed0.json
```

Useful options:

```bash
--limit-contexts N   # pilot on the first N selected contexts
--force              # rerun answers even if results already exist
--rebuild-index      # ignore completion markers and rebuild/resume indexes
--manifest PATH      # use a different subset manifest
```

For reproducing the reported scores, do **not** use `--force`, `--rebuild-index`, or a different manifest unless you intentionally want a new experiment.

---

## Evaluate

### EN.QA

The upstream ComoRAG QA evaluator computes normalized Exact Match and token-overlap F1.

Example:

```bash
python script/eval_qa.py \
  evaluation_inputs/enqa_subset \
  --output evaluation_results/enqa_subset
```

### EN.MC

```bash
python scripts/evaluate_enmc_subset.py
```

This produces both:

- official InfiniteBench-style MC accuracy,
- strict explicit-choice accuracy and response-format diagnostics.

---

## Reproducibility records

The repository preserves compact reproducibility records in:

```text
experiment_manifests/reproducibility_2026-09-25/
```

Important files include:

```text
experiment_config.txt
final_scores.txt
pip_freeze.txt
python_version.txt
torch_environment.txt
windows_version.txt
subset_manifest_sha256.txt
final_results_sha256.txt
final_evaluation_sha256.txt
critical_code_sha256.txt
```

These records capture the experimental configuration, package environment, score summary, and hashes of the frozen inputs/outputs/code used for the reported run.

---

## Repository structure for this experiment

```text
ComoRAG/
├── run_gemini_infinitebench.py
├── main_gemini.py
├── requirements.txt
├── scripts/
│   ├── prepare_infinitebench.py
│   ├── select_infinitebench_subset.py
│   ├── evaluate_enmc_subset.py
│   ├── find_bad_enmc_stop_outputs.py
│   ├── remove_bad_stop_cache.py
│   ├── recover_enmc_stop_bug.py
│   └── apply_enmc_stop_bug_recovery.py
├── evaluation_results/
│   ├── enqa_subset/evaluation_summary.json
│   └── enmc_subset/evaluation_summary.json
├── experiment_manifests/
│   ├── infinitebench_subset_seed0.json
│   ├── enmc_stop_bug_recovery.json
│   └── reproducibility_2026-09-25/
└── src/comorag/
    ├── ComoRAG.py
    ├── llm/openai_gpt.py
    └── utils/
```

---

## Important limitations

- The Gemini experiment uses a **budget-constrained subset**, whereas the paper's reference results use the complete benchmark.
- This is an **end-to-end model substitution**, not a controlled final-answer-only swap. Gemini participates in model-dependent indexing and reasoning stages.
- Gemini API safety/content filters can affect intermediate summarization and final answering.
- The reconstructed 6,000-token V:S:E:H allocation uses integer budgets of `3692:923:923:462` to approximate the paper's `8:2:2:1` ratio.
- The Gemini OpenAI-compatible endpoint used here does not receive the configured seed parameter.
- The implementation was developed and evaluated on native Windows; exact behavior may differ across operating systems, CUDA setups, package versions, or future Gemini API revisions.

---

## Original ComoRAG paper

If you use this repository, please also cite the original ComoRAG work:

```bibtex
@article{wang2025comorag,
  title={ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning},
  author={Wang, Juyuan and Zhao, Rongchen and Wei, Wei and Wang, Yufeng and Yu, Mo and Zhou, Jie and Xu, Jin and Xu, Liyan},
  journal={arXiv preprint arXiv:2508.10419},
  year={2025}
}
```

Original project:

- Paper: https://arxiv.org/abs/2508.10419
- Official repository: https://github.com/EternityJune25/ComoRAG

---

## Acknowledgements

This work builds directly on the original ComoRAG implementation and its upstream dependencies. The purpose of this fork is to document a seminar replication/extension using Gemini 3.1 Flash-Lite and a reproducible InfiniteBench subset, not to replace or rebrand the original project.
