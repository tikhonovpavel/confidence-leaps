# confidence-leaps

A single, self-contained, crash-safe pipeline that — given **a model name and a
set of multiple-choice prompts** — for every example:

1. loads the model once,
2. generates a **reasoning trace**,
3. splits the trace into **chunks** on reasoning-shift trigger words,
4. forces an answer after every chunk prefix and reads off `P(A/B/C/D)`,
5. computes the **per-chunk confidences** plus derived scalars: max confidence,
   biggest probability **leap** (jump), biggest drop, max correct-answer jump,
   number of answer changes, first-correct / stabilization indices, and the
   leap early-stopping verdict.

This is the standalone extraction tool behind the "confidence leaps" analysis.
Reasoning is separated from the final answer **by special-token id** resolved
from the model's own tokenizer (not by fragile string matching), with explicit
overrides and a fail-fast guard when markers can't be resolved.

## Install

```bash
pip install -r requirements.txt
```

Requires a GPU for the 32B-class reasoning models used in the paper; smaller
models (e.g. `Qwen/Qwen3-8B`) run on a single consumer GPU.

## Usage

```bash
# Built-in datasets (GPQA / LogiQA)
python extract_confidence_traces.py \
    --model_name "Qwen/Qwen3-8B" \
    --dataset gpqa --limit 20 \
    --run_dir runs/qwen3_gpqa

# Your own multiple-choice tasks (see example_tasks.jsonl)
python extract_confidence_traces.py \
    --model_name "Qwen/Qwen3-8B" \
    --prompts_file example_tasks.jsonl \
    --run_dir runs/custom
```

### Your own tasks

The probing reads `P(A/B/C/D)`, so tasks must be multiple-choice with options
A–D. Each line of a `.jsonl` (or each record of a `.json` list) is either:

```json
{"question": "Find the root of 2x + 6 = 0.", "choices": ["-3","3","-6","12"], "correct_letter": "A"}
{"prompt": "Find the root of 2x + 6 = 0.\n\nChoices:\nA) -3\nB) 3\nC) -6\nD) 12", "correct_letter": "A"}
```

`correct_letter` is optional (metrics that need it become null without it). A
ready example is in [`example_tasks.jsonl`](example_tasks.jsonl).

### Resume after a crash

Results are appended (and `fsync`'d) line-by-line to `run_dir/results.jsonl`, so
a crashed run keeps everything computed so far. Re-run the **same** command with
`--resume` to skip the examples already finished:

```bash
python extract_confidence_traces.py --run_dir runs/custom --resume
```

## Configurable variables (CLI flags worth varying)

| Flag | What it controls |
|------|------------------|
| `--model_name`, `--device_map` | which model, how it's sharded |
| `--dataset` / `--prompts_file` | prompt source (`gpqa`/`logiqa`, or your file) |
| `--dataset_config`, `--dataset_split`, `--limit`, `--offset` | which/how many examples |
| `--gen_system_prompt`, `--gen_max_new_tokens`, `--gen_temperature` | reasoning-trace generation |
| `--triggers` | **chunking split words**, e.g. `--triggers '["Wait","But","However"]'` |
| `--model_flavour` (`qwen`/`oss`), `--min_len_to_split` | chunking mode |
| `--reasoning_open_token`, `--reasoning_close_token` | override the reasoning delimiter tokens if your model spells them differently (e.g. `<|/think|>`) |
| `--forcing_system_prompt` | the system prompt used when forcing the A–D answer |
| `--max_prompt_tokens`, `--max_chunks` | length guards (skip over-long traces) |
| `--leap_threshold` | probability-jump threshold for the leap heuristic |
| `--run_dir`, `--resume` | output folder / resume mode |

## How reasoning is delimited

The pipeline splits a generated sequence into *reasoning* and *final answer* by
the model's reasoning **close token**, resolved in this order:

1. explicit `--reasoning_open_token` / `--reasoning_close_token`,
2. known literal pairs (`<think>`/`</think>`, `<|think|>`/`<|/think|>`, …),
3. a scan of the tokenizer's special vocab for think-like tokens.

If none resolve — and the model isn't a harmony-style model (`--model_flavour
oss`) — the run **fails fast before loading the weights**, rather than silently
producing empty traces.

## Output layout

```
run_dir/
  config.json        # the exact configuration of the run
  results.jsonl      # one JSON record per example (written incrementally)
  summary.json       # dataset-level aggregates, written at the end
```

Each `results.jsonl` record (`status == "ok"`) contains: `prompt`, `question`,
`choices`, `correct_letter`, `natural_answer`, `reasoning_trace`, `chunks`,
`letter_probs` (per-chunk `P(A/B/C/D)`), `per_chunk_top_confidence`,
`per_chunk_correct_prob`, `max_confidence`, `max_jump`, `max_drop`,
`max_correct_jump`, `num_changes`, `first_correct_index`, `stabilized_index`,
`leap_present` / `leap_stop_index` / `leap_predicted_letter`, etc. Examples that
can't be processed are still recorded with a `status` (`no_think`, `no_chunks`,
`too_many_chunks`, `prompt_too_long`, `error`) so nothing is silently dropped.
