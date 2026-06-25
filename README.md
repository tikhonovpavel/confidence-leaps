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

## Example output

Real output from
`python extract_confidence_traces.py --model_name "Qwen/Qwen3-8B" --dataset gpqa --limit 5 --max_prompt_tokens 16384 --max_chunks 60 --run_dir runs/gpqa`
(a GPQA physics question — `⟨10σ_z + 5σ_x⟩` for a spin-½ superposition).

This is a textbook **confidence leap**: the model commits to the wrong answer
`A` for three chunks, then at chunk 3 — which literally opens with the trigger
word *"Wait"* — it rechecks the algebra and `P(correct=B)` jumps from `0.32`
to `0.98`, after which it stays certain. The probability trajectory:

```
chunk:          0     1     2     3      4    5  …  21
prediction:     A     A     A     B      B    B  …  B
P(correct=B): 0.233 0.304 0.318 0.985  0.999 1.0 … 1.0
                            └──── leap (Δ=0.667) ────┘
```

One `results.jsonl` record (the `reasoning_trace` / `chunks` text is abbreviated
here with `…`; everything else is verbatim):

```json
{
  "example_index": 2,
  "status": "ok",
  "question": "A spin-half particle is in a linear superposition 0.5|↑⟩ + (√3/2)|↓⟩ …",
  "choices": ["…", "…", "…", "…"],
  "correct_letter": "B",
  "natural_answer": "B",
  "reasoning_trace": "Okay, let's see. I need to find the expectation value of the operator 10σ_z + 5σ …",
  "chunks": [
    "…",
    "But ⟨↑|↓⟩ is zero, and ⟨↓|↑⟩ is also zero. So the cross terms vanish. Therefore, ⟨σ_z⟩ = (0.25)(1) + ((3/4)(-1)) = 0.2…",
    "Wait, let me check that again. Let me compute each term:\n\nFirst term: 0.5 * (√3)/2 * ⟨↑|↑⟩ = 0.5*(√3)/2 *1 = √3/4 ≈ 0.43…"
  ],
  "num_chunks": 22,
  "letter_probs": [
    "…",
    {"A": 0.3178, "B": 0.3178, "C": 0.2475, "D": 0.1169},
    {"A": 0.0140, "B": 0.9846, "C": 0.0002, "D": 0.0012}
  ],
  "per_chunk_prediction": ["A", "A", "A", "B", "B", "B", "…", "B"],
  "per_chunk_correct_prob": [0.233, 0.304, 0.318, 0.985, 0.999, 1.0, "…", 1.0],
  "max_confidence": 1.0,
  "mean_confidence": 0.918,
  "max_jump": {"chunk_index": 3, "letter": "B", "delta": 0.6668, "prob_before": 0.3178, "prob_after": 0.9846},
  "max_correct_jump": {"chunk_index": 3, "delta": 0.6668},
  "final_prediction": "B",
  "final_confidence": 1.0,
  "first_prediction_index": 0,
  "first_correct_index": 3,
  "stabilized_index": 3,
  "stabilized_value": "B",
  "num_changes": 1,
  "correct_at_first_chunk": false,
  "overall_correct": true,
  "leap_threshold": 0.5,
  "leap_present": true,
  "leap_stop_index": 3,
  "leap_predicted_letter": "B",
  "leap_correct": true
}
```

The matching `summary.json` for the 5-example run:

```json
{
  "num_examples": 5,
  "status_counts": {"ok": 4, "too_many_chunks": 1},
  "num_ok": 4,
  "mean_num_chunks": 39.5,
  "mean_max_confidence": 0.9996,
  "mean_max_jump": 0.338,
  "leap_present_rate": 0.25,
  "natural_accuracy": 0.75,
  "final_chunk_accuracy": 0.75
}
```

Here `leap_stop_index: 3` means the early-stopping heuristic would have halted
the chain at chunk 3 (out of 22) with the correct answer — the remaining 18
chunks are spent only re-confirming a decision already made. One of the five
examples was skipped (`too_many_chunks`) by the `--max_chunks 60` guard, and is
still recorded in `results.jsonl` rather than dropped.
