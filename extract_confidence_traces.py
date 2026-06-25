"""
End-to-end confidence-trace extractor.

Given a model name and a set of prompts (an MCQ dataset or a custom file), this
pipeline does, for every example:

    1. loads the model once (reusing `ModelForcing` from the paper code),
    2. generates a reasoning trace (reusing the tested generation prompt),
    3. splits the trace into chunks (reusing `split_reasoning_chain`),
    4. forces an answer after every chunk prefix and reads off P(A/B/C/D)
       (reusing `compute_letter_probs_only` / `ModelForcing`),
    5. computes the per-chunk confidences and a set of derived scalars
       (max confidence, max probability leap, stabilization, etc.).

Everything that touches the model or the chunking is imported from the existing,
already-debugged modules in the repo root -- this file only orchestrates them,
adds crash-safe incremental writes, resume, a tqdm bar and a configurable CLI.

Results are written incrementally (one JSON line per example, fsync'd) into a
dedicated run folder so a crashed run can be resumed from where it stopped.

Example:
    python confidence_pipeline/extract_confidence_traces.py \
        --model_name "Qwen/Qwen3-32B" \
        --dataset gpqa \
        --limit 20 \
        --run_dir confidence_runs/qwen3_gpqa

    # resume the same run after a crash:
    python confidence_pipeline/extract_confidence_traces.py \
        --run_dir confidence_runs/qwen3_gpqa --resume
"""

import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional

import fire
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# This package is self-contained: the building blocks below are vendored copies
# living next to this file, so only the `confidence_pipeline/` folder is needed.
# Put this file's own directory on the path so the local copies are found whether
# the file is run as a script or imported as a module.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# --- vendored, already-debugged building blocks --------------------------------
from rc_utils import (  # noqa: E402
    split_reasoning_chain,
    extract_reasoning_from_prompt,
    compose_user_prompt,
)
from precompute_forced_solution_metrics import (  # noqa: E402
    ModelForcing,
    compute_letter_probs_only,
    compute_metrics,
)
from leap_heuristic import simulate_jump_heuristic  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Prompt loading
# ----------------------------------------------------------------------------
def load_examples(
    dataset: Optional[str],
    prompts_file: Optional[str],
    dataset_config: str,
    dataset_split: str,
) -> List[Dict]:
    """Returns a list of normalized examples.

    Each example dict has:
        prompt          -- full MCQ user prompt (question + choices)
        question        -- raw question text (optional)
        choices         -- list of choice strings (optional)
        correct_letter  -- 'A'..'D' or None
    Source is either a named dataset (gpqa/logiqa, reusing the repo Dataset
    classes) or a json/jsonl file of records.
    """
    if dataset and prompts_file:
        raise ValueError("Pass either --dataset or --prompts_file, not both.")

    examples: List[Dict] = []

    if dataset:
        key = dataset.lower()
        if key == "gpqa":
            from gpqa_dataset import GPQADataset
            ds = GPQADataset(config_name=dataset_config, split=dataset_split)
        elif key == "logiqa":
            from logiqa_dataset import LogiqaDataset
            ds = LogiqaDataset(split=dataset_split)
        else:
            raise ValueError(f"Unknown dataset '{dataset}'. Use 'gpqa' or 'logiqa'.")
        for i in range(len(ds)):
            item = ds[i]
            examples.append({
                "prompt": item["prompt"],
                "question": item.get("question"),
                "choices": item.get("choices"),
                "correct_letter": item.get("answer_letter"),
            })
        return examples

    if prompts_file:
        records = _read_records(prompts_file)
        for rec in records:
            if isinstance(rec, str):
                examples.append({"prompt": rec, "question": rec,
                                 "choices": None, "correct_letter": None})
                continue
            question = rec.get("question")
            choices = rec.get("choices")
            prompt = rec.get("prompt")
            if not prompt:
                # Build the MCQ prompt from question + choices the same way the
                # forced-solution code expects it.
                prompt = compose_user_prompt(question or "", json.dumps(choices) if choices else "")
            letter = rec.get("correct_letter") or rec.get("answer_letter")
            examples.append({
                "prompt": prompt,
                "question": question,
                "choices": choices,
                "correct_letter": (letter or "").strip().upper() or None,
            })
        return examples

    raise ValueError("Provide a prompt source: --dataset gpqa|logiqa or --prompts_file path.")


def _read_records(path: str) -> List:
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("examples", data.get("data", []))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list (or jsonl) of records.")
    return data


# ----------------------------------------------------------------------------
# Reasoning-trace generation (reuses the tested generation recipe)
# ----------------------------------------------------------------------------
def _token_id(tok, token_str: str) -> Optional[int]:
    """Resolve a marker to a single vocab id, or None if it isn't one real token.

    Used to find the reasoning-delimiter tokens (e.g. </think>) by *id* instead
    of by fragile string matching. Returns None when the marker is not a single
    dedicated token in this tokenizer (e.g. gpt-oss has no </think>).
    """
    tid = tok.convert_tokens_to_ids(token_str)
    if tid is None:
        return None
    unk = getattr(tok, "unk_token_id", None)
    if unk is not None and tid == unk:
        return None
    return tid


# Patterns for auto-discovering reasoning-delimiter tokens in a tokenizer's
# *added/special* vocab, so we don't hardcode one spelling. Matches e.g.
# <think>/</think>, <|think|>/<|/think|>, <|thinking|>, <|reasoning|>, and
# their *_start/_end and start_/end_ variants.
_THINK_WORD_RE = re.compile(r"think|thought|reason", re.IGNORECASE)
_CLOSE_HINT_RE = re.compile(r"</|/\s*\w|_end|end_|\bend\b|close|stop|finish", re.IGNORECASE)


def _scan_think_markers(tok) -> tuple:
    """Scan the tokenizer's added/special vocab for think-like open/close tokens.

    Returns (open_id, close_id), either of which may be None. This is what makes
    detection robust to spellings other than literal </think> (e.g. <|/think|>).
    """
    try:
        added = dict(tok.get_added_vocab())  # {token_str: id}
    except Exception:
        added = {}
    # Also consider declared special tokens, in case they aren't in added_vocab.
    for s in getattr(tok, "all_special_tokens", []) or []:
        if s not in added:
            tid = tok.convert_tokens_to_ids(s)
            if isinstance(tid, int):
                added[s] = tid

    opens, closes = [], []
    for s, tid in added.items():
        if not _THINK_WORD_RE.search(s):
            continue
        (closes if _CLOSE_HINT_RE.search(s) else opens).append((s, tid))
    open_id = opens[0][1] if opens else None
    close_id = closes[0][1] if closes else None
    return open_id, close_id


def resolve_think_marker_ids(tok, open_override: Optional[str], close_override: Optional[str]) -> tuple:
    """Resolve the reasoning open/close marker *token ids* for this tokenizer.

    Priority: explicit CLI overrides -> known literal pairs -> generic scan of
    the tokenizer's special vocab. Returns (open_id, close_id); close_id None
    means "no usable reasoning-close token" (caller uses the text fallback,
    e.g. gpt-oss harmony).
    """
    if close_override:
        return _token_id(tok, open_override) if open_override else None, _token_id(tok, close_override)
    for o, c in [("<think>", "</think>"), ("<|think|>", "<|/think|>"), ("<|thinking|>", "<|/thinking|>")]:
        cid = _token_id(tok, c)
        if cid is not None:
            return _token_id(tok, o), cid
    return _scan_think_markers(tok)


def split_reasoning_and_answer(tok, gen_ids, open_id: Optional[int], close_id: Optional[int]) -> Optional[tuple]:
    """Split generated token ids into (reasoning_text, answer_text) on the
    reasoning-close token id. Returns None if there is no usable close token in
    the output (caller should fall back to the model-aware text extractor).
    """
    if close_id is None:
        return None
    ids = gen_ids.tolist()
    if close_id not in ids:
        return None
    ci = ids.index(close_id)
    start = 0
    if open_id is not None and open_id in ids[:ci]:
        start = ids.index(open_id) + 1
    reasoning = tok.decode(ids[start:ci], skip_special_tokens=True).strip()
    answer = tok.decode(ids[ci + 1:], skip_special_tokens=True)
    return reasoning, answer


@torch.no_grad()
def generate_reasoning_trace(
    forcing: ModelForcing,
    user_prompt: str,
    gen_system_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, Optional[str]]:
    """Generate a reasoning trace with the loaded model and return the raw text,
    the extracted <think> content and the natural answer letter.

    We let the model think in its native way: for Qwen3 (non-QwQ) we explicitly
    enable thinking mode; QwQ / DeepSeek-R1 / gpt-oss emit their own thinking
    section under the standard generation prompt.
    """
    tok = forcing.tokenizer
    model = forcing.model
    name = forcing.model_name.lower()

    messages = []
    if gen_system_prompt:  # empty/None -> no system message (don't suppress reasoning)
        messages.append({"role": "system", "content": gen_system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    template_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if "qwen3" in name and "qwq" not in name:
        # Qwen3 gates its <think>...</think> block behind enable_thinking.
        template_kwargs["enable_thinking"] = True
    prompt = tok.apply_chat_template(messages, **template_kwargs)

    inputs = tok(prompt, return_tensors="pt").to(forcing.device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, attention_mask=inputs["attention_mask"])
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)

    output_ids = model.generate(inputs["input_ids"], **gen_kwargs)

    gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    full_text = tok.decode(output_ids[0], skip_special_tokens=False)

    # Preferred: split on the reasoning-close *token id* (exact, model-agnostic
    # within the <think> family). Fall back to the repo's model-aware text
    # extractor for tokenizers without </think> (e.g. gpt-oss harmony channels).
    open_id = getattr(forcing, "_think_open_id", None)
    close_id = getattr(forcing, "_think_close_id", None)
    split = split_reasoning_and_answer(tok, gen_ids, open_id, close_id)
    if split is not None:
        think, answer_text = split
    else:
        think = extract_reasoning_from_prompt(full_text, forcing.model_name)
        gen_text = tok.decode(gen_ids, skip_special_tokens=False)
        answer_text = gen_text.split("</think>", 1)[1] if "</think>" in gen_text else gen_text

    matches = re.findall(r"[A-D]", answer_text)
    natural_answer = matches[-1] if matches else None

    return {"full_text": full_text, "think": think, "natural_answer": natural_answer}


# ----------------------------------------------------------------------------
# Derived confidence metrics
# ----------------------------------------------------------------------------
def compute_confidence_summary(
    letter_probs: List[Dict[str, float]],
    correct_letter: Optional[str],
    leap_threshold: float,
) -> Dict:
    """Per-chunk confidences + derived scalars (max confidence, max leap, ...).

    The stability metrics (num_changes, stabilization, first-correct, ...) reuse
    `compute_metrics`; the leap detection reuses `simulate_jump_heuristic`.
    """
    letters = ["A", "B", "C", "D"]

    per_chunk_top_letter: List[Optional[str]] = []
    per_chunk_top_conf: List[Optional[float]] = []
    per_chunk_correct_prob: List[Optional[float]] = []

    for probs in letter_probs:
        if probs:
            top_letter = max(probs, key=probs.get)
            per_chunk_top_letter.append(top_letter)
            per_chunk_top_conf.append(float(probs[top_letter]))
        else:
            per_chunk_top_letter.append(None)
            per_chunk_top_conf.append(None)
        if correct_letter and probs:
            per_chunk_correct_prob.append(float(probs.get(correct_letter, 0.0)))
        else:
            per_chunk_correct_prob.append(None)

    predictions = per_chunk_top_letter  # argmax prediction per chunk

    # --- biggest single-step probability leap of *any* letter (the "leap") ----
    max_jump = None  # dict
    max_drop = None
    max_correct_jump = None
    for k in range(1, len(letter_probs)):
        prev, cur = letter_probs[k - 1], letter_probs[k]
        if not prev or not cur:
            continue
        for letter in letters:
            delta = (cur.get(letter, 0.0) or 0.0) - (prev.get(letter, 0.0) or 0.0)
            if max_jump is None or delta > max_jump["delta"]:
                max_jump = {"chunk_index": k, "letter": letter, "delta": float(delta),
                            "prob_before": float(prev.get(letter, 0.0) or 0.0),
                            "prob_after": float(cur.get(letter, 0.0) or 0.0)}
            if max_drop is None or delta < max_drop["delta"]:
                max_drop = {"chunk_index": k, "letter": letter, "delta": float(delta)}
        if correct_letter:
            cdelta = (cur.get(correct_letter, 0.0) or 0.0) - (prev.get(correct_letter, 0.0) or 0.0)
            if max_correct_jump is None or cdelta > max_correct_jump["delta"]:
                max_correct_jump = {"chunk_index": k, "delta": float(cdelta)}

    # --- max confidence over the whole trace ----------------------------------
    valid_conf = [(i, c) for i, c in enumerate(per_chunk_top_conf) if c is not None]
    if valid_conf:
        max_conf_idx, max_conf = max(valid_conf, key=lambda x: x[1])
        min_conf_idx, min_conf = min(valid_conf, key=lambda x: x[1])
        mean_conf = sum(c for _, c in valid_conf) / len(valid_conf)
    else:
        max_conf_idx = max_conf = min_conf_idx = min_conf = mean_conf = None

    # --- stability metrics (reused) -------------------------------------------
    stability = compute_metrics(predictions, correct_letter)

    # --- leap early-stopping heuristic (reused) -------------------------------
    leap = simulate_jump_heuristic(letter_probs, leap_threshold)
    if leap is not None:
        leap_stop_index, leap_letter = leap
        leap_present = True
        leap_correct = (correct_letter is not None) and (leap_letter == correct_letter)
    else:
        leap_stop_index, leap_letter, leap_present, leap_correct = None, None, False, None

    final_prediction = predictions[-1] if predictions else None
    final_conf = per_chunk_top_conf[-1] if per_chunk_top_conf else None

    return {
        "num_chunks": len(letter_probs),
        # per-chunk arrays
        "letter_probs": letter_probs,
        "per_chunk_prediction": predictions,
        "per_chunk_top_confidence": per_chunk_top_conf,
        "per_chunk_correct_prob": per_chunk_correct_prob,
        # derived scalars
        "max_confidence": max_conf,
        "max_confidence_chunk": max_conf_idx,
        "min_confidence": min_conf,
        "min_confidence_chunk": min_conf_idx,
        "mean_confidence": mean_conf,
        "max_jump": max_jump,
        "max_drop": max_drop,
        "max_correct_jump": max_correct_jump,
        "final_prediction": final_prediction,
        "final_confidence": final_conf,
        # stability (reused compute_metrics)
        "first_prediction_index": stability["first_prediction_index"],
        "first_correct_index": stability["first_correct_index"],
        "stabilized_index": stability["stabilized_index"],
        "stabilized_value": stability["stabilized_value"],
        "num_changes": stability["num_changes"],
        "correct_at_first_chunk": stability["correct_at_first_chunk"],
        "overall_correct": stability["overall_correct"],
        # leap heuristic (reused simulate_jump_heuristic)
        "leap_threshold": leap_threshold,
        "leap_present": leap_present,
        "leap_stop_index": leap_stop_index,
        "leap_predicted_letter": leap_letter,
        "leap_correct": leap_correct,
    }


# ----------------------------------------------------------------------------
# Crash-safe run directory helpers
# ----------------------------------------------------------------------------
def _load_done_indices(results_path: str) -> set:
    done = set()
    if not os.path.exists(results_path):
        return done
    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["example_index"])
            except Exception:
                continue
    return done


def _append_jsonl(path: str, record: Dict):
    """Append one record and flush+fsync so a crash keeps everything so far."""
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def run(
    # --- model -------------------------------------------------------------
    model_name: str = "Qwen/Qwen3-32B",
    device_map: str = "auto",
    # --- prompt source -----------------------------------------------------
    dataset: Optional[str] = None,           # 'gpqa' | 'logiqa'
    prompts_file: Optional[str] = None,      # .json / .jsonl of records
    dataset_config: str = "gpqa_diamond",
    dataset_split: str = "train",
    limit: Optional[int] = None,
    offset: int = 0,
    # --- generation (the reasoning trace) ----------------------------------
    # Empty by default: a "don't explain" system prompt would suppress the very
    # reasoning we want. Override only if you have a reason to.
    gen_system_prompt: str = "",
    gen_max_new_tokens: int = 12800,
    gen_temperature: float = 0.6,
    # --- forced-solution probing -------------------------------------------
    forcing_system_prompt: str = "Answer only with a letter of a correct choice.",
    max_prompt_tokens: int = 8192,
    max_chunks: int = 40,
    # --- reasoning-boundary tokens (auto-detected; override if your model
    #     spells them differently, e.g. --reasoning_close_token '<|/think|>') ---
    reasoning_open_token: Optional[str] = None,
    reasoning_close_token: Optional[str] = None,
    # --- chunking (the variables most worth varying) -----------------------
    triggers: Optional[List[str]] = None,    # override chunk split words, e.g. --triggers '["Wait","But"]'
    model_flavour: Optional[str] = None,     # 'qwen' | 'oss'; inferred from model_name if None
    min_len_to_split: int = 100,             # only used for oss flavour
    # --- metrics -----------------------------------------------------------
    leap_threshold: float = 0.5,
    # --- run / resume ------------------------------------------------------
    run_dir: str = "confidence_runs/run",
    resume: bool = False,
):
    """Generate reasoning traces and extract per-chunk confidences + derived metrics.

    Anything a user would reasonably want to vary is a CLI flag: the model, the
    prompt source, the generation settings, the *chunking trigger words*, the
    chunking flavour, the forced-solution system prompt, the leap threshold, and
    the prompt/chunk length caps. Outputs go into a dedicated `run_dir`, written
    incrementally so the run is crash-safe and resumable.
    """
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.jsonl")
    config_path = os.path.join(run_dir, "config.json")

    config = {
        "model_name": model_name, "device_map": device_map,
        "dataset": dataset, "prompts_file": prompts_file,
        "dataset_config": dataset_config, "dataset_split": dataset_split,
        "limit": limit, "offset": offset,
        "gen_system_prompt": gen_system_prompt,
        "gen_max_new_tokens": gen_max_new_tokens, "gen_temperature": gen_temperature,
        "forcing_system_prompt": forcing_system_prompt,
        "max_prompt_tokens": max_prompt_tokens, "max_chunks": max_chunks,
        "triggers": triggers, "model_flavour": model_flavour,
        "min_len_to_split": min_len_to_split, "leap_threshold": leap_threshold,
        "reasoning_open_token": reasoning_open_token,
        "reasoning_close_token": reasoning_close_token,
    }
    # On a fresh run, persist the config; on resume, just keep the original.
    if not (resume and os.path.exists(config_path)):
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    # --- prompts -----------------------------------------------------------
    examples = load_examples(dataset, prompts_file, dataset_config, dataset_split)
    if offset:
        examples = examples[offset:]
    if limit is not None:
        examples = examples[:limit]
    logger.info("Loaded %d examples.", len(examples))

    done = _load_done_indices(results_path) if resume else set()
    if resume and done:
        logger.info("Resuming: %d examples already done, skipping them.", len(done))
    if not resume and os.path.exists(results_path):
        logger.warning(
            "results.jsonl already exists in %s and --resume is False; new results "
            "will be appended. Use --resume to skip finished examples.", run_dir,
        )

    flavour = model_flavour or ("oss" if "gpt-oss" in model_name.lower() else "qwen")

    # --- pre-flight: resolve reasoning markers BEFORE loading the weights -----
    # We split reasoning by token id and refuse to guess, so fail fast (without
    # paying for a multi-GB model load) if the special tokens can't be resolved.
    pre_tok = AutoTokenizer.from_pretrained(model_name)
    open_id, close_id = resolve_think_marker_ids(pre_tok, reasoning_open_token, reasoning_close_token)

    if reasoning_close_token and close_id is None:
        raise RuntimeError(
            f"--reasoning_close_token {reasoning_close_token!r} is not a single token in "
            f"the vocabulary of '{model_name}'. Pass a marker that exists as one special token."
        )
    if close_id is None:
        if flavour == "oss":
            logger.info(
                "No </think>-style token for %s; using the harmony text extractor "
                "(expected for gpt-oss).", model_name,
            )
        else:
            raise RuntimeError(
                f"Could not resolve reasoning open/close special tokens for '{model_name}' "
                f"(open={open_id}, close={close_id}). The pipeline splits reasoning by token "
                f"id and refuses to guess. Either pass --reasoning_open_token / "
                f"--reasoning_close_token with the markers this model uses (inspect the "
                f"tokenizer's special tokens), or set --model_flavour oss if it delimits "
                f"reasoning with harmony channels instead of <think>-style tokens."
            )
    else:
        logger.info(
            "Reasoning markers resolved by token id: open=%s close=%s (decoded: %r / %r)",
            open_id, close_id,
            pre_tok.decode([open_id]) if open_id is not None else None,
            pre_tok.decode([close_id]),
        )

    # --- model (loaded once) ----------------------------------------------
    forcing = ModelForcing(model_name, device_map=device_map)
    forcing._think_open_id = open_id
    forcing._think_close_id = close_id

    pending = [(i, ex) for i, ex in enumerate(examples) if i not in done]
    for example_index, ex in tqdm(pending, desc="Examples", initial=0, total=len(pending)):
        try:
            record = _process_one(
                forcing=forcing,
                example_index=example_index,
                ex=ex,
                gen_system_prompt=gen_system_prompt,
                gen_max_new_tokens=gen_max_new_tokens,
                gen_temperature=gen_temperature,
                forcing_system_prompt=forcing_system_prompt,
                max_prompt_tokens=max_prompt_tokens,
                max_chunks=max_chunks,
                triggers=triggers,
                flavour=flavour,
                min_len_to_split=min_len_to_split,
                leap_threshold=leap_threshold,
            )
        except Exception as e:  # noqa: BLE001 - keep the run alive, record the error
            logger.exception("Example %d failed: %s", example_index, e)
            record = {"example_index": example_index, "status": "error", "error": str(e)}
        _append_jsonl(results_path, record)

    _write_summary(results_path, os.path.join(run_dir, "summary.json"))
    logger.info("Done. Results in %s", results_path)
    return 0


def _write_summary(results_path: str, summary_path: str):
    """Aggregate the per-example results into a small dataset-level summary."""
    recs = [json.loads(l) for l in open(results_path) if l.strip()]
    ok = [r for r in recs if r.get("status") == "ok"]
    statuses: Dict[str, int] = {}
    for r in recs:
        statuses[r.get("status", "?")] = statuses.get(r.get("status", "?"), 0) + 1

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    labeled = [r for r in ok if r.get("correct_letter")]
    summary = {
        "num_examples": len(recs),
        "status_counts": statuses,
        "num_ok": len(ok),
        "mean_num_chunks": _mean([r.get("num_chunks") for r in ok]),
        "mean_max_confidence": _mean([r.get("max_confidence") for r in ok]),
        "mean_max_jump": _mean([(r.get("max_jump") or {}).get("delta") for r in ok]),
        "leap_present_rate": _mean([1.0 if r.get("leap_present") else 0.0 for r in ok]),
        "natural_accuracy": _mean([1.0 if r.get("natural_answer") == r.get("correct_letter") else 0.0 for r in labeled]) if labeled else None,
        "final_chunk_accuracy": _mean([1.0 if r.get("overall_correct") else 0.0 for r in labeled]) if labeled else None,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary: %s", json.dumps(summary, ensure_ascii=False))


def _process_one(
    *,
    forcing: ModelForcing,
    example_index: int,
    ex: Dict,
    gen_system_prompt: str,
    gen_max_new_tokens: int,
    gen_temperature: float,
    forcing_system_prompt: str,
    max_prompt_tokens: int,
    max_chunks: int,
    triggers: Optional[List[str]],
    flavour: str,
    min_len_to_split: int,
    leap_threshold: float,
) -> Dict:
    user_prompt = ex["prompt"]
    correct_letter = (ex.get("correct_letter") or "").strip().upper() or None

    # 1. generate the reasoning trace
    gen = generate_reasoning_trace(
        forcing, user_prompt, gen_system_prompt, gen_max_new_tokens, gen_temperature,
    )
    think = gen["think"]
    if not think:
        return {"example_index": example_index, "status": "no_think",
                "prompt": user_prompt, "natural_answer": gen["natural_answer"]}

    # 2. chunk it (reuse split_reasoning_chain, with optional custom triggers)
    chunks = split_reasoning_chain(
        think, model_flavour=flavour, min_len_to_split=min_len_to_split, triggers=triggers,
    )
    if not chunks:
        return {"example_index": example_index, "status": "no_chunks",
                "prompt": user_prompt, "think": think}
    if len(chunks) > max_chunks:
        return {"example_index": example_index, "status": "too_many_chunks",
                "prompt": user_prompt, "num_chunks": len(chunks)}

    # length guard (same spirit as precompute_forced_solution_metrics)
    templated = forcing._prepare_templated_prompt(forcing_system_prompt, user_prompt, think)
    num_tokens = len(forcing.tokenizer.encode(templated))
    if num_tokens > max_prompt_tokens:
        return {"example_index": example_index, "status": "prompt_too_long",
                "prompt": user_prompt, "num_prompt_tokens": num_tokens}

    # 3. force an answer after each chunk prefix -> P(A/B/C/D) per chunk
    letter_probs = compute_letter_probs_only(forcing, user_prompt, chunks, forcing_system_prompt)

    # 4. derived confidence summary
    summary = compute_confidence_summary(letter_probs, correct_letter, leap_threshold)

    return {
        "example_index": example_index,
        "status": "ok",
        "prompt": user_prompt,
        "question": ex.get("question"),
        "choices": ex.get("choices"),
        "correct_letter": correct_letter,
        "natural_answer": gen["natural_answer"],
        "reasoning_trace": think,
        "chunks": chunks,
        **summary,
    }


if __name__ == "__main__":
    sys.exit(fire.Fire(run))
