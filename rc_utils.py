import json
import re
from typing import List, Optional


# Shared constant used for Qwen3 forced-solution prompting
QWEN3_SPECIAL_STOPPING_PROMPT = (
    "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n"
)


TRIGGERS_QWEN = ['Wait', 'Alternatively', 'Hmm', 'Perhaps', 'Maybe', 'But', 'However']
TRIGGERS_OSS = sorted(
    list(
        set(
            TRIGGERS_QWEN
            + [
                'Actually', 'Could be', 'What if', 'Suppose', "Let's consider", "Let's try",
                "Let's analyze", "Let's attempt", "Let's assume", "Let's check",
                "Let's start by", 'Thus', 'So', 'Therefore', 'This suggests',
                'This means', 'Yes', 'No', 'Correct', 'Incorrect',
                'Alternatively', 'Now', 'First', 'Firstly', 'Then',
            ]
        )
    )
)


def split_reasoning_chain(
    reasoning_chain: str, model_flavour: str = 'qwen', min_len_to_split: int = 100,
    triggers: Optional[List[str]] = None,
) -> List[str]:
    """Splits a reasoning chain into chunks based on trigger words at paragraph starts.

    Logic mirrors the viewer implementation: paragraphs are separated by blank lines;
    a new chunk starts when a paragraph begins with one of the trigger words.

    For 'oss' model_flavour, it uses a conditional split: a new chunk is created
    only if the current chunk's length exceeds `min_len_to_split`.

    `triggers` lets callers override the default trigger-word list. When None, the
    flavour-specific defaults (TRIGGERS_QWEN / TRIGGERS_OSS) are used (back-compat).
    """
    if not reasoning_chain:
        return []

    if model_flavour != 'oss':
        triggers = triggers if triggers is not None else TRIGGERS_QWEN
        trigger_pattern = re.compile(r'^\s*(?:' + '|'.join(triggers) + r')\b', re.IGNORECASE)
        paragraphs = re.split(r'\n\s*\n', reasoning_chain.strip())
        if not paragraphs or not paragraphs[0]:
            return []
        final_chunks: List[str] = []
        current_chunk_paragraphs: List[str] = [paragraphs[0]]
        for paragraph in paragraphs[1:]:
            paragraph_stripped = paragraph.strip()
            if not paragraph_stripped:
                continue
            if trigger_pattern.match(paragraph_stripped):
                final_chunks.append("\n\n".join(current_chunk_paragraphs).strip())
                current_chunk_paragraphs = [paragraph]
            else:
                current_chunk_paragraphs.append(paragraph)
        if current_chunk_paragraphs:
            final_chunks.append("\n\n".join(current_chunk_paragraphs).strip())
        return [chunk for chunk in final_chunks if chunk]
    
    # --- Conditional splitting logic for OSS model ---
    triggers = triggers if triggers is not None else TRIGGERS_OSS
    trigger_pattern = re.compile(r'^\s*(?:' + '|'.join(re.escape(t) for t in triggers) + r')\b', re.IGNORECASE)
    
    paragraphs = re.split(r'(\n\s*\n)', reasoning_chain.strip())
    if not paragraphs or not paragraphs[0]:
        return []
    
    final_chunks = []
    current_chunk_parts = []
    
    i = 0
    while i < len(paragraphs):
        part = paragraphs[i]
        if not current_chunk_parts and part.isspace():
            i += 1
            continue

        is_trigger_para = (i % 2 == 0) and trigger_pattern.match(part.strip())
        current_len = len("".join(current_chunk_parts))

        if is_trigger_para and current_len > min_len_to_split:
            final_chunks.append("".join(current_chunk_parts).strip())
            current_chunk_parts = [part]
        else:
            current_chunk_parts.append(part)
        
        i += 1
        
    if current_chunk_parts:
        final_chunks.append("".join(current_chunk_parts).strip())
        
    return [chunk for chunk in final_chunks if chunk]


def extract_think_content(full_prompt_text: str) -> Optional[str]:
    """Return inner text strictly inside <think>...</think> if present; else None.
    """
    if not full_prompt_text:
        return None
    open_tag = "<think>"
    close_tag = "</think>"
    open_pos = full_prompt_text.find(open_tag)
    if open_pos == -1:
        return None
    close_pos = full_prompt_text.find(close_tag, open_pos + len(open_tag))
    if close_pos == -1:
        return None
    inner = full_prompt_text[open_pos + len(open_tag):close_pos]
    return inner.strip()


def extract_reasoning_from_prompt(full_prompt_text: str, model_path: str) -> Optional[str]:
    """Extracts reasoning/think content based on the model path."""
    model_path = model_path or ""
    if 'gpt-oss-20b' in model_path.lower():
        # OSS model uses <|start|>analysis<|message|>...<|end|>
        match = re.search(r"<\|start\|>analysis<\|message\|>(.*?)(<\|end\|>)", full_prompt_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    else:
        # Default to Qwen/Deepseek <think>...</think>
        return extract_think_content(full_prompt_text)


def parse_choices_text(choices_text: str) -> str:
    if not choices_text:
        return ""
    parsed = None
    try:
        parsed = json.loads(choices_text)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        lines = []
        for key in sorted(parsed.keys()):
            lines.append(f"{key}) {parsed[key]}")
        return "\n".join(lines)
    if isinstance(parsed, list):
        lines = []
        for idx, value in enumerate(parsed):
            letter = chr(ord('A') + idx)
            lines.append(f"{letter}) {value}")
        return "\n".join(lines)
    return choices_text.strip()


def compose_user_prompt(question_text: str, choices_text: str) -> str:
    choices_block = parse_choices_text(choices_text)
    if choices_block:
        return f"{question_text}\n\nChoices:\n{choices_block}"
    return question_text


