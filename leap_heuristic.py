"""Leap (probability-jump) early-stopping heuristic.

`simulate_jump_heuristic` is copied verbatim from the repo's
`analyze_probability_jump_heuristic.py` so this package stays self-contained
without pulling in pandas/matplotlib. Logic is unchanged.
"""

from typing import Dict, List, Optional, Tuple


def simulate_jump_heuristic(
    letter_probs: List[Optional[Dict[str, float]]],
    threshold: float
) -> Optional[Tuple[int, str]]:
    """
    Simulates an early stopping heuristic based on a sudden jump in a letter's probability.

    Triggers if P(letter)_k - P(letter)_(k-1) > threshold for any letter.

    Returns (stop_index, predicted_letter) if the rule triggers, otherwise None.
    """
    if not letter_probs or len(letter_probs) < 2:
        return None

    for k in range(1, len(letter_probs)):
        probs_k_minus_1 = letter_probs[k-1]
        probs_k = letter_probs[k]

        if not isinstance(probs_k_minus_1, dict) or not isinstance(probs_k, dict):
            continue

        # Find all letters that jumped above the threshold
        jumped_letters = []
        for letter in "ABCD":
            prob_before = probs_k_minus_1.get(letter, 0.0) or 0.0
            prob_after = probs_k.get(letter, 0.0) or 0.0
            if prob_after - prob_before > threshold:
                jumped_letters.append((letter, prob_after))

        if jumped_letters:
            # If multiple letters jump, pick the one with the highest final probability
            best_letter, _ = max(jumped_letters, key=lambda item: item[1])
            return k, best_letter

    return None
