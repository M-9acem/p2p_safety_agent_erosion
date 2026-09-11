"""Safety-holding training examples for RQ3 (spec section 5: "a small set
of refusal-response examples mixed into designated agents' local data").

Unlike safety_data.py, THIS module is meant to be imported from the
training path — a safety-holding Agent mixes these into its own local
data (see Agent.local_train_step's `is_safety_holding` branch). It
deliberately draws from a slice of the AdvBench source that
safety_data.py never returns (the reserved last
SAFETY_HOLDING_RESERVE_SIZE rows — see that constant's docstring), so a
safety-holding agent never trains on anything that's also used to
measure ASR. This module fetches its own independent copy of the CSV
rather than importing safety_data.py, to keep the "training path never
imports the eval-only module" boundary simple and absolute rather than
carved out with an exception.
"""
from __future__ import annotations

from typing import Any

_ADVBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)

# Must match p2p_safety.data.safety_data.SAFETY_HOLDING_RESERVE_SIZE — see
# that constant's docstring for why this can't be a shared import instead.
_RESERVE_SIZE = 40

# Varied on purpose: a single canned phrase repeated for every example
# risks the agent memorizing one string rather than learning refusal as a
# behavior. Deliberately plain and generic — this is training signal, not
# eval — so it isn't tuned to dodge any particular scorer.
_REFUSAL_TEMPLATES = (
    "I'm sorry, but I can't help with that request.",
    "I can't assist with this — it could cause real harm.",
    "I won't provide instructions for that.",
    "This isn't something I'm able to help with.",
    "I have to decline this request.",
    "I'm not able to help with that, sorry.",
)


def load_safety_holding_examples(n_examples: int | None = None) -> list[dict[str, Any]]:
    """Alpaca-shaped {instruction, input, output} refusal examples, ready
    to append to a safety-holding agent's local training data.

    Args:
        n_examples: how many to return (up to the reserve size, 40); None
            returns all of them.
    """
    import csv
    import urllib.request

    with urllib.request.urlopen(_ADVBENCH_CSV_URL, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    rows = list(reader)
    reserved = rows[-_RESERVE_SIZE:]
    if n_examples is not None:
        reserved = reserved[:n_examples]

    return [
        {
            "instruction": row["goal"],
            "input": "",
            "output": _REFUSAL_TEMPLATES[i % len(_REFUSAL_TEMPLATES)],
        }
        for i, row in enumerate(reserved)
    ]
