from p2p_safety.eval.capability import (
    _format_mmlu_prompt,
    _parse_mmlu_answer,
    _token_f1,
    run_held_out_task,
)


def test_format_mmlu_prompt_letters_choices():
    prompt = _format_mmlu_prompt("What is 2+2?", ["3", "4", "5", "6"])
    assert "A. 3" in prompt
    assert "B. 4" in prompt
    assert "C. 5" in prompt
    assert "D. 6" in prompt
    assert "What is 2+2?" in prompt


def test_parse_mmlu_answer_clean():
    assert _parse_mmlu_answer("B") == "B"
    assert _parse_mmlu_answer("The answer is C.") == "C"
    assert _parse_mmlu_answer("  a  ") == "A"  # case-insensitive


def test_parse_mmlu_answer_word_boundary_not_substring():
    # "A" must not match inside a word like "AS" or "CAT"
    assert _parse_mmlu_answer("As far as I can tell...") is None


def test_parse_mmlu_answer_unparseable_returns_none():
    assert _parse_mmlu_answer("I'm not sure.") is None
    assert _parse_mmlu_answer("") is None


def test_token_f1_identical_is_one():
    assert _token_f1("the cat sat", "the cat sat") == 1.0


def test_token_f1_no_overlap_is_zero():
    assert _token_f1("apples oranges", "bicycle rocket") == 0.0


def test_token_f1_partial_overlap():
    f1 = _token_f1("the cat sat on the mat", "the cat sat")
    assert 0.0 < f1 < 1.0


def test_token_f1_empty_prediction_vs_nonempty_reference():
    assert _token_f1("", "some reference") == 0.0


def test_token_f1_ignores_punctuation_and_case():
    assert _token_f1("The Cat, sat!", "the cat sat") == 1.0


def test_run_held_out_task_uses_reference_output():
    def fake_generate(prompts: list[str]) -> list[str]:
        return ["the cat sat" for _ in prompts]

    held_out = [
        {"instruction": "describe the scene", "input": "", "output": "the cat sat"},
        {"instruction": "describe the scene", "input": "", "output": "completely different words"},
    ]
    result = run_held_out_task(fake_generate, held_out)
    assert result["n_examples"] == 2
    assert 0.0 < result["mean_f1"] < 1.0  # one perfect match, one mismatch, averaged
