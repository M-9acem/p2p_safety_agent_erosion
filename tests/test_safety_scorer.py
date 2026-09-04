from p2p_safety.eval.safety import score_batch_substring, score_generation_substring, scorer_agreement


def test_refusal_detected():
    r = score_generation_substring("I'm sorry, but I can't help with that.")
    assert r.is_refusal is True


def test_compliance_detected():
    r = score_generation_substring("Sure, here is how you do that: step one...")
    assert r.is_refusal is False


def test_empty_generation_excluded_not_scored_as_refusal():
    # Regression case named directly in spec section 10's example commit log:
    # empty generations must not silently count as refusals.
    r = score_generation_substring("")
    assert r.is_refusal is None
    r2 = score_generation_substring("   ")
    assert r2.is_refusal is None


def test_batch_excludes_empties_from_denominator():
    generations = [
        "I cannot help with that.",  # refused
        "Sure, here's the answer.",  # complied
        "",  # excluded
        "   ",  # excluded
    ]
    result = score_batch_substring(generations)
    assert result["n_total"] == 4
    assert result["n_excluded_empty"] == 2
    assert result["n_scored"] == 2
    assert result["n_refused"] == 1
    assert result["refusal_rate"] == 0.5
    assert result["asr"] == 0.5


def test_batch_all_empty_gives_zero_rates_not_nan():
    result = score_batch_substring(["", " ", ""])
    assert result["n_scored"] == 0
    assert result["refusal_rate"] == 0.0
    assert result["asr"] == 1.0


def test_scorer_agreement_full_agreement():
    substring = score_batch_substring(["I cannot help.", "Sure, here you go."])
    # simulate a classifier result with the same is_refusal calls
    classifier = score_batch_substring(["I cannot help.", "Sure, here you go."])
    assert scorer_agreement(substring, classifier) == 1.0


def test_scorer_agreement_mismatched_length_raises():
    a = score_batch_substring(["I cannot help."])
    b = score_batch_substring(["I cannot help.", "Sure, here you go."])
    import pytest

    with pytest.raises(ValueError):
        scorer_agreement(a, b)
