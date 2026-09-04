import numpy as np
import pytest

from p2p_safety.average import (
    Adapter,
    adapter_drift,
    average_adapters,
    average_delta,
    average_params,
    materialize_delta,
    refactor_svd,
)

RNG = np.random.default_rng(0)


def _random_adapter(modules: list[str], r: int, in_dim: int, out_dim: int, seed: int) -> Adapter:
    rng = np.random.default_rng(seed)
    return {
        m: {"A": rng.normal(size=(r, in_dim)), "B": rng.normal(size=(out_dim, r))}
        for m in modules
    }


def test_svd_refactor_reconstruction_error_bounded_at_full_rank():
    # When r matches the true rank of dW, reconstruction should be near-exact.
    A = RNG.normal(size=(8, 16))
    B = RNG.normal(size=(32, 8))
    alpha = 16
    dW = materialize_delta(A, B, alpha, r=8)

    A2, B2, err = refactor_svd(dW, r=8, alpha=alpha)
    assert err < 1e-8
    dW_hat = materialize_delta(A2, B2, alpha, r=8)
    np.testing.assert_allclose(dW_hat, dW, atol=1e-6)


def test_svd_refactor_error_bounded_and_decreasing_in_rank():
    # dW here has rank up to 16 (sum of two rank-8 updates); refactoring at
    # lower rank should have higher error than at the true rank, and error
    # must never exceed 1 (a sane upper bound for a *relative* Frobenius
    # error on a reasonably well-conditioned matrix).
    A1, B1 = RNG.normal(size=(8, 16)), RNG.normal(size=(32, 8))
    A2, B2 = RNG.normal(size=(8, 16)), RNG.normal(size=(32, 8))
    alpha = 16
    dW = materialize_delta(A1, B1, alpha, 8) + materialize_delta(A2, B2, alpha, 8)

    _, _, err_low = refactor_svd(dW, r=4, alpha=alpha)
    _, _, err_high = refactor_svd(dW, r=16, alpha=alpha)
    assert 0.0 <= err_high < err_low
    assert err_high < 1e-6  # r=16 spans the true rank of dW


def test_params_mode_and_delta_mode_disagree_in_general():
    # The whole point of section 4's subtlety: mean(B) @ mean(A) != mean(B @ A).
    modules = ["q_proj"]
    a1 = _random_adapter(modules, r=4, in_dim=8, out_dim=8, seed=1)
    a2 = _random_adapter(modules, r=4, in_dim=8, out_dim=8, seed=2)
    alpha, r = 8, 4

    params_result = average_params([a1, a2])
    delta_result, _ = average_delta([a1, a2], alpha=alpha, r=r)

    dW_params = materialize_delta(params_result["q_proj"]["A"], params_result["q_proj"]["B"], alpha, r)
    dW_delta = materialize_delta(delta_result["q_proj"]["A"], delta_result["q_proj"]["B"], alpha, r)

    assert not np.allclose(dW_params, dW_delta)


def test_delta_mode_matches_true_mean_dW_at_full_rank():
    # delta mode should reconstruct the *exact* mean update when the true
    # combined rank doesn't exceed r (this is the "correct" half of the
    # subtlety). average_delta's r does double duty — it's both the rank
    # the adapters were trained at (used to materialize each dW) and the
    # target refactor rank — so this only holds losslessly when the sum
    # doesn't exceed rank r. Force that by sharing A across both adapters:
    # B1 @ A + B2 @ A = (B1 + B2) @ A still has rank <= r.
    modules = ["q_proj"]
    r, in_dim, out_dim, alpha = 8, 16, 32, 16
    rng = np.random.default_rng(1)
    shared_A = rng.normal(size=(r, in_dim))
    a1 = {"q_proj": {"A": shared_A, "B": rng.normal(size=(out_dim, r))}}
    a2 = {"q_proj": {"A": shared_A, "B": rng.normal(size=(out_dim, r))}}

    true_mean_dW = (
        materialize_delta(a1["q_proj"]["A"], a1["q_proj"]["B"], alpha, r)
        + materialize_delta(a2["q_proj"]["A"], a2["q_proj"]["B"], alpha, r)
    ) / 2

    result, errors = average_delta([a1, a2], alpha=alpha, r=r)
    dW_hat = materialize_delta(result["q_proj"]["A"], result["q_proj"]["B"], alpha, r)
    np.testing.assert_allclose(dW_hat, true_mean_dW, atol=1e-6)
    assert errors["q_proj"] < 1e-8


def test_two_agent_complete_graph_reduces_to_plain_averaging():
    # Phase 2 acceptance check, verified by hand (spec section 8).
    modules = ["q_proj", "v_proj"]
    a1 = _random_adapter(modules, r=4, in_dim=8, out_dim=8, seed=10)
    a2 = _random_adapter(modules, r=4, in_dim=8, out_dim=8, seed=11)

    result = average_params([a1, a2])
    for m in modules:
        np.testing.assert_allclose(result[m]["A"], (a1[m]["A"] + a2[m]["A"]) / 2)
        np.testing.assert_allclose(result[m]["B"], (a1[m]["B"] + a2[m]["B"]) / 2)


def test_identical_adapters_average_to_themselves_both_modes():
    modules = ["q_proj"]
    a = _random_adapter(modules, r=4, in_dim=8, out_dim=8, seed=5)
    alpha, r = 8, 4

    params_result = average_params([a, a, a])
    for m in modules:
        np.testing.assert_allclose(params_result[m]["A"], a[m]["A"])
        np.testing.assert_allclose(params_result[m]["B"], a[m]["B"])

    delta_result, errors = average_delta([a, a, a], alpha=alpha, r=r)
    dW_orig = materialize_delta(a["q_proj"]["A"], a["q_proj"]["B"], alpha, r)
    dW_new = materialize_delta(delta_result["q_proj"]["A"], delta_result["q_proj"]["B"], alpha, r)
    np.testing.assert_allclose(dW_new, dW_orig, atol=1e-6)


def test_adapter_drift_zero_for_identical_adapters():
    modules = ["q_proj"]
    a = _random_adapter(modules, r=4, in_dim=8, out_dim=8, seed=5)
    assert adapter_drift(a, a, alpha=8, r=4) == pytest.approx(0.0, abs=1e-8)


def test_adapter_drift_positive_for_different_adapters():
    modules = ["q_proj"]
    a1 = _random_adapter(modules, r=4, in_dim=8, out_dim=8, seed=1)
    a2 = _random_adapter(modules, r=4, in_dim=8, out_dim=8, seed=2)
    assert adapter_drift(a1, a2, alpha=8, r=4) > 0


def test_average_adapters_dispatch():
    modules = ["q_proj"]
    a1 = _random_adapter(modules, r=4, in_dim=8, out_dim=8, seed=1)
    a2 = _random_adapter(modules, r=4, in_dim=8, out_dim=8, seed=2)

    _, info = average_adapters([a1, a2], mode="params")
    assert info == {}

    _, info = average_adapters([a1, a2], mode="delta", alpha=8, r=4)
    assert "reconstruction_error" in info

    with pytest.raises(ValueError):
        average_adapters([a1, a2], mode="bogus")

    with pytest.raises(ValueError):
        average_adapters([a1, a2], mode="delta")  # missing alpha/r


def test_mismatched_target_modules_raises():
    a1 = _random_adapter(["q_proj"], r=4, in_dim=8, out_dim=8, seed=1)
    a2 = _random_adapter(["k_proj"], r=4, in_dim=8, out_dim=8, seed=2)
    with pytest.raises(ValueError):
        average_params([a1, a2])


def test_refactor_svd_device_matches_numpy_path():
    # Skipped where torch isn't installed (this scaffold's local venv) —
    # runs for real on the cluster, where svd_device="cuda" is what a
    # real p2p_network run actually uses (CPU numpy SVD measured ~16s on
    # an MLP-projection-sized real matrix; ~200 such per agent per round
    # made delta mode impractical until this existed).
    pytest.importorskip("torch")
    A = RNG.normal(size=(8, 32)).astype(np.float64)
    B = RNG.normal(size=(24, 8)).astype(np.float64)
    dW = materialize_delta(A, B, alpha=16, r=8)

    A_cpu, B_cpu, err_cpu = refactor_svd(dW, r=8, alpha=16, svd_device=None)
    A_dev, B_dev, err_dev = refactor_svd(dW, r=8, alpha=16, svd_device="cpu")

    # SVD sign/ordering can differ slightly between backends; compare the
    # reconstructed dW, not A/B directly.
    dW_hat_cpu = materialize_delta(A_cpu, B_cpu, alpha=16, r=8)
    dW_hat_dev = materialize_delta(A_dev, B_dev, alpha=16, r=8)
    np.testing.assert_allclose(dW_hat_cpu, dW_hat_dev, atol=1e-4)
    assert abs(err_cpu - err_dev) < 1e-4
