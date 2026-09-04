"""Adapter averaging: params-mode and delta-mode.

Spec section 4's "one technical subtlety that matters": averaging LoRA A and
B matrices separately is NOT the same as averaging the effective weight
updates, because mean(B) @ mean(A) != mean(B @ A). Both modes are
implemented behind `average_mode`; the main experiment runs in "delta".

An adapter is represented as:
    dict[module_name, dict["A" | "B", array]]
where for module m, A has shape (r, in_features) and B has shape
(out_features, r), matching the PEFT LoRA convention dW = (alpha/r) * B @ A.

This module is written against plain arrays (numpy here) so the averaging
math is unit-testable without a GPU or the transformers/peft stack.

The SVD in refactor_svd is the one place that turned out to matter for
real model sizes: a single CPU numpy SVD on an MLP-projection-sized
matrix (1536x8960, Qwen2.5-1.5B) took ~16s in practice — with ~200 LoRA
layers per agent, that's tens of minutes per agent per round. `svd_device`
routes just that one operation through torch on a GPU when given (e.g.
"cuda"), a ~40x speedup measured on the same matrix size, while leaving
everything else — including every existing caller and test that doesn't
pass it — on plain numpy.
"""
from __future__ import annotations

from typing import Any

import numpy as np

Adapter = dict[str, dict[str, np.ndarray]]


def _module_names(adapters: list[Adapter]) -> list[str]:
    names = set(adapters[0].keys())
    for a in adapters[1:]:
        if set(a.keys()) != names:
            raise ValueError("adapters have mismatched target modules")
    return sorted(names)


def average_params(adapters: list[Adapter]) -> Adapter:
    """params-mode: average A across adapters, average B across adapters,
    separately. Cheap, common, technically wrong (see module docstring)."""
    if not adapters:
        raise ValueError("adapters must be non-empty")
    out: Adapter = {}
    for m in _module_names(adapters):
        out[m] = {
            "A": np.mean([a[m]["A"] for a in adapters], axis=0),
            "B": np.mean([a[m]["B"] for a in adapters], axis=0),
        }
    return out


def materialize_delta(A: np.ndarray, B: np.ndarray, alpha: float, r: int) -> np.ndarray:
    """dW = (alpha / r) * B @ A for one module."""
    return (alpha / r) * (B @ A)


def refactor_svd(
    dW: np.ndarray, r: int, alpha: float, svd_device: str | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Refactor a dense update dW back into rank-r LoRA A, B via truncated SVD.

    dW ~= U_r S_r V_r^T. Absorb the singular values evenly into A and B so
    that (alpha/r) * B @ A reconstructs dW as closely as rank r allows:
        B = U_r sqrt(S_r), A = sqrt(S_r) V_r^T, scaled by sqrt(r/alpha).

    Args:
        svd_device: None (default) does the SVD on CPU via numpy — what
            every test uses. Pass e.g. "cuda" to do just this step via
            torch on that device instead (see module docstring for why).

    Returns:
        (A, B, relative_reconstruction_error) where the error is
        ||dW_hat - dW||_F / ||dW||_F for dW_hat = (alpha/r) * B @ A.
    """
    if svd_device is not None:
        import torch

        dW_t = torch.from_numpy(dW).to(svd_device)
        U_t, S_t, Vt_t = torch.linalg.svd(dW_t, full_matrices=False)
        U, S, Vt = U_t.cpu().numpy(), S_t.cpu().numpy(), Vt_t.cpu().numpy()
    else:
        U, S, Vt = np.linalg.svd(dW, full_matrices=False)
    r_eff = min(r, S.shape[0])
    U_r, S_r, Vt_r = U[:, :r_eff], S[:r_eff], Vt[:r_eff, :]

    scale = np.sqrt(r / alpha)
    sqrt_S = np.sqrt(S_r)
    B = U_r * sqrt_S[None, :] * scale
    A = sqrt_S[:, None] * Vt_r * scale

    if r_eff < r:
        pad_out = r - r_eff
        B = np.pad(B, ((0, 0), (0, pad_out)))
        A = np.pad(A, ((0, pad_out), (0, 0)))

    dW_hat = materialize_delta(A, B, alpha, r)
    denom = np.linalg.norm(dW)
    rel_err = float(np.linalg.norm(dW_hat - dW) / denom) if denom > 0 else 0.0
    return A, B, rel_err


def average_delta(
    adapters: list[Adapter], alpha: float, r: int, svd_device: str | None = None
) -> tuple[Adapter, dict[str, float]]:
    """delta-mode: materialize dW per module per adapter, average the dW's,
    refactor back to rank r via truncated SVD.

    Returns:
        (new_adapter, reconstruction_errors) where reconstruction_errors
        maps module name to the relative SVD reconstruction error, for the
        "bounded reconstruction error" test required by spec section 9.
    """
    if not adapters:
        raise ValueError("adapters must be non-empty")
    out: Adapter = {}
    errors: dict[str, float] = {}
    for m in _module_names(adapters):
        dWs = [materialize_delta(a[m]["A"], a[m]["B"], alpha, r) for a in adapters]
        dW_mean = np.mean(dWs, axis=0)
        A, B, err = refactor_svd(dW_mean, r, alpha, svd_device=svd_device)
        out[m] = {"A": A, "B": B}
        errors[m] = err
    return out, errors


def average_adapters(
    adapters: list[Adapter],
    mode: str,
    alpha: float | None = None,
    r: int | None = None,
    svd_device: str | None = None,
) -> tuple[Adapter, dict[str, Any]]:
    """Dispatch on `average_mode` ("params" or "delta"). This is the entry
    point Agent.average_step (Phase 2) should call.

    Returns (new_adapter, info) where info is empty for "params" mode and
    holds per-module reconstruction errors for "delta" mode.
    """
    if mode == "params":
        return average_params(adapters), {}
    if mode == "delta":
        if alpha is None or r is None:
            raise ValueError("delta mode requires alpha and r")
        adapter, errors = average_delta(adapters, alpha, r, svd_device=svd_device)
        return adapter, {"reconstruction_error": errors}
    raise ValueError(f"unknown average_mode: {mode}")


def adapter_drift(a: Adapter, b: Adapter, alpha: float, r: int) -> float:
    """Mean pairwise distance between two agents' effective dW, averaged
    over target modules (spec section 6, "Adapter drift" metric)."""
    modules = _module_names([a, b])
    dists = []
    for m in modules:
        dW_a = materialize_delta(a[m]["A"], a[m]["B"], alpha, r)
        dW_b = materialize_delta(b[m]["A"], b[m]["B"], alpha, r)
        dists.append(float(np.linalg.norm(dW_a - dW_b)))
    return float(np.mean(dists))
