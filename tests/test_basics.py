"""Sanity checks. Run: python -m pytest -q   (or python tests/test_basics.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aif_inductor.model_spec import flatten, validate
from aif_inductor.scoring import score
from aif_inductor.tmaze import generate_trajectories, true_model


def test_true_model_is_valid():
    spec, warns = validate(true_model())
    assert warns == []


def test_flat_model_is_normalised():
    spec, _ = validate(true_model())
    A, B, D = flatten(spec)
    for a in A:
        assert np.allclose(a.sum(0), 1)
    assert np.allclose(B.sum(0), 1) and np.isclose(D.sum(), 1)


def test_likelihood_ignores_state_relabelling():
    """Swapping the order of the context states must not change the score."""
    spec, _ = validate(true_model())
    perm = dict(spec)
    perm["A"] = [a[:, :, ::-1] for a in spec["A"]]
    perm["D"] = [spec["D"][0], spec["D"][1][::-1]]
    data = generate_trajectories(20, seed=1)
    assert np.isclose(score(spec, data)[0], score(perm, data)[0])


def test_true_model_beats_context_free_model():
    spec, _ = validate(true_model())
    data = generate_trajectories(50, seed=2)
    flat = dict(spec)
    flat["A"] = [a.mean(axis=2, keepdims=True).repeat(2, axis=2) for a in spec["A"]]
    assert score(spec, data)[0] > score(flat, data)[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
