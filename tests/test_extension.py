"""Checks for the extension: cue_first demonstrations and cue_link.
Run: python tests/test_extension.py"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aif_inductor.analysis import cue_link
from aif_inductor.induce import run_code
from aif_inductor.llm import MOCK_PROPOSALS
from aif_inductor.model_spec import flatten, validate
from aif_inductor.tmaze import generate_trajectories, true_model


def test_random_data_unchanged():
    """Adding a policy must not change the data every earlier run used."""
    data = json.dumps([generate_trajectories(30, seed=s) for s in range(3)])
    assert hashlib.sha256(data.encode()).hexdigest()[:16] == "63b8fdbdc4ebec84"


def test_cue_first_actions():
    for ep in generate_trajectories(200, seed=4, policy="cue_first"):
        assert ep["actions"][0] == 3 and ep["actions"][1] in (1, 2)
        assert ep["obs"][1][0] == 3 and ep["obs"][2][0] == ep["actions"][1]


def test_cue_first_format_matches_random():
    a = generate_trajectories(3, seed=0)[0]
    b = generate_trajectories(3, seed=0, policy="cue_first")[0]
    assert a.keys() == b.keys() == {"obs", "actions"}
    assert len(a["obs"]) == len(b["obs"]) and len(a["actions"]) == len(b["actions"])
    assert all(len(o) == 3 and all(isinstance(x, int) for x in o) for o in b["obs"])


def test_cue_first_uses_both_arms():
    arms = [ep["actions"][1] for ep in generate_trajectories(200, seed=5, policy="cue_first")]
    assert 60 < arms.count(1) < 140


def test_unknown_policy_rejected():
    try:
        generate_trajectories(1, policy="oracle")
    except ValueError:
        return
    raise AssertionError("unknown policy accepted")


def test_true_model_is_linked():
    r = cue_link(validate(true_model())[0])
    assert r["linked"] and r["correct_sign"] and 0.8 < r["link"] < 0.9


def test_mock_models():
    got = [cue_link(validate(run_code(p.split("```python")[1].split("```")[0]))[0])
           for p in MOCK_PROPOSALS]
    assert [g["linked"] for g in got] == [False, False, True]
    assert abs(got[0]["link"]) < 1e-6 and abs(got[1]["link"]) < 1e-6


def merged_true_model():
    """The true model as one 8-state controlled factor."""
    A, B, D = flatten(validate(true_model())[0], eps=0)
    return dict(factors=[("joint", 8, True)], A=A, A_deps=[[0], [0], [0]], B=[B], D=[D])


def test_merged_factor_model_is_linked():
    spec = validate(merged_true_model())[0]
    assert cue_link(spec)["linked"] and cue_link(spec)["correct_sign"]


def test_wrong_sign_is_linked_but_flagged():
    m = true_model()
    m["A"][2] = m["A"][2][[0, 2, 1]]  # swap the two cue values
    r = cue_link(validate(m)[0])
    assert r["linked"] and not r["correct_sign"] and r["link"] < -0.8


def test_relabelling_does_not_change_link():
    m = true_model()
    m["A"] = [a[:, :, ::-1] for a in m["A"]]
    assert np.isclose(cue_link(validate(m)[0])["link"], cue_link(validate(true_model())[0])["link"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
