"""Checking and flattening a candidate model.

A candidate model is a dict:
  factors : list of (name, size, controlled) ; exactly one factor has controlled=True
  A       : list with one array per observation modality; A[m].shape == (n_obs_m, *sizes of A_deps[m])
  A_deps  : list of lists of factor indices each modality depends on
  B       : list with one array per factor; B[f].shape == (size_f, size_f, n_actions if controlled else 1)
  D       : list with one prior vector per factor

This is the same factorised format pymdp uses, so a valid candidate can be handed
straight to a pymdp Agent.
"""
import itertools
import numpy as np

N_OBS = [4, 3, 3]
N_ACT = 4
MAX_JOINT_STATES = 64  # flatten is quadratic in this; the true model needs 8


class InvalidModel(Exception):
    pass


def validate(spec, n_obs=N_OBS, n_act=N_ACT):
    """Return a cleaned copy of the spec (float arrays, columns normalised),
    plus a list of warnings. Raise InvalidModel with a readable message if broken."""
    warnings = []
    try:
        factors = [(str(n), int(s), bool(c)) for n, s, c in spec["factors"]]
        A = [np.asarray(a, dtype=float) for a in spec["A"]]
        A_deps = [[int(f) for f in deps] for deps in spec["A_deps"]]
        B = [np.asarray(b, dtype=float) for b in spec["B"]]
        D = [np.asarray(d, dtype=float) for d in spec["D"]]
    except Exception as e:
        raise InvalidModel(f"could not read model dict: {e!r}")

    sizes = [s for _, s, _ in factors]
    if not sizes or any(s < 1 for s in sizes):
        raise InvalidModel("need at least one factor, each of size 1 or more")
    if int(np.prod(sizes, dtype=object)) > MAX_JOINT_STATES:
        raise InvalidModel(f"joint state space is too large: factor sizes {sizes}, "
                           f"limit {MAX_JOINT_STATES} joint states")
    n_f = len(factors)
    if sum(c for _, _, c in factors) != 1:
        raise InvalidModel("exactly one factor must be controlled by the action")
    if len(A) != len(n_obs) or len(A_deps) != len(n_obs):
        raise InvalidModel(f"need exactly {len(n_obs)} A arrays and A_deps entries")
    if len(B) != n_f or len(D) != n_f:
        raise InvalidModel("need one B array and one D vector per factor")

    for m, (a, deps) in enumerate(zip(A, A_deps)):
        if any(f < 0 or f >= n_f for f in deps) or len(set(deps)) != len(deps):
            raise InvalidModel(f"A_deps[{m}]={deps} refers to unknown or repeated factors")
        want = (n_obs[m], *[sizes[f] for f in deps])
        if a.shape != want:
            raise InvalidModel(f"A[{m}] has shape {a.shape}, expected {want}")
        if (a < 0).any() or not np.isfinite(a).all():
            raise InvalidModel(f"A[{m}] has negative or non-finite entries")
        col = a.sum(0)
        if (col <= 0).any():
            raise InvalidModel(f"A[{m}] has an all-zero column")
        if not np.allclose(col, 1, atol=1e-3):
            warnings.append(f"A[{m}] columns did not sum to 1; renormalised")
        A[m] = a / col

    for f, (b, (name, s, ctrl)) in enumerate(zip(B, factors)):
        want = (s, s, n_act if ctrl else 1)
        if b.shape != want:
            raise InvalidModel(f"B[{f}] ({name}) has shape {b.shape}, expected {want}")
        if (b < 0).any() or not np.isfinite(b).all():
            raise InvalidModel(f"B[{f}] has negative or non-finite entries")
        col = b.sum(0)
        if (col <= 0).any():
            raise InvalidModel(f"B[{f}] has an all-zero column")
        if not np.allclose(col, 1, atol=1e-3):
            warnings.append(f"B[{f}] columns did not sum to 1; renormalised")
        B[f] = b / col

    for f, d in enumerate(D):
        if d.shape != (sizes[f],) or (d < 0).any() or d.sum() <= 0:
            raise InvalidModel(f"D[{f}] must be a non-negative vector of length {sizes[f]}")
        D[f] = d / d.sum()

    return dict(factors=factors, A=A, A_deps=A_deps, B=B, D=D), warnings


def flatten(spec, eps=1e-3):
    """Turn a factorised model into a flat HMM over joint states.

    Returns A_joint (list of (n_obs_m, S)), B_joint (S, S, n_act), D_joint (S,).
    eps mixes in a little uniform noise so that no observation is ever
    impossible (otherwise one bad step gives log 0 = -inf).
    """
    sizes = [s for _, s, _ in spec["factors"]]
    ctrl = [c for _, _, c in spec["factors"]]
    joint = list(itertools.product(*[range(s) for s in sizes]))
    S = len(joint)

    A_joint = []
    for a, deps in zip(spec["A"], spec["A_deps"]):
        Aj = np.stack([a[(slice(None), *[js[f] for f in deps])] for js in joint], axis=1)
        Aj = (1 - eps) * Aj + eps / Aj.shape[0]
        A_joint.append(Aj)

    n_act = spec["B"][ctrl.index(True)].shape[2]
    B_joint = np.ones((S, S, n_act))
    for act in range(n_act):
        for i, nxt in enumerate(joint):
            for j, cur in enumerate(joint):
                p = 1.0
                for f, b in enumerate(spec["B"]):
                    p *= b[nxt[f], cur[f], act if ctrl[f] else 0]
                B_joint[i, j, act] = p
    B_joint = (1 - eps) * B_joint + eps / S

    D_joint = np.ones(S)
    for i, js in enumerate(joint):
        for f, d in enumerate(spec["D"]):
            D_joint[i] *= d[js[f]]
    return A_joint, B_joint, D_joint


def n_free_params(spec):
    """Rough count of free parameters (for reporting model size in the trace)."""
    n = sum(a.size - a[0].size for a in spec["A"])
    n += sum(b.size - b[0].size for b in spec["B"])
    n += sum(d.size - 1 for d in spec["D"])
    return int(n)


def describe(spec):
    """One-line structural summary, used in the change trace."""
    fac = ", ".join(f"{n}[{s}]{'*' if c else ''}" for n, s, c in spec["factors"])
    deps = ", ".join(f"o{m}<-{d}" for m, d in enumerate(spec["A_deps"]))
    return f"factors: {fac} | deps: {deps}"
