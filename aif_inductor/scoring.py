"""Exact scoring of a candidate model by the HMM forward algorithm.

Pinductor needs a particle filter and a hand-made distance kernel because its
LLM writes arbitrary simulator code. Here the model is a small discrete POMDP,
so we can compute the exact probability of the observed data:

  log p(o_{0:T} | a_{0:T-1}, model) = sum_t log p(o_t | o_{<t}, a_{<t}, model)

Each term is one step of the forward algorithm (Bayesian filtering).
It does not care how the LLM orders or names its hidden states.
"""
import numpy as np

from .model_spec import flatten


def forward(A_joint, B_joint, D_joint, episode):
    """Run the forward algorithm on one episode.

    Returns per-step log-likelihoods, per-step per-modality predictive
    distributions, and the per-step posterior beliefs."""
    belief = D_joint.copy()
    logps, preds, posts = [], [], []
    for t, o in enumerate(episode["obs"]):
        if t > 0:
            belief = B_joint[:, :, episode["actions"][t - 1]] @ belief  # predict
        pred = [A @ belief for A in A_joint]  # predictive dist of each modality
        lik = np.ones_like(belief)
        for A, o_m in zip(A_joint, o):
            lik *= A[o_m]
        p = float(lik @ belief)
        logps.append(np.log(p))
        preds.append(pred)
        belief = lik * belief / p  # update (Bayes' rule)
        posts.append(belief)
    return np.array(logps), preds, posts


def score(spec, data, eps=1e-3):
    """Mean log-likelihood per observation step (higher is better), plus details."""
    A_j, B_j, D_j = flatten(spec, eps)
    per_ep = [forward(A_j, B_j, D_j, ep) for ep in data]
    all_logps = np.concatenate([lp for lp, _, _ in per_ep])
    return float(all_logps.mean()), per_ep


def diagnostics(spec, data, k=5, eps=1e-3):
    """Text feedback for the LLM: the k worst-predicted steps and where the
    lost probability is concentrated (which modality)."""
    _, per_ep = score(spec, data, eps)
    rows = []
    modality_loss = np.zeros(len(data[0]["obs"][0]))
    for i, (ep, (lps, preds, _)) in enumerate(zip(data, per_ep)):
        for t, (lp, pred) in enumerate(zip(lps, preds)):
            o = ep["obs"][t]
            for m, o_m in enumerate(o):
                modality_loss[m] -= np.log(pred[m][o_m])
            rows.append((lp, i, t, o, pred))
    rows.sort(key=lambda r: r[0])

    lines = ["Share of surprise (negative log-prob) by modality: "
             + ", ".join(f"o{m}: {100 * l / modality_loss.sum():.0f}%"
                         for m, l in enumerate(modality_loss))]
    lines.append(f"The {k} observations your model found most surprising:")
    for lp, i, t, o, pred in rows[:k]:
        ep = data[i]
        hist = ", ".join(f"a={a}" for a in ep["actions"][:t]) or "start"
        pred_txt = "; ".join(
            f"o{m}: predicted {np.round(p, 3).tolist()}, saw {o[m]}" for m, p in enumerate(pred)
        )
        lines.append(
            f"- episode {i}, step {t} (history: {hist}; earlier obs: {ep['obs'][:t]}): "
            f"p(obs)={np.exp(lp):.4f}. {pred_txt}"
        )
    return "\n".join(lines)
