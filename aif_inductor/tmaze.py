"""Ground-truth T-maze world and a data generator.

The world (hidden from the LLM):
  hidden factor 0: location in {0: centre, 1: left arm, 2: right arm, 3: cue spot}
  hidden factor 1: context  in {0: reward is left, 1: reward is right}, fixed per episode

Observations (the LLM only sees these, with neutral names o0, o1, o2):
  o0: location sensor, 4 values, always equals the true location
  o1: arm signal, 3 values {0: nothing, 1: reward, 2: punishment}
  o2: cue-spot signal, 3 values {0: nothing, 1: cue A, 2: cue B}

Actions: 4 values, "go to location k". The two arms are absorbing:
once in an arm, the agent stays there whatever it does.
"""
import numpy as np

N_LOC, N_CTX, N_ACT = 4, 2, 4
REWARD_PROB = 0.98
CUE_VALIDITY = 0.95
EPISODE_LEN = 3  # number of actions per episode (so 4 observations)


def true_model():
    """The true generative model, written in the same format the LLM must produce."""
    A_loc = np.zeros((N_LOC, N_LOC, N_CTX))
    for l in range(N_LOC):
        A_loc[l, l, :] = 1.0

    A_arm = np.zeros((3, N_LOC, N_CTX))
    A_arm[0, [0, 3], :] = 1.0
    for arm, good_ctx in [(1, 0), (2, 1)]:
        for ctx in range(N_CTX):
            p = REWARD_PROB if ctx == good_ctx else 1 - REWARD_PROB
            A_arm[1, arm, ctx] = p
            A_arm[2, arm, ctx] = 1 - p

    A_cue = np.zeros((3, N_LOC, N_CTX))
    A_cue[0, [0, 1, 2], :] = 1.0
    A_cue[1, 3, 0], A_cue[2, 3, 0] = CUE_VALIDITY, 1 - CUE_VALIDITY
    A_cue[2, 3, 1], A_cue[1, 3, 1] = CUE_VALIDITY, 1 - CUE_VALIDITY

    B_loc = np.zeros((N_LOC, N_LOC, N_ACT))
    for a in range(N_ACT):
        for s in range(N_LOC):
            B_loc[s if s in (1, 2) else a, s, a] = 1.0
    B_ctx = np.eye(N_CTX)[:, :, None]

    D_loc = np.array([1.0, 0, 0, 0])
    D_ctx = np.array([0.5, 0.5])

    return dict(
        factors=[("location", N_LOC, True), ("context", N_CTX, False)],
        A=[A_loc, A_arm, A_cue],
        A_deps=[[0, 1], [0, 1], [0, 1]],
        B=[B_loc, B_ctx],
        D=[D_loc, D_ctx],
    )


class TMazeEnv:
    """Samples observations from the true model. Never exposes the context to agents."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.m = true_model()

    def _observe(self):
        loc, ctx = self.loc, self.ctx
        return tuple(int(self.rng.choice(A.shape[0], p=A[:, loc, ctx])) for A in self.m["A"])

    def reset(self):
        self.loc = 0
        self.ctx = int(self.rng.integers(N_CTX))
        return self._observe()

    def step(self, action):
        self.loc = int(np.argmax(self.m["B"][0][:, self.loc, action]))
        return self._observe()


def generate_trajectories(n_episodes, seed=0, policy="random"):
    """Return a list of episodes. Each episode is a dict with 'obs' (list of
    (o0,o1,o2) tuples, length T+1) and 'actions' (list of ints, length T).
    Hidden states are NOT recorded, matching Pinductor's setting.

    policy="random": every action uniform over the 4 actions.
    policy="cue_first": go to the cue spot (action 3), then to an arm chosen
    uniformly (action 1 or 2), then a uniform action. Arms are absorbing, so
    the last action does not move the agent. The policy never reads the context."""
    if policy not in ("random", "cue_first"):
        raise ValueError(f"unknown policy {policy!r}")
    env = TMazeEnv(seed)
    rng = np.random.default_rng(seed + 10_000)
    data = []
    for _ in range(n_episodes):
        obs = [env.reset()]
        actions = []
        for _ in range(EPISODE_LEN):
            if policy == "cue_first" and len(actions) == 0:
                a = 3
            elif policy == "cue_first" and len(actions) == 1:
                a = int(rng.integers(1, 3))
            else:
                a = int(rng.integers(N_ACT))
            actions.append(a)
            obs.append(env.step(a))
        data.append(dict(obs=obs, actions=actions))
    return data


def format_trajectories(data, max_episodes=None):
    """Plain-text rendering of episodes for the LLM prompt."""
    lines = []
    for i, ep in enumerate(data[:max_episodes]):
        parts = [f"o={ep['obs'][0]}"]
        for a, o in zip(ep["actions"], ep["obs"][1:]):
            parts.append(f"a={a} -> o={o}")
        lines.append(f"episode {i}: " + " | ".join(parts))
    return "\n".join(lines)
