"""Plug a model (true or induced) into a pymdp active inference agent and
measure how well it does in the real T-maze.

The agent plans by minimising expected free energy:
  G(policy) = risk (distance from preferred outcomes) + ambiguity
so it goes to the cue spot first when that resolves uncertainty about the arms.
Preferences C are given (they are the task, not part of the world model):
the agent likes o1=1 and dislikes o1=2. The penalty is larger than the reward,
so a blind 50/50 guess is unattractive and visiting the cue is worth it.

Where pymdp 1.0.5 computes this (pymdp/control.py):
  compute_neg_efe_policy (line 622) adds, per time step (line 700),
  compute_info_gain (line 467), which is H[q(o)] minus ambiguity, and
  compute_expected_utility (line 501), which is E_q(o)[C].
Risk and ambiguity are never computed separately: risk = -utility - H[q(o)],
so -G = utility + info gain. C is not normalised, which shifts every policy
by the same constant and does not change the choice.
"""
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from pymdp.agent import Agent

from .tmaze import EPISODE_LEN, N_ACT, TMazeEnv

C_ARM = [0.0, 2.0, -4.0]


def make_agent(spec, policy_len=1):
    ctrl_f = [c for _, _, c in spec["factors"]].index(True)
    A = [jnp.asarray(a)[None] for a in spec["A"]]
    B = [jnp.asarray(b)[None] for b in spec["B"]]
    D = [jnp.asarray(d)[None] for d in spec["D"]]
    C = [jnp.zeros((1, 4)), jnp.asarray([C_ARM]), jnp.zeros((1, 3))]
    num_controls = [b.shape[-1] for b in spec["B"]]
    return Agent(A=A, B=B, C=C, D=D, A_dependencies=spec["A_deps"],
                 B_dependencies=[[f] for f in range(len(spec["B"]))],
                 num_controls=num_controls, control_fac_idx=[ctrl_f],
                 policy_len=policy_len), ctrl_f


def first_arm_outcome(env_ctx, locs):
    """1 if the first arm the agent entered was the rewarded one, 0 if wrong,
    None if it never entered an arm."""
    for l in locs:
        if l in (1, 2):
            return int((l == 1 and env_ctx == 0) or (l == 2 and env_ctx == 1))
    return None


@eqx.filter_jit
def _step(agent, obs, prior, key):
    """One perception-planning-action cycle, compiled once per agent."""
    qs = agent.infer_states(obs, prior)
    q_pi, _ = agent.infer_policies(qs)
    action = agent.sample_action(q_pi, rng_key=jr.split(key, 1))
    return action, agent.update_empirical_prior(action, qs)


def run_agent(spec, n_episodes=100, seed=0):
    """Returns dict with success rate (first arm correct), arm-entry rate,
    cue-visit rate, mean return (reward obs minus punishment obs), and
    cue_then_success: visited the cue before entering an arm, and that arm
    was the rewarded one."""
    agent, ctrl_f = make_agent(spec)
    env = TMazeEnv(seed + 777)
    key = jr.PRNGKey(seed)
    stats = {"success": [], "entered_arm": [], "visited_cue": [], "return": [],
             "cue_then_success": []}
    for _ in range(n_episodes):
        o = env.reset()
        prior = agent.D
        locs, ret = [], 0
        for _ in range(EPISODE_LEN):
            obs = [jnp.array([x]) for x in o]
            key, sub = jr.split(key)
            action, prior = _step(agent, obs, prior, sub)
            a = int(action[0, ctrl_f])
            o = env.step(a)
            locs.append(env.loc)
            ret += (o[1] == 1) - (o[1] == 2)
        out = first_arm_outcome(env.ctx, locs)
        stats["entered_arm"].append(out is not None)
        stats["success"].append(out == 1)
        stats["visited_cue"].append(3 in locs)
        stats["return"].append(ret)
        stats["cue_then_success"].append(out == 1 and cue_before_arm(locs))
    return {k: float(np.mean(v)) for k, v in stats.items()}


def cue_before_arm(locs):
    for l in locs:
        if l == 3:
            return True
        if l in (1, 2):
            return False
    return False


def run_random(n_episodes=100, seed=0):
    env = TMazeEnv(seed + 777)
    rng = np.random.default_rng(seed)
    stats = {"success": [], "entered_arm": [], "visited_cue": [], "return": []}
    for _ in range(n_episodes):
        env.reset()
        locs, ret = [], 0
        for _ in range(EPISODE_LEN):
            o = env.step(int(rng.integers(N_ACT)))
            locs.append(env.loc)
            ret += (o[1] == 1) - (o[1] == 2)
        out = first_arm_outcome(env.ctx, locs)
        stats["entered_arm"].append(out is not None)
        stats["success"].append(out == 1)
        stats["visited_cue"].append(3 in locs)
        stats["return"].append(ret)
    return {k: float(np.mean(v)) for k, v in stats.items()}
