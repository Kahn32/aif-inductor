"""Does a model link the cue to the reward? Answered from the model's own
predictions, so it works however the model names or merges its states.

The question asked of the model: go to the cue spot (action 3) and see cue
value c, then go to the left arm (action 1). How likely is the reward
signal (o1 = 1)? If the cue is linked, that probability depends on c.

  link = p(o1=1 | cue 1, left) - p(o1=1 | cue 2, left)

The true model gives about +0.86. A model with no link gives 0. A model that
links the cue the wrong way round gives a negative value.
We call a model linked when |link| >= LINK_THRESHOLD. 0.5 sits well clear of
both 0 and 0.86, so small numeric noise cannot flip the verdict, and a model
whose cue predicts the arm only weakly does not count.
"""
import numpy as np

from .model_spec import flatten

LINK_THRESHOLD = 0.5
CUE_ACTION, LEFT_ACTION = 3, 1


def cue_link(spec, eps=1e-3):
    """Return dict(link=float, linked=bool, correct_sign=bool)."""
    A, B, D = flatten(spec, eps)
    at_cue = B[:, :, CUE_ACTION] @ D
    p_reward = []
    for cue in (1, 2):
        post = A[2][cue] * at_cue
        post = post / post.sum()
        in_left = B[:, :, LEFT_ACTION] @ post
        p_reward.append(float(A[1][1] @ in_left))
    link = p_reward[0] - p_reward[1]
    return dict(link=link, linked=abs(link) >= LINK_THRESHOLD, correct_sign=link > 0)
