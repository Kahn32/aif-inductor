"""LLM clients. OpenRouterLLM calls a real model; MockLLM replays scripted
proposals so the whole pipeline can be tested offline for free."""
import os
import time

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMUnavailable(RuntimeError):
    """The LLM cannot be reached now; stop cleanly and keep what exists."""


class OpenRouterLLM:
    """Minimal chat client for any OpenAI-compatible endpoint.

    Default: OpenRouter. Free models there have IDs ending in ":free"
    (no credit needed; limits are 20 requests/minute and about 50 requests/day).
    Set OPENROUTER_API_KEY in your shell.

    Local alternative, also free: Ollama. Set LLM_BASE_URL=http://localhost:11434/v1/chat/completions
    and pass the Ollama model name (e.g. a Qwen coder model). No key needed then."""

    def __init__(self, model, temperature=0.7, max_tokens=4000, min_gap_s=3.5,
                 max_attempts=3, retry_wait_s=10, rate_wait_s=60, seed=None,
                 reasoning_effort=None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.url = os.environ.get("LLM_BASE_URL", OPENROUTER_URL)
        self.key = os.environ.get("OPENROUTER_API_KEY", "")
        if self.url == OPENROUTER_URL and not self.key:
            raise RuntimeError("Set OPENROUTER_API_KEY first (export OPENROUTER_API_KEY=...)")
        self.min_gap_s = min_gap_s  # stay under 20 requests/minute
        self._last = 0.0
        self.last_served = None
        self.seed = seed  # best effort: providers may ignore it
        self.reasoning_effort = reasoning_effort  # reasoning tokens count against max_tokens
        self.max_attempts = max_attempts
        self.retry_wait_s = retry_wait_s
        self.rate_wait_s = rate_wait_s
        # calls = successful replies; attempts = every HTTP request sent
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "attempts": 0}

    def __call__(self, system, user):
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        for attempt in range(self.max_attempts):
            wait = self.min_gap_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            self.usage["attempts"] += 1
            try:
                r = requests.post(self.url, json=body, timeout=300,
                                  headers={"Authorization": f"Bearer {self.key}"} if self.key else {})
                if r.status_code == 429:
                    if "per-day" in r.text or "free-models-per-day" in r.text:
                        raise LLMUnavailable("Daily free-model limit reached. Try again tomorrow (resets 00:00 UTC).")
                    print(f"  rate limited (429), waiting {self.rate_wait_s} s...")
                    time.sleep(self.rate_wait_s)
                    continue
                r.raise_for_status()
                j = r.json()
                u = j.get("usage", {})
                self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                self.usage["completion_tokens"] += u.get("completion_tokens", 0)
                self.usage["calls"] += 1
                choice = j["choices"][0]
                self.last_served = {"model": j.get("model"), "provider": j.get("provider"),
                                    "finish_reason": choice.get("finish_reason")}
                return choice["message"]["content"]
            except Exception as e:
                if isinstance(e, LLMUnavailable):
                    raise
                if attempt == self.max_attempts - 1:
                    raise LLMUnavailable(f"LLM call failed: {e!r}") from e
                print(f"  LLM call failed ({e!r}), retrying...")
                time.sleep(self.retry_wait_s * (attempt + 1))
        raise LLMUnavailable(f"LLM call failed after {self.max_attempts} attempts (rate limits?)")


# Scripted proposals for offline testing. They mimic a plausible run:
# 1) no hidden context, 2) context found but cue not linked to it, 3) correct.

_COMMON_B = '''
    B_loc = np.zeros((4, 4, 4))
    for a in range(4):
        for s in range(4):
            B_loc[s if s in (1, 2) else a, s, a] = 1.0
'''

MOCK_PROPOSALS = [
    '''REASON: First guess. o0 looks like a position sensor that copies the position. o1 only fires at positions 1 and 2, with 1 and 2 about equally often, so I model it as a coin flip there. o2 only fires at position 3, also roughly 50/50. Positions 1 and 2 seem to be traps you cannot leave.
```python
import numpy as np
def build_model():
    factors = [("position", 4, True)]
    A0 = np.eye(4)
    A1 = np.zeros((3, 4)); A1[0, [0, 3]] = 1; A1[1, [1, 2]] = 0.5; A1[2, [1, 2]] = 0.5
    A2 = np.zeros((3, 4)); A2[0, [0, 1, 2]] = 1; A2[1, 3] = 0.5; A2[2, 3] = 0.5
''' + _COMMON_B + '''
    D = [np.array([1.0, 0, 0, 0])]
    return dict(factors=factors, A=[A0, A1, A2], A_deps=[[0], [0], [0]], B=[B_loc], D=D)
```''',
    '''REASON: The worst steps are o1 values at positions 1 and 2. Inside one episode, o1 at an arm keeps the same value step after step, but which value differs between episodes, and an episode where position 1 gives 1 is one where position 2 would give 2. That looks like a hidden binary variable fixed per episode. I add a 2-state "side" factor that never changes and make o1 depend on it.
```python
import numpy as np
def build_model():
    factors = [("position", 4, True), ("side", 2, False)]
    A0 = np.zeros((4, 4, 2))
    for l in range(4): A0[l, l, :] = 1
    A1 = np.zeros((3, 4, 2)); A1[0, [0, 3], :] = 1
    A1[1, 1, 0] = 0.97; A1[2, 1, 0] = 0.03; A1[2, 1, 1] = 0.97; A1[1, 1, 1] = 0.03
    A1[1, 2, 1] = 0.97; A1[2, 2, 1] = 0.03; A1[2, 2, 0] = 0.97; A1[1, 2, 0] = 0.03
    A2 = np.zeros((3, 4)); A2[0, [0, 1, 2]] = 1; A2[1, 3] = 0.5; A2[2, 3] = 0.5
''' + _COMMON_B + '''
    B_side = np.eye(2)[:, :, None]
    D = [np.array([1.0, 0, 0, 0]), np.array([0.5, 0.5])]
    return dict(factors=factors, A=[A0, A1, A2], A_deps=[[0, 1], [0, 1], [0]], B=[B_loc, B_side], D=D)
```''',
    '''REASON: Most remaining surprise is in o2, and o2 at position 3 predicts which arm later gives o1=1: o2=1 goes with side 0 and o2=2 with side 1, right most of the time. I make o2 depend on the side factor, with about 95% reliability.
```python
import numpy as np
def build_model():
    factors = [("position", 4, True), ("side", 2, False)]
    A0 = np.zeros((4, 4, 2))
    for l in range(4): A0[l, l, :] = 1
    A1 = np.zeros((3, 4, 2)); A1[0, [0, 3], :] = 1
    A1[1, 1, 0] = 0.97; A1[2, 1, 0] = 0.03; A1[2, 1, 1] = 0.97; A1[1, 1, 1] = 0.03
    A1[1, 2, 1] = 0.97; A1[2, 2, 1] = 0.03; A1[2, 2, 0] = 0.97; A1[1, 2, 0] = 0.03
    A2 = np.zeros((3, 4, 2)); A2[0, [0, 1, 2], :] = 1
    A2[1, 3, 0] = 0.95; A2[2, 3, 0] = 0.05; A2[2, 3, 1] = 0.95; A2[1, 3, 1] = 0.05
''' + _COMMON_B + '''
    B_side = np.eye(2)[:, :, None]
    D = [np.array([1.0, 0, 0, 0]), np.array([0.5, 0.5])]
    return dict(factors=factors, A=[A0, A1, A2], A_deps=[[0, 1], [0, 1], [0, 1]], B=[B_loc, B_side], D=D)
```''',
]


class MockLLM:
    def __init__(self, stop_after=None):
        self.i = 0
        self.stop_after = stop_after  # simulate hitting the daily cap
        self.model = "mock"
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "attempts": 0}

    def __call__(self, system, user):
        if self.stop_after is not None and self.i >= self.stop_after:
            raise LLMUnavailable("Simulated daily limit.")
        out = MOCK_PROPOSALS[min(self.i, len(MOCK_PROPOSALS) - 1)]
        self.i += 1
        self.usage["calls"] += 1
        self.usage["attempts"] += 1
        return out
