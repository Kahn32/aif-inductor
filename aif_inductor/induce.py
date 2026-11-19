"""The induction loop: LLM proposes a model as code, we score it exactly,
send back diagnostics, and log every change with its reason and effect."""
import difflib
import hashlib
import json
import pickle
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from .llm import LLMUnavailable
from .model_spec import InvalidModel, describe, n_free_params, validate
from .scoring import diagnostics, score
from .tmaze import format_trajectories

SYSTEM = textwrap.dedent("""
    You are a scientist building a world model of an unknown environment from data.
    You write the model as a small discrete POMDP in Python with numpy only.
    You reason carefully from the data, and you change the model one idea at a time.
""").strip()

TASK = textwrap.dedent("""
    An agent acts in an unknown environment. Each episode starts fresh.
    At every step it receives an observation with three parts o=(o0, o1, o2):
      o0 takes values 0..3, o1 takes values 0..2, o2 takes values 0..2.
    It has 4 actions, a in {{0,1,2,3}}. You never see the hidden state.

    Write a function build_model() that returns a dict:
      factors : list of (name, size, controlled). Hidden state factors.
                Exactly ONE factor must have controlled=True (the action moves it).
      A       : list of 3 numpy arrays, one per observation part.
                A[m][o, s_i, s_j, ...] = p(o_m = o | the factors listed in A_deps[m]).
                Shape: (n_values_of_o_m, sizes of the factors in A_deps[m] in that order).
      A_deps  : list of 3 lists of factor indices.
      B       : one array per factor. B[f][s_next, s, a] = p(next | current, action).
                Shape (size, size, 4) if controlled, else (size, size, 1).
      D       : one initial-state probability vector per factor.
    Probabilities over the first axis must sum to 1.

    Data (o is the observation, a the action taken next):
    {data}

    {feedback}

    Reply with exactly:
    REASON: <2-4 sentences: what you are changing and which evidence made you change it>
    ```python
    <complete code defining build_model()>
    ```
""").strip()

FEEDBACK = textwrap.dedent("""
    Your current best model is below. On the training data it scores {score:.4f}
    (mean log-probability per observation step; 0 would be perfect).
    ```python
    {code}
    ```
    Diagnostics:
    {diag}
    Propose ONE improved version. Change the structure (factors, dependencies)
    if the evidence calls for it, not only the numbers.
""").strip()


def prompt_hash():
    """Fingerprint of the prompt templates used by a run."""
    return hashlib.sha256((SYSTEM + TASK + FEEDBACK).encode()).hexdigest()[:12]


def parse_reply(text):
    if not text or not text.strip():
        raise InvalidModel("the reply was empty")
    reason = re.search(r"REASON:\s*(.+?)(?:```|$)", text, re.S)
    code = re.search(r"```python\s*(.+?)```", text, re.S)
    if not code:
        raise InvalidModel("no ```python code block in the reply")
    return (reason.group(1).strip() if reason else ""), code.group(1).strip()


def run_code(code, timeout=20):
    """Execute LLM code in a separate Python process (so a crash or infinite
    loop cannot take down the run) and return the dict from build_model().
    The child gets a minimal environment, so API keys in the parent's
    environment are not passed on. This is still not a security sandbox:
    the code can read any file your user account can read."""
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "cand.py"
        out = Path(d) / "out.pkl"
        src.write_text(code + textwrap.dedent(f"""

            if __name__ == "__main__":
                import pickle, numpy as _np
                _m = build_model()
                _m = {{k: ([_np.asarray(x).tolist() for x in v] if k in ("A", "B", "D") else v)
                       for k, v in _m.items()}}
                pickle.dump(_m, open({str(out)!r}, "wb"))
        """))
        r = subprocess.run([sys.executable, "-I", str(src)], capture_output=True,
                           text=True, timeout=timeout, cwd=d,
                           env={"PATH": "/usr/bin:/bin", "HOME": d})
        if r.returncode != 0:
            raise InvalidModel("code crashed:\n" + r.stderr[-1500:])
        return pickle.load(open(out, "rb"))


def induce(llm, train, test, rounds=5, log_path=None, verbose=True):
    """Greedy refinement: always refine the best model so far.
    Returns the best spec and the full trace (list of dicts)."""
    trace, best = [], None
    feedback = "This is your first attempt."
    for r in range(rounds + 1):
        prompt = TASK.format(data=format_trajectories(train), feedback=feedback)
        entry = {"round": r, "parent": best["round"] if best else None}
        try:
            reply = llm(SYSTEM, prompt)
        except LLMUnavailable as e:
            entry.update(ok=False, error=str(e)[:1500], accepted=False, stopped=True)
            if verbose:
                print(f"[round {r}] STOPPED: {entry['error'][:200]}")
            trace.append(entry)
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            break
        entry["served"] = getattr(llm, "last_served", None)
        try:
            reason, code = parse_reply(reply)
            entry.update(reason=reason, code=code)
            spec, warns = validate(run_code(code))
            s_train, _ = score(spec, train)
            s_test, _ = score(spec, test)
            entry.update(ok=True, warnings=warns, train=s_train, test=s_test,
                         structure=describe(spec), n_params=n_free_params(spec), spec=spec)
            if best is not None:
                entry["diff"] = "\n".join(difflib.unified_diff(
                    best["code"].splitlines(), code.splitlines(),
                    "parent", "child", lineterm="", n=0))
                entry["delta_test"] = s_test - best["test"]
            # selection uses held-out score, so bigger models are not automatically favoured
            if best is None or s_test > best["test"]:
                best = dict(entry)
                entry["accepted"] = True
            else:
                entry["accepted"] = False
        except Exception as e:  # a bad candidate must never end the run
            entry.update(ok=False, error=f"{type(e).__name__}: {e}"[:1500], accepted=False)

        if verbose:
            if entry.get("ok"):
                print(f"[round {r}] train {entry['train']:.4f}  held-out {entry['test']:.4f}  "
                      f"{'ACCEPTED' if entry['accepted'] else 'rejected'}  | {entry['structure']}")
            else:
                print(f"[round {r}] FAILED: {entry['error'][:200]}")
        trace.append(entry)
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps({k: v for k, v in entry.items() if k != "spec"}) + "\n")

        if best is not None:
            feedback = FEEDBACK.format(score=best["train"], code=best["code"],
                                       diag=diagnostics(best["spec"], train))
            if not entry.get("ok"):
                feedback += f"\nYour last reply failed with this error, avoid it:\n{entry['error'][:800]}"
        elif not entry.get("ok"):
            feedback = f"Your last reply failed with this error, fix it:\n{entry['error'][:800]}"
    return (best["spec"] if best else None), trace
