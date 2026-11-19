"""Run one induction experiment end to end.

  python scripts/run.py --mock                         # offline test, no API key
  python scripts/run.py --model <openrouter-slug> --seed 0 --rounds 5
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aif_inductor.analysis import cue_link
from aif_inductor.evaluate import run_agent, run_random
from aif_inductor.induce import induce, prompt_hash
from aif_inductor.llm import MockLLM, OpenRouterLLM
from aif_inductor.model_spec import validate
from aif_inductor.runs import prepare_run_dir, run_tag
from aif_inductor.scoring import score
from aif_inductor.tmaze import generate_trajectories, true_model

p = argparse.ArgumentParser()
p.add_argument("--model", help="OpenRouter model slug")
p.add_argument("--mock", action="store_true")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--n-train", type=int, default=30)
p.add_argument("--n-test", type=int, default=100)
p.add_argument("--n-final", type=int, default=100, help="final test episodes, never used for selection")
p.add_argument("--eval-episodes", type=int, default=100)
p.add_argument("--demo-policy", default="random", choices=["random", "cue_first"],
               help="policy behind the training data; held-out and final test stay random")
p.add_argument("--max-tokens", type=int, default=4000, help="includes hidden reasoning tokens")
p.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"])
p.add_argument("--mock-stop-after", type=int, help="testing: simulate the daily cap")
p.add_argument("--force", action="store_true", help="replace existing logs for this run")
args = p.parse_args()

llm = MockLLM(args.mock_stop_after) if args.mock else OpenRouterLLM(
    args.model, seed=args.seed, max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort)
tag = run_tag("mock" if args.mock else args.model, args.seed, args.n_train, args.demo_policy)
out = prepare_run_dir(Path("runs") / tag, force=args.force)

train = generate_trajectories(args.n_train, seed=args.seed, policy=args.demo_policy)
test = generate_trajectories(args.n_test, seed=args.seed + 5000)
final_test = generate_trajectories(args.n_final, seed=args.seed + 9000)  # not passed to induce
truth, _ = validate(true_model())
true_final = score(truth, final_test)[0]
print(f"true model: train {score(truth, train)[0]:.4f}  held-out {score(truth, test)[0]:.4f}  "
      f"final test {true_final:.4f}")

t0 = time.time()
best, trace = induce(llm, train, test, rounds=args.rounds, log_path=out / "trace.jsonl")



def evaluate(spec):
    """An induced model that pymdp rejects is recorded, not fatal."""
    try:
        return run_agent(spec, args.eval_episodes, args.seed)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:500]}


def git_state():
    root = Path(__file__).resolve().parents[1]
    def git(*cmd):
        return subprocess.run(["git", *cmd], cwd=root, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


def fmt(stats):
    return "  ".join(f"{m}={x:.2f}" if isinstance(x, float) else f"{m}={x}" for m, x in stats.items())


stopped = next((e["error"] for e in trace if e.get("stopped")), None)
print("\nevaluating agents in the real environment...")
results = {"true_model": evaluate(truth),
           "random": run_random(args.eval_episodes, args.seed)}
if best is not None:
    results["induced"] = evaluate(best)
# agent performance of every accepted model, to show structure -> behaviour
per_round = []
for e in trace:
    if e.get("accepted"):
        r = evaluate(e["spec"])
        final = score(e["spec"], final_test)[0]
        per_round.append({"round": e["round"], "test": e["test"], "final_test": final,
                          "cue_link": cue_link(e["spec"]), **r})
        print(f"  round {e['round']} model: held-out {e['test']:.4f}  final test {final:.4f}  {fmt(r)}")
for k, v in results.items():
    print(f"  {k:11s} " + fmt(v))

summary = dict(status="partial" if stopped else "complete", stop_reason=stopped,
               prompt_hash=prompt_hash(), git=git_state(),
               seed=args.seed, model=llm.model, rounds=args.rounds,
               demo_policy=args.demo_policy, n_train=args.n_train,
               cue_link=cue_link(best) if best is not None else None,
               max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
               true_train=score(truth, train)[0], true_test=score(truth, test)[0], true_final_test=true_final,
               induced_final_test=score(best, final_test)[0] if best is not None else None,
               results=results, per_round=per_round, usage=llm.usage, seconds=time.time() - t0)
json.dump(summary, open(out / "summary.json", "w"), indent=2)
print(f"\nsaved to {out}/  (trace.jsonl, summary.json)")
