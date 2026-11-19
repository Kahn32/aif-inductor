"""Make the figure and the human-readable change trace from one or more runs.

  python scripts/report.py runs/mock_n30_random_seed0
  python scripts/report.py runs/<model>_n30_random_seed*      # several seeds -> variance shown
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run_dirs = [Path(p) for p in sys.argv[1:]]
assert run_dirs, "give at least one run directory"
out = run_dirs[0].parent / "report"
out.mkdir(exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
finals, true_test, rand_succ, true_succ = [], [], [], []
for d in run_dirs:
    trace = [json.loads(l) for l in open(d / "trace.jsonl")]
    summ = json.load(open(d / "summary.json"))
    true_test.append(summ["true_test"])
    rand_succ.append(summ["results"]["random"]["success"])
    true_succ.append(summ["results"]["true_model"]["success"])
    best, xs, ys = None, [], []
    for e in trace:
        if e.get("ok") and (best is None or e["test"] > best):
            best = e["test"]
        xs.append(e["round"])
        ys.append(best if best is not None else float("nan"))
    ax1.plot(xs, ys, marker="o", alpha=0.8, label=d.name)
    if "success" in summ["results"].get("induced", {}):
        finals.append(summ["results"]["induced"]["success"])
    pr = [r for r in summ.get("per_round", []) if "success" in r]
    ax2.plot([r["round"] for r in pr], [r["success"] for r in pr], marker="s", alpha=0.8)

ax1.axhline(sum(true_test) / len(true_test), ls="--", c="k", label="true model")
ax1.set_xlabel("refinement round")
ax1.set_ylabel("best held-out log-lik per step")
ax1.set_title("Model fit during induction")
ax1.legend(fontsize=7)
ax2.axhline(sum(true_succ) / len(true_succ), ls="--", c="k", label="agent with true model")
ax2.axhline(sum(rand_succ) / len(rand_succ), ls=":", c="gray", label="random agent")
ax2.set_xlabel("round of accepted model")
ax2.set_ylabel("success rate (first arm correct)")
ax2.set_ylim(-0.05, 1.05)
ax2.set_title("EFE agent using each accepted model")
ax2.legend(fontsize=7)
from matplotlib.ticker import MaxNLocator
for ax in (ax1, ax2):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.tight_layout()
fig.savefig(out / "induction.png", dpi=150)

# Change trace, written for a human reader
lines = ["# Change trace\n"]
for d in run_dirs:
    lines.append(f"## {d.name}\n")
    for e in (json.loads(l) for l in open(d / "trace.jsonl")):
        if not e.get("ok"):  # includes rounds stopped by the LLM limit
            lines.append(f"**Round {e['round']}: failed.** `{e['error'][:200]}`\n")
            continue
        verdict = "accepted" if e["accepted"] else "rejected"
        delta = f", change {e['delta_test']:+.4f}" if "delta_test" in e else ""
        lines.append(f"**Round {e['round']} ({verdict}).** held-out {e['test']:.4f}{delta}, "
                     f"{e['n_params']} free params  \n`{e['structure']}`  \n"
                     f"Reason given: {e['reason']}\n")
(out / "trace.md").write_text("\n".join(lines))

if len(finals) > 1:
    import statistics as st
    print(f"final success over {len(finals)} seeds: mean {st.mean(finals):.2f}, "
          f"sd {st.stdev(finals):.2f}, values {finals}")
print(f"wrote {out}/induction.png and {out}/trace.md")
