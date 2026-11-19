"""Plot the sweep: cue link and agent behaviour against training size,
one line per demonstration policy.

  python scripts/plot_sweep.py results/sweep_seed0.csv [more csv files]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from_files = [Path(p) for p in sys.argv[1:]]
assert from_files, "give at least one sweep csv"
rows = [r for f in from_files for r in csv.DictReader(open(f))]

panels = [("link", "cue link of best model\n(true model about 0.86)"),
          ("cue_then_success", "agent: cue visited, then right arm")]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (key, label) in zip(axes, panels):
    for policy, marker in [("random", "o"), ("cue_first", "s")]:
        pts = defaultdict(list)
        for r in rows:
            if r["policy"] == policy and r[key] not in ("", "None"):
                pts[int(r["n_train"])].append(float(r[key]))
        xs = sorted(pts)
        for x in xs:  # every seed as a dot, the line through the means
            ax.scatter([x] * len(pts[x]), pts[x], marker=marker, alpha=0.5,
                       color="C0" if policy == "random" else "C1")
        ax.plot(xs, [sum(pts[x]) / len(pts[x]) for x in xs], marker=marker,
                color="C0" if policy == "random" else "C1", label=policy)
    ax.set_xscale("log", base=2)
    ax.set_xticks([5, 10, 20, 40], ["5", "10", "20", "40"])
    ax.set_xlabel("training episodes")
    ax.set_ylabel(label)
    ax.legend(fontsize=8)
axes[0].axhline(0.5, ls=":", c="gray")
axes[0].set_ylim(-1, 1)
axes[0].axhline(-0.5, ls=":", c="gray")
axes[1].set_ylim(-0.05, 1.05)
n_seeds = len({r["seed"] for r in rows})
fig.suptitle(f"Extension sweep, {n_seeds} seed{'s' if n_seeds > 1 else ''}", fontsize=10)
fig.tight_layout()
out = from_files[0].with_suffix(".png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
