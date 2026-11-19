"""Extension sweep: how much data makes the cue-to-reward link findable?

Runs the frozen setup for 4 training sizes x 2 demonstration policies.

  python scripts/sweep.py --mock                # offline dry run, no requests
  python scripts/sweep.py --seed 0              # real: 8 runs, at least 48 requests

Finished settings are skipped, so a stopped sweep can be resumed by running
the same command again. Before each real run it asks OpenRouter how many free
requests are left and stops if a run might not fit.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aif_inductor.runs import run_tag

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
FROZEN = ["--rounds", "5", "--max-tokens", "16000", "--reasoning-effort", "none"]
SIZES = [5, 10, 20, 40]
POLICIES = ["random", "cue_first"]
PER_RUN = 6  # 1 + 5 rounds, if no call is retried

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--mock", action="store_true")
p.add_argument("--margin", type=int, default=1, help="spare requests kept for retries")
p.add_argument("--out", default="results")
args = p.parse_args()


def requests_left():
    r = requests.get("https://openrouter.ai/api/v1/key", timeout=30,
                     headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    r.raise_for_status()
    return r.json()["data"]["free_model_daily_requests"]["remaining"]


rows = []
for policy in POLICIES:
    for n in SIZES:
        run_dir = ROOT / "runs" / run_tag("mock" if args.mock else MODEL, args.seed, n, policy)
        if not (run_dir / "summary.json").exists():
            if not args.mock:
                left = requests_left()
                if left < PER_RUN + args.margin:
                    print(f"stopping: {left} requests left, a run needs {PER_RUN} plus {args.margin} spare")
                    break
            cmd = [sys.executable, "scripts/run.py", "--seed", str(args.seed),
                   "--n-train", str(n), "--demo-policy", policy]
            cmd += ["--mock", "--rounds", "3", "--eval-episodes", "20"] if args.mock else ["--model", MODEL, *FROZEN]
            print(f"running {policy} n={n}", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=False)
        else:
            print(f"skipping {policy} n={n}: already done")
    else:
        continue
    break

for policy in POLICIES:
    for n in SIZES:
        f = ROOT / "runs" / run_tag("mock" if args.mock else MODEL, args.seed, n, policy) / "summary.json"
        if not f.exists():
            continue
        s = json.load(open(f))
        ind = s["results"].get("induced", {})
        link = s.get("cue_link") or {}
        failed = sum(1 for l in open(f.parent / "trace.jsonl") if not json.loads(l).get("ok"))
        rows.append(dict(policy=policy, n_train=n, seed=s["seed"], status=s["status"],
                         final_test=s["induced_final_test"], true_final_test=s["true_final_test"],
                         link=link.get("link"), linked=link.get("linked"), correct_sign=link.get("correct_sign"),
                         success=ind.get("success"), visited_cue=ind.get("visited_cue"),
                         cue_then_success=ind.get("cue_then_success"), failed_rounds=failed,
                         requests=s["usage"]["attempts"],
                         log=str(f.relative_to(ROOT))))

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
name = f"sweep_{'mock_' if args.mock else ''}seed{args.seed}.csv"
if rows:
    with open(out / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
print(f"{len(rows)} of {len(SIZES) * len(POLICIES)} settings done, table in {out / name}")
