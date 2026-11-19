# Seed results

Setup: prompt `cfd6cbd2e60f`, `nvidia/nemotron-3-ultra-550b-a55b:free`, reasoning off, 5 refinement rounds, 30 random-policy training episodes, 100 held-out episodes, 100 final-test episodes, 100 agent episodes. Every run made 6 requests, 48 in total, with no retries.

Seeds 0 to 4 ran first. Seeds 5 to 7 were added the same night with identical settings because 20 free requests were left. No seed was rerun or dropped.

Scores are mean log-probability per observation step, higher is better. Cue link is defined in `aif_inductor/analysis.py`: the model's own predicted change in reward probability at the left arm between the two cue values. The true model gives +0.86; 0 means no link; a model counts as linked at 0.5 or more in size.

## Best model per seed

| Seed | Best round | Final test | True final test | Cue link | Agent success | True model success | Random success | Entered arm | Visited cue | First linked round | Failed rounds | Requests | Log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 0 | 5 | -0.2856 | -0.2303 | -0.00 | 0.00 | 0.97 | 0.34 | 0.00 | 1.00 | 0 | 1 | 6 | [log](logs/seed0/summary.json) |
| 1 | 4 | -0.5499 | -0.2531 | +1.00 | 0.61 | 0.97 | 0.45 | 1.00 | 0.00 | 3 | 1 | 6 | [log](logs/seed1/summary.json) |
| 2 | 4 | -1.7866 | -0.2142 | +0.00 | 0.00 | 0.94 | 0.31 | 0.00 | 1.00 | never | 2 | 6 | [log](logs/seed2/summary.json) |
| 3 | 4 | -0.5674 | -0.2196 | +0.00 | 0.00 | 0.96 | 0.50 | 0.00 | 0.00 | never | 1 | 6 | [log](logs/seed3/summary.json) |
| 4 | 4 | -0.4426 | -0.2573 | +0.00 | 0.00 | 0.95 | 0.42 | 0.00 | 1.00 | never | 1 | 6 | [log](logs/seed4/summary.json) |
| 5 | 5 | -0.4461 | -0.2607 | +0.00 | 0.53 | 0.96 | 0.44 | 1.00 | 0.00 | never | 2 | 6 | [log](logs/seed5/summary.json) |
| 6 | 1 | -0.6495 | -0.2286 | +0.00 | 0.00 | 0.95 | 0.42 | 0.00 | 1.00 | never | 3 | 6 | [log](logs/seed6/summary.json) |
| 7 | 2 | -0.4106 | -0.2469 | +0.00 | 0.57 | 0.95 | 0.41 | 1.00 | 0.00 | never | 1 | 6 | [log](logs/seed7/summary.json) |

Agent success over 8 seeds: mean 0.21, standard deviation 0.30, values 0.00, 0.61, 0.00, 0.00, 0.00, 0.53, 0.00, 0.57. True model: 0.94 to 0.97.
Final-test score: mean -0.6423, standard deviation 0.4755.
12 of 48 rounds failed. Figure: `figures/seeds_induction.png`. Every round's reason: `results/trace.md`.

## Every accepted model

| Seed | Round | Cue link | Visited cue | Success |
|---|---:|---:|---:|---:|
| 0 | 0 | +1.00 | 1.0 | 0.49 |
| 0 | 3 | +0.00 | 1.0 | 0.0 |
| 0 | 5 | -0.00 | 1.0 | 0.0 |
| 1 | 0 | +0.17 | 0.0 | 0.39 |
| 1 | 3 | +1.00 | 0.0 | 0.61 |
| 1 | 4 | +1.00 | 0.0 | 0.61 |
| 2 | 1 | +0.00 | agent error | agent error |
| 2 | 2 | +0.00 | 0.0 | 0.0 |
| 2 | 3 | +0.00 | 0.0 | 0.0 |
| 2 | 4 | +0.00 | 1.0 | 0.0 |
| 3 | 0 | -0.00 | 0.0 | 0.51 |
| 3 | 1 | -0.11 | 0.0 | 0.0 |
| 3 | 3 | +0.00 | 1.0 | 0.51 |
| 3 | 4 | +0.00 | 0.0 | 0.0 |
| 4 | 0 | -0.00 | 0.0 | 0.0 |
| 4 | 1 | -0.00 | 1.0 | 0.0 |
| 4 | 2 | -0.00 | 1.0 | 0.0 |
| 4 | 3 | +0.00 | 0.0 | 0.0 |
| 4 | 4 | +0.00 | 1.0 | 0.0 |
| 5 | 0 | +0.00 | 0.0 | 0.0 |
| 5 | 3 | +0.00 | 0.0 | 0.0 |
| 5 | 4 | +0.00 | 0.0 | 0.53 |
| 5 | 5 | +0.00 | 0.0 | 0.53 |
| 6 | 0 | +0.00 | 0.0 | 0.49 |
| 6 | 1 | +0.00 | 1.0 | 0.0 |
| 7 | 0 | +0.00 | 0.0 | 0.0 |
| 7 | 2 | +0.00 | 0.0 | 0.57 |

## What this shows

- No selected model solved the task. Mean success was 0.21 against 0.94 or more for the true model.
- Two seeds produced a linked model. Seed 0 found one in round 0 (link +1.00), then selection replaced it with better-scoring models that had no link. Seed 1's best model is linked (+1.00), but its agent never visited the cue.
- So better fit did not mean better behaviour. Seed 0's final model came closest to the true model's score, has no link, and its agent never entered an arm.
- The plan's claim that unlinked models never visit the cue is false here. Seeds 0, 2, 4 and 6 have unlinked best models that visit the cue in every episode.
- Seeds 1, 5 and 7 scored 0.53 to 0.61 without visiting the cue. That is a committed guess, 0.09 to 0.16 above random actions on 100 episodes.
- An earlier version of this file used factor dependencies to judge the link. That was wrong: seed 4 shares a factor between the two signals and still has zero link.

## Limits

- Eight seeds, one task, one free model.
- The prompt was kept after looking at an earlier seed 0 run. Model settings were chosen after test runs on seeds 0 and 100. Those runs are not counted here.
- The provider may ignore the seed. An earlier seed 0 run with the same prompt gave different models.
