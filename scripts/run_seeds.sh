#!/bin/sh
# The frozen five-seed experiment. Needs OPENROUTER_API_KEY and 30 free requests.
# Reasoning is off because this model returned empty replies or stalled with it on.
# A failed seed is kept and reported. Existing runs are never overwritten.
cd "$(dirname "$0")/.." || exit 1
for seed in 0 1 2 3 4; do
  .venv/bin/python scripts/run.py --model nvidia/nemotron-3-ultra-550b-a55b:free \
    --seed "$seed" --rounds 5 --max-tokens 16000 --reasoning-effort none
  echo "seed $seed exit code $?"
done
