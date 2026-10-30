"""Backtest the model on completed games, old (v1) vs new (v2).

    python backtest.py 2025 2 19          # season, first week, last week + 1
    python backtest.py 2026 2 9 --quick   # fewer simulations, faster

Reports, for each model:
  team level   - expected vs actual offensive TDs per team-game (Poisson deviance, error, correlation)
  player level - Brier score and log loss on "did he score?" (lower is better), how often each game's
                 3 most likely scorers scored, and how the 20%+ picks (the Slate Sniper list) hit
  calibration  - predicted vs actual scoring rate by probability bucket
"""
import sys

import engine
import compare as C

args = [a for a in sys.argv[1:] if not a.startswith("--")]
season, w0, w1 = (int(x) for x in args[:3]) if len(args) >= 3 else (2025, 2, 19)
n_sims = 1500 if "--quick" in sys.argv else 4000
weeks = range(w0, w1)
configs = {"old": engine.Params.legacy(), "new": engine.Params()}

print(f"Season {season}, weeks {w0}-{w1 - 1}\n\nTEAM LEVEL (expected vs actual offensive TDs)")
for name, p in configs.items():
    m = C.team_metrics(C.team_eval(season, weeks, p))
    print(f"  {name:4s} Poisson deviance {m['poisson_dev']:.4f} · avg error {m['mae']:.3f} TDs · "
          f"correlation {m['corr']:.3f}")

print("\nPLAYER LEVEL (simulating every game, this takes a few minutes)")
df = C.player_eval(season, weeks, configs, n_sims=n_sims)
base = df["scored"].mean()
print(f"  {len(df)} player-games, {df['game_id'].nunique()} games, base scoring rate {base:.1%}")
for name in configs:
    m = C.player_metrics(df, name)
    print(f"  {name:4s} Brier {m['brier']:.4f} · log loss {m['log_loss']:.4f} · top-3 hit {m['top3_hit']:.1%} · "
          f"20%+ picks: {m['n_20plus']} scored {m['hit_20plus']:.1%} (predicted {m['pred_20plus']:.1%})")
for name in configs:
    print(f"\nCalibration, {name}:")
    print(C.calibration(df, name).round(3).to_string())
