"""Check model calibration on completed games:  python backtest.py 2025 6 12"""
import sys
import engine

season, w0, w1 = (int(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (2025, 6, 12)
cal, brier, base, tm = engine.backtest(season, range(w0, w1), n_sims=3000)
print(cal.round(3))
print(f"Brier {brier:.4f}  vs base-rate {base:.4f}  (lower is better)")
print(f"Team TDs/game predicted {tm.pred.mean():.2f}, actual {tm.actual.mean():.2f}")
