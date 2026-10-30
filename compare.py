"""Backtest harness: old (v1) vs new (v2) model on completed games."""
import numpy as np
import pandas as pd
import engine


def actual_tds(season):
    full = engine._load("pbp", [season])
    t = full[((full["rush_touchdown"] == 1) | (full["pass_touchdown"] == 1)) & (full["td_team"] == full["posteam"])
             & (full["two_point_attempt"] != 1)]
    return t


def team_eval(season, weeks, p):
    """Team-level only (no simulation): expected offensive TDs vs actual, per team-game."""
    sched = engine.get_schedule(season)
    games = sched[sched["week"].isin(weeks) & sched["result"].notna()]
    at = actual_tds(season)
    tds = at.groupby(["game_id", "posteam"]).size()
    rtds = at[at["rush_touchdown"] == 1].groupby(["game_id", "posteam"]).size()
    rows = []
    for wk, gw in games.groupby("week"):
        pbp = engine._prep(season, int(wk), p)
        rates, lg_r, lg_p = engine.team_td_rates(pbp, p)
        for _, g in gw.iterrows():
            for off, dfn, home in ((g["away_team"], g["home_team"], False), (g["home_team"], g["away_team"], True)):
                lr, lp, _ = engine.expected_tds_v2(pbp, rates, lg_r, lg_p, g, off, dfn, home, season, int(wk), p)
                rows.append({"season": season, "week": wk, "game_id": g["game_id"], "team": off,
                             "pred": lr + lp, "actual": tds.get((g["game_id"], off), 0),
                             "pred_r": lr, "pred_p": lp, "act_r": rtds.get((g["game_id"], off), 0)})
    return pd.DataFrame(rows)


def poisson_dev(pred, y):
    pred = np.maximum(pred, 1e-6)
    t = np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / pred), 0.0)
    return float(2 * np.mean(t - (y - pred)))


def team_metrics(df):
    return {"poisson_dev": poisson_dev(df["pred"].values, df["actual"].values),
            "mae": float(np.mean(np.abs(df["pred"] - df["actual"]))),
            "corr": float(np.corrcoef(df["pred"], df["actual"])[0, 1]),
            "split_dev": poisson_dev(df["pred_r"].values, df["act_r"].values)
                         + poisson_dev(df["pred_p"].values, (df["actual"] - df["act_r"]).values)}


def player_eval(season, weeks, configs, n_sims=4000):
    """Simulate every completed game under each config. Returns one row per (game, player) with each
    config's probability (players a config doesn't list get 0.3%) and whether he scored."""
    sched = engine.get_schedule(season)
    games = sched[sched["week"].isin(weeks) & sched["result"].notna()]
    scorers = actual_tds(season).groupby("game_id")["td_player_id"].apply(set).to_dict()
    out = []
    for gid in games["game_id"]:
        per = {}
        info = {}
        for name, p in configs.items():
            p.n_sims = n_sims
            res, _ = engine.run_game(gid, p, seed=1)
            per[name] = res.set_index("player_id")["p_anytime"]
            info.update(res.set_index("player_id")[["player", "team", "pos"]].to_dict("index"))
        ids = sorted(set().union(*[s.index for s in per.values()]))
        sc = scorers.get(gid, set())
        for pid in ids:
            row = {"season": season, "game_id": gid, "player_id": pid, **info[pid], "scored": int(pid in sc)}
            for name, s in per.items():
                row[name] = float(s.get(pid, 0.003))
            out.append(row)
    return pd.DataFrame(out)


def player_metrics(df, col):
    p = df[col].clip(0.001, 0.999)
    y = df["scored"]
    brier = float(((p - y) ** 2).mean())
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    # top-3 per game: how often the model's 3 most likely scorers actually scored
    top = df.assign(_p=p).sort_values("_p", ascending=False).groupby("game_id").head(3)
    # 20%+ picks (the Slate Sniper cut): hit rate vs average predicted
    hi = df[df[col] >= 0.20]
    return {"brier": brier, "log_loss": ll, "top3_hit": float(top["scored"].mean()),
            "n_20plus": int(len(hi)), "hit_20plus": float(hi["scored"].mean()),
            "pred_20plus": float(hi[col].mean())}


def calibration(df, col, bins=(0, .1, .2, .3, .4, .5, 1)):
    b = pd.cut(df[col], list(bins))
    return df.groupby(b, observed=True).agg(n=("scored", "size"), predicted=(col, "mean"), actual=("scored", "mean"))
