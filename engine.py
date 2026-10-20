"""
NFL Anytime-Touchdown Monte Carlo engine.

Data: nflverse (play-by-play, schedules, rosters, injuries) via nflreadpy — free,
updated nightly during the season.

Model overview
--------------
1. TEAM LEVEL  - Expected offensive rushing and passing TDs for each side, from
   a recency-weighted, shrunk blend of the offense's TD rate and the opponent
   defense's TD-allowed rate (multiplicative / log5 style), plus home field.
2. PLAYER LEVEL - Each player's share of his team's rushing and receiving TDs,
   built from opportunity (carries, inside-10 carries, targets, red-zone
   targets) and realized TD share, recency-weighted, with a multi-season
   finishing-skill multiplier (shrunk) and an opponent-vs-position adjustment
   (e.g. a defense that leaks TE touchdowns).
3. AVAILABILITY - Players listed Out / IR are removed and their share is
   redistributed; Doubtful / Questionable are discounted.
4. SIMULATION  - Each sim draws a shared game-environment factor (gamma) so
   rush and pass TDs are correlated, draws team TD counts (Poisson-gamma =
   negative binomial), then allocates each TD to a player (multinomial).
   Output: P(anytime TD), P(2+ TDs), expected TDs, fair American odds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import nflreadpy as nfl

# ----------------------------------------------------------------------------
# Tunable parameters
# ----------------------------------------------------------------------------
@dataclass
class Params:
    n_sims: int = 20000
    seasons_back: int = 3                 # current + 2 prior seasons of history
    prior_season_weight: float = 0.35     # weight of last season's games vs this season
    older_season_weight: float = 0.12
    recent_games_boost: float = 1.6       # extra weight on a team's last 4 games
    team_shrink_games: float = 5.0        # pseudo-games of league average (team rates)
    home_edge: float = 0.04               # +4% TDs home, -4% away
    game_dispersion: float = 9.0          # gamma shape; lower = more volatile game scripts
    # player share blend
    rush_w: dict = field(default_factory=lambda: {"carry": 0.30, "i10": 0.45, "td": 0.25})
    rec_w: dict = field(default_factory=lambda: {"tgt": 0.35, "rz": 0.40, "td": 0.25})
    skill_shrink_opps: float = 40.0       # pseudo red-zone opps for finishing-skill shrinkage
    skill_clip: tuple = (0.75, 1.35)
    pos_adj_shrink: float = 4.0           # pseudo-TDs for opponent-vs-position shrinkage
    pos_adj_clip: tuple = (0.7, 1.4)
    injury_mult: dict = field(default_factory=lambda: {
        "Out": 0.0, "Doubtful": 0.2, "Questionable": 0.85})
    share_temper: float = 0.7             # <1 flattens shares (backtest showed raw shares too concentrated)
    unlisted_share: float = 0.07          # TDs going to depth players / trick plays not on the list
    redistribute_frac: float = 0.8        # share of an injured player's opps that flows to teammates on this list


# ----------------------------------------------------------------------------
# Data loading (cached in-process)
# ----------------------------------------------------------------------------
_CACHE: dict = {}
PBP_COLS = ["game_id", "season", "week", "posteam", "defteam", "play_type", "two_point_attempt",
            "rush_touchdown", "pass_touchdown", "td_team", "td_player_id", "td_player_name",
            "rusher_player_id", "receiver_player_id", "yardline_100"]


def _load(name, seasons):
    key = (name, tuple(seasons))
    if key not in _CACHE:
        fn = {"pbp": nfl.load_pbp, "sched": nfl.load_schedules,
              "rost": nfl.load_rosters_weekly, "inj": nfl.load_injuries}[name]
        try:
            if name == "pbp":
                # load one season at a time, keep only needed columns (cuts memory ~10x)
                parts = []
                for yr in seasons:
                    df = fn([yr])
                    parts.append(df.select([c for c in PBP_COLS if c in df.columns]).to_pandas())
                    del df
                _CACHE[key] = pd.concat(parts, ignore_index=True)
            else:
                _CACHE[key] = fn(list(seasons)).to_pandas()
        except Exception:
            # injuries / rosters can be missing for a not-yet-started season
            _CACHE[key] = pd.DataFrame()
    return _CACHE[key]


def current_season() -> int:
    return nfl.get_current_season()


def get_schedule(season: int) -> pd.DataFrame:
    s = _load("sched", [season]).copy()
    s["label"] = (s["away_team"] + " @ " + s["home_team"] + "  —  Wk " + s["week"].astype(str)
                  + ", " + s["gameday"].astype(str))
    return s


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _game_weights(pbp: pd.DataFrame, season: int, p: Params) -> pd.Series:
    """Weight per (game, team-on-offense) by season recency, plus recent-form boost."""
    sw = {season: 1.0, season - 1: p.prior_season_weight}
    w = pbp["season"].map(sw).fillna(p.older_season_weight)
    return w


def _prep(season: int, week: int, p: Params):
    # identical for every game in the same week, so build it once per week
    key = ("prep", season, week, p.seasons_back, p.prior_season_weight,
           p.older_season_weight, p.recent_games_boost)
    if key not in _CACHE:
        _CACHE[key] = _build_prep(season, week, p)
    return _CACHE[key]


def _build_prep(season: int, week: int, p: Params):
    seasons = list(range(season - p.seasons_back + 1, season + 1))
    pbp = _load("pbp", seasons)
    # only use information available before the game being simulated
    pbp = pbp[(pbp["season"] < season) | (pbp["week"] < week)].copy()
    pbp = pbp[pbp["play_type"].isin(["run", "pass"]) & (pbp["two_point_attempt"] != 1)]
    pbp["w"] = _game_weights(pbp, season, p)
    # recent-form boost: a team's 4 most recent games this season
    cur = pbp["season"] == season
    if cur.any():
        recent = (pbp[cur].groupby("posteam")["week"]
                  .apply(lambda s: set(sorted(s.unique())[-4:])).to_dict())
        is_recent = [cur_i and wk in recent.get(t, ()) for cur_i, wk, t
                     in zip(cur, pbp["week"], pbp["posteam"])]
        pbp.loc[is_recent, "w"] *= p.recent_games_boost
    pbp["rtd"] = ((pbp["rush_touchdown"] == 1) & (pbp["td_team"] == pbp["posteam"])).astype(int)
    pbp["ptd"] = ((pbp["pass_touchdown"] == 1) & (pbp["td_team"] == pbp["posteam"])).astype(int)
    return pbp


# ----------------------------------------------------------------------------
# 1) Team-level expected TDs
# ----------------------------------------------------------------------------
def team_td_rates(pbp: pd.DataFrame, p: Params):
    g = (pbp.groupby(["game_id", "posteam", "defteam"])
         .agg(rtd=("rtd", "sum"), ptd=("ptd", "sum"), w=("w", "mean")).reset_index())
    lg_r = np.average(g["rtd"], weights=g["w"])
    lg_p = np.average(g["ptd"], weights=g["w"])

    def shrunk(side, col, lg):
        a = g.groupby(side).apply(lambda d: pd.Series({
            "num": (d[col] * d["w"]).sum(), "den": d["w"].sum()}), include_groups=False)
        return (a["num"] + p.team_shrink_games * lg) / (a["den"] + p.team_shrink_games)

    rates = pd.DataFrame({
        "off_r": shrunk("posteam", "rtd", lg_r), "off_p": shrunk("posteam", "ptd", lg_p),
        "def_r": shrunk("defteam", "rtd", lg_r), "def_p": shrunk("defteam", "ptd", lg_p)})
    return rates, lg_r, lg_p


def expected_tds(rates, lg_r, lg_p, off, dfn, is_home, p: Params):
    hfa = 1 + p.home_edge if is_home else 1 - p.home_edge
    lam_r = lg_r * (rates.at[off, "off_r"] / lg_r) * (rates.at[dfn, "def_r"] / lg_r) * hfa
    lam_p = lg_p * (rates.at[off, "off_p"] / lg_p) * (rates.at[dfn, "def_p"] / lg_p) * hfa
    return lam_r, lam_p


# ----------------------------------------------------------------------------
# 2) Opponent vs position (receiving + rushing TDs allowed by scorer position)
# ----------------------------------------------------------------------------
def opp_position_mults(pbp, positions, dfn, p: Params):
    td = pbp[(pbp["rtd"] == 1) | (pbp["ptd"] == 1)].copy()
    td["pos"] = td["td_player_id"].map(positions).fillna("OTH")
    td["kind"] = np.where(td["rtd"] == 1, "rush", "rec")
    # weighted games faced by this defense, so seasons are weighted consistently
    gw = pbp.groupby(["game_id", "defteam"])["w"].mean().reset_index()
    opp_games = gw.loc[gw["defteam"] == dfn, "w"].sum()
    games_all = gw["w"].sum()
    out = {}
    meaningful = {("rush", "RB"), ("rush", "QB"), ("rec", "WR"), ("rec", "TE"), ("rec", "RB")}
    for (kind, pos), d in td.groupby(["kind", "pos"]):
        if (kind, pos) not in meaningful:
            continue
        lg_rate = d["w"].sum() / games_all            # weighted TDs per team-game
        opp_td = d.loc[d["defteam"] == dfn, "w"].sum()
        exp = lg_rate * opp_games
        k = p.pos_adj_shrink
        mult = (opp_td + k) / (exp + k)
        out[(kind, pos)] = float(np.clip(mult, *p.pos_adj_clip))
    return out


# ----------------------------------------------------------------------------
# 3) Player shares
# ----------------------------------------------------------------------------
def player_shares(pbp, team, season, positions, names, pos_mult, p: Params):
    tp = pbp[(pbp["posteam"] == team) & (pbp["season"] == season)]
    if tp.empty:  # early season fallback: last season for this team
        tp = pbp[(pbp["posteam"] == team) & (pbp["season"] == season - 1)]
    # recency-weight within season (later weeks count more)
    wk = tp["week"]
    rw = 1 + (wk - wk.min()) / max(1, (wk.max() - wk.min())) if len(tp) else wk
    tp = tp.assign(rw=rw)

    runs = tp[tp["play_type"] == "run"]
    passes = tp[tp["play_type"] == "pass"]

    def share(df, idcol, mask=None):
        d = df if mask is None else df[mask]
        s = d.groupby(idcol)["rw"].sum()
        return s / s.sum() if s.sum() > 0 else s

    carry = share(runs, "rusher_player_id")
    i10 = share(runs, "rusher_player_id", runs["yardline_100"] <= 10)
    rtd = share(runs, "rusher_player_id", runs["rtd"] == 1)
    tgt = share(passes, "receiver_player_id")
    rz = share(passes, "receiver_player_id", passes["yardline_100"] <= 20)
    ptd = share(passes, "receiver_player_id", passes["ptd"] == 1)

    ids = sorted(set(carry.index) | set(tgt.index))
    df = pd.DataFrame(index=ids)
    for nm, s in [("carry", carry), ("i10", i10), ("rtd", rtd),
                  ("tgt", tgt), ("rz", rz), ("ptd", ptd)]:
        df[nm] = s.reindex(ids).fillna(0.0)

    rw_, cw = p.rush_w, p.rec_w
    df["rush_share"] = rw_["carry"] * df["carry"] + rw_["i10"] * df["i10"] + rw_["td"] * df["rtd"]
    df["rec_share"] = cw["tgt"] * df["tgt"] + cw["rz"] * df["rz"] + cw["td"] * df["ptd"]

    # finishing skill from multi-season history (TDs per red-zone opportunity vs league)
    rzp = pbp[pbp["yardline_100"] <= 20]
    opp = pd.concat([rzp["rusher_player_id"], rzp["receiver_player_id"]]).value_counts()
    tds = pbp.loc[(pbp["rtd"] == 1) | (pbp["ptd"] == 1), "td_player_id"].value_counts()
    lg_conv = tds.sum() / max(1, opp.sum())
    k = p.skill_shrink_opps
    skill = ((tds.reindex(ids).fillna(0) + k * lg_conv)
             / ((opp.reindex(ids).fillna(0) + k) * lg_conv)).clip(*p.skill_clip)
    df["skill"] = skill
    df["pos"] = [positions.get(i, "OTH") for i in ids]
    df["name"] = [names.get(i, i) for i in ids]
    df["rush_share"] *= df["skill"] * [pos_mult.get(("rush", x), 1.0) for x in df["pos"]]
    df["rec_share"] *= df["skill"] * [pos_mult.get(("rec", x), 1.0) for x in df["pos"]]
    for c in ["rush_share", "rec_share"]:
        df[c] = df[c] ** p.share_temper
        if df[c].sum() > 0:
            df[c] *= (1 - p.unlisted_share) / df[c].sum()
    # drop negligible contributors (keep them in 'other')
    df = df[(df["rush_share"] > 0.004) | (df["rec_share"] > 0.004)]
    return df


def apply_availability(df, team, season, week, p: Params, overrides=None):
    """Discount / remove players by injury report and roster status."""
    notes = []
    df = df.copy()
    df["status"] = "Active"
    df["avail"] = 1.0
    rost = _load("rost", [season])
    inj = _load("inj", [season])

    if not rost.empty:
        r = rost[rost["season"] == season]
        wk_avail = r.loc[r["week"] <= week, "week"]
        latest = r[r["week"] == (wk_avail.max() if len(wk_avail) else r["week"].max())]
        cur_team = latest.set_index("gsis_id")["team"].to_dict()
        cur_stat = latest.set_index("gsis_id")["status"].to_dict()
        for pid in df.index:
            t = cur_team.get(pid)
            if t is not None and t != team:
                df.at[pid, "avail"] = 0.0
                df.at[pid, "status"] = f"Now on {t}"
            elif cur_stat.get(pid) not in (None, "ACT"):
                df.at[pid, "avail"] = 0.0
                df.at[pid, "status"] = f"Roster: {cur_stat.get(pid)}"

    if not inj.empty and (inj["week"] == week).any():
        rep = inj[(inj["week"] == week) & (inj["team"] == team)]
        for _, row in rep.iterrows():
            pid, st = row["gsis_id"], row["report_status"]
            if pid in df.index and st in p.injury_mult and df.at[pid, "avail"] > 0:
                df.at[pid, "avail"] = p.injury_mult[st]
                df.at[pid, "status"] = st
    else:
        notes.append(f"Week {week} injury report not published yet — only roster status (IR, trades, cuts) applied.")

    for pid, mult in (overrides or {}).items():
        if pid in df.index:
            df.at[pid, "avail"] = mult
            df.at[pid, "status"] = "Manual: OUT" if mult == 0 else f"Manual: x{mult}"

    for c in ["rush_share", "rec_share"]:
        lost = (df[c] * (1 - df["avail"])).sum()
        df[c] = df[c] * df["avail"]
        rem = df[c].sum()
        if rem > 0:
            # most of the vacated opportunity flows to remaining listed players
            df[c] *= (rem + p.redistribute_frac * lost) / rem
    return df, notes


# ----------------------------------------------------------------------------
# 4) Monte Carlo
# ----------------------------------------------------------------------------
def simulate_team(df, lam_r, lam_p, env, rng):
    n = len(env)
    nr = rng.poisson(lam_r * env)
    npass = rng.poisson(lam_p * env)
    pr = np.append(df["rush_share"].values, max(0.0, 1 - df["rush_share"].sum()))
    pp = np.append(df["rec_share"].values, max(0.0, 1 - df["rec_share"].sum()))
    pr, pp = pr / pr.sum(), pp / pp.sum()
    counts = rng.multinomial(nr, pr)[:, :-1] + rng.multinomial(npass, pp)[:, :-1]
    return counts, nr + npass


def run_game(game_id: str, p: Params | None = None, seed: int | None = None, overrides=None):
    p = p or Params()
    season = int(game_id[:4])
    sched = get_schedule(season)
    g = sched[sched["game_id"] == game_id].iloc[0]
    week, home, away = int(g["week"]), g["home_team"], g["away_team"]

    pbp = _prep(season, week, p)
    rost_all = pd.concat([_load("rost", list(range(season - p.seasons_back + 1, season + 1)))])
    rost_all = rost_all.sort_values(["season", "week"])
    positions = rost_all.drop_duplicates("gsis_id", keep="last").set_index("gsis_id")["position"].to_dict()
    names = rost_all.drop_duplicates("gsis_id", keep="last").set_index("gsis_id")["full_name"].to_dict()

    rates, lg_r, lg_p = team_td_rates(pbp, p)
    rng = np.random.default_rng(seed)
    env = rng.gamma(p.game_dispersion, 1 / p.game_dispersion, p.n_sims)

    results, team_info, notes = [], {}, []
    hits = {}   # player_id -> did he score in each sim (for same-game parlay correlation)
    for off, dfn, is_home in [(away, home, False), (home, away, True)]:
        lam_r, lam_p = expected_tds(rates, lg_r, lg_p, off, dfn, is_home, p)
        pm = opp_position_mults(pbp, positions, dfn, p)
        df = player_shares(pbp, off, season, positions, names, pm, p)
        df, n_ = apply_availability(df, off, season, week, p, overrides)
        notes += [f"{off}: {x}" for x in n_]
        team_env = env * rng.gamma(p.game_dispersion * 2, 1 / (p.game_dispersion * 2), p.n_sims)
        counts, team_tds = simulate_team(df, lam_r, lam_p, team_env, rng)
        for j, pid in enumerate(df.index):
            hits[pid] = counts[:, j] >= 1
        team_info[off] = {"exp_rush_td": lam_r, "exp_pass_td": lam_p,
                          "sim_mean_td": team_tds.mean(),
                          "opp_pos_mult": {f"{k[0]}-{k[1]}": round(v, 2) for k, v in pm.items()
                                           if k[1] in ("RB", "WR", "TE", "QB")}}
        out = pd.DataFrame({
            "player": df["name"].values, "player_id": df.index.values, "team": off, "pos": df["pos"].values,
            "status": df["status"].values,
            "p_anytime": (counts >= 1).mean(0), "p_2plus": (counts >= 2).mean(0),
            "exp_tds": counts.mean(0),
            "rush_share": df["rush_share"].values, "rec_share": df["rec_share"].values,
            "skill_mult": df["skill"].values})
        results.append(out)

    res = pd.concat(results).sort_values("p_anytime", ascending=False).reset_index(drop=True)
    res = res[res["p_anytime"] > 0.005]
    res["fair_odds"] = res["p_anytime"].apply(prob_to_american)
    meta = {"game": g.to_dict(), "teams": team_info, "notes": list(dict.fromkeys(notes)),
            "league_rush_td_pg": lg_r, "league_pass_td_pg": lg_p,
            "data_through": f"{season} week {week - 1}",
            "n_sims": p.n_sims,
            # bit-packed per-sim scoring indicators (8x smaller); unpack with parlays.unpack_hits
            "hits": {pid: np.packbits(hits[pid]) for pid in res["player_id"]}}
    return res, meta


def prob_to_american(prob: float) -> str:
    if prob <= 0 or prob >= 1:
        return "—"
    return f"{-100 * prob / (1 - prob):.0f}" if prob >= 0.5 else f"+{100 * (1 - prob) / prob:.0f}"


# ----------------------------------------------------------------------------
# Backtest: calibration on completed games
# ----------------------------------------------------------------------------
def backtest(season: int, weeks, p: Params | None = None, n_sims: int = 4000):
    p = p or Params()
    p.n_sims = n_sims
    sched = get_schedule(season)
    games = sched[sched["week"].isin(weeks) & sched["result"].notna()]
    full = _load("pbp", [season])
    rows, team_rows = [], []
    for gid in games["game_id"]:
        res, meta = run_game(gid, p, seed=1)
        gp = full[full["game_id"] == gid]
        tdp = gp[((gp["rush_touchdown"] == 1) | (gp["pass_touchdown"] == 1))
                 & (gp["td_team"] == gp["posteam"])]
        res["scored"] = res["player_id"].isin(set(tdp["td_player_id"])).astype(int)
        res["game_id"] = gid
        for t, info in meta["teams"].items():
            team_rows.append({"pred": info["sim_mean_td"], "actual": int((tdp["posteam"] == t).sum())})
        rows.append(res)
    bt = pd.concat(rows)
    bt["bin"] = pd.cut(bt["p_anytime"], [0, .1, .2, .3, .4, .5, .6, 1])
    cal = bt.groupby("bin", observed=True).agg(n=("scored", "size"), predicted=("p_anytime", "mean"),
                                               actual=("scored", "mean"))
    brier = ((bt["p_anytime"] - bt["scored"]) ** 2).mean()
    base = ((bt["scored"].mean() - bt["scored"]) ** 2).mean()
    tm = pd.DataFrame(team_rows)
    return cal, brier, base, tm
