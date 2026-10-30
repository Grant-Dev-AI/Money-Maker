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

    # ---- v2 upgrades (each can be switched off; Params.legacy() turns them all off) ----
    # (5) smooth recency: weight = 0.5 ** (weeks_ago / half_life); each offseason counts as extra weeks
    use_decay: bool = True
    team_half_life: float = 12.0          # weeks (tuned on 2024)
    offseason_gap: float = 8.0            # extra "weeks ago" added per offseason (roster turnover)
    player_half_life: float = 8.0         # weeks, within the current season (2.5-12 tested: all within noise)
    # (5) snap share: a player's recent snap % vs his own average scales his share (role changes)
    use_snaps: bool = False               # tested on 2024: no measurable gain, so off by default
    snap_recent_games: int = 2
    snap_beta: float = 0.5                # mult = (recent / average) ** beta
    snap_clip: tuple = (0.6, 1.5)
    # lineup-aware shares: recognize a starter back in the lineup (see lineup_shares)
    use_lineup: bool = True
    lineup_gamma: float = 3.0             # how hard to down-weight games a current starter missed
    lineup_shrink_games: float = 2.0      # pseudo-games of the prior role for small samples
    lineup_last_season: bool = True       # prior = role on this team last season (else: plain share this season)
    # (2) opponent-adjusted team ratings: ridge Poisson regression  log E[TD] = mu + off + def + home
    use_glm: bool = True
    glm_ridge: float = 16.0               # penalty on team ratings (pseudo-games pulling toward average)
    # (3) red-zone decomposition: E[TD] = E[RZ trips] x P(TD | trip) + E[TDs from outside the 20]
    use_rz: bool = True
    rz_blend: float = 1.0                 # weight of the red-zone estimate vs the direct TD-count estimate (tuned: 1.0)
    # (1) Vegas: blend the team total implied by the spread + total (geometric blend)
    use_vegas: bool = True
    vegas_weight: float = 0.9             # tuned on 2024 (1.0 was marginally better; 0.9 keeps the model in the mix)

    @classmethod
    def legacy(cls, **kw):
        """The original (v1) model: every v2 upgrade switched off."""
        return cls(use_decay=False, use_snaps=False, use_glm=False, use_rz=False, use_vegas=False,
                   use_lineup=False, **kw)


# ----------------------------------------------------------------------------
# Data loading (cached in-process)
# ----------------------------------------------------------------------------
_CACHE: dict = {}
PBP_COLS = ["game_id", "season", "week", "posteam", "defteam", "play_type", "two_point_attempt",
            "rush_touchdown", "pass_touchdown", "td_team", "td_player_id", "td_player_name",
            "rusher_player_id", "receiver_player_id", "yardline_100", "home_team", "fixed_drive"]


def _load(name, seasons):
    key = (name, tuple(seasons))
    if key not in _CACHE:
        fn = {"pbp": nfl.load_pbp, "sched": nfl.load_schedules,
              "rost": nfl.load_rosters_weekly, "inj": nfl.load_injuries,
              "snaps": nfl.load_snap_counts, "players": lambda _: nfl.load_players()}[name]
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
           p.older_season_weight, p.recent_games_boost, p.use_decay, p.team_half_life, p.offseason_gap)
    if key not in _CACHE:
        _CACHE[key] = _build_prep(season, week, p)
    return _CACHE[key]


def _build_prep(season: int, week: int, p: Params):
    seasons = list(range(season - p.seasons_back + 1, season + 1))
    pbp = _load("pbp", seasons)
    # only use information available before the game being simulated
    pbp = pbp[(pbp["season"] < season) | (pbp["week"] < week)].copy()
    pbp = pbp[pbp["play_type"].isin(["run", "pass"]) & (pbp["two_point_attempt"] != 1)]
    if p.use_decay:
        # smooth exponential decay by weeks before the target game; each offseason adds extra distance
        weeks_ago = (season - pbp["season"]) * (22 + p.offseason_gap) + week - pbp["week"]
        pbp["w"] = 0.5 ** (weeks_ago.clip(lower=0) / p.team_half_life)
    else:
        pbp["w"] = _game_weights(pbp, season, p)
    # recent-form boost: a team's 4 most recent games this season (v1 only)
    cur = pbp["season"] == season
    if cur.any() and not p.use_decay:
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
# 1b) v2 team model: opponent-adjusted ratings, red-zone decomposition, Vegas
# ----------------------------------------------------------------------------
def team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, offense): rush/pass TDs, red-zone trips, TDs inside / outside the 20, weight."""
    d = pbp.assign(td=pbp["rtd"] + pbp["ptd"], inside=pbp["yardline_100"] <= 20)
    d["rz_td"] = d["td"] * d["inside"]
    d["rz_drive"] = np.where(d["inside"], d["fixed_drive"], np.nan)
    g = (d.groupby(["game_id", "season", "week", "posteam", "defteam"])
         .agg(rtd=("rtd", "sum"), ptd=("ptd", "sum"), td=("td", "sum"), rz_td=("rz_td", "sum"),
              trips=("rz_drive", "nunique"), w=("w", "mean"), home_team=("home_team", "first"))
         .reset_index())
    g["long_td"] = g["td"] - g["rz_td"]
    g["home"] = (g["posteam"] == g["home_team"]).astype(float)
    return g


def _design(g, teams):
    """Columns: [intercept, offense one-hot, defense one-hot, home]."""
    ix = {t: i for i, t in enumerate(teams)}
    n, k = len(g), len(teams)
    X = np.zeros((n, 2 + 2 * k))
    X[:, 0] = 1
    X[np.arange(n), 1 + g["posteam"].map(ix).values] = 1
    X[np.arange(n), 1 + k + g["defteam"].map(ix).values] = 1
    X[:, -1] = g["home"].values
    return X


def _fit_glm(X, y, w, ridge, family="poisson", trials=None, iters=50):
    """Ridge-penalized GLM by Newton's method. Team ratings are penalized; intercept and home are not.
    poisson: log link on counts y.  binomial: logit link, y successes out of `trials`."""
    pen = np.full(X.shape[1], ridge)
    pen[0] = pen[-1] = 1e-6
    b = np.zeros(X.shape[1])
    if family == "poisson":
        b[0] = np.log(max(np.average(y, weights=w), 1e-6))
    else:
        b[0] = np.log(max(y.sum(), 1e-6) / max(trials.sum() - y.sum(), 1e-6))
    for _ in range(iters):
        eta = X @ b
        if family == "poisson":
            mu = np.exp(eta)
            grad = X.T @ (w * (y - mu)) - pen * b
            H = (X * (w * mu)[:, None]).T @ X + np.diag(pen)
        else:
            pr = 1 / (1 + np.exp(-eta))
            grad = X.T @ (w * (y - trials * pr)) - pen * b
            H = (X * (w * trials * pr * (1 - pr))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(H, grad)
        b += step
        if np.abs(step).max() < 1e-7:
            break
    return b


def _team_models(pbp, p: Params):
    """Fit the v2 team-level models once per (season, week, params)."""
    key = ("team_models", id(pbp), p.glm_ridge)
    if key in _CACHE:
        return _CACHE[key]
    g = team_games(pbp)
    teams = sorted(set(g["posteam"]) | set(g["defteam"]))
    X = _design(g, teams)
    w = g["w"].values
    m = {"teams": teams,
         "rush": _fit_glm(X, g["rtd"].values, w, p.glm_ridge),
         "pass": _fit_glm(X, g["ptd"].values, w, p.glm_ridge),
         "trips": _fit_glm(X, g["trips"].values.astype(float), w, p.glm_ridge),
         "conv": _fit_glm(X, g["rz_td"].values.astype(float), w, p.glm_ridge, "binomial",
                          trials=g["trips"].values.astype(float)),
         "long": _fit_glm(X, g["long_td"].values.astype(float), w, p.glm_ridge)}
    _CACHE[key] = m
    return m


def _glm_eta(b, teams, off, dfn, is_home):
    k = len(teams)
    ix = {t: i for i, t in enumerate(teams)}
    eta = b[0] + b[-1] * float(is_home)
    if off in ix:
        eta += b[1 + ix[off]]
    if dfn in ix:
        eta += b[1 + k + ix[dfn]]
    return eta


def _vegas_fit(season, week, p: Params):
    """Map Vegas-implied team points to offensive TDs: TDs = a + b * points, fit on past games."""
    key = ("vegas_fit", season, week, p.seasons_back)
    if key in _CACHE:
        return _CACHE[key]
    seasons = list(range(season - p.seasons_back + 1, season + 1))
    s = _load("sched", seasons)
    s = s[s["total_line"].notna() & s["spread_line"].notna() & s["result"].notna()
          & ((s["season"] < season) | (s["week"] < week))]
    pbp = _load("pbp", seasons)
    off = pbp[pbp["play_type"].isin(["run", "pass"]) & (pbp["two_point_attempt"] != 1)
              & ((pbp["rush_touchdown"] == 1) | (pbp["pass_touchdown"] == 1))
              & (pbp["td_team"] == pbp["posteam"])]
    tds = off.groupby(["game_id", "posteam"]).size()
    rows = []
    for _, r in s.iterrows():
        for team, pts in ((r["home_team"], (r["total_line"] + r["spread_line"]) / 2),
                          (r["away_team"], (r["total_line"] - r["spread_line"]) / 2)):
            rows.append((pts, tds.get((r["game_id"], team), 0)))
    a = np.array(rows, float)
    if len(a) < 50:
        fit = (0.0, 2.45 / 22.5)          # fallback: ~league average TDs per implied point
    else:
        b, c = np.polyfit(a[:, 0], a[:, 1], 1)
        fit = (c, b)
    _CACHE[key] = fit
    return fit


def implied_points(g, team):
    """Vegas-implied points for `team` in schedule row g (nflverse spread_line > 0 = home favored)."""
    tot, spr = g.get("total_line"), g.get("spread_line")
    if tot is None or spr is None or pd.isna(tot) or pd.isna(spr):
        return None
    return (tot + spr) / 2 if team == g["home_team"] else (tot - spr) / 2


def expected_tds_v2(pbp, rates, lg_r, lg_p, g, off, dfn, is_home, season, week, p: Params):
    """Expected rushing and passing TDs with the v2 upgrades that are switched on."""
    if p.use_glm or p.use_rz:
        m = _team_models(pbp, p)
    if p.use_glm:
        lam_r = float(np.exp(_glm_eta(m["rush"], m["teams"], off, dfn, is_home)))
        lam_p = float(np.exp(_glm_eta(m["pass"], m["teams"], off, dfn, is_home)))
    else:
        lam_r, lam_p = expected_tds(rates, lg_r, lg_p, off, dfn, is_home, p)
    total = lam_r + lam_p
    rush_frac = lam_r / total if total > 0 else 0.45
    info = {"direct": total}
    if p.use_rz:
        trips = np.exp(_glm_eta(m["trips"], m["teams"], off, dfn, is_home))
        conv = 1 / (1 + np.exp(-_glm_eta(m["conv"], m["teams"], off, dfn, is_home)))
        long_ = np.exp(_glm_eta(m["long"], m["teams"], off, dfn, is_home))
        rz_total = float(trips * conv + long_)
        info.update(rz_trips=float(trips), rz_conv=float(conv), rz_total=rz_total)
        total = p.rz_blend * rz_total + (1 - p.rz_blend) * total
    if p.use_vegas:
        pts = implied_points(g, off)
        if pts is not None:
            a, b = _vegas_fit(season, week, p)
            vegas = max(a + b * pts, 0.3)
            info.update(implied_pts=float(pts), vegas_td=float(vegas))
            total = float(np.exp(p.vegas_weight * np.log(vegas) + (1 - p.vegas_weight) * np.log(total)))
    info["final"] = total
    return total * rush_frac, total * (1 - rush_frac), info


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
def snap_mults(season, week, team, p: Params):
    """{gsis_id: multiplier} from each player's recent offensive snap % vs his own decayed average
    this season (only games he played, so a return from injury isn't read as a demotion)."""
    sn = _load("snaps", [season])
    pl = _load("players", [0])
    if sn.empty or pl.empty:
        return {}
    sn = sn[(sn["season"] == sn["season"].max()) & (sn["team"] == team) & (sn["week"] < week)
            & (sn["game_type"] == "REG") & sn["offense_pct"].notna()]
    if sn.empty:
        return {}
    pfr2gsis = pl.dropna(subset=["pfr_id", "gsis_id"]).set_index("pfr_id")["gsis_id"].to_dict()
    out = {}
    for pfr, d in sn.groupby("pfr_player_id"):
        gid = pfr2gsis.get(pfr)
        if gid is None or len(d) < p.snap_recent_games + 1:
            continue                         # not enough history to call it a role change
        d = d.sort_values("week")
        w = 0.5 ** ((d["week"].max() - d["week"]) / p.player_half_life)
        avg = np.average(d["offense_pct"], weights=w)
        recent = d["offense_pct"].tail(p.snap_recent_games).mean()
        if avg > 0.02:
            out[gid] = float(np.clip((max(recent, 0.01) / avg) ** p.snap_beta, *p.snap_clip))
    return out


_COMPS = {"carry": ("run", "rusher_player_id", None), "i10": ("run", "rusher_player_id", "i10"),
          "rtd": ("run", "rusher_player_id", "rtd"), "tgt": ("pass", "receiver_player_id", None),
          "rz": ("pass", "receiver_player_id", "rz"), "ptd": ("pass", "receiver_player_id", "ptd")}


def _comp_counts(tp):
    """{component: (games x players count matrix, games team-total series)}."""
    out = {}
    for c, (pt, col, m) in _COMPS.items():
        d = tp[tp["play_type"] == pt]
        if m == "i10":
            d = d[d["yardline_100"] <= 10]
        elif m == "rz":
            d = d[d["yardline_100"] <= 20]
        elif m in ("rtd", "ptd"):
            d = d[d[m] == 1]
        mat = d.groupby(["game_id", col]).size().unstack(fill_value=0)
        out[c] = (mat, d.groupby("game_id").size())
    return out


def _snap_presence(tp):
    """games x gsis_id matrix: 1 if the player took an offensive snap, 0 if the team's snap report
    for that game doesn't list him, NaN where there is no snap report for the game."""
    season, team = int(tp["season"].iloc[0]), tp["posteam"].iloc[0]
    key = ("snap_presence", season, team, tuple(sorted(tp["game_id"].unique())))
    if key in _CACHE:
        return _CACHE[key]
    sn, pl = _load("snaps", [season]), _load("players", [0])
    out = None
    if not sn.empty and not pl.empty:
        sn = sn[(sn["team"] == team) & sn["game_id"].isin(tp["game_id"].unique())]
        if not sn.empty:
            m = pl.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]
            sn = sn.assign(gsis=sn["pfr_player_id"].map(m)).dropna(subset=["gsis"])
            sn = sn[sn["offense_snaps"].fillna(0) > 0]
            out = (sn.assign(v=1.0).pivot_table(index="game_id", columns="gsis", values="v", aggfunc="max")
                   .fillna(0.0))
    _CACHE[key] = out
    return out


def lineup_shares(tp, prev, p: Params):
    """Lineup-aware opportunity shares (fixes 'starter is back' misses).
    1. Each player's share is measured only over games he played (a suspension or injury
       doesn't count as zero carries).
    2. Past games where someone in the current lineup (played the team's latest game) was missing
       are down-weighted by how big his role is, so the weeks a backup started while the starter
       was out count less once the starter is back.
    3. Small samples are shrunk toward the player's role on this team last season (or, without
       one, toward his plain share this season), then shares are re-normalized to add up."""
    games = tp.groupby("game_id")["week"].first().sort_values()
    gid = games.index
    gw = tp.groupby("game_id")["rw"].first().reindex(gid).values
    cc = _comp_counts(tp)
    ids = sorted(set(cc["carry"][0].columns) | set(cc["tgt"][0].columns))
    cnt = {c: m.reindex(index=gid, columns=ids, fill_value=0).values.astype(float) for c, (m, _) in cc.items()}
    tot = {c: t.reindex(gid, fill_value=0).values.astype(float) for c, (_, t) in cc.items()}
    app = ((cnt["carry"] + cnt["tgt"]) > 0).astype(float)          # games x players: did he play
    on_field = _snap_presence(tp)
    if on_field is not None:
        # snap counts know who was on the field even without a touch (a WR with no targets still played)
        on = on_field.reindex(index=gid, columns=ids).values
        app = np.where(np.isnan(on), app, np.maximum(app, np.nan_to_num(on)))

    def ingame(c, w):
        num = (cnt[c] * w[:, None]).sum(0)
        den = (app * (tot[c] * w)[:, None]).sum(0)
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    active = app[-1] > 0                                           # in the lineup in the latest game
    weights = {}
    for side, base in (("run", "carry"), ("pass", "tgt")):
        s0 = ingame(base, gw)
        missing = ((1 - app) * (s0 * active)[None, :]).sum(1)     # role size of active players absent that game
        weights[side] = gw * np.exp(-p.lineup_gamma * missing)

    prior_cc = _comp_counts(prev) if len(prev) else {}
    df = pd.DataFrame(index=ids)
    for c, (pt, _, _) in _COMPS.items():
        w = weights[pt]
        s = ingame(c, w)
        n_eff = (app * w[:, None]).sum(0) / max(w.max(), 1e-9)    # effective games played
        total_w = (tot[c] * w).sum()
        raw = (cnt[c] * w[:, None]).sum(0) / total_w if total_w > 0 else np.zeros(len(ids))
        prior = raw.copy()
        if p.lineup_last_season and c in prior_cc and prior_cc[c][1].sum() > 0:
            pm, pt_ = prior_cc[c]
            last = (pm.sum(0) / pt_.sum()).reindex(ids)
            prior = np.where(last.notna(), last.fillna(0).values, raw)
        k = p.lineup_shrink_games
        shr = (n_eff * s + k * prior) / (n_eff + k)
        target = 1.0 if total_w > 0 else 0.0
        df[c] = shr * target / shr.sum() if shr.sum() > 0 else 0.0
    return df


def player_shares(pbp, team, season, positions, names, pos_mult, p: Params, snap_mult=None):
    tp = pbp[(pbp["posteam"] == team) & (pbp["season"] == season)]
    if tp.empty:  # early season fallback: last season for this team
        tp = pbp[(pbp["posteam"] == team) & (pbp["season"] == season - 1)]
    # recency-weight within season (later weeks count more)
    wk = tp["week"]
    if p.use_decay and len(tp):
        rw = 0.5 ** ((wk.max() - wk) / p.player_half_life)
    else:
        rw = 1 + (wk - wk.min()) / max(1, (wk.max() - wk.min())) if len(tp) else wk
    tp = tp.assign(rw=rw)

    runs = tp[tp["play_type"] == "run"]
    passes = tp[tp["play_type"] == "pass"]

    if p.use_lineup and len(tp):
        prev = pbp[(pbp["posteam"] == team) & (pbp["season"] == tp["season"].iloc[0] - 1)]
        df = lineup_shares(tp, prev, p)
        ids = list(df.index)
    else:
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
    if snap_mult:
        sm = np.array([snap_mult.get(i, 1.0) for i in ids])
        df["snap_mult"] = sm
        df["rush_share"] *= sm
        df["rec_share"] *= sm
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
        lam_r, lam_p, tinfo = expected_tds_v2(pbp, rates, lg_r, lg_p, g, off, dfn, is_home, season, week, p)
        pm = opp_position_mults(pbp, positions, dfn, p)
        sm = snap_mults(season, week, off, p) if p.use_snaps else None
        df = player_shares(pbp, off, season, positions, names, pm, p, sm)
        df, n_ = apply_availability(df, off, season, week, p, overrides)
        notes += [f"{off}: {x}" for x in n_]
        team_env = env * rng.gamma(p.game_dispersion * 2, 1 / (p.game_dispersion * 2), p.n_sims)
        counts, team_tds = simulate_team(df, lam_r, lam_p, team_env, rng)
        for j, pid in enumerate(df.index):
            hits[pid] = counts[:, j] >= 1
        team_info[off] = {"exp_rush_td": lam_r, "exp_pass_td": lam_p,
                          "sim_mean_td": team_tds.mean(), "team_model": tinfo,
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
