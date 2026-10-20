"""
Suggested anytime-TD parlays.

Each leg's payout is the live Kalshi multiplier (Yes ask + fee) when a market exists,
otherwise the model's fair payout (1 / model probability). Parlay payout = product of legs.

Hit probability comes from the model. Legs from different games are treated as independent;
two legs from the same game use the joint probability from the simulation, which keeps
the game-script correlation (in a shootout, everyone's odds go up together).

Rules for every parlay: at most 2 players from one game, no two players from the same team,
and no two suggestions with the same set of players. Each one is the highest model hit
probability that fits its target:
  Easy Money  - 2 players,   about 5x
  Cash Cow    - 2 players,   about 7x
  10 Bands    - 2-3 players, about 10x
  Longshotta  - 2-4 players, 12-15x
"""
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# name: (legs allowed, payout band, fallback band if nothing fits, target payout for tie-breaks)
SPECS = {
    "Easy Money": ((2,), (4.5, 5.5), (4.0, 6.0), 5.0),
    "Cash Cow": ((2,), (6.5, 7.5), (6.0, 8.0), 7.0),
    "10 Bands": ((2, 3), (9.0, 11.0), (8.5, 12.0), 10.0),
    "Longshotta": ((2, 3, 4), (12.0, 15.0), (11.0, 16.0), 13.5),
}
MIN_LEG_P = 0.12            # ignore legs the model gives under 12%
POOL_SIZE = 45              # most likely scorers considered for 3-4 leg parlays
MAX_PER_GAME = 2


@dataclass
class Leg:
    pid: str
    player: str
    team: str
    pos: str
    game_id: str
    game: str
    p: float
    payout: float
    source: str             # "Kalshi" or "Model"
    hits: np.ndarray = field(repr=False)


@dataclass
class Parlay:
    name: str
    legs: list
    payout: float
    hit_prob: float
    note: str = ""

    @property
    def ev(self):
        return self.hit_prob * self.payout - 1

    @property
    def ids(self):
        return frozenset(l.pid for l in self.legs)


def unpack_hits(packed, n):
    return np.unpackbits(packed, count=n).astype(bool)


def shootout_scores(sims: dict, games: pd.DataFrame) -> pd.DataFrame:
    """Rank games by projected scoring and how even the two offenses are."""
    rows = []
    for _, g in games.iterrows():
        if g["game_id"] not in sims:
            continue
        t = sims[g["game_id"]][1]["teams"]
        a, h = t[g["away_team"]]["sim_mean_td"], t[g["home_team"]]["sim_mean_td"]
        total, balance = a + h, min(a, h) / max(a, h)
        rows.append({"game_id": g["game_id"], "game": f"{g['away_team']} @ {g['home_team']}",
                     "proj_tds": total, "balance": balance,
                     # mostly total scoring, with a bump for both offenses being able to keep pace
                     "shootout": total * (0.7 + 0.3 * balance)})
    return pd.DataFrame(rows).sort_values("shootout", ascending=False).reset_index(drop=True)


def build_legs(sims: dict, games: pd.DataFrame, kalshi_mult: dict) -> list[Leg]:
    """kalshi_mult: {player_id: payout multiplier incl. fee} for players with a live market."""
    legs = []
    for _, g in games.iterrows():
        if g["game_id"] not in sims:
            continue
        res, meta = sims[g["game_id"]]
        n = meta["n_sims"]
        for _, r in res[res["p_anytime"] >= MIN_LEG_P].iterrows():
            k = kalshi_mult.get(r["player_id"])
            legs.append(Leg(r["player_id"], r["player"], r["team"], r["pos"], g["game_id"],
                            f"{g['away_team']} @ {g['home_team']}", float(r["p_anytime"]),
                            float(k) if k else 1 / float(r["p_anytime"]), "Kalshi" if k else "Model",
                            unpack_hits(meta["hits"][r["player_id"]], n)))
    return sorted(legs, key=lambda l: -l.p)


def _hit_prob(combo):
    by_game = {}
    for l in combo:
        by_game.setdefault(l.game_id, []).append(l)
    prob = 1.0
    for ls in by_game.values():
        prob *= ls[0].p if len(ls) == 1 else float(np.mean(ls[0].hits & ls[1].hits))
    return prob


def _search(legs, sizes, band, target, exclude=()):
    best = None
    for k in sizes:
        for combo in itertools.combinations(legs, k):
            payout = float(np.prod([l.payout for l in combo]))
            if not band[0] <= payout <= band[1]:
                continue
            if len({l.team for l in combo}) < k:      # no two players from the same team
                continue
            if max(Counter(l.game_id for l in combo).values()) > MAX_PER_GAME:
                continue
            ids = frozenset(l.pid for l in combo)
            if ids in exclude:
                continue
            prob = _hit_prob(combo)
            # tie-break toward the target payout
            key = (round(prob, 4), -abs(payout - target))
            if best is None or key > best[0]:
                best = (key, combo, payout, prob)
    return best


def suggest(sims: dict, games: pd.DataFrame, kalshi_mult: dict) -> tuple[list[Parlay], pd.DataFrame]:
    legs = build_legs(sims, games, kalshi_mult)
    out, used = [], set()
    for name, (sizes, band, wide, target) in SPECS.items():
        # 2-leg searches can use every leg; 3-4 leg searches use the most likely scorers only
        pool = legs if max(sizes) == 2 else legs[:POOL_SIZE]
        hit = _search(pool, sizes, band, target, used) or _search(pool, sizes, wide, target, used)
        if not hit:
            continue
        lo, hi = band
        note = (f"Best model hit chance for {'2' if sizes == (2,) else f'{sizes[0]}–{sizes[-1]}'} players "
                f"at {lo:g}–{hi:g}x" + ("." if lo <= hit[2] <= hi else " (nothing fit, so the range was widened)."))
        out.append(Parlay(name, list(hit[1]), hit[2], hit[3], note))
        used.add(out[-1].ids)
    return out, shootout_scores(sims, games)
