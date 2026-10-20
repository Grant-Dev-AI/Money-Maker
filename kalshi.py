"""
Kalshi market data for NFL anytime-touchdown markets.

Uses Kalshi's public market-data API (no account or API key needed to read prices).
Docs: https://docs.kalshi.com

Flow:
1. Find the NFL anytime-TD series (auto-discovered by title, or set manually).
2. Find the event (game) in that series matching the selected matchup.
3. Read each player market's Yes ask price = cost to buy "Yes, he scores".
4. Match Kalshi player names to model players and compute edge / EV.
"""
from __future__ import annotations

import difflib
import math
import re
import unicodedata

import pandas as pd
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
# Kalshi's NFL player touchdown props live in "Pro Football Touchdowns" (KXNFLTD):
# one market per player per threshold (1+, 2+, 3+), plus a D/ST market per team.
PREFERRED_SERIES = ["KXNFLTD", "KXNFLANYTD"]
TIMEOUT = 15

# Kalshi sometimes uses different team codes than nflverse
TEAM_ALIASES = {
    "LA": ["LA", "LAR"], "LAR": ["LA", "LAR"], "WAS": ["WAS", "WSH"], "JAX": ["JAX", "JAC"],
    "LV": ["LV", "OAK"], "ARI": ["ARI", "ARZ"],
}
TEAM_NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers",
    "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos",
    "DET": "Lions", "GB": "Packers", "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars",
    "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants", "NYJ": "Jets",
    "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers",
    "TEN": "Titans", "WAS": "Commanders",
}


class KalshiError(Exception):
    pass


def _get(path, params=None):
    try:
        r = requests.get(f"{BASE}{path}", params=params, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise KalshiError(f"Could not reach Kalshi: {e}") from e
    if r.status_code != 200:
        raise KalshiError(f"Kalshi returned HTTP {r.status_code} for {path}: {r.text[:200]}")
    return r.json()


def _paged(path, key, params):
    out, cursor = [], None
    for _ in range(20):
        p = dict(params, limit=200)
        if cursor:
            p["cursor"] = cursor
        js = _get(path, p)
        out += js.get(key, [])
        cursor = js.get("cursor")
        if not cursor:
            break
    return out


# ----------------------------------------------------------------------------
# 1) Series discovery
# ----------------------------------------------------------------------------
def find_td_series() -> list[dict]:
    """Return candidate NFL anytime-touchdown series, best match first."""
    js = _get("/series", {"category": "Sports"})
    series = js.get("series", []) or []
    cands = []
    for s in series:
        title = (s.get("title") or "").lower()
        tick = (s.get("ticker") or "").upper()
        tags = " ".join(s.get("tags") or []).lower()
        is_nfl = ("nfl" in title or "pro football" in title or "football" in tags
                  or tick.startswith("KXNFL"))
        if not (is_nfl and "touchdown" in title):
            continue
        if any(x in title for x in ["first", "last", "season", "2+", "two or more", "total", "passing",
                                     "d/st", "team", "half", "next", "protected", "longest", "shortest",
                                     "multiple", "career", "leader", "quarter", "time", "rushing",
                                     "receiving"]):
            continue
        score = 2 if "anytime" in title else 1
        cands.append({"ticker": s.get("ticker"), "title": s.get("title"), "score": score})
    return sorted(cands, key=lambda c: -c["score"])


# ----------------------------------------------------------------------------
# 2) Event (game) matching
# ----------------------------------------------------------------------------
def _codes(team):
    return TEAM_ALIASES.get(team, [team])


def find_event(series_ticker: str, away: str, home: str):
    """Find the open event in a series for the given matchup."""
    events = _paged("/events", "events", {"series_ticker": series_ticker, "status": "open"})
    return _match_event(events, away, home), events


def _match_event(events, away: str, home: str):
    best = None
    for ev in events:
        tick = (ev.get("event_ticker") or "").upper()
        text = f"{ev.get('title', '')} {ev.get('sub_title', '')}".lower()
        suffix = tick.split("-")[-1] if "-" in tick else tick
        code_hit = any(a in suffix for a in _codes(away)) and any(h in suffix for h in _codes(home))
        # check full matchup string at end of ticker to avoid e.g. NE matching NEW
        strict = any(suffix.endswith(a + h) for a in _codes(away) for h in _codes(home))
        name_hit = (TEAM_NAMES.get(away, "~").lower() in text and TEAM_NAMES.get(home, "~").lower() in text)
        score = 3 * strict + 2 * name_hit + code_hit
        if score and (best is None or score > best[0]):
            best = (score, ev)
    return best[1] if best else None


# ----------------------------------------------------------------------------
# 3) Markets / prices
# ----------------------------------------------------------------------------
def _price(m, field):
    """Kalshi gives prices in cents (int) and/or dollars (string). Return dollars or None."""
    d = m.get(f"{field}_dollars")
    if d not in (None, ""):
        try:
            return float(d)
        except ValueError:
            pass
    c = m.get(field)
    if c is None:
        return None
    return float(c) / 100.0


def _num(m, field):
    """Kalshi sends counts as ints ('volume') or fixed-point strings ('volume_fp')."""
    for k in (field, f"{field}_fp"):
        v = m.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


_TITLE_RE = re.compile(r"will (.+?) (score|record)", re.I)
_THRESH_PATTERNS = [
    re.compile(r"(\d+)\s*\+"),                         # '1+', '2+ TDs'
    re.compile(r"at least (\d+)", re.I),                # 'at least 2 touchdowns'
    re.compile(r"(\d+) or more", re.I),                 # '2 or more'
    re.compile(r"over (\d+(?:\.\d+)?)", re.I),          # 'over 1.5'
]


def _threshold(m):
    """Number of TDs the market requires (1 = anytime TD). None if it can't be determined."""
    fs = m.get("floor_strike")
    if isinstance(fs, (int, float)):
        return math.floor(fs) + 1 if fs != int(fs) else int(fs)   # 0.5 -> 1, 1.5 -> 2, 1 -> 1
    text = " ".join(str(m.get(k) or "") for k in ("yes_sub_title", "subtitle", "title", "rules_primary"))
    for pat in _THRESH_PATTERNS:
        mt = pat.search(text)
        if mt:
            v = float(mt.group(1))
            return math.floor(v) + 1 if "over" in pat.pattern else int(v)
    if re.search(r"\b(a|any|anytime) touchdown\b", text, re.I):
        return 1
    mt = re.search(r"-(\d)$", m.get("ticker") or "")         # ticker ending '-1', '-2', ...
    return int(mt.group(1)) if mt else None


def _player_from_market(m):
    for k in ("yes_sub_title", "subtitle", "no_sub_title"):
        v = (m.get(k) or "").strip()
        if v and v.lower() not in ("yes", "no"):
            v = re.sub(r"\s*:\s*\d+\+?\s*$", "", v)          # drop trailing ': 1+' style suffix
            v = re.sub(r"\s*\(.*?\)\s*$", "", v)              # drop trailing '(DET)'
            return v
    mt = _TITLE_RE.search(m.get("title") or "")
    return mt.group(1) if mt else (m.get("title") or m.get("ticker"))


def get_markets(event_ticker: str) -> pd.DataFrame:
    mk = _paged("/markets", "markets", {"event_ticker": event_ticker})
    rows = []
    for m in mk:
        if m.get("status") not in (None, "active", "open", "initialized"):
            continue
        label = f"{m.get('yes_sub_title', '')} {m.get('title', '')}"
        if "D/ST" in label.upper() or "DEFENSE" in label.upper():
            continue  # team defense / special teams markets, not player props
        rows.append({
            "kalshi_name": _player_from_market(m),
            "tds_required": _threshold(m),
            "market_title": m.get("yes_sub_title") or m.get("title"),
            "ticker": m.get("ticker"),
            "yes_ask": _price(m, "yes_ask"),
            "yes_bid": _price(m, "yes_bid"),
            "last": _price(m, "last_price"),
            "volume": _num(m, "volume"),
            "open_interest": _num(m, "open_interest"),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4) Name matching + EV
# ----------------------------------------------------------------------------
def _norm(name: str) -> str:
    n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[.\-'’]", " ", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def match_names(kalshi_names, model_names) -> dict:
    """Map kalshi name -> model name."""
    model_norm = {_norm(m): m for m in model_names}
    out = {}
    for k in kalshi_names:
        kn = _norm(k)
        if kn in model_norm:
            out[k] = model_norm[kn]
            continue
        parts = kn.split()
        if len(parts) >= 2:
            # 'j gibbs' -> first initial + last name
            hits = [m for mn, m in model_norm.items()
                    if mn.split()[-1] == parts[-1] and mn[0] == parts[0][0]]
            if len(hits) == 1:
                out[k] = hits[0]
                continue
        close = difflib.get_close_matches(kn, list(model_norm), n=1, cutoff=0.85)
        if close:
            out[k] = model_norm[close[0]]
    return out


def anytime_only(markets: pd.DataFrame) -> pd.DataFrame:
    """Keep only 1+ touchdown (anytime) markets: one per player.

    Kalshi's touchdown events can list 1+, 2+, 3+ TD markets for each player.
    Markets known to need 2+ TDs are dropped. If any are still duplicated for a player,
    keep the highest-priced one, since P(1+ TD) is always >= P(2+ TDs).
    """
    mk = markets.copy()
    if "tds_required" in mk:
        known_one = mk["tds_required"] == 1
        if known_one.any():
            mk = mk[known_one]
        else:
            mk = mk[mk["tds_required"].isna() | (mk["tds_required"] <= 1)]
    mk = (mk.assign(_px=mk["yes_ask"].fillna(mk["last"]).fillna(0))
            .sort_values("_px", ascending=False)
            .drop_duplicates("kalshi_name").drop(columns="_px"))
    return mk


def kalshi_fee(price: float) -> float:
    """Kalshi taker fee per contract: 0.07 x P x (1-P), rounded up to the cent
    (charged per order; per-contract approximation for a single contract)."""
    if price is None or price <= 0 or price >= 1:
        return 0.0
    return math.ceil(0.07 * price * (1 - price) * 100) / 100


def build_ev_table(res: pd.DataFrame, markets: pd.DataFrame, include_fees: bool = True,
                   thin_spread: float = 0.10, thin_volume: int = 100) -> tuple[pd.DataFrame, list]:
    if markets.empty:
        return pd.DataFrame(), []
    markets = anytime_only(markets)
    mapping = match_names(markets["kalshi_name"].tolist(), res["player"].tolist())
    mk = markets.copy()
    mk["player"] = mk["kalshi_name"].map(mapping)
    unmatched = mk.loc[mk["player"].isna(), "kalshi_name"].tolist()
    mk = mk.dropna(subset=["player"])
    df = mk.merge(res[["player", "team", "pos", "status", "p_anytime"]], on="player", how="left")
    df = df[df["yes_ask"].notna() & (df["yes_ask"] > 0) & (df["yes_ask"] < 1)]
    df["fee"] = df["yes_ask"].apply(kalshi_fee) if include_fees else 0.0
    df["cost"] = df["yes_ask"] + df["fee"]                     # all-in price per $1 contract
    df["market_pct"] = df["cost"]                              # implied probability you pay
    df["edge_pts"] = df["p_anytime"] - df["market_pct"]
    df["ev_per_dollar"] = df["p_anytime"] / df["cost"] - 1
    df["multiplier"] = 1 / df["cost"]
    spread = df["yes_ask"] - df["yes_bid"].fillna(0)
    df["liquidity"] = ["Thin" if (s > thin_spread or v < thin_volume) else "OK"
                       for s, v in zip(spread, df["volume"])]
    df = df.sort_values("ev_per_dollar", ascending=False).reset_index(drop=True)
    return df, unmatched


def fetch_game(away: str, home: str, series_override: str | None = None):
    """Convenience: discover series, find event, return (markets_df, info dict)."""
    info = {"series_candidates": [], "series": None, "event": None, "events_seen": []}
    if series_override:
        tickers = [series_override.strip().upper()]
    else:
        try:
            cands = find_td_series()
        except KalshiError:
            cands = []
        info["series_candidates"] = cands
        tickers = PREFERRED_SERIES + [c["ticker"] for c in cands if c["ticker"] not in PREFERRED_SERIES]
        if not tickers:
            raise KalshiError("No NFL anytime-touchdown series found on Kalshi. "
                              "If you know the series ticker, enter it in the sidebar.")
    for t in tickers:
        ev, events = find_event(t, away, home)
        info["events_seen"] += [e.get("event_ticker") for e in events][:40]
        if ev:
            info["series"], info["event"] = t, ev
            return get_markets(ev["event_ticker"]), info
    raise KalshiError(f"No open Kalshi touchdown event found for {away} @ {home}. "
                      "Markets usually open a few days before kickoff.")


def fetch_week(matchups, series_override: str | None = None) -> dict:
    """Markets for many games at once. Each series' open-event list is fetched only once.

    matchups: list of (away, home). Returns {(away, home): (markets_df, series, event)};
    games with no open Kalshi event are left out.
    """
    tickers = [series_override.strip().upper()] if series_override else list(PREFERRED_SERIES)
    events_by_series, out, errors = {}, {}, []

    def events(t):
        if t not in events_by_series:
            try:
                events_by_series[t] = _paged("/events", "events", {"series_ticker": t, "status": "open"})
            except KalshiError as e:
                events_by_series[t] = []
                errors.append(str(e))
        return events_by_series[t]

    def try_series(series_list, games):
        for away, home in games:
            for t in series_list:
                ev = _match_event(events(t), away, home)
                if ev:
                    out[(away, home)] = (get_markets(ev["event_ticker"]), t, ev)
                    break

    try_series(tickers, matchups)
    missing = [g for g in matchups if g not in out]
    if missing and not series_override:
        try:
            extra = [c["ticker"] for c in find_td_series() if c["ticker"] not in tickers]
        except KalshiError:
            extra = []
        try_series(extra, missing)
    if not out and errors:   # couldn't talk to Kalshi at all: say so instead of "no markets"
        raise KalshiError(errors[0])
    return out


if __name__ == "__main__":
    # Quick diagnostic:  python kalshi.py GB DET
    import sys
    away, home = (sys.argv[1:3] if len(sys.argv) >= 3 else ("GB", "DET"))
    print("Series candidates:", find_td_series())
    try:
        mk, info = fetch_game(away, home)
        print("Using series", info["series"], "event", info["event"]["event_ticker"])
        print(mk.to_string())
    except KalshiError as e:
        print("ERROR:", e)
