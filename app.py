"""
Money Printing Machine — NFL anytime-TD model + Kalshi edge finder.
Run:  streamlit run app.py
"""
import html
import random
from datetime import datetime

import pandas as pd
import streamlit as st

import engine
import kalshi
import parlays

st.set_page_config(page_title="Money Printing Machine", page_icon="🏈", layout="wide",
                   initial_sidebar_state="collapsed")

MIN_MODEL_PCT = 0.15     # arbitrage list: ignore players the model gives under 15%
TOP_N = 8                # parlay view: scorers shown per game
SEED = 7

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Graduate&family=Barlow+Condensed:wght@400;500;600;700&display=swap');
:root{
  --field:#0c3a1f; --field2:#0f4424; --chalk:rgba(255,255,255,.07);
  --ink:#f3f6ef; --muted:#a9c2ad; --cash:#8fd16a; --gold:#f5c542; --red:#ff7a6b;
  --card:rgba(6,28,14,.78); --line:rgba(255,255,255,.12);
}
html, body, [data-testid="stAppViewContainer"], .stApp{
  background:
    repeating-linear-gradient(90deg, transparent 0 calc(10% - 2px), var(--chalk) calc(10% - 2px) 10%),
    repeating-linear-gradient(90deg, var(--field) 0 10%, var(--field2) 10% 20%) !important;
  color:var(--ink); font-family:'Barlow Condensed', sans-serif;
}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"], [data-testid="collapsedControl"]{display:none;}
.block-container{padding-top:2rem; max-width:1300px; position:relative; z-index:2;}
h1,h2,h3,p,label,span,div{font-family:'Barlow Condensed', sans-serif;}

/* ---------- home ---------- */
.mpm-title{font-family:'Graduate', 'Arial Black', Impact, sans-serif; text-align:center; color:var(--gold);
  font-size:clamp(2.6rem, 8vw, 6.2rem); line-height:1.02; margin:7vh 0 1.2rem; letter-spacing:.02em;
  text-shadow:0 4px 0 #6b4e00, 0 10px 30px rgba(0,0,0,.55);}
.mpm-sub{text-align:center; color:var(--muted); font-size:1.15rem; letter-spacing:.3em;
  text-transform:uppercase; margin-bottom:3.5rem;}
.st-key-go_parlay button, .st-key-go_arb button{
  min-height:150px; border-radius:18px; border:3px solid var(--gold) !important;
  background:linear-gradient(180deg, rgba(10,45,22,.92), rgba(4,22,10,.95));
  box-shadow:0 10px 30px rgba(0,0,0,.45), inset 0 0 0 1px rgba(255,255,255,.06);
  transition:transform .15s ease, box-shadow .15s ease, border-color .15s ease;}
.st-key-go_parlay, .st-key-go_arb, .mpm-title, .mpm-sub{position:relative; z-index:3;}
.st-key-go_parlay button *, .st-key-go_arb button *{white-space:normal !important; overflow:visible !important;
  text-overflow:clip !important;}
.st-key-go_parlay button p, .st-key-go_arb button p{
  font-family:'Graduate', 'Arial Black', Impact, sans-serif; font-size:clamp(1.3rem, 2.4vw, 1.9rem);
  color:var(--ink); line-height:1.2;}
.stButton button, .stDownloadButton button{background:rgba(6,28,14,.85); color:var(--ink);
  border:1px solid var(--gold);}
.stButton button:hover, .stDownloadButton button:hover{border-color:var(--cash); color:var(--cash);}
[data-baseweb="select"] > div{background:rgba(6,28,14,.85); color:var(--ink); border-color:var(--gold);}
.st-key-go_parlay button:hover, .st-key-go_arb button:hover{
  transform:translateY(-4px); border-color:var(--cash) !important; box-shadow:0 16px 40px rgba(0,0,0,.55);}

.rain{position:fixed; inset:0; pointer-events:none; overflow:hidden; z-index:0;}
.rain span{position:absolute; top:-12vh; animation-name:fall; animation-timing-function:linear;
  animation-iteration-count:infinite; filter:drop-shadow(0 6px 6px rgba(0,0,0,.35)); opacity:.75;}
@keyframes fall{
  0%{transform:translate(0,-10vh) rotate(0deg);}
  50%{transform:translate(var(--sway),55vh) rotate(calc(var(--spin) / 2));}
  100%{transform:translate(0,120vh) rotate(var(--spin));}
}
@media (prefers-reduced-motion: reduce){ .rain{display:none;} }

/* ---------- result pages ---------- */
.page-title{font-family:'Graduate', 'Arial Black', Impact, sans-serif; color:var(--gold); font-size:2.3rem; margin:0;
  text-shadow:0 3px 0 #6b4e00;}
.page-note{color:var(--muted); font-size:1.05rem; margin:.2rem 0 1rem;}
.card{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px 16px 8px;
  margin-bottom:18px; backdrop-filter:blur(2px);}
.card-head{display:flex; justify-content:space-between; align-items:baseline; gap:10px;
  border-bottom:1px solid var(--line); padding-bottom:8px; margin-bottom:6px;}
.matchup{font-family:'Graduate', 'Arial Black', Impact, sans-serif; font-size:1.35rem; color:var(--ink);}
.kick{color:var(--muted); font-size:1rem; white-space:nowrap;}
.final{color:var(--gold); font-weight:600;}
table.tbl{width:100%; border-collapse:collapse; font-size:1.05rem;}
.tbl th{color:var(--muted); font-weight:500; text-align:left; font-size:.85rem; letter-spacing:.08em;
  text-transform:uppercase; padding:4px 6px;}
.tbl td{padding:5px 6px; border-top:1px solid rgba(255,255,255,.06); color:var(--ink);}
.tbl td.num, .tbl th.num{text-align:right; font-variant-numeric:tabular-nums;}
.tbl .rk{color:var(--muted); width:24px;}
.tbl .tm{color:var(--muted); font-size:.92rem;}
.tbl .flag{color:var(--red); font-size:.85rem; margin-left:6px;}
.bar{position:relative; height:8px; background:rgba(255,255,255,.08); border-radius:4px; min-width:70px;}
.bar i{position:absolute; left:0; top:0; bottom:0; background:var(--cash); border-radius:4px;}
.edge{color:var(--cash); font-weight:700;}
.thin{color:var(--gold); font-size:.85rem;}
.stButton button{font-family:'Barlow Condensed', sans-serif;}
.section{font-family:'Graduate', 'Arial Black', Impact, sans-serif; color:var(--ink); font-size:1.5rem;
  margin:1.2rem 0 .6rem; letter-spacing:.04em;}
.big{font-family:'Graduate', 'Arial Black', Impact, sans-serif; font-size:1.6rem; color:var(--ink);}
.src{font-size:.75rem; color:var(--muted); border:1px solid var(--line); border-radius:4px; padding:0 4px;}
.stats{display:flex; justify-content:space-between; gap:8px; margin-top:8px; padding-top:8px;
  border-top:1px solid var(--line); color:var(--muted); font-size:1.05rem;}
.stats b{color:var(--ink);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def money_rain(n=28):
    rnd = random.Random(42)
    items = ["🏈", "💵", "💸", "🏈", "💵", "💰"]
    spans = []
    for i in range(n):
        spans.append(
            f'<span style="left:{rnd.uniform(0, 97):.1f}%;font-size:{rnd.uniform(1.3, 2.7):.2f}rem;'
            f'animation-duration:{rnd.uniform(7, 15):.1f}s;animation-delay:-{rnd.uniform(0, 15):.1f}s;'
            f'--sway:{rnd.uniform(-60, 60):.0f}px;--spin:{rnd.choice([-1, 1]) * rnd.randint(180, 720)}deg">'
            f'{items[i % len(items)]}</span>')
    st.markdown(f'<div class="rain">{"".join(spans)}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data (cached)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def schedule(season):
    return engine.get_schedule(season)


@st.cache_data(ttl=3600, show_spinner=False)
def simulate_game(game_id):
    # game script (9.0) and home-field edge (4%) use the model's default settings
    return engine.run_game(game_id, engine.Params(), seed=SEED)


@st.cache_data(ttl=60, show_spinner=False)
def kalshi_week(matchups):
    return kalshi.fetch_week(list(matchups))


def _kick_label_safe(g):
    """'Sun Oct 25 · 1:00 PM ET' (built by hand: strftime's no-padding flags differ on Windows)."""
    try:
        d = datetime.strptime(f"{g['gameday']} {g.get('gametime') or '00:00'}", "%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(g["gameday"])
    return f"{d:%a %b} {d.day} · {d.hour % 12 or 12}:{d:%M} {'AM' if d.hour < 12 else 'PM'} ET"


def week_games(season, week):
    s = schedule(season)
    g = s[s["week"] == week].copy()
    g["_t"] = pd.to_datetime(g["gameday"].astype(str) + " " + g["gametime"].fillna("00:00").astype(str),
                             errors="coerce")
    return g.sort_values(["_t", "game_id"]).reset_index(drop=True)


def run_week(games):
    """Simulate every game in the week, with a progress bar. Returns {game_id: (res, meta)}."""
    out, errors = {}, {}
    bar = st.progress(0.0, text="Warming up the printer… (first run downloads ~3 seasons of play-by-play)")
    for i, g in games.iterrows():
        bar.progress(i / len(games), text=f"Simulating {g['away_team']} @ {g['home_team']} ({i + 1}/{len(games)})")
        try:
            out[g["game_id"]] = simulate_game(g["game_id"])
        except Exception as e:  # keep the rest of the week running
            errors[g["game_id"]] = f"{type(e).__name__}: {e}"
    bar.empty()
    return out, errors


def week_picker():
    season = engine.current_season()
    sched = schedule(season)
    upcoming = sched[sched["result"].isna()]
    default = int(upcoming["week"].min()) if len(upcoming) else int(sched["week"].max())
    weeks = sorted(int(w) for w in sched["week"].unique())
    if "week" not in st.session_state:
        st.session_state.week = default
    week = st.selectbox("Week", weeks, key="week", format_func=lambda w: f"Week {w}",
                        label_visibility="collapsed")
    return season, week


def header(title, note):
    c = st.columns([1.2, 6, 1.4])
    if c[0].button("🏠 Home", width="stretch"):
        st.session_state.page = "home"
        st.rerun()
    c[1].markdown(f'<p class="page-title">{title}</p><p class="page-note">{note}</p>', unsafe_allow_html=True)
    with c[2]:
        return week_picker()


def pct(x):
    return f"{x * 100:.1f}%"


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_home():
    money_rain()
    st.markdown('<div class="mpm-title">MONEY PRINTING<br>MACHINE</div>'
                '<div class="mpm-sub">NFL anytime touchdown model</div>', unsafe_allow_html=True)
    _, a, gap, b, _ = st.columns([1, 4, 0.4, 4, 1])
    if a.button("🏈  Touchdown Parlay Money Maker", key="go_parlay", width="stretch"):
        st.session_state.page = "parlay"
        st.rerun()
    if b.button("💸  Arbitrage Money Maker", key="go_arb", width="stretch"):
        st.session_state.page = "arb"
        st.rerun()


def kalshi_multipliers(live, sims):
    """{player_id: Kalshi payout multiplier incl. fee} for live games, plus a status note."""
    try:
        mk = kalshi_week(tuple(zip(live["away_team"], live["home_team"])))
    except kalshi.KalshiError:
        return {}, "Kalshi unreachable, so every leg is priced at the model's fair odds."
    mult = {}
    for _, g in live.iterrows():
        key = (g["away_team"], g["home_team"])
        if key not in mk or g["game_id"] not in sims:
            continue
        res = sims[g["game_id"]][0]
        ev, _ = kalshi.build_ev_table(res, mk[key][0], include_fees=True)
        if not ev.empty:
            ids = dict(zip(res["player"], res["player_id"]))
            mult.update({ids[p]: m for p, m in zip(ev["player"], ev["multiplier"]) if p in ids})
    if not mult:
        return mult, "No Kalshi markets yet, so every leg is priced at the model's fair odds."
    return mult, "Legs tagged K use the live Kalshi payout (fee included); M uses the model's fair odds."


PARLAY_STYLE = {"Easy Money": ("💰", "var(--cash)"), "Cash Cow": ("🐄", "var(--gold)"),
                "10 Bands": ("💵", "#c79bff"), "Longshotta": ("🎯", "#7fc8ff")}


def suggested_parlays(games, sims):
    live = games[games["result"].isna()]
    st.markdown('<p class="section">Suggested parlays</p>', unsafe_allow_html=True)
    if live.empty:
        st.info("Every game this week is final. Pick an upcoming week for parlay suggestions.")
        return
    with st.spinner("Building parlays…"):
        mult, note = kalshi_multipliers(live, sims)
        picks, _ = parlays.suggest(sims, live, mult)
    if not picks:
        st.info("Couldn't build a ~10x parlay within the rules for this week.")
        return
    cols = st.columns(2)
    for i, pr in enumerate(picks):
        c = cols[i % 2]
        icon, color = PARLAY_STYLE[pr.name]
        legs = "".join(
            f'<tr><td>{html.escape(l.player)}<div class="tm">{l.team} · {l.pos} · {l.game}</div></td>'
            f'<td class="num">{pct(l.p)}</td>'
            f'<td class="num">{l.payout:.2f}x <span class="src">{"K" if l.source == "Kalshi" else "M"}</span></td></tr>'
            for l in pr.legs)
        ev_cls = "edge" if pr.ev > 0 else "tm"
        c.markdown(
            f'<div class="card parlay" style="border-top:4px solid {color}">'
            f'<div class="card-head"><span class="matchup" style="color:{color}">{icon} {pr.name}</span>'
            f'<span class="big">{pr.payout:.1f}x</span></div>'
            f'<table class="tbl"><tr><th>Leg</th><th class="num">TD %</th><th class="num">Payout</th></tr>{legs}</table>'
            f'<div class="stats"><span>Model hit chance <b>{pct(pr.hit_prob)}</b></span>'
            f'<span>EV / $1 <b class="{ev_cls}">{pr.ev:+.0%}</b></span></div>'
            f'<p class="page-note" style="font-size:.95rem;margin:4px 0 6px">{html.escape(pr.note)}</p></div>',
            unsafe_allow_html=True)
    st.caption(f"{note} Rules: at most 2 players per game, no two from the same team. Hit chance uses the "
               "simulation's joint odds for same-game legs, so game-script correlation is included.")


def page_parlay():
    season, week = header("🏈 Touchdown Parlay Money Maker",
                          f"Top {TOP_N} most likely anytime-TD scorers per game, ranked by model probability.")
    games = week_games(season, week)
    if games.empty:
        st.info("No games on the schedule for this week.")
        return
    sims, errors = run_week(games)
    suggested_parlays(games, sims)

    st.markdown('<p class="section">Game-by-game</p>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, g in games.iterrows():
        gid = g["game_id"]
        away, home = g["away_team"], g["home_team"]
        if pd.notna(g.get("result")):
            when = f'<span class="final">FINAL {away} {int(g["away_score"])} – {home} {int(g["home_score"])}</span>'
        else:
            when = html.escape(_kick_label_safe(g))
        body = ""
        if gid in sims:
            res, meta = sims[gid]
            top = res.nlargest(TOP_N, "p_anytime")
            rows = []
            for r, (_, p) in enumerate(top.iterrows(), 1):
                flag = (f'<span class="flag">{html.escape(str(p["status"]))}</span>'
                        if p["status"] not in ("Active",) else "")
                rows.append(
                    f'<tr><td class="rk">{r}</td><td>{html.escape(p["player"])}{flag}</td>'
                    f'<td class="tm">{p["team"]} · {p["pos"]}</td>'
                    f'<td><div class="bar"><i style="width:{min(100, p["p_anytime"] * 100 / 0.8):.0f}%"></i></div></td>'
                    f'<td class="num">{pct(p["p_anytime"])}</td><td class="num tm">{p["fair_odds"]}</td></tr>')
            t = meta["teams"]
            body = ('<table class="tbl"><tr><th></th><th>Player</th><th>Team</th><th></th>'
                    '<th class="num">TD %</th><th class="num">Fair</th></tr>' + "".join(rows) + "</table>"
                    f'<p class="page-note" style="font-size:.95rem;margin-top:6px">Expected TDs: '
                    f'{away} {t[away]["sim_mean_td"]:.2f} · {home} {t[home]["sim_mean_td"]:.2f}</p>')
        else:
            body = f'<p class="page-note">Couldn\'t simulate this game: {html.escape(errors.get(gid, ""))}</p>'
        cols[i % 2].markdown(
            f'<div class="card"><div class="card-head"><span class="matchup">{away} @ {home}</span>'
            f'<span class="kick">{when}</span></div>{body}</div>', unsafe_allow_html=True)


def page_arb():
    season, week = header("💸 Arbitrage Money Maker",
                          f"Every player with a model TD probability of at least {MIN_MODEL_PCT:.0%} and a "
                          "positive edge over Kalshi, ranked by edge (Model % − Kalshi %, fee included).")
    games = week_games(season, week)
    live = games[games["result"].isna()]
    if live.empty:
        st.info("Every game this week is final, so there are no open Kalshi markets. Pick an upcoming week.")
        return

    c = st.columns([1.3, 6])
    if c[0].button("↻ Refresh prices", width="stretch"):
        kalshi_week.clear()
    sims, errors = run_week(live)

    try:
        with st.spinner("Pulling Kalshi prices…"):
            mk = kalshi_week(tuple(zip(live["away_team"], live["home_team"])))
    except kalshi.KalshiError as e:
        st.warning(f"Kalshi: {e}")
        return

    rows, no_market, unmatched_all = [], [], []
    for _, g in live.iterrows():
        key = (g["away_team"], g["home_team"])
        if g["game_id"] not in sims:
            continue
        if key not in mk:
            no_market.append(f"{key[0]} @ {key[1]}")
            continue
        res, _ = sims[g["game_id"]]
        ev, unmatched = kalshi.build_ev_table(res, mk[key][0], include_fees=True)
        unmatched_all += unmatched
        if not ev.empty:
            ev["game"] = f"{key[0]} @ {key[1]}"
            ev["kick"] = _kick_label_safe(g)
            rows.append(ev)

    c[1].caption(f"Kalshi 1+ TD markets · prices cached 60 s · {len(mk)} of {len(live)} remaining games have markets"
                 + (f" · no market yet: {', '.join(no_market)}" if no_market else ""))
    if not rows:
        st.info("No Kalshi player markets matched the model for this week yet. Markets usually open a few days "
                "before kickoff.")
        return

    allp = pd.concat(rows, ignore_index=True)
    board = (allp[(allp["p_anytime"] >= MIN_MODEL_PCT) & (allp["edge_pts"] > 0)]
             .sort_values("edge_pts", ascending=False).reset_index(drop=True))
    if board.empty:
        st.info("No players with a positive edge right now. Kalshi is pricing everyone at or above the model.")
        return
    if (board["edge_pts"] > 0.25).any():
        st.warning("Some edges are over 25 pts. Edges that large usually mean a data mismatch or a stale market. "
                   "Check the price on Kalshi before betting.")

    trs = []
    for r, p in board.iterrows():
        flag = f'<span class="flag">{html.escape(str(p["status"]))}</span>' if p["status"] != "Active" else ""
        liq = '<span class="thin">Thin</span>' if p["liquidity"] == "Thin" else "OK"
        trs.append(
            f'<tr><td class="rk">{r + 1}</td><td>{html.escape(p["player"])}{flag}</td>'
            f'<td class="tm">{p["team"]} · {p["pos"]}</td><td class="tm">{p["game"]}</td>'
            f'<td class="num">{pct(p["p_anytime"])}</td><td class="num">{pct(p["market_pct"])}</td>'
            f'<td class="num edge">+{p["edge_pts"] * 100:.1f}</td><td class="num">{p["ev_per_dollar"]:+.0%}</td>'
            f'<td class="num">{p["multiplier"]:.2f}x</td><td class="num">{liq}</td></tr>')
    st.markdown(
        '<div class="card"><table class="tbl"><tr><th></th><th>Player</th><th>Team</th><th>Game</th>'
        '<th class="num">Model</th><th class="num">Kalshi</th><th class="num">Edge (pts)</th>'
        '<th class="num">EV / $1</th><th class="num">Payout</th><th class="num">Liquidity</th></tr>'
        + "".join(trs) + "</table></div>", unsafe_allow_html=True)

    out = board[["player", "team", "pos", "game", "kick", "status", "p_anytime", "market_pct", "edge_pts",
                 "ev_per_dollar", "multiplier", "liquidity", "volume", "ticker"]]
    st.download_button("Download list (CSV)", out.to_csv(index=False), f"arbitrage_week{week}.csv", "text/csv")
    if errors or unmatched_all:
        with st.expander("Troubleshooting"):
            for gid, e in errors.items():
                st.write(f"Simulation failed for {gid}: {e}")
            if unmatched_all:
                st.write("Kalshi players not matched to the model (usually no usage in the model, or name "
                         "differences):", ", ".join(sorted(set(unmatched_all))))


PAGES = {"home": page_home, "parlay": page_parlay, "arb": page_arb}
PAGES[st.session_state.setdefault("page", "home")]()
