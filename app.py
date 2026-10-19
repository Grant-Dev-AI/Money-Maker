"""
NFL Anytime TD Monte Carlo — Streamlit front end.
Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

import engine
import kalshi

st.set_page_config(page_title="NFL TD Simulator", page_icon="🏈", layout="wide")
st.title("🏈 NFL Anytime Touchdown Simulator")
st.caption("Monte Carlo model on nflverse play-by-play: team TD rates × opponent defense, "
           "player opportunity shares, finishing skill, opponent-vs-position, injuries.")


@st.cache_data(ttl=3600, show_spinner=False)
def schedule(season):
    return engine.get_schedule(season)


@st.cache_data(ttl=60, show_spinner=False)
def kalshi_markets(away, home, series_override):
    return kalshi.fetch_game(away, home, series_override or None)


@st.cache_data(ttl=3600, show_spinner=False)
def simulate(game_id, n_sims, dispersion, home_edge, seed, overrides_tuple):
    p = engine.Params(n_sims=n_sims, game_dispersion=dispersion, home_edge=home_edge)
    return engine.run_game(game_id, p, seed=seed, overrides=dict(overrides_tuple))


# ---------------- Sidebar: game selection ----------------
with st.sidebar:
    st.header("1. Select game")
    cur = engine.current_season()
    season = st.selectbox("Season", list(range(cur, cur - 4, -1)))
    with st.spinner("Loading schedule…"):
        sched = schedule(season)
    upcoming = sched[sched["result"].isna()]
    default_week = int(upcoming["week"].min()) if len(upcoming) else int(sched["week"].max())
    weeks = sorted(sched["week"].unique())
    week = st.selectbox("Week", weeks, index=weeks.index(default_week))
    wk_games = sched[sched["week"] == week]
    label = st.selectbox("Game", wk_games["label"])
    game = wk_games[wk_games["label"] == label].iloc[0]

    st.header("2. Settings")
    n_sims = st.select_slider("Simulations", [5000, 10000, 20000, 50000, 100000], value=20000)
    dispersion = st.slider("Game-script stability", 3.0, 20.0, 9.0, 0.5,
                           help="Lower = more volatile games (more blowouts / shootouts).")
    home_edge = st.slider("Home-field TD edge", 0.0, 0.10, 0.04, 0.01)
    seed = st.number_input("Random seed", value=7, step=1)

    st.header("3. Kalshi")
    use_kalshi = st.toggle("Compare to live Kalshi prices", value=True)
    include_fees = st.checkbox("Include Kalshi trading fee", value=True,
                               help="Adds Kalshi's taker fee (about 0.07 × P × (1−P) per contract) to the price.")
    min_edge = st.slider("Highlight edges of at least (pts)", 0, 15, 5)
    hide_thin = st.checkbox("Hide thin markets", value=False,
                            help="Thin = bid/ask spread over 10¢ or under 100 contracts traded.")
    series_override = st.text_input("Series ticker (optional)", "",
                                    help="Leave blank to auto-detect Kalshi's NFL anytime-TD series.")

if "overrides" not in st.session_state:
    st.session_state.overrides = {}

run = st.sidebar.button("▶ Run simulation", type="primary", width="stretch")
if run or st.session_state.get("last_game") == game["game_id"]:
    st.session_state.last_game = game["game_id"]
    with st.spinner("Pulling play-by-play history and simulating… (first run downloads ~3 seasons of data)"):
        res, meta = simulate(game["game_id"], n_sims, dispersion, home_edge, int(seed),
                             tuple(sorted(st.session_state.overrides.items())))

    away, home = game["away_team"], game["home_team"]
    st.subheader(f"{away} @ {home} — Week {game['week']}, {game['gameday']} {game.get('gametime') or ''}")
    if pd.notna(game.get("result")):
        st.info(f"Final: {away} {int(game['away_score'])} – {home} {int(game['home_score'])} "
                "(simulation uses only data from before this game)")

    c = st.columns(4)
    for i, t in enumerate([away, home]):
        ti = meta["teams"][t]
        c[2 * i].metric(f"{t} expected TDs", f"{ti['sim_mean_td']:.2f}")
        c[2 * i + 1].metric(f"{t} rush / pass", f"{ti['exp_rush_td']:.2f} / {ti['exp_pass_td']:.2f}")
    for n in meta["notes"]:
        st.warning(n)

    # ---------------- Kalshi EV ranking ----------------
    if use_kalshi:
        st.markdown("### 💲 Best value vs Kalshi (highest EV first)")
        cols = st.columns([1, 5])
        if cols[0].button("↻ Refresh prices"):
            kalshi_markets.clear()
        try:
            with st.spinner("Fetching Kalshi prices…"):
                mk, kinfo = kalshi_markets(away, home, series_override.strip())
            ev, unmatched = kalshi.build_ev_table(res, mk, include_fees=include_fees)
            if hide_thin and not ev.empty:
                ev = ev[ev["liquidity"] == "OK"]
            cols[1].caption(f"Kalshi event **{kinfo['event'].get('title', kinfo['event']['event_ticker'])}** · "
                            f"series {kinfo['series']} · 1+ TD (anytime) markets only · prices cached 60 s · Kalshi % = Yes ask"
                            + (" + fee" if include_fees else ""))
            if ev.empty:
                st.warning("Found the Kalshi event but couldn't match any player markets to the model.")
            else:
                if (ev["edge_pts"] > 0.25).any():
                    st.warning("Some edges are over 25 pts. Edges that large usually mean a data mismatch "
                               "or a stale/empty market — check the price on Kalshi before betting.")
                n_pos = int((ev["edge_pts"] >= min_edge / 100).sum())
                if n_pos:
                    st.success(f"{n_pos} player(s) with an edge of {min_edge}+ pts.")
                else:
                    st.info(f"No players with an edge of {min_edge}+ pts right now.")
                show_ev = ev[["player", "team", "pos", "status", "p_anytime", "market_pct", "edge_pts",
                              "ev_per_dollar", "multiplier", "liquidity", "volume"]]

                def _hl(row):
                    good = row["edge_pts"] >= min_edge / 100
                    return ["background-color: rgba(46,160,67,0.18)" if good else ""] * len(row)

                st.dataframe(
                    show_ev.style.apply(_hl, axis=1), hide_index=True, width="stretch",
                    height=min(600, 38 + 35 * len(show_ev)),
                    column_config={
                        "player": "Player", "team": "Team", "pos": "Pos", "status": "Status",
                        "p_anytime": st.column_config.NumberColumn("Model %", format="percent"),
                        "market_pct": st.column_config.NumberColumn("Kalshi %", format="percent"),
                        "edge_pts": st.column_config.NumberColumn("Edge", format="percent",
                                                                  help="Model % minus Kalshi %"),
                        "ev_per_dollar": st.column_config.NumberColumn(
                            "EV per $1", format="percent",
                            help="Average profit per $1 if the model is right: Model % ÷ Kalshi % − 1"),
                        "multiplier": st.column_config.NumberColumn("Payout", format="%.2fx"),
                        "liquidity": "Liquidity",
                        "volume": st.column_config.NumberColumn("Volume", format="%d"),
                    })
                st.download_button("Download EV list (CSV)", ev.to_csv(index=False),
                                   f"kalshi_ev_{game['game_id']}.csv", "text/csv")
            with st.expander("Kalshi details / troubleshooting"):
                if unmatched:
                    st.write("Kalshi markets not matched to a model player (usually players with no "
                             "usage in the model, or name format differences):", ", ".join(unmatched))
                st.write("Raw Kalshi markets:")
                st.dataframe(mk, hide_index=True, width="stretch")
                st.json({k: v for k, v in kinfo.items() if k != "event"} | {"event_ticker": kinfo["event"]["event_ticker"]},
                        expanded=False)
        except kalshi.KalshiError as e:
            st.warning(f"Kalshi: {e}")
        except Exception as e:  # unexpected API format
            st.error(f"Kalshi data error: {type(e).__name__}: {e}. Copy this message to troubleshoot.")

    # ---------------- Full model results ----------------
    st.markdown("### All players — model probabilities")
    team_f = st.radio("Show", ["Both", away, home], horizontal=True)
    view = res if team_f == "Both" else res[res["team"] == team_f]
    show = view[["player", "team", "pos", "status", "p_anytime", "p_2plus", "exp_tds",
                 "fair_odds", "rush_share", "rec_share", "skill_mult"]].copy()
    st.dataframe(
        show, hide_index=True, width="stretch", height=min(700, 38 + 35 * len(show)),
        column_config={
            "player": "Player", "team": "Team", "pos": "Pos", "status": "Status",
            "p_anytime": st.column_config.ProgressColumn("Anytime TD", format="percent", min_value=0, max_value=1),
            "p_2plus": st.column_config.NumberColumn("2+ TDs", format="percent"),
            "exp_tds": st.column_config.NumberColumn("Exp. TDs", format="%.2f"),
            "fair_odds": "Fair odds",
            "rush_share": st.column_config.NumberColumn("Rush TD share", format="percent"),
            "rec_share": st.column_config.NumberColumn("Rec TD share", format="percent"),
            "skill_mult": st.column_config.NumberColumn("Finishing mult", format="%.2f"),
        })
    st.download_button("Download CSV", res.to_csv(index=False),
                       f"td_sim_{game['game_id']}.csv", "text/csv")

    # ---------------- Edge finder ----------------
    with st.expander("🧮 Manual check for one player (any market)"):
        st.caption("Enter a payout multiplier (total return per \\$1, stake included) to check a single player.")
        cc = st.columns(4)
        pl = cc[0].selectbox("Player", res["player"])
        mult = cc[1].number_input("Payout multiplier (x)", min_value=1.01, value=2.50, step=0.05, format="%.2f")
        implied = 1 / mult
        mp = float(res.loc[res["player"] == pl, "p_anytime"].iloc[0])
        ev1 = mp * mult - 1
        cc[2].metric("Model vs implied", f"{mp:.1%} vs {implied:.1%}", f"{(mp - implied) * 100:+.1f} pts")
        cc[3].metric("EV per \\$1", f"{ev1:+.1%}")

    # ---------------- Manual injuries ----------------
    with st.expander("🩹 Manual injury overrides (re-run after changes)"):
        st.caption("Official injury reports come out Wed–Fri. Mark players OUT here to redistribute their usage.")
        ids = dict(zip(res["player"] + " (" + res["team"] + ")", res["player_id"]))
        current = [k for k, v in ids.items() if v in st.session_state.overrides]
        sel = st.multiselect("Mark OUT", list(ids), default=current)
        if st.button("Apply overrides"):
            st.session_state.overrides = {ids[s]: 0.0 for s in sel}
            st.rerun()

    with st.expander("📊 Opponent vs position multipliers"):
        st.caption("How each defense allows TDs to a position vs league average (1.00 = average). "
                   "Shifts TD share between positions within the offense.")
        st.dataframe(pd.DataFrame({f"{t} offense vs {o} D": meta["teams"][t]["opp_pos_mult"]
                                   for t, o in [(away, home), (home, away)]}).round(2))

    with st.expander("ℹ️ Methodology"):
        st.markdown(engine.__doc__)
        st.markdown(f"Data through **{meta['data_through']}**. League average: "
                    f"{meta['league_rush_td_pg']:.2f} rush TD / {meta['league_pass_td_pg']:.2f} pass TD per team-game "
                    "(recency-weighted).")
else:
    st.info("Pick a game in the sidebar and click **Run simulation**.")
