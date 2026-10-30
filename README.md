# Money Printing Machine — NFL Anytime Touchdown Model

Monte Carlo model that estimates each player's probability of scoring a touchdown, plus a Kalshi edge finder.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens in your browser on the homepage with three buttons:

- **Touchdown Parlay Money Maker**: four suggested parlays (never two players from the same team or more than 2 from one game), then every game of the week in kickoff order with its top 8 most likely anytime-TD scorers. Each parlay is the highest model hit chance for its target:
  - *Easy Money*: 2 players, about 5x
  - *Cash Cow*: 2 players, about 7x
  - *10 Bands*: 2–3 players, about 10x
  - *Longshotta*: 2–4 players, 12–15x

  Legs are priced at the live Kalshi payout (fee included) when a market exists, otherwise at the model's fair odds. Hit chance comes from the model and uses the simulation's joint odds for two legs in the same game.

  Each parlay has a green **Refresh parlay** button. It swaps 1, 2 or 3 players (picked at random, never more than the parlay has) for the best replacements that keep the payout in that parlay's range and follow the same rules. Swapped-out players don't come back, and you won't see a lineup repeat (or duplicate another card) until every option has been shown. Refreshed lineups stick for that week until you restart the app, and are repriced with live Kalshi odds on every reload.
- **Arbitrage Money Maker**: every player across the week's remaining games with a model TD probability of at least 15% and a positive edge vs Kalshi, ranked by edge (Model % − Kalshi %, fee included). Download as CSV.
- **Slate Sniper** 🎯: the week split into its 5 slates (Thursday Night, Sunday Morning, Sunday Afternoon, Sunday Night, Monday Night). Each slate lists every player the model gives at least a 20% anytime-TD chance, ranked from most to least likely across all games in that slate. Sunday Morning is the 1:00 PM ET (10 AM PT) window plus any early international game; Sunday Afternoon is the 4:05/4:25 PM ET window; Sunday Night is 7 PM ET or later. Thanksgiving games land in Thursday, Monday doubleheaders in Monday. Any Wednesday/Friday/Saturday game shows in a "Bonus Games" section at the end. The 20% cutoff is `SLATE_MIN_PCT` at the top of `app.py`.

All three default to the current week; a dropdown at the top right switches weeks. Game script (stability 9.0) and home field (+4% TDs) use the model defaults. The first run downloads ~3 seasons of nflverse play-by-play (about 30 seconds); after that a full week simulates in a few seconds.

## What it does

1. **Pulls data** (nflverse, free, updated nightly): play-by-play for the current + 2 prior seasons, schedules, weekly rosters, injury reports. Only data from *before* the selected game is used.
2. **Team TDs**: each offense's rushing and passing TD rate × the opponent defense's TD-allowed rate (relative to league average), recency-weighted (this season > last season, last 4 games boosted), shrunk toward league average, plus home field.
3. **Player shares**: carries, inside-10 carries, targets, red-zone targets and realized TD share this season (later weeks weighted more); a multi-season *finishing skill* multiplier (TDs per red-zone opportunity vs league, shrunk); and an *opponent-vs-position* adjustment (e.g., a defense that gives up TE touchdowns).
4. **Availability**: players on IR / traded / cut are removed; Out / Doubtful / Questionable from the official report are discounted once that week's report is out. You can also mark players OUT manually.
5. **Simulation**: 20k games by default. A shared game-script factor correlates rush and pass TDs; team TD counts are negative-binomial; each TD is allocated to a player. Outputs P(anytime TD), P(2+), expected TDs, fair American odds, and a sportsbook edge calculator.

## Kalshi comparison

The Arbitrage page pulls Kalshi's NFL 1+ touchdown markets for every game that hasn't been played yet:

- **Kalshi** = the Yes ask price (what it costs to buy "he scores") plus Kalshi's taker fee.
- **Edge** = Model % − Kalshi %, in percentage points. The list is sorted by this.
- **EV / $1** = Model % ÷ Kalshi % − 1, the average profit per dollar if the model is right.
- **Liquidity** flags thin markets (bid/ask spread over 10¢ or under 100 contracts traded).

No Kalshi account or API key is needed. Prices refresh every 60 seconds or when you click **Refresh prices**. Markets usually open a few days before kickoff.

To check the connection from the terminal: `python kalshi.py GB DET`.

## v2 upgrades (October 2026)

The team-level TD forecast was rebuilt. Each piece has a switch in `Params`, and `Params.legacy()` turns them all off to get the original model back.

1. **Vegas lines** (`use_vegas`): each team's implied points, (total ± spread) ÷ 2, converted to expected TDs with a line fit on past games, then blended 90/10 with the model's own estimate. Lines come free with the nflverse schedule.
2. **Opponent-adjusted ratings** (`use_glm`): a ridge-penalized Poisson regression, log E[TDs] = league + offense + defense + home, fit across all games at once, so a defense that faced weak offenses isn't overrated.
3. **Red-zone decomposition** (`use_rz`): expected TDs = expected red-zone trips × P(TD | trip) + TDs from outside the 20, each opponent-adjusted the same way. More stable than counting TDs.
4. **Smooth recency** (`use_decay`): exponential decay, 12-week half-life for team stats, with each offseason counting as 8 extra weeks; 8-week half-life for player shares. Replaces the old season weights and last-4-games boost.
   **Snap share** (`use_snaps`) is built in but **off**: a player's recent snap % vs his own average scaling his share. On 2024 it made no measurable difference.

Settings were tuned on 2024 only, then tested on 2025 and 2026, which weren't used for tuning:

| | 2025 wks 2–18 (256 games) | | 2026 wks 2–7 (91 games) | |
|---|---|---|---|---|
| | old | **new** | old | **new** |
| Team TDs: correlation with actual | 0.318 | **0.404** | 0.281 | **0.398** |
| Team TDs: avg error | 1.084 | **1.048** | 1.037 | **1.013** |
| Player Brier score (lower = better) | 0.1276 | **0.1261** | 0.1358 | **0.1348** |
| Player log loss (lower = better) | 0.4083 | **0.4043** | 0.4369 | **0.4338** |
| Each game's top 3 picks who scored | 39.1% | **41.3%** | **37.0%** | 36.3% |

Resampling by game, the new model has the better log loss in 100% of resamples on 2025 and 96% on 2026. The top-3 dip on 2026 is within noise for 91 games. Most of the gain comes from the Vegas lines. Without them, upgrades 2–4 mainly improve the rush/pass split, which helps player picks a little.

## Returning starters (`use_lineup`)

Fixes misses like Week 8 2026. Josh Jacobs was back from a suspension and starting, but the model still liked MarShawn Lloyd because Lloyd had started weeks 1–6. Player shares are now lineup-aware:

1. A player's share is measured only over games he was on the field (from snap counts), so a suspension or injury doesn't count as zero carries.
2. Past games where someone in the current lineup (anyone who played the team's latest game) was missing count less, scaled by the size of his role (`lineup_gamma` = 3). Once a starter is back, the weeks his backup started stop driving the numbers.
3. Small samples lean on the player's role on the same team last season (`lineup_shrink_games` = 2 pseudo-games), then shares are re-split to add up.

It kicks in once the player has played his first game back. It doesn't predict a big first game. CAR @ GB: Jacobs 22% → 48%, Lloyd 40% → 30%.

Tuned on 2024, then tested on 2025 and 2026 against the version without it. "Starter back" games are team-games where a player with a real role (8+ carries or 5+ targets) in the last game had missed 2+ earlier games:

| | 2025 | | 2026 wks 2–7 | |
|---|---|---|---|---|
| | before | **after** | before | **after** |
| Brier, all players | 0.1250 | **0.1244** | 0.1348 | **0.1324** |
| Log loss, all players | 0.4011 | **0.3963** | 0.4331 | **0.4222** |
| Log loss, "starter back" games | 0.3842 | **0.3808** | 0.3894 | **0.3844** |
| Top-3 picks who scored | **41.5%** | 39.7% | 37.0% | **37.3%** |

Top-3 hit rate went +1.5 pts in 2024 (tuning season), −1.8 in 2025 and +0.3 in 2026, which nets to about zero and is within noise. Probability accuracy improved in all three seasons.

## Validation

`python backtest.py 2025 2 19` re-runs the old and new model side by side on completed games and prints team-level error, player-level Brier / log loss / top-3 hit rate / how the 20%+ picks hit, and calibration tables. Add `--quick` for fewer simulations. Uses `compare.py`.

## Files

- `engine.py` — data loading, model, simulation, backtest (all tunable knobs are in `Params`)
- `app.py` — Streamlit UI (homepage, parlay view, arbitrage view, slate sniper)
- `parlays.py` — suggested-parlay builder
- `kalshi.py` — Kalshi market data (per game or whole week), name matching, EV ranking
- `backtest.py` — old vs new comparison on completed games
- `compare.py` — backtest metrics used by `backtest.py`

## Known limitations

- Weather isn't modeled directly (the Vegas total captures most of it).
- Backtests use nflverse's betting lines, which are close to closing lines. Lines earlier in the week are a bit less sharp.
- The 20%+ picks score a little more often than predicted (about 33% vs 31% on 2025), so top-end probabilities run slightly conservative.
- Defensive / special-teams TDs aren't modeled (players only score offensive TDs here).
- Rookies or newly signed players with no snaps for this team get no share until they play.
