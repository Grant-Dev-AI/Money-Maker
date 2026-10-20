# Money Printing Machine — NFL Anytime Touchdown Model

Monte Carlo model that estimates each player's probability of scoring a touchdown, plus a Kalshi edge finder.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens in your browser on the homepage with two buttons:

- **Touchdown Parlay Money Maker**: four suggested parlays (never two players from the same team or more than 2 from one game), then every game of the week in kickoff order with its top 8 most likely anytime-TD scorers. Each parlay is the highest model hit chance for its target:
  - *Easy Money*: 2 players, about 5x
  - *Cash Cow*: 2 players, about 7x
  - *10 Bands*: 2–3 players, about 10x
  - *Longshotta*: 2–4 players, 12–15x

  Legs are priced at the live Kalshi payout (fee included) when a market exists, otherwise at the model's fair odds. Hit chance comes from the model and uses the simulation's joint odds for two legs in the same game.
- **Arbitrage Money Maker**: every player across the week's remaining games with a model TD probability of at least 15% and a positive edge vs Kalshi, ranked by edge (Model % − Kalshi %, fee included). Download as CSV.

Both default to the current week; a dropdown at the top right switches weeks. Game script (stability 9.0) and home field (+4% TDs) use the model defaults. The first run downloads ~3 seasons of nflverse play-by-play (about 30 seconds); after that a full week simulates in a few seconds.

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

## Validation

`python backtest.py 2025 13 18` re-runs the model on completed games and reports calibration. On 2025 weeks 13–17 (out of sample from the tuning weeks), predicted vs actual anytime-TD rates lined up within a few points in every bucket, and Brier score was 0.122 vs 0.140 for a base-rate guess.

## Files

- `engine.py` — data loading, model, simulation, backtest (all tunable knobs are in `Params`)
- `app.py` — Streamlit UI (homepage, parlay view, arbitrage view)
- `parlays.py` — suggested-parlay builder
- `kalshi.py` — Kalshi market data (per game or whole week), name matching, EV ranking
- `backtest.py` — calibration check

## Known limitations

- Doesn't use Vegas totals/spreads, weather, or snap counts — adding a game total would likely be the biggest single improvement.
- Defensive / special-teams TDs aren't modeled (players only score offensive TDs here).
- Rookies or newly signed players with no snaps for this team get no share until they play.
