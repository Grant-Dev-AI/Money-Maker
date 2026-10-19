# NFL Anytime Touchdown Simulator

Monte Carlo model that estimates each player's probability of scoring a touchdown in a selected NFL game.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens in your browser. Pick **Season → Week → Game**, click **Run simulation**. The first run downloads ~3 seasons of nflverse play-by-play (about 30 seconds); after that it's cached.

## What it does

1. **Pulls data** (nflverse, free, updated nightly): play-by-play for the current + 2 prior seasons, schedules, weekly rosters, injury reports. Only data from *before* the selected game is used.
2. **Team TDs**: each offense's rushing and passing TD rate × the opponent defense's TD-allowed rate (relative to league average), recency-weighted (this season > last season, last 4 games boosted), shrunk toward league average, plus home field.
3. **Player shares**: carries, inside-10 carries, targets, red-zone targets and realized TD share this season (later weeks weighted more); a multi-season *finishing skill* multiplier (TDs per red-zone opportunity vs league, shrunk); and an *opponent-vs-position* adjustment (e.g., a defense that gives up TE touchdowns).
4. **Availability**: players on IR / traded / cut are removed; Out / Doubtful / Questionable from the official report are discounted once that week's report is out. You can also mark players OUT manually.
5. **Simulation**: 20k games by default. A shared game-script factor correlates rush and pass TDs; team TD counts are negative-binomial; each TD is allocated to a player. Outputs P(anytime TD), P(2+), expected TDs, fair American odds, and a sportsbook edge calculator.

## Kalshi comparison

With **Compare to live Kalshi prices** on (sidebar), the app pulls Kalshi's NFL anytime-touchdown markets for the selected game and ranks players from highest to lowest expected value:

- **Kalshi %** = the Yes ask price (what it costs to buy "he scores"), plus Kalshi's trading fee if that box is checked.
- **Edge** = Model % − Kalshi %.
- **EV per $1** = Model % ÷ Kalshi % − 1, the average profit per dollar if the model is right.
- **Liquidity** flags thin markets (bid/ask spread over 10¢ or under 100 contracts traded).

No Kalshi account or API key is needed. Prices refresh every 60 seconds or when you click **Refresh prices**. Markets usually open a few days before kickoff.

To check the connection from the terminal: `python kalshi.py GB DET`.

## Validation

`python backtest.py 2025 13 18` re-runs the model on completed games and reports calibration. On 2025 weeks 13–17 (out of sample from the tuning weeks), predicted vs actual anytime-TD rates lined up within a few points in every bucket, and Brier score was 0.122 vs 0.140 for a base-rate guess.

## Files

- `engine.py` — data loading, model, simulation, backtest (all tunable knobs are in `Params`)
- `app.py` — Streamlit UI
- `kalshi.py` — Kalshi market data, name matching, EV ranking
- `backtest.py` — calibration check

## Known limitations

- Doesn't use Vegas totals/spreads, weather, or snap counts — adding a game total would likely be the biggest single improvement.
- Defensive / special-teams TDs aren't modeled (players only score offensive TDs here).
- Rookies or newly signed players with no snaps for this team get no share until they play.
