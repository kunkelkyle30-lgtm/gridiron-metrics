# Gridiron Metrics: Dynasty Lab

Static site (`index.html`) hosted on GitHub Pages. Four tools:

- **My Team**: connect a Sleeper league; position grades on market value and current production, title odds from a rest-of-season simulation, and waiver/trade moves ranked by how much they change your title odds.
- **League**: title and playoff odds for every team, plus each manager's trading tendencies from Sleeper trade history.
- **Edges**: players where FantasyPros expert consensus and the FantasyCalc trade market disagree most (falls back to experts vs. production).
- **College**: NFL hit rates by draft capital (2000–2023), rookie report cards vs. draft-slot expectations, and a college watchlist ranked on share of team production.

Live data (Sleeper, DynastyProcess, FantasyCalc) loads in the visitor's browser. College and draft data is prebuilt into `data/college.json` by `scripts/build_college.py` from cfbfastR and nflverse files.
