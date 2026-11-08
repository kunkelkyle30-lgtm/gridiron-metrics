"""Build data/college.json for the Gridiron Metrics Rookie & College Lab.

Inputs (downloaded separately, paths passed as args):
  --cfb DIR     cfbfastR-data player_stats/csv files (player_stats_YYYY.csv) and
                rosters/rds/cfb_rosters_YYYY.rds
  --nfl DIR     nflverse draft_picks.csv, player_stats_season.csv,
                stats_player_reg_YYYY.csv, plus DynastyProcess db_playerids.csv
                and values.csv

Outputs one JSON file with:
  hitrates   NFL outcomes by position and draft-capital bucket (drafts 2000-2023)
  rookies    2024-2026 draft classes: draft capital, landing spot, college
             profile, NFL production vs. what their draft slot predicts
  prospects  current college players at QB/RB/WR/TE scored on production
"""
import argparse, json, math, os, datetime
import pandas as pd
import numpy as np

SKILL = ["QB", "RB", "WR", "TE"]
POWER = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"}
TOP_N = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}


def bucket(pick):
    if pick <= 12: return "Top 12"
    if pick <= 32: return "Picks 13–32"
    if pick <= 64: return "Round 2"
    if pick <= 100: return "Round 3"
    if pick <= 170: return "Rounds 4–5"
    return "Rounds 6–7"

BUCKETS = ["Top 12", "Picks 13–32", "Round 2", "Round 3", "Rounds 4–5", "Rounds 6–7"]


# ---------------- College aggregation ----------------
COLS = ["game_id", "season", "team", "conference", "reception_player_id", "reception_player", "reception_yds",
        "target_player_id", "target_player", "completion_player_id", "completion_player", "completion_yds",
        "incompletion_player_id", "interception_thrown_player_id", "rush_player_id", "rush_player", "rush_yds",
        "touchdown_player_id"]


def college_season(path):
    d = pd.read_csv(path, usecols=COLS, low_memory=False)
    for c in ["reception_player_id", "target_player_id", "completion_player_id", "incompletion_player_id",
              "interception_thrown_player_id", "rush_player_id", "touchdown_player_id"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["reception_yds", "completion_yds", "rush_yds"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    season = int(pd.to_numeric(d["season"], errors="coerce").dropna().mode()[0])
    rec = d[d.reception_player_id.notna()].copy()
    # On passing TDs the source credits the QB as touchdown_player, so any TD on a catch counts for the receiver
    rec["td"] = rec.touchdown_player_id.notna().astype(int)
    tgt = d[d.target_player_id.notna()]
    rush = d[d.rush_player_id.notna()].copy()
    rush["td"] = rush.touchdown_player_id.notna().astype(int)
    comp = d[d.completion_player_id.notna()].copy()
    comp["td"] = comp.touchdown_player_id.notna().astype(int)
    inc = d[d.incompletion_player_id.notna()]
    ints = d[d.interception_thrown_player_id.notna()]

    # team totals
    tt = pd.DataFrame({
        "t_rec_yds": rec.groupby("team").reception_yds.sum(),
        "t_rec_td": rec.groupby("team").td.sum(),
        "t_targets": rec.groupby("team").size().add(tgt.groupby("team").size(), fill_value=0),
        "t_rush_att": rush.groupby("team").size(),
        "t_rush_yds": rush.groupby("team").rush_yds.sum(),
        "t_games": d.groupby("team").game_id.nunique(),
    }).fillna(0)

    g = {}
    def add(df, idcol, namecol, fields):
        if df.empty: return
        grp = df.groupby(idcol)
        for pid, sub in grp:
            pid = int(pid)
            r = g.setdefault(pid, {"id": pid, "games": set(), "teams": {}, "conf": None})
            r["name"] = sub[namecol].iloc[0] if namecol and namecol in sub else r.get("name")
            r["games"].update(sub.game_id.tolist())
            for tm, n in sub.team.value_counts().items(): r["teams"][tm] = r["teams"].get(tm, 0) + n
            if r["conf"] is None: r["conf"] = sub.conference.mode().iloc[0] if sub.conference.notna().any() else None
            for k, fn in fields.items(): r[k] = r.get(k, 0) + fn(sub)
    add(rec, "reception_player_id", "reception_player", {"rec": len, "rec_yds": lambda s: s.reception_yds.sum(), "rec_td": lambda s: s.td.sum()})
    add(tgt, "target_player_id", "target_player", {"inc_tgt": len})
    add(rush, "rush_player_id", "rush_player", {"rush_att": len, "rush_yds": lambda s: s.rush_yds.sum(), "rush_td": lambda s: s.td.sum()})
    add(comp, "completion_player_id", "completion_player", {"cmp": len, "pass_yds": lambda s: s.completion_yds.sum(), "pass_td": lambda s: s.td.sum()})
    add(inc, "incompletion_player_id", None, {"inc": len})
    add(ints, "interception_thrown_player_id", None, {"int": len})

    out = {}
    for pid, r in g.items():
        team = max(r["teams"], key=r["teams"].get)
        T = tt.loc[team] if team in tt.index else None
        gp = len(r["games"])
        x = {k: float(r.get(k, 0)) for k in ["rec", "rec_yds", "rec_td", "inc_tgt", "rush_att", "rush_yds", "rush_td", "cmp", "pass_yds", "pass_td", "inc", "int"]}
        x["targets"] = x["rec"] + x["inc_tgt"]
        x["att"] = x["cmp"] + x["inc"] + x["int"]
        x.update({"season": season, "team": team, "conf": r["conf"], "games": gp, "name": r.get("name")})
        if T is not None:
            x["tgt_share"] = x["targets"] / T.t_targets if T.t_targets else 0
            x["yds_share"] = x["rec_yds"] / T.t_rec_yds if T.t_rec_yds else 0
            x["td_share"] = x["rec_td"] / T.t_rec_td if T.t_rec_td else 0
            x["rush_share"] = x["rush_att"] / T.t_rush_att if T.t_rush_att else 0
            x["team_games"] = float(T.t_games)
        out[pid] = x
    return season, out


def receiver_metrics(x):
    gp = max(1, x["games"])
    dom = (x.get("yds_share", 0) + x.get("td_share", 0)) / 2
    return {"dom": dom, "tgt_share": x.get("tgt_share", 0), "ypg": x["rec_yds"] / gp,
            "ypt": x["rec_yds"] / x["targets"] if x["targets"] else 0, "rec": x["rec"], "rec_yds": x["rec_yds"], "rec_td": x["rec_td"]}


def metrics_for(pos, x):
    gp = max(1, x["games"])
    if pos in ("WR", "TE"):
        return receiver_metrics(x)
    if pos == "RB":
        return {"scrim_ypg": (x["rush_yds"] + x["rec_yds"]) / gp, "rush_share": x.get("rush_share", 0),
                "rec_share": x.get("yds_share", 0), "td_pg": (x["rush_td"] + x["rec_td"]) / gp,
                "ypc": x["rush_yds"] / x["rush_att"] if x["rush_att"] else 0,
                "rush_yds": x["rush_yds"], "rec_yds": x["rec_yds"], "tds": x["rush_td"] + x["rec_td"]}
    if pos == "QB":
        return {"pass_ypg": x["pass_yds"] / gp, "ypa": x["pass_yds"] / x["att"] if x["att"] else 0,
                "tdint_pg": (x["pass_td"] - x["int"]) / gp, "rush_ypg": x["rush_yds"] / gp,
                "pass_yds": x["pass_yds"], "pass_td": x["pass_td"], "int": x["int"], "rush_yds": x["rush_yds"]}
    return {}

WEIGHTS = {
    "WR": {"dom": 0.4, "tgt_share": 0.25, "ypg": 0.2, "ypt": 0.15},
    "TE": {"dom": 0.35, "tgt_share": 0.25, "ypg": 0.25, "ypt": 0.15},
    "RB": {"scrim_ypg": 0.4, "rush_share": 0.2, "rec_share": 0.2, "td_pg": 0.2},
    "QB": {"pass_ypg": 0.3, "ypa": 0.25, "tdint_pg": 0.25, "rush_ypg": 0.2},
}


def score_pool(rows, pos):
    """Percentile composite within position, discounted 12% outside power conferences."""
    if not rows: return
    w = WEIGHTS[pos]
    df = pd.DataFrame([r["m"] for r in rows])
    pct = {k: df[k].rank(pct=True).values for k in w}
    for i, r in enumerate(rows):
        s = sum(w[k] * pct[k][i] for k in w) * 100
        if r.get("conf") not in POWER and r.get("team") != "Notre Dame": s *= 0.88
        r["score"] = round(s, 1)


# ---------------- NFL ----------------
def nfl_seasons(nfl):
    parts = []
    old = pd.read_csv(os.path.join(nfl, "player_stats_season.csv"), low_memory=False,
                      usecols=["season", "season_type", "player_id", "position", "games", "fantasy_points_ppr"])
    parts.append(old[old.season_type == "REG"])
    for f in sorted(os.listdir(nfl)):
        if f.startswith("stats_player_reg_"):
            d = pd.read_csv(os.path.join(nfl, f), low_memory=False)
            keep = ["season", "player_id", "position", "games", "fantasy_points_ppr", "recent_team", "targets", "carries", "target_share"]
            parts.append(d[[c for c in keep if c in d.columns]])
    s = pd.concat(parts, ignore_index=True)
    s = s.drop_duplicates(["season", "player_id"], keep="last")
    s["ppg"] = s.fantasy_points_ppr / s.games.clip(lower=1)
    s["pos_rank"] = s.groupby(["season", "position"]).fantasy_points_ppr.rank(ascending=False, method="min")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfb", required=True); ap.add_argument("--nfl", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # College seasons
    col = {}
    for f in sorted(os.listdir(a.cfb)):
        if f.startswith("player_stats_") and f.endswith(".csv"):
            yr, data = college_season(os.path.join(a.cfb, f)); col[yr] = data
            print("college", yr, len(data))
    cur = max(col)

    # Rosters for the current college season
    import rdata, warnings
    warnings.filterwarnings("ignore")
    ros = rdata.conversion.convert(rdata.parser.parse_file(os.path.join(a.cfb, f"cfb_rosters_{cur}.rds")))
    ros = pd.DataFrame(ros)
    ros["athlete_id"] = pd.to_numeric(ros.athlete_id, errors="coerce")
    ros = ros[ros.position.isin(SKILL)].dropna(subset=["athlete_id"])
    rmap = {int(r.athlete_id): r for r in ros.itertuples()}

    # Prospects: current college season production
    pools = {p: [] for p in SKILL}
    for pid, x in col[cur].items():
        r = rmap.get(pid)
        if r is None: continue
        pos = r.position
        if x["games"] < 5: continue
        if pos in ("WR", "TE") and x["targets"] < 25: continue
        if pos == "RB" and x["rush_att"] + x["rec"] < 50: continue
        if pos == "QB" and x["att"] < 100: continue
        try: yr = int(r.year)
        except Exception: yr = None
        if yr not in (1, 2, 3, 4, 5): yr = None
        prev = col.get(cur - 1, {}).get(pid)
        row = {"id": pid, "name": f"{r.first_name} {r.last_name}".strip(), "pos": pos, "team": x["team"], "conf": x["conf"],
               "year": yr, "ht": int(r.height) if pd.notna(r.height) and str(r.height).replace('.', '').isdigit() else None,
               "wt": int(r.weight) if pd.notna(r.weight) and str(r.weight).replace('.', '').isdigit() else None,
               "games": x["games"], "m": metrics_for(pos, x),
               "prev": metrics_for(pos, prev) if prev and prev["games"] >= 4 else None,
               "prev_team": prev["team"] if prev else None}
        pools[pos].append(row)
    prospects = []
    for pos, rows in pools.items():
        score_pool(rows, pos)
        rows.sort(key=lambda r: -r["score"])
        prospects.extend(rows[:120])
    for r in prospects:
        r["m"] = {k: round(v, 3) for k, v in r["m"].items()}
        if r["prev"]: r["prev"] = {k: round(v, 3) for k, v in r["prev"].items()}

    # NFL history: hit rates by draft capital
    seasons = nfl_seasons(a.nfl)
    dr = pd.read_csv(os.path.join(a.nfl, "draft_picks.csv"), low_memory=False)
    dr = dr[dr.position.isin(SKILL)]
    hist = dr[(dr.season >= 2000) & (dr.season <= 2023)].copy()
    by_pid = {pid: g for pid, g in seasons.groupby("player_id")}
    rows = []
    for r in hist.itertuples():
        g = by_pid.get(r.gsis_id)
        out = {"pos": r.position, "b": bucket(int(r.pick)), "hit": 0, "elite": 0, "y": {}}
        if g is not None:
            for k in (1, 2, 3):
                s = g[g.season == r.season + k - 1]
                if len(s) and s.games.iloc[0] >= 4: out["y"][k] = float(s.ppg.iloc[0])
            first3 = g[(g.season >= r.season) & (g.season <= r.season + 2)]
            if len(first3):
                best = first3.pos_rank.min()
                out["hit"] = int(best <= TOP_N[r.position]); out["elite"] = int(best <= TOP_N[r.position] / 2)
        rows.append(out)
    hitrates = {}
    for pos in SKILL:
        hitrates[pos] = []
        for b in BUCKETS:
            sub = [x for x in rows if x["pos"] == pos and x["b"] == b]
            if not sub: continue
            med = {}
            for k in (1, 2, 3):
                v = [x["y"][k] for x in sub if k in x["y"]]
                med[k] = round(float(np.median(v)), 1) if v else None
            hitrates[pos].append({"bucket": b, "n": len(sub), "hit": round(sum(x["hit"] for x in sub) / len(sub), 3),
                                  "elite": round(sum(x["elite"] for x in sub) / len(sub), 3), "ppg": med})

    # Rookies 2024-2026: draft capital, landing spot, college profile, NFL vs. expectation
    ids = pd.read_csv(os.path.join(a.nfl, "db_playerids.csv"), low_memory=False, dtype=str)
    ids = ids.set_index("gsis_id")
    vals = pd.read_csv(os.path.join(a.nfl, "values.csv"), dtype=str)
    vmap = {r.fp_id: r for r in vals.itertuples() if r.fp_id and r.fp_id != "NA"}
    vmax1 = pd.to_numeric(vals[vals.pos != "PICK"].value_1qb).max(); vmax2 = pd.to_numeric(vals[vals.pos != "PICK"].value_2qb).max()
    last_nfl = int(seasons.season.max())
    team_now = seasons[seasons.season == last_nfl]
    rookies = []
    for r in dr[dr.season >= 2024].itertuples():
        b = bucket(int(r.pick))
        exp = next((h for h in hitrates[r.position] if h["bucket"] == b), None)
        info = ids.loc[r.gsis_id] if r.gsis_id in ids.index else None
        if isinstance(info, pd.DataFrame): info = info.iloc[0]
        sleeper = info["sleeper_id"] if info is not None and pd.notna(info["sleeper_id"]) else None
        fp = info["fantasypros_id"] if info is not None and pd.notna(info["fantasypros_id"]) else None
        espn = info["espn_id"] if info is not None and pd.notna(info["espn_id"]) else None
        bday = info["birthdate"] if info is not None and pd.notna(info["birthdate"]) else None
        v = vmap.get(fp)
        g = by_pid.get(r.gsis_id)
        nfl = []
        if g is not None:
            for s in g.sort_values("season").itertuples():
                if s.season < r.season: continue
                yi = int(s.season - r.season + 1)
                e = exp["ppg"].get(yi) if exp else None
                nfl.append({"season": int(s.season), "yr": yi, "g": int(s.games), "ppg": round(float(s.ppg), 1), "rank": int(s.pos_rank),
                            "exp": e, "team": getattr(s, "recent_team", None) if isinstance(getattr(s, "recent_team", None), str) else None,
                            "tgt_share": round(float(s.target_share), 3) if hasattr(s, "target_share") and pd.notna(s.target_share) else None,
                            "carries": int(s.carries) if hasattr(s, "carries") and pd.notna(s.carries) else None})
        # college final season profile (ESPN ids carry over from college to NFL)
        cprof = None
        if espn:
            try: e_id = int(float(espn))
            except Exception: e_id = None
            for yr in (r.season - 1, r.season - 2):
                x = col.get(yr, {}).get(e_id) if e_id else None
                if x and x["games"] >= 4:
                    cprof = {"season": yr, "team": x["team"], "conf": x["conf"], "games": x["games"], "m": {k: round(v2, 3) for k, v2 in metrics_for(r.position, x).items()}}
                    break
        age = None
        if bday:
            try:
                bd = datetime.date.fromisoformat(bday[:10]); age = round(((datetime.date(int(r.season), 4, 25) - bd).days) / 365.25, 1)
            except Exception: pass
        rookies.append({"name": r.pfr_player_name, "pos": r.position, "class": int(r.season), "round": int(r.round), "pick": int(r.pick),
                        "bucket": b, "team": r.team, "college": r.college, "draft_age": age, "sleeper_id": sleeper,
                        "value": {"v1": round(100 * float(v.value_1qb) / vmax1, 1), "v2": round(100 * float(v.value_2qb) / vmax2, 1)} if v is not None else None,
                        "hit_rate": exp["hit"] if exp else None, "elite_rate": exp["elite"] if exp else None,
                        "nfl": nfl, "college_profile": cprof})

    out = {"built": datetime.date.today().isoformat(), "college_season": cur, "nfl_season": last_nfl,
           "values_date": vals.scrape_date.dropna().iloc[0] if "scrape_date" in vals else None,
           "buckets": BUCKETS, "hitrates": hitrates, "rookies": rookies, "prospects": prospects}
    with open(a.out, "w") as f: json.dump(out, f, separators=(",", ":"), default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else str(o))
    print("rookies", len(rookies), "prospects", len(prospects), "bytes", os.path.getsize(a.out))


if __name__ == "__main__":
    main()
