import hashlib
import json
import math
import os
import shutil
import time
import warnings
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

PLACE_SLUG = {
    "札幌": "sapporo", "函館": "hakodate", "福島": "fukushima",
    "新潟": "niigata", "東京": "tokyo", "中山": "nakayama",
    "中京": "chukyo", "京都": "kyoto", "阪神": "hanshin", "小倉": "kokura",
}

CLASS_MAP = {23: 1, 43: 2, 67: 3, 115: 4, 131: 4, 147: 5, 163: 5, 179: 6, 195: 7}
SEX_MAP = {"牡": 1, "牝": 2, "セ": 3, "騙": 3}
TRACK_STATE_MAP = {"良": 1, "稍": 2, "重": 3, "不": 4}

RAW_COLUMNS = [
    "年", "月", "日", "場所", "レース番号", "クラスコード", "芝・ダ",
    "トラックコード", "距離", "馬場状態", "馬名", "性別", "年齢",
    "斤量", "頭数", "馬番", "確定着順", "異常コード", "着差タイム",
    "走破タイム(秒)", "補正タイム", "通過順4角", "上がり3Fタイム",
    "馬体重", "血統登録番号", "レースID(新)", "PCI", "RPCI",
    "枠番", "重量コード", "年齢限定(競走種別コード)", "トラックコード(JV)",
]

HIST_METRICS = [
    "finish", "margin", "time_sec", "corrected_time", "corner4",
    "final3f", "bodyweight", "carried", "pci", "rpci", "speed1000",
]

MIN_BEST = {"finish", "margin", "time_sec", "corner4", "final3f", "speed1000"}

RANK_GRID = [
    {"leaves": 7, "depth": 3, "minc": 30, "col": 0.8, "l2": 1},
    {"leaves": 15, "depth": 4, "minc": 30, "col": 0.8, "l2": 1},
    {"leaves": 31, "depth": 5, "minc": 20, "col": 0.8, "l2": 2},
    {"leaves": 15, "depth": -1, "minc": 50, "col": 0.8, "l2": 3},
]

SPECIAL_CFGS = [
    ("lgb", 2, 20), ("lgb", 3, 30), ("lgb", 4, 40),
    ("et", 3, 20), ("et", 4, 30), ("rf", 3, 20),
]

SCAN_CFGS = [
    ("lgb", 1, 20), ("lgb", 2, 20), ("lgb", 3, 30),
    ("et", 2, 20), ("et", 3, 30), ("rf", 2, 20),
]

RANK_FOLDS = [(2011, 2016, 2017, 2018), (2011, 2018, 2019, 2020), (2011, 2020, 2021, 2022)]


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_status(root: Path, **kwargs):
    p = root / "output" / "status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    base = {}
    if p.exists():
        try:
            base = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            base = {}
    base.update(kwargs)
    base["updated_at_epoch"] = time.time()
    p.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")


def dist_band(x):
    if pd.isna(x):
        return np.nan
    x = int(x)
    if x <= 1299:
        return 0
    if x <= 1699:
        return 1
    if x <= 1999:
        return 2
    return 3


def load_history(zip_path: Path, root: Path):
    cache = root / "cache" / "history_base.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_pickle(cache)

    parts = []
    with zipfile.ZipFile(zip_path) as z:
        csvs = [n for n in z.namelist() if n.upper().endswith(".CSV")]
        for i, name in enumerate(csvs, 1):
            with z.open(name) as fh:
                try:
                    d = pd.read_csv(fh, encoding="cp932", usecols=RAW_COLUMNS, low_memory=False)
                except Exception:
                    continue
            parts.append(d)
            if i % 50 == 0:
                print(f"[load] {i}/{len(csvs)} files", flush=True)
    if not parts:
        raise RuntimeError("No CSVs could be loaded from dataset ZIP")

    df = pd.concat(parts, ignore_index=True)
    del parts

    numeric_cols = [
        "年", "月", "日", "レース番号", "クラスコード", "距離", "年齢", "斤量",
        "頭数", "馬番", "確定着順", "着差タイム", "走破タイム(秒)", "補正タイム",
        "通過順4角", "上がり3Fタイム", "馬体重", "PCI", "RPCI", "枠番",
        "重量コード", "年齢限定(競走種別コード)", "トラックコード(JV)", "血統登録番号",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["year_full"] = np.where(df["年"] < 100, 2000 + df["年"], df["年"]).astype("int16")
    df["_date"] = pd.to_datetime(dict(year=df["year_full"], month=df["月"], day=df["日"]), errors="coerce")
    df["race_id"] = df["レースID(新)"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["horse_id"] = df["血統登録番号"].astype("Int64").astype(str)
    missing_horse = df["血統登録番号"].isna()
    df.loc[missing_horse, "horse_id"] = "N:" + df.loc[missing_horse, "馬名"].astype(str)

    df["class_level"] = df["クラスコード"].map(CLASS_MAP)
    df["finish"] = pd.to_numeric(df["確定着順"], errors="coerce")
    df["margin"] = pd.to_numeric(df["着差タイム"], errors="coerce")
    df["time_sec"] = pd.to_numeric(df["走破タイム(秒)"], errors="coerce")
    df["corrected_time"] = pd.to_numeric(df["補正タイム"], errors="coerce")
    df["corner4"] = pd.to_numeric(df["通過順4角"], errors="coerce")
    df["final3f"] = pd.to_numeric(df["上がり3Fタイム"], errors="coerce")
    df["bodyweight"] = pd.to_numeric(df["馬体重"], errors="coerce")
    df["carried"] = pd.to_numeric(df["斤量"], errors="coerce")
    df["pci"] = pd.to_numeric(df["PCI"], errors="coerce")
    df["rpci"] = pd.to_numeric(df["RPCI"], errors="coerce")
    df["speed1000"] = df["time_sec"] / pd.to_numeric(df["距離"], errors="coerce") * 1000.0
    df["dist_band"] = pd.to_numeric(df["距離"], errors="coerce").map(dist_band)
    df["sex_code"] = df["性別"].map(SEX_MAP)
    df["track_state_code"] = df["馬場状態"].map(TRACK_STATE_MAP)
    df["draw_pct"] = pd.to_numeric(df["枠番"], errors="coerce") / 8.0
    df["gate_pct"] = pd.to_numeric(df["馬番"], errors="coerce") / pd.to_numeric(df["頭数"], errors="coerce")
    df["target_win"] = (df["finish"] == 1).astype("int8")
    df["target_top2"] = (df["finish"].between(1, 2)).astype("int8")
    df["target_top3"] = (df["finish"].between(1, 3)).astype("int8")

    # Research history uses only completed rows before building history features.
    df = df[df["finish"] > 0].copy()
    df.sort_values(["horse_id", "_date", "race_id", "馬番"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_pickle(cache)
    print(f"[load] completed rows={len(df):,}", flush=True)
    return df


def add_group_features(df: pd.DataFrame, keys, prefix: str, add_last3=False):
    gb = df.groupby(keys, sort=False, observed=True)
    df[f"{prefix}_n"] = gb.cumcount().astype("int32")

    for m in HIST_METRICS:
        shifted = gb[m].shift(1)
        df[f"{prefix}_last_{m}"] = shifted.astype("float32")

        valid = df[m].notna().astype("int32")
        filled = df[m].fillna(0.0).astype("float64")
        csum = filled.groupby([df[k] for k in keys], sort=False).cumsum() - filled
        ccnt = valid.groupby([df[k] for k in keys], sort=False).cumsum() - valid
        mean = csum / ccnt.replace(0, np.nan)
        df[f"{prefix}_mean_{m}"] = mean.astype("float32")

        tmp_name = f"__sh_{prefix}_{m}"
        df[tmp_name] = shifted
        gb2 = df.groupby(keys, sort=False, observed=True)[tmp_name]
        if m in MIN_BEST:
            best = gb2.cummin()
        else:
            best = gb2.cummax()
        df[f"{prefix}_best_{m}"] = best.astype("float32")
        df.drop(columns=[tmp_name], inplace=True)

        if add_last3:
            # This transform is intentionally explicit and deterministic.
            last3 = gb[m].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            df[f"{prefix}_last3_{m}"] = last3.astype("float32")
    return df


def build_features(base: pd.DataFrame, root: Path):
    cache = root / "cache" / "adult_dirt_features_v2.pkl"
    if cache.exists():
        return pd.read_pickle(cache)

    df = base.copy()
    print("[features] g", flush=True)
    df = add_group_features(df, ["horse_id"], "g", add_last3=True)
    print("[features] b", flush=True)
    df = add_group_features(df, ["horse_id", "芝・ダ", "dist_band"], "b")
    print("[features] e", flush=True)
    df = add_group_features(df, ["horse_id", "場所", "芝・ダ", "距離"], "e")
    print("[features] x", flush=True)
    df = add_group_features(df, ["horse_id", "場所", "芝・ダ", "距離", "class_level"], "x")

    prev_date = df.groupby("horse_id", sort=False)["_date"].shift(1)
    df["days_since"] = (df["_date"] - prev_date).dt.days.astype("float32")
    df["body_change"] = (df["bodyweight"] - df["g_last_bodyweight"]).astype("float32")
    df["carry_change"] = (df["carried"] - df["g_last_carried"]).astype("float32")

    adult = df[
        (df["芝・ダ"] == "ダ")
        & (pd.to_numeric(df["年齢"], errors="coerce") >= 3)
        & (df["class_level"].notna())
        & (df["場所"].isin(PLACE_SLUG))
    ].copy()

    # Race-level history coverage from horse-level prior-history counts.
    for pref in ["g", "b", "e", "x"]:
        adult[f"race_{pref}_cov"] = adult.groupby("race_id", sort=False)[f"{pref}_n"].transform(
            lambda s: (s > 0).mean()
        ).astype("float32")

    adult["month"] = pd.to_numeric(adult["月"], errors="coerce").astype("float32")
    adult["distance"] = pd.to_numeric(adult["距離"], errors="coerce").astype("float32")
    adult["age"] = pd.to_numeric(adult["年齢"], errors="coerce").astype("float32")
    adult["field_n"] = pd.to_numeric(adult["頭数"], errors="coerce").astype("float32")
    adult["horse_no"] = pd.to_numeric(adult["馬番"], errors="coerce").astype("float32")
    adult["frame_no"] = pd.to_numeric(adult["枠番"], errors="coerce").astype("float32")
    adult["weight_code"] = pd.to_numeric(adult["重量コード"], errors="coerce").astype("float32")
    adult["age_limit_code"] = pd.to_numeric(adult["年齢限定(競走種別コード)"], errors="coerce").astype("float32")
    adult["track_code_jv"] = pd.to_numeric(adult["トラックコード(JV)"], errors="coerce").astype("float32")

    keep_raw = [
        "race_id", "year_full", "場所", "距離", "馬番", "馬名", "horse_id",
        "finish", "target_win", "target_top2", "target_top3", "class_level",
    ]
    current = [
        "month", "distance", "age", "carried", "field_n", "horse_no", "bodyweight",
        "frame_no", "weight_code", "age_limit_code", "track_code_jv", "class_level",
        "days_since", "body_change", "carry_change", "draw_pct", "gate_pct",
        "sex_code", "track_state_code", "race_g_cov", "race_b_cov", "race_e_cov", "race_x_cov",
    ]
    hist = [
        c for c in adult.columns
        if c.startswith(("g_", "b_", "e_", "x_")) and c not in {"g_n", "b_n", "e_n", "x_n"}
    ]
    features = current + ["g_n", "b_n", "e_n", "x_n"] + hist
    features = list(dict.fromkeys([c for c in features if c in adult.columns]))

    out = adult[keep_raw + features].copy()
    for c in features:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    out.to_pickle(cache)

    spec = {
        "version": "ADULT_DIRT_V2",
        "history_rows_completed_only": True,
        "odds_popularity_used": False,
        "class_map": CLASS_MAP,
        "distance_bands": {"0": "<=1299", "1": "1300-1699", "2": "1700-1999", "3": ">=2000"},
        "groups": {
            "g": ["horse"],
            "b": ["horse", "surface", "distance_band"],
            "e": ["horse", "venue", "surface", "exact_distance"],
            "x": ["horse", "venue", "surface", "exact_distance", "class_level"],
        },
        "features": features,
    }
    (root / "output" / "FEATURE_SPEC.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[features] adult dirt rows={len(out):,} features={len(features)}", flush=True)
    return out


def ranker_model(cfg, n_estimators, seed):
    return LGBMRanker(
        objective="lambdarank", metric="ndcg", label_gain=[0, 1],
        n_estimators=n_estimators, learning_rate=0.04,
        num_leaves=cfg["leaves"], max_depth=cfg["depth"],
        min_child_samples=cfg["minc"], colsample_bytree=cfg["col"],
        reg_lambda=cfg["l2"], random_state=seed, n_jobs=2, verbosity=-1,
    )


def clf_model(cfg, seed, n_estimators=180):
    kind, depth, leaf = cfg
    if kind == "lgb":
        return LGBMClassifier(
            n_estimators=n_estimators, learning_rate=0.035,
            num_leaves=max(3, 2 ** depth - 1), max_depth=depth,
            min_child_samples=leaf, colsample_bytree=0.85, reg_lambda=3,
            class_weight="balanced", random_state=seed, verbosity=-1, n_jobs=2,
        )
    if kind == "et":
        return ExtraTreesClassifier(
            n_estimators=n_estimators, max_depth=depth, min_samples_leaf=leaf,
            max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=2,
        )
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=depth, min_samples_leaf=leaf,
        max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=2,
    )


def sorted_groups(sub):
    s = sub.sort_values(["race_id", "馬番"]).copy()
    return s, s.groupby("race_id", sort=False).size().tolist()


def prep(sub, features, med=None):
    x = sub[features].replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = x.median()
    return x.fillna(med).astype("float32"), med


def cap_at_k(sub, k, score_col="score"):
    vals = []
    for _, g in sub.groupby("race_id", sort=False):
        vals.append(int(g.nlargest(k, score_col)["target_top3"].sum() >= 3))
    return float(np.mean(vals)) if vals else np.nan


def top1_actual_top3_hit(sub, score_col):
    vals = []
    for _, g in sub.groupby("race_id", sort=False):
        idx = g[score_col].idxmax()
        vals.append(int(sub.loc[idx, "target_top3"] == 1))
    return float(np.mean(vals)) if vals else np.nan


def entropy_from_scores(x):
    x = np.asarray(x, dtype=float)
    ex = np.exp(x - np.nanmax(x))
    p = ex / np.nansum(ex)
    return float(-(p * np.log(p + 1e-12)).sum()), p


def wilson_lower(successes, n, z=1.6448536269514722):
    if n <= 0:
        return 0.0
    phat = successes / n
    den = 1 + z * z / n
    center = phat + z * z / (2 * n)
    adj = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (center - adj) / den


def feature_columns(df):
    excluded = {
        "race_id", "year_full", "場所", "距離", "馬番", "馬名", "horse_id",
        "finish", "target_win", "target_top2", "target_top3",
        "rank_score", "rank_pos", "sp_target_win", "sp_target_top2", "sp_target_top3",
    }
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def make_oof_rank(df, features, cfg):
    parts = []
    for fi, (a, b, c, d) in enumerate(RANK_FOLDS):
        tr = df[df.year_full.between(a, b)]
        va = df[df.year_full.between(c, d)].copy()
        if tr.race_id.nunique() < 20 or va.race_id.nunique() < 5:
            continue
        trs, groups = sorted_groups(tr)
        xtr, med = prep(trs, features)
        xv, _ = prep(va, features, med)
        m = ranker_model(cfg, 220, 500 + fi)
        m.fit(xtr, trs.target_top3.astype(int), group=groups)
        va["rank_score"] = m.predict(xv)
        va["rank_pos"] = va.groupby("race_id")["rank_score"].rank(method="first", ascending=False)
        parts.append(va)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def select_ranker(df, features):
    rows = []
    for ci, cfg in enumerate(RANK_GRID):
        fold_vals = []
        ok = True
        for fi, (a, b, c, d) in enumerate(RANK_FOLDS):
            tr = df[df.year_full.between(a, b)]
            va = df[df.year_full.between(c, d)].copy()
            if tr.race_id.nunique() < 20 or va.race_id.nunique() < 5:
                ok = False
                break
            trs, groups = sorted_groups(tr)
            xtr, med = prep(trs, features)
            xv, _ = prep(va, features, med)
            m = ranker_model(cfg, 150, 100 + ci * 10 + fi)
            m.fit(xtr, trs.target_top3.astype(int), group=groups)
            va["score"] = m.predict(xv)
            fold_vals.append([cap_at_k(va, k) for k in (6, 8, 10, 12)])
        if ok:
            rows.append({
                "idx": ci, **cfg,
                "min8": min(v[1] for v in fold_vals),
                "avg8": float(np.mean([v[1] for v in fold_vals])),
                "avg6_12": float(np.mean(fold_vals)),
            })
    if not rows:
        return None, pd.DataFrame()
    sel = pd.DataFrame(rows).sort_values(
        ["min8", "avg8", "avg6_12"], ascending=False
    ).reset_index(drop=True)
    sel["selected"] = False
    sel.loc[0, "selected"] = True
    return RANK_GRID[int(sel.iloc[0]["idx"])], sel


def select_specialists(df, features):
    selected = {}
    tables = {}
    for target in ["target_win", "target_top2", "target_top3"]:
        rows = []
        for ci, cfg in enumerate(SPECIAL_CFGS):
            hits, aucs = [], []
            ok = True
            for fi, (a, b, c, d) in enumerate(RANK_FOLDS):
                tr = df[df.year_full.between(a, b)]
                va = df[df.year_full.between(c, d)].copy()
                if tr[target].nunique() < 2 or va.race_id.nunique() < 5:
                    ok = False
                    break
                med = tr[features].median()
                m = clf_model(cfg, 1000 + ci * 10 + fi, 130)
                m.fit(tr[features].fillna(med), tr[target])
                va["s"] = m.predict_proba(va[features].fillna(med))[:, 1]
                hits.append(top1_actual_top3_hit(va, "s"))
                aucs.append(roc_auc_score(va[target], va["s"]) if va[target].nunique() > 1 else 0.5)
            if ok:
                rows.append({
                    "idx": ci, "cfg": str(cfg), "min_hit": min(hits),
                    "avg_hit": float(np.mean(hits)), "avg_auc": float(np.mean(aucs)),
                })
        if not rows:
            return None, {}
        tab = pd.DataFrame(rows).sort_values(
            ["min_hit", "avg_hit", "avg_auc"], ascending=False
        ).reset_index(drop=True)
        tab["selected"] = False
        tab.loc[0, "selected"] = True
        selected[target] = SPECIAL_CFGS[int(tab.iloc[0]["idx"])]
        tables[target] = tab
    return selected, tables


def attach_specialist_predictions(df, features, selected, test=False):
    if test:
        tr = df[df.year_full.between(2011, 2022)]
        te = df[df.year_full.between(2023, 2026)].copy()
        med = tr[features].median()
        models = {}
        for ti, target in enumerate(["target_win", "target_top2", "target_top3"]):
            m = clf_model(selected[target], 3000 + ti, 240)
            m.fit(tr[features].fillna(med), tr[target])
            te["sp_" + target] = m.predict_proba(te[features].fillna(med))[:, 1]
            models[target] = m
        return te, models, med

    parts = []
    for fi, (a, b, c, d) in enumerate(RANK_FOLDS):
        tr = df[df.year_full.between(a, b)]
        va = df[df.year_full.between(c, d)].copy()
        if tr.empty or va.empty:
            continue
        med = tr[features].median()
        for ti, target in enumerate(["target_win", "target_top2", "target_top3"]):
            m = clf_model(selected[target], 2000 + fi * 10 + ti, 180)
            m.fit(tr[features].fillna(med), tr[target])
            va["sp_" + target] = m.predict_proba(va[features].fillna(med))[:, 1]
        parts.append(va)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def mark_and_race_features(horses, k):
    rows = []
    marked_parts = []
    for rid, g in horses.groupby("race_id", sort=False):
        gg = g.copy()
        chosen = []
        for score in ["sp_target_win", "sp_target_top2", "sp_target_top3"]:
            cand = gg.loc[~gg.index.isin(chosen)]
            idx = cand[score].idxmax()
            chosen.append(idx)
        gg["mark"] = 0
        for j, idx in enumerate(chosen, 1):
            gg.loc[idx, "mark"] = j

        rs = gg.sort_values("rank_pos")
        top = rs.head(k)
        full3 = int(top.target_top3.sum() >= 3)
        axis_hit = int(gg.loc[chosen, "target_top3"].sum() >= 1)
        formation = int(full3 and axis_hit)
        scores = rs.rank_score.to_numpy(float)
        ent, prob = entropy_from_scores(scores)

        row = {
            "race_id": str(rid), "year": int(gg.year_full.iloc[0]),
            "full3": full3, "axis_hit": axis_hit, "formation": formation,
            "field_n": len(gg), "class_level": float(gg.class_level.iloc[0]),
            "month": float(gg["month"].iloc[0]), "weight_code": float(gg["weight_code"].iloc[0])
                if pd.notna(gg["weight_code"].iloc[0]) else np.nan,
            "race_g_cov": float(gg.race_g_cov.iloc[0]),
            "race_b_cov": float(gg.race_b_cov.iloc[0]),
            "race_e_cov": float(gg.race_e_cov.iloc[0]),
            "race_x_cov": float(gg.race_x_cov.iloc[0]),
            "rank_std": float(np.nanstd(scores)), "rank_entropy": ent,
            "top3_mass": float(prob[:3].sum()),
            "topk_mass": float(prob[: min(k, len(prob))].sum()),
            "k": int(k),
        }
        for pref, score in [
            ("w", "sp_target_win"), ("q", "sp_target_top2"), ("t", "sp_target_top3")
        ]:
            vals = np.sort(gg[score].to_numpy(float))[::-1]
            row[pref + "1"] = vals[0]
            row[pref + "2"] = vals[1] if len(vals) > 1 else np.nan
            row[pref + "gap"] = vals[0] - vals[1] if len(vals) > 1 else np.nan
        for i in range(10):
            row[f"r{i+1}"] = float(scores[i]) if i < len(scores) else np.nan
        rows.append(row)
        marked_parts.append(gg)
    return pd.DataFrame(rows), pd.concat(marked_parts, ignore_index=True)


def choose_scan(oof):
    all_candidates = []
    by_k = {}
    for k in (6, 8, 10, 12):
        races, _ = mark_and_race_features(oof, k)
        by_k[k] = races
        tr = races[races.year.between(2017, 2020)].copy()
        va = races[races.year.between(2021, 2022)].copy()
        if tr.empty or va.empty or tr.formation.nunique() < 2:
            continue
        sf = [
            c for c in races.columns
            if c not in {"race_id", "year", "full3", "axis_hit", "formation"}
            and pd.api.types.is_numeric_dtype(races[c])
        ]
        med = tr[sf].median()
        a = tr[sf].fillna(med)
        b = va[sf].fillna(med)
        minplay = max(5, int(math.ceil(0.15 * len(va))))
        for ci, cfg in enumerate(SCAN_CFGS):
            m = clf_model(cfg, 5000 + k * 10 + ci, 180)
            m.fit(a, tr.formation)
            p = m.predict_proba(b)[:, 1]
            thresholds = np.unique(np.quantile(p, np.linspace(0.25, 0.995, 80)))
            for thr in thresholds:
                z = p >= thr
                n = int(z.sum())
                if n < minplay:
                    continue
                succ = int(va.formation.to_numpy()[z].sum())
                precision = succ / n
                full3 = float(va.full3.to_numpy()[z].mean())
                axis = float(va.axis_hit.to_numpy()[z].mean())
                all_candidates.append({
                    "k": k, "cfg_idx": ci, "cfg": str(cfg), "thr": float(thr),
                    "play": n, "precision": precision, "full3": full3, "axis": axis,
                    "wilson90": wilson_lower(succ, n),
                })
    if not all_candidates:
        return None, pd.DataFrame(), by_k
    tab = pd.DataFrame(all_candidates).sort_values(
        ["wilson90", "precision", "full3", "axis", "play"],
        ascending=False,
    ).reset_index(drop=True)
    tab["selected"] = False
    tab.loc[0, "selected"] = True
    return tab.iloc[0], tab, by_k


def cell_name(place, distance):
    return f"{PLACE_SLUG.get(place, 'other')}_d{int(distance)}"


def train_cell(cell_df, features, root: Path, place, distance):
    name = cell_name(place, distance)
    out = root / "output" / "cells" / name
    out.mkdir(parents=True, exist_ok=True)
    model_dir = root / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)

    rank_cfg, rank_sel = select_ranker(cell_df, features)
    if rank_cfg is None:
        return {"cell": name, "venue": place, "distance": int(distance), "status": "SKIP_RANK"}

    rank_sel.to_csv(out / "RANK_SELECTION.csv", index=False)
    oof_rank = make_oof_rank(cell_df, features, rank_cfg)
    if oof_rank.empty:
        return {"cell": name, "venue": place, "distance": int(distance), "status": "SKIP_OOF"}

    tr = cell_df[cell_df.year_full.between(2011, 2022)]
    te = cell_df[cell_df.year_full.between(2023, 2026)].copy()
    trs, groups = sorted_groups(tr)
    xtr, rank_med = prep(trs, features)
    xt, _ = prep(te, features, rank_med)
    final_ranker = ranker_model(rank_cfg, 320, 777)
    final_ranker.fit(xtr, trs.target_top3.astype(int), group=groups)
    te["rank_score"] = final_ranker.predict(xt)
    te["rank_pos"] = te.groupby("race_id")["rank_score"].rank(method="first", ascending=False)

    selected_special, special_tabs = select_specialists(cell_df, features)
    if selected_special is None:
        return {"cell": name, "venue": place, "distance": int(distance), "status": "SKIP_SPECIAL"}
    for target, tab in special_tabs.items():
        tab.to_csv(out / f"{target}_SELECTION.csv", index=False)

    oof_sp = attach_specialist_predictions(cell_df, features, selected_special, test=False)
    if oof_sp.empty:
        return {"cell": name, "venue": place, "distance": int(distance), "status": "SKIP_SPECIAL_OOF"}
    rank_cols = oof_rank[["race_id", "馬番", "rank_score", "rank_pos"]]
    oof = oof_sp.merge(rank_cols, on=["race_id", "馬番"], how="inner")

    test_sp, special_models, special_med = attach_specialist_predictions(
        cell_df, features, selected_special, test=True
    )
    test = test_sp.merge(
        te[["race_id", "馬番", "rank_score", "rank_pos"]], on=["race_id", "馬番"], how="inner"
    )

    scan_pick, scan_table, _ = choose_scan(oof)
    if scan_pick is None:
        return {"cell": name, "venue": place, "distance": int(distance), "status": "SKIP_SCAN"}
    scan_table.to_csv(out / "SCAN_SELECTION.csv", index=False)

    k = int(scan_pick["k"])
    ro, oof_marked = mark_and_race_features(oof, k)
    rt, test_marked = mark_and_race_features(test, k)
    sf = [
        c for c in ro.columns
        if c not in {"race_id", "year", "full3", "axis_hit", "formation"}
        and pd.api.types.is_numeric_dtype(ro[c])
    ]
    scan_med = ro[sf].median()
    scan_cfg = SCAN_CFGS[int(scan_pick["cfg_idx"])]
    scan_model = clf_model(scan_cfg, 6000, 280)
    scan_model.fit(ro[sf].fillna(scan_med), ro.formation)
    rt["scan_prob"] = scan_model.predict_proba(rt[sf].fillna(scan_med))[:, 1]
    rt["play"] = rt["scan_prob"] >= float(scan_pick["thr"])
    played = rt[rt.play].copy()

    selected_n = int(len(played))
    capture = float(played.formation.mean()) if selected_n else np.nan
    pool_full3 = float(played.full3.mean()) if selected_n else np.nan
    axis_hit = float(played.axis_hit.mean()) if selected_n else np.nan
    coverage = selected_n / len(rt) if len(rt) else 0.0

    if selected_n < 8:
        tier, status = "INSUFFICIENT", "SKIP"
    elif capture >= 0.98:
        tier, status = "S", "ADOPT"
    elif capture >= 0.95:
        tier, status = "A+", "ADOPT"
    elif capture >= 0.90:
        tier, status = "A", "ADOPT"
    else:
        tier, status = "BELOW_A", "SKIP"

    rt.to_csv(out / "TEST_RACES.csv", index=False)
    test_marked.to_csv(out / "TEST_HORSES.csv", index=False)
    if selected_n:
        played.groupby("year").agg(
            races=("race_id", "size"),
            formation=("formation", "mean"),
            full3=("full3", "mean"),
            axis_hit=("axis_hit", "mean"),
        ).to_csv(out / "BY_YEAR.csv")

    joblib.dump(final_ranker, model_dir / "ranker.joblib", compress=3)
    joblib.dump(rank_med, model_dir / "rank_median.joblib", compress=3)
    joblib.dump(special_med, model_dir / "special_median.joblib", compress=3)
    for target, model in special_models.items():
        joblib.dump(model, model_dir / f"{target}.joblib", compress=3)
    joblib.dump(scan_model, model_dir / "scan.joblib", compress=3)
    joblib.dump(scan_med, model_dir / "scan_median.joblib", compress=3)

    spec = {
        "cell": name, "venue": place, "distance": int(distance),
        "features": features, "rank_cfg": rank_cfg,
        "special_cfg": {k_: list(v) for k_, v in selected_special.items()},
        "scan_cfg": list(scan_cfg), "scan_features": sf,
        "scan_threshold": float(scan_pick["thr"]), "candidate_k": k,
        "marks": ["target_win", "target_top2", "target_top3"],
        "odds_popularity_used": False,
        "train_years": [2011, 2022], "test_years": [2023, 2026],
    }
    (model_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "cell": name, "venue": place, "distance": int(distance), "status": status, "tier": tier,
        "test_races": int(len(rt)), "selected_races": selected_n, "coverage": coverage,
        "capture_3of3": capture, "pool_full3": pool_full3, "axis_hit": axis_hit,
        "candidate_k": k, "scan_threshold": float(scan_pick["thr"]),
        "val_precision": float(scan_pick["precision"]), "val_wilson90": float(scan_pick["wilson90"]),
    }


def build_manifest(root: Path):
    records = []
    for base in [root / "models", root / "output"]:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.name != "MANIFEST_SHA256.csv":
                records.append({
                    "path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                })
    man = pd.DataFrame(records).sort_values("path")
    man.to_csv(root / "output" / "MANIFEST_SHA256.csv", index=False)
    return man


def make_runtime_zip(root: Path):
    target = root / "output" / "JRA_ADULT_DIRT_RUNTIME_V2.zip"
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for relbase in ["models", "output/REGISTRY.csv", "output/FEATURE_SPEC.json", "output/MANIFEST_SHA256.csv"]:
            p = root / relbase
            if p.is_dir():
                for q in p.rglob("*"):
                    if q.is_file():
                        z.write(q, q.relative_to(root))
            elif p.exists():
                z.write(p, p.relative_to(root))
    return target


def run_training(root: Path):
    root = Path(root)
    (root / "output").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    write_status(root, phase="STARTING", complete=False, error=None)
    try:
        zip_path = root / "input" / "central.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"dataset missing: {zip_path}")
        write_status(root, phase="LOADING_DATA")
        base = load_history(zip_path, root)
        write_status(root, phase="BUILDING_FEATURES", completed_rows=int(len(base)))
        adult = build_features(base, root)
        features = feature_columns(adult)

        race_meta = adult.groupby(["場所", "距離", "year_full"], sort=False)["race_id"].nunique().reset_index(name="races")
        race_meta.to_csv(root / "output" / "CELL_YEAR_META.csv", index=False)

        cells = adult.groupby(["場所", "距離"], sort=False)
        registry = []
        eligible = []
        for (place, distance), g in cells:
            train_races = g[g.year_full.between(2011, 2022)].race_id.nunique()
            test_races = g[g.year_full.between(2023, 2026)].race_id.nunique()
            if train_races >= 40 and test_races >= 10:
                eligible.append((place, distance, train_races, test_races))

        write_status(root, phase="TRAINING_CELLS", eligible_cells=len(eligible), completed_cells=0)
        for i, (place, distance, train_races, test_races) in enumerate(eligible, 1):
            name = cell_name(place, distance)
            print(f"[cell] {i}/{len(eligible)} {name} train={train_races} test={test_races}", flush=True)
            try:
                result = train_cell(
                    adult[(adult["場所"] == place) & (adult["距離"] == distance)].copy(),
                    features, root, place, distance
                )
            except Exception as exc:
                result = {
                    "cell": name, "venue": place, "distance": int(distance),
                    "status": "ERROR", "error": repr(exc),
                }
                print(f"[cell:error] {name}: {exc!r}", flush=True)
            registry.append(result)
            pd.DataFrame(registry).to_csv(root / "output" / "REGISTRY.csv", index=False)
            write_status(
                root, phase="TRAINING_CELLS", eligible_cells=len(eligible),
                completed_cells=i, current_cell=name
            )

        reg = pd.DataFrame(registry)
        reg.to_csv(root / "output" / "REGISTRY.csv", index=False)
        build_manifest(root)
        runtime_zip = make_runtime_zip(root)
        runtime_sha = sha256_file(runtime_zip)
        summary = {
            "version": "JRA_ADULT_DIRT_RUNTIME_V2",
            "eligible_cells": len(eligible),
            "adopt_cells": int((reg.get("status") == "ADOPT").sum()) if not reg.empty else 0,
            "skip_cells": int((reg.get("status") == "SKIP").sum()) if not reg.empty else 0,
            "error_cells": int((reg.get("status") == "ERROR").sum()) if not reg.empty else 0,
            "runtime_zip": runtime_zip.name,
            "runtime_sha256": runtime_sha,
            "odds_popularity_used": False,
        }
        (root / "output" / "SUMMARY.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_status(root, phase="COMPLETE", complete=True, summary=summary)
        print("[complete]", json.dumps(summary, ensure_ascii=False), flush=True)
        return summary
    except Exception as exc:
        write_status(root, phase="FAILED", complete=False, error=repr(exc))
        print(f"[fatal] {exc!r}", flush=True)
        raise


if __name__ == "__main__":
    root = Path(os.environ.get("JRA_WORKDIR", "/data/jra"))
    run_training(root)
