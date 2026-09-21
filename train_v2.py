import gc
import hashlib
import json
import math
import os
import shutil
import time
import traceback
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
PLACE_CODE = {name: i + 1 for i, name in enumerate(PLACE_SLUG)}
CODE_PLACE = {v: k for k, v in PLACE_CODE.items()}
SURFACE_MAP = {"芝": 1, "ダ": 2}
LAST3_METRICS = {"finish", "margin", "time_sec", "corrected_time", "corner4", "final3f", "pci", "rpci"}

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

# Memory-safe simple97 family: the empirically stronger adult-dirt reconstruction
# used eight history metrics. Current bodyweight/carried remain current-condition
# features, but are not duplicated into all four historical blocks.
HIST_METRICS = [
    "finish", "margin", "time_sec", "corrected_time",
    "corner4", "final3f", "pci", "rpci",
]

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
    cache = root / "cache" / "history_base_flat_allrunners_v2_3.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        print(f"[load] using compact cache {cache}", flush=True)
        return pd.read_pickle(cache)

    source_zip = zip_path
    with zipfile.ZipFile(zip_path) as outer:
        direct_csvs = [n for n in outer.namelist() if n.upper().endswith(".CSV")]
        direct_ok = False
        for name in direct_csvs[:5]:
            try:
                with outer.open(name) as fh:
                    cols = set(pd.read_csv(fh, encoding="cp932", nrows=0).columns)
                if set(RAW_COLUMNS).issubset(cols):
                    direct_ok = True
                    break
            except Exception:
                pass
        if not direct_ok:
            nested_infos = sorted(
                [i for i in outer.infolist() if i.filename.lower().endswith(".zip")],
                key=lambda i: i.file_size,
                reverse=True,
            )
            for info in nested_infos:
                candidate = root / "cache" / Path(info.filename).name
                if not candidate.exists() or candidate.stat().st_size != info.file_size:
                    print(f"[load] extracting nested zip {info.filename} bytes={info.file_size}", flush=True)
                    with outer.open(info) as src, candidate.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                try:
                    with zipfile.ZipFile(candidate) as nz:
                        names = [n for n in nz.namelist() if n.upper().endswith(".CSV")]
                        matched = False
                        for name in names[:10]:
                            try:
                                with nz.open(name) as fh:
                                    cols = set(pd.read_csv(fh, encoding="cp932", nrows=0).columns)
                                if set(RAW_COLUMNS).issubset(cols):
                                    matched = True
                                    break
                            except Exception:
                                pass
                    if matched:
                        source_zip = candidate
                        print(f"[load] selected nested source {candidate.name}", flush=True)
                        break
                except Exception:
                    continue

    parts = []
    with zipfile.ZipFile(source_zip) as z:
        csvs = [n for n in z.namelist() if n.upper().endswith(".CSV")]
        for i, name in enumerate(csvs, 1):
            with z.open(name) as fh:
                try:
                    d = pd.read_csv(fh, encoding="cp932", usecols=RAW_COLUMNS, low_memory=False)
                except Exception:
                    continue

            # Pre-race-safe field definition:
            # 1=出走取消, 2=発走除外, 3=競走除外 never started and are known
            # before the race starts, so remove them. Keep 4+ (競走中止/失格/etc.)
            # because those horses did start and belong in the pre-race field.
            abnormal = pd.to_numeric(d["異常コード"], errors="coerce").fillna(0).astype("int8")
            track_code_raw = pd.to_numeric(d["トラックコード"], errors="coerce")
            # JRA source audit: track_code 2/3 are obstacle races (2=turf obstacle,
            # 3=dirt obstacle). Exclude them BEFORE building any horse history so
            # g/b/e/x are flat-racing histories only.
            flat_row = ~track_code_raw.isin([2, 3])
            valid_race = d["レースID(新)"].notna() & flat_row & ~abnormal.isin([1, 2, 3])
            d = d.loc[valid_race].copy()
            if d.empty:
                continue

            n = pd.DataFrame(index=d.index)
            def num(col):
                return pd.to_numeric(d[col], errors="coerce")

            year = num("年")
            n["year_full"] = np.where(year < 100, 2000 + year, year).astype("int16")
            n["month"] = num("月").fillna(0).astype("int8")
            n["day"] = num("日").fillna(0).astype("int8")
            n["place_code"] = d["場所"].map(PLACE_CODE).fillna(0).astype("int8")
            n["surface_code"] = d["芝・ダ"].map(SURFACE_MAP).fillna(0).astype("int8")
            n["distance"] = num("距離").fillna(0).astype("int16")
            n["class_code"] = num("クラスコード").fillna(0).astype("int16")
            n["class_level"] = n["class_code"].map(CLASS_MAP).fillna(0).astype("int8")
            n["age"] = num("年齢").fillna(0).astype("int8")
            n["carried"] = num("斤量").astype("float32")
            n["field_n"] = num("頭数").fillna(0).astype("int8")
            n["horse_no"] = num("馬番").fillna(0).astype("int8")
            n["finish"] = num("確定着順").fillna(0).astype("int8")
            n["margin"] = num("着差タイム").astype("float32")
            n["time_sec"] = num("走破タイム(秒)").astype("float32")
            n["corrected_time"] = num("補正タイム").astype("float32")
            n["corner4"] = num("通過順4角").astype("float32")
            n["final3f"] = num("上がり3Fタイム").astype("float32")
            n["bodyweight"] = num("馬体重").astype("float32")
            n["pci"] = num("PCI").astype("float32")
            n["rpci"] = num("RPCI").astype("float32")
            n["frame_no"] = num("枠番").fillna(0).astype("int8")
            n["weight_code"] = num("重量コード").astype("float32")
            n["age_limit_code"] = num("年齢限定(競走種別コード)").astype("float32")
            n["track_code_jv"] = num("トラックコード(JV)").astype("float32")
            n["track_state_code"] = d["馬場状態"].map(TRACK_STATE_MAP).fillna(0).astype("int8")
            n["sex_code"] = d["性別"].map(SEX_MAP).fillna(0).astype("int8")

            runner_id_text = (
                d["レースID(新)"].astype("string").fillna("").str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )
            horse_no_text = pd.to_numeric(d["馬番"], errors="coerce").fillna(0).astype("int16").astype(str).str.zfill(2)
            suffix_ok = runner_id_text.str[-2:].eq(horse_no_text)
            if not bool(suffix_ok.all()):
                bad_n = int((~suffix_ok).sum())
                raise RuntimeError(f"race id suffix/horse number mismatch rows={bad_n}")

            # レースID(新) is runner-level: the trailing 2 digits are 馬番.
            # Strip them to obtain one stable key shared by every runner in a race.
            race_key_text = runner_id_text.str[:-2]
            if not bool(race_key_text.str.fullmatch(r"\d{16}").all()):
                raise RuntimeError("race key format mismatch: expected 16 numeric digits")
            n["race_id"] = pd.to_numeric(race_key_text, errors="raise").astype("int64")

            horse = num("血統登録番号")
            fb = (
                pd.util.hash_pandas_object(d["馬名"].fillna(""), index=False).to_numpy(dtype="uint64")
                & np.uint64(0x7FFFFFFFFFFFFFFF)
            ).astype("int64")
            harr = np.where(
                horse.notna().to_numpy(),
                horse.fillna(0).astype("int64").to_numpy(),
                -fb - 1,
            )
            n["horse_id"] = harr.astype("int64")

            dates = pd.to_datetime(
                dict(year=n["year_full"], month=n["month"], day=n["day"]), errors="coerce"
            )
            n["date_days"] = (dates - pd.Timestamp("2000-01-01")).dt.days.fillna(-1).astype("int32")
            n["dist_band"] = n["distance"].map(dist_band).fillna(-1).astype("int8")
            n["draw_pct"] = (n["frame_no"].astype("float32") / 8.0).astype("float32")
            n["gate_pct"] = (
                n["horse_no"].astype("float32") / n["field_n"].replace(0, np.nan).astype("float32")
            ).astype("float32")
            n["speed1000"] = (
                n["time_sec"] / n["distance"].replace(0, np.nan).astype("float32") * 1000.0
            ).astype("float32")
            n["completed"] = (n["finish"] > 0).astype("int8")
            n["target_win"] = (n["finish"] == 1).astype("int8")
            n["target_top2"] = n["finish"].between(1, 2).astype("int8")
            n["target_top3"] = n["finish"].between(1, 3).astype("int8")
            n = n[n["race_id"] > 0]
            parts.append(n.reset_index(drop=True))

            if i % 50 == 0:
                rows = sum(len(x) for x in parts)
                print(f"[load] {i}/{len(csvs)} files compact_rows={rows:,}", flush=True)

    if not parts:
        raise RuntimeError("No horse-level CSVs could be loaded from dataset ZIP or nested ZIPs")

    df = pd.concat(parts, ignore_index=True, copy=False)
    del parts
    df.sort_values(["horse_id", "date_days", "race_id", "horse_no"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_pickle(cache)
    print(
        f"[load] all-runner compact rows={len(df):,} memory_mb={df.memory_usage(deep=True).sum()/1e6:.1f}",
        flush=True,
    )
    return df


def add_group_features_to_target(base, adult, mask, keys, prefix, add_last3=False):
    key_series = [base[k] for k in keys]
    completed = base["completed"].astype("int32")

    # History count = only prior completed races in the matching context.
    comp_cum = completed.groupby(key_series, sort=False, dropna=False).cumsum() - completed
    adult[f"{prefix}_n"] = comp_cum.loc[mask].to_numpy(dtype="int32")
    del comp_cum

    for m in HIST_METRICS:
        # Prior result metrics are known at prediction time. Current/future
        # non-completions never enter the historical aggregate.
        masked = base[m].where(base["completed"].eq(1))

        shifted = masked.groupby(key_series, sort=False, dropna=False).shift(1)
        last = shifted.groupby(key_series, sort=False, dropna=False).ffill()
        adult[f"{prefix}_last_{m}"] = last.loc[mask].to_numpy(dtype="float32")

        valid = masked.notna().astype("int32")
        filled = masked.fillna(0.0).astype("float64")
        csum = filled.groupby(key_series, sort=False, dropna=False).cumsum() - filled
        ccnt = valid.groupby(key_series, sort=False, dropna=False).cumsum() - valid
        mean = csum / ccnt.replace(0, np.nan)
        adult[f"{prefix}_mean_{m}"] = mean.loc[mask].to_numpy(dtype="float32")

        if add_last3 and m in LAST3_METRICS:
            # Rolling over the previous three race rows, averaging only completed
            # values. This is pre-race safe because all inputs are prior rows.
            last3 = masked.groupby(key_series, sort=False, dropna=False).transform(
                lambda s: s.shift(1).rolling(3, min_periods=1).mean()
            )
            adult[f"{prefix}_last3_{m}"] = last3.loc[mask].to_numpy(dtype="float32")
            del last3

        del masked, shifted, last, valid, filled, csum, ccnt, mean
        gc.collect()

    del completed
    gc.collect()
    return adult

def build_features(base: pd.DataFrame, root: Path):
    cache = root / "cache" / "adult_dirt_features_simple97_flat_noleak_v2_3.pkl"
    if cache.exists():
        try:
            print(f"[features] using cache {cache}", flush=True)
            return pd.read_pickle(cache)
        except Exception as exc:
            print(f"[features] invalid cache removed: {cache.name} {exc!r}", flush=True)
            cache.unlink(missing_ok=True)

    mask = (
        (base["surface_code"] == 2)
        & (base["age"] >= 3)
        & (base["class_level"] > 0)
        & (base["place_code"] > 0)
    )
    raw_keep = [
        "race_id", "year_full", "place_code", "distance", "horse_no", "horse_id",
        "finish", "target_win", "target_top2", "target_top3", "class_level",
        "month", "age", "carried", "field_n", "bodyweight", "frame_no",
        "weight_code", "age_limit_code", "track_code_jv", "draw_pct", "gate_pct",
        "sex_code", "track_state_code", "date_days",
    ]
    adult = base.loc[mask, raw_keep].copy().reset_index(drop=True)
    print(
        f"[features] adult target rows={len(adult):,} base_mb={base.memory_usage(deep=True).sum()/1e6:.1f}",
        flush=True,
    )

    print("[features] g", flush=True)
    adult = add_group_features_to_target(base, adult, mask, ["horse_id"], "g", add_last3=True)
    print("[features] b", flush=True)
    adult = add_group_features_to_target(
        base, adult, mask, ["horse_id", "surface_code", "dist_band"], "b"
    )
    print("[features] e", flush=True)
    adult = add_group_features_to_target(
        base, adult, mask, ["horse_id", "place_code", "surface_code", "distance"], "e"
    )
    print("[features] x", flush=True)
    adult = add_group_features_to_target(
        base, adult, mask,
        ["horse_id", "place_code", "surface_code", "distance", "class_level"], "x"
    )

    horse_group = base.groupby("horse_id", sort=False)
    prev_days = horse_group["date_days"].shift(1)
    prev_body = horse_group["bodyweight"].shift(1)
    prev_carried = horse_group["carried"].shift(1)
    adult["days_since"] = (
        base.loc[mask, "date_days"].to_numpy(dtype="float32")
        - prev_days.loc[mask].to_numpy(dtype="float32")
    )
    adult["body_change"] = (
        adult["bodyweight"].to_numpy(dtype="float32")
        - prev_body.loc[mask].to_numpy(dtype="float32")
    )
    adult["carry_change"] = (
        adult["carried"].to_numpy(dtype="float32")
        - prev_carried.loc[mask].to_numpy(dtype="float32")
    )
    del horse_group, prev_days, prev_body, prev_carried
    gc.collect()

    for pref in ["g", "b", "e", "x"]:
        adult[f"race_{pref}_cov"] = adult.groupby("race_id", sort=False)[f"{pref}_n"].transform(
            lambda s: (s > 0).mean()
        ).astype("float32")

    current = [
        "month", "distance", "age", "carried", "field_n", "horse_no", "bodyweight",
        "frame_no", "weight_code", "age_limit_code", "track_code_jv", "class_level",
        "days_since", "body_change", "carry_change", "draw_pct", "gate_pct",
        "sex_code", "track_state_code", "race_g_cov", "race_b_cov", "race_e_cov", "race_x_cov",
    ]
    hist = [
        col for col in adult.columns
        if col.startswith(("g_", "b_", "e_", "x_")) and col not in {"g_n", "b_n", "e_n", "x_n"}
    ]
    features = current + ["g_n", "b_n", "e_n", "x_n"] + hist
    features = list(dict.fromkeys([col for col in features if col in adult.columns]))

    keep_raw = [
        "race_id", "year_full", "place_code", "distance", "horse_no", "horse_id",
        "finish", "target_win", "target_top2", "target_top3", "class_level",
    ]
    output_columns = keep_raw + [col for col in features if col not in keep_raw]
    out = adult[output_columns].copy()
    if out.columns.duplicated().any():
        raise RuntimeError("duplicate feature columns detected")
    for col in features:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    cache_tmp = cache.with_suffix(cache.suffix + ".part")
    cache_tmp.unlink(missing_ok=True)
    out.to_pickle(cache_tmp)
    os.replace(cache_tmp, cache)

    spec = {
        "version": "ADULT_DIRT_V2_3_FLAT_NOLEAK",
        "history_rows_completed_only": True,
        "current_race_keeps_all_runners": True,
        "postrace_finish_filter_on_current_race": False,
        "prestart_nonrunners_excluded_abnormal_codes": [1, 2, 3],
        "obstacle_track_codes_excluded_before_history": [2, 3],
        "race_key_source": "レースID(新) stripped trailing 2-digit horse number",
        "odds_popularity_used": False,
        "compact_numeric_loader": True,
        "feature_family": "simple97_memory_safe",
        "history_aggregates": ["last", "mean"],
        "g_extra_aggregate": "last3",
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
    print(
        f"[features] adult dirt rows={len(out):,} features={len(features)} "
        f"memory_mb={out.memory_usage(deep=True).sum()/1e6:.1f}",
        flush=True,
    )
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
    s = sub.sort_values(["race_id", "horse_no"]).copy()
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
        "race_id", "year_full", "place_code", "distance", "horse_no", "horse_id", "horse_id",
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
    short_races = 0
    for rid, g in horses.groupby("race_id", sort=False):
        gg = g.copy()
        if len(gg) < 3:
            # A 3-of-3 objective is undefined for fewer than three available rows.
            # Keep this fail-closed and visible rather than crashing.
            short_races += 1
            continue
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
    if short_races:
        print(f"[scan] skipped_short_races={short_races}", flush=True)
    marked = pd.concat(marked_parts, ignore_index=True) if marked_parts else pd.DataFrame()
    return pd.DataFrame(rows), marked


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


def cell_name(place_code, distance):
    place = CODE_PLACE.get(int(place_code), "other")
    return f"{PLACE_SLUG.get(place, 'other')}_d{int(distance)}"


def train_cell(cell_df, features, root: Path, place_code, distance):
    place = CODE_PLACE.get(int(place_code), str(place_code))
    name = cell_name(place_code, distance)
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
    rank_cols = oof_rank[["race_id", "horse_no", "rank_score", "rank_pos"]]
    oof = oof_sp.merge(rank_cols, on=["race_id", "horse_no"], how="inner")

    test_sp, special_models, special_med = attach_specialist_predictions(
        cell_df, features, selected_special, test=True
    )
    test = test_sp.merge(
        te[["race_id", "horse_no", "rank_score", "rank_pos"]], on=["race_id", "horse_no"], how="inner"
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
    # Preserve forward calibration: the threshold was selected from a model
    # trained on 2017-2020 and evaluated on 2021-2022. Recreate that exact
    # training regime for the frozen 2023-2026 test instead of refitting on
    # 2021-2022 and reusing an incompatible probability threshold.
    scan_train = ro[ro.year.between(2017, 2020)].copy()
    scan_med = scan_train[sf].median()
    scan_cfg_idx = int(scan_pick["cfg_idx"])
    scan_cfg = SCAN_CFGS[scan_cfg_idx]
    scan_seed = 5000 + k * 10 + scan_cfg_idx
    scan_model = clf_model(scan_cfg, scan_seed, 180)
    scan_model.fit(scan_train[sf].fillna(scan_med), scan_train.formation)
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
        "scan_train_years": [2017, 2020], "scan_threshold_validation_years": [2021, 2022],
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
    target = root / "output" / "JRA_ADULT_DIRT_RUNTIME_V2_3.zip"
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


def finalize_existing_v2_3(root: Path):
    root = Path(root)
    reg_path = root / "output" / "REGISTRY.csv"
    status_path = root / "output" / "status.json"
    if not reg_path.exists():
        raise RuntimeError("cannot finalize: REGISTRY.csv missing")

    reg = pd.read_csv(reg_path)
    eligible_expected = None
    if status_path.exists():
        try:
            s = json.loads(status_path.read_text(encoding="utf-8"))
            eligible_expected = s.get("eligible_cells")
        except Exception:
            eligible_expected = None
    if eligible_expected is None:
        eligible_expected = len(reg)

    if len(reg) != int(eligible_expected):
        raise RuntimeError(
            f"cannot finalize incomplete registry rows={len(reg)} expected={eligible_expected}"
        )

    # Reclaim persistent-volume space. Training is complete, so these caches and
    # invalid pre-v2.3 archives are no longer required to reproduce inference.
    for p in [
        root / "cache",
        root / "archive_invalid_pre_v2_3",
        root / "archive_invalid_v2",
    ]:
        if p.exists():
            print(f"[finalize] removing {p}", flush=True)
            shutil.rmtree(p)

    runtime_zip = root / "output" / "JRA_ADULT_DIRT_RUNTIME_V2_3.zip"
    runtime_zip.unlink(missing_ok=True)
    runtime_zip.with_suffix(runtime_zip.suffix + ".part").unlink(missing_ok=True)

    build_manifest(root)
    runtime_zip = make_runtime_zip(root)
    runtime_sha = sha256_file(runtime_zip)

    status_counts = reg["status"].fillna("UNKNOWN").value_counts().to_dict()
    summary = {
        "version": "JRA_ADULT_DIRT_RUNTIME_V2_3",
        "eligible_cells": int(len(reg)),
        "adopt_cells": int((reg["status"] == "ADOPT").sum()),
        "skip_cells": int((reg["status"] == "SKIP").sum()),
        "skip_rank_cells": int((reg["status"] == "SKIP_RANK").sum()),
        "error_cells": int((reg["status"] == "ERROR").sum()),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "runtime_zip": runtime_zip.name,
        "runtime_bytes": int(runtime_zip.stat().st_size),
        "runtime_sha256": runtime_sha,
        "odds_popularity_used": False,
        "finalized_from_existing_registry": True,
    }
    (root / "output" / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_status(root, phase="COMPLETE", complete=True, summary=summary)
    print("[complete]", json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def prepare_v2_3_workspace(root: Path):
    expected = "JRA_ADULT_DIRT_RUNTIME_V2_3"
    status_path = root / "output" / "status.json"
    existing_version = None
    if status_path.exists():
        try:
            s = json.loads(status_path.read_text(encoding="utf-8"))
            existing_version = (s.get("summary") or {}).get("version")
        except Exception:
            existing_version = None

    # If this volume still contains the invalid pre-v2.3 artifacts, isolate them
    # automatically. Input data and cache are intentionally preserved.
    if existing_version != expected:
        archive = root / "archive_invalid_pre_v2_3"
        archive.mkdir(parents=True, exist_ok=True)
        for name in ["models", "output"]:
            src = root / name
            if not src.exists():
                continue
            dst = archive / f"{name}_pre_v2_3"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))

    (root / "output").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)

    # Reclaim the small Railway volume from caches that can no longer be used by
    # v2.3. The raw input ZIP and the authoritative v2.3 all-runner cache remain.
    obsolete = [
        root / "cache" / "history_base_compact.pkl",
        root / "cache" / "adult_dirt_features_simple97_v2.pkl",
        root / "cache" / "history_base_allrunners_v2_2.pkl",
        root / "cache" / "adult_dirt_features_simple97_noleak_v2_2.pkl",
        root / "cache" / "adult_dirt_features_simple97_noleak_v2_2.pkl.part",
    ]
    for p in obsolete:
        if p.exists():
            print(f"[cleanup] removing obsolete cache {p.name} bytes={p.stat().st_size}", flush=True)
            p.unlink()

    target_cache = root / "cache" / "adult_dirt_features_simple97_flat_noleak_v2_3.pkl"
    target_part = target_cache.with_suffix(target_cache.suffix + ".part")
    # A cache file from a previously failed ENOSPC write is not authoritative.
    if target_part.exists():
        target_part.unlink()


def run_training(root: Path):
    root = Path(root)
    prepare_v2_3_workspace(root)
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

        race_sizes = adult.groupby("race_id", sort=False).size()
        if len(race_sizes) == 0:
            raise RuntimeError("no adult dirt races after feature build")
        if int(race_sizes.min()) < 3:
            bad = int((race_sizes < 3).sum())
            raise RuntimeError(f"race grouping integrity failure: short_races={bad} min_runners={int(race_sizes.min())}")
        print(
            f"[audit] race_groups={len(race_sizes):,} min_runners={int(race_sizes.min())} "
            f"max_runners={int(race_sizes.max())}",
            flush=True,
        )

        race_meta = adult.groupby(["place_code", "distance", "year_full"], sort=False)["race_id"].nunique().reset_index(name="races")
        race_meta.to_csv(root / "output" / "CELL_YEAR_META.csv", index=False)

        cells = adult.groupby(["place_code", "distance"], sort=False)
        registry = []
        eligible = []
        for (place_code, distance), g in cells:
            train_races = g[g.year_full.between(2011, 2022)].race_id.nunique()
            test_races = g[g.year_full.between(2023, 2026)].race_id.nunique()
            if train_races >= 40 and test_races >= 10:
                eligible.append((place_code, distance, train_races, test_races))

        write_status(root, phase="TRAINING_CELLS", eligible_cells=len(eligible), completed_cells=0)
        for i, (place_code, distance, train_races, test_races) in enumerate(eligible, 1):
            name = cell_name(place_code, distance)
            print(f"[cell] {i}/{len(eligible)} {name} train={train_races} test={test_races}", flush=True)
            try:
                result = train_cell(
                    adult[(adult["place_code"] == place_code) & (adult["distance"] == distance)].copy(),
                    features, root, place_code, distance
                )
            except Exception as exc:
                traceback.print_exc()
                result = {
                    "cell": name, "venue": CODE_PLACE.get(int(place_code), str(place_code)), "distance": int(distance),
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
            "version": "JRA_ADULT_DIRT_RUNTIME_V2_3",
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
