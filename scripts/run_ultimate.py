import os, sys, time, math, warnings, pathlib, itertools
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no GUI windows
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from catboost import CatBoostRegressor, Pool

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge

try:
    import pygeohash as pgh
    GEOHASH_AVAIL = True
except ImportError:
    GEOHASH_AVAIL = False

try:
    import shap
    SHAP_AVAIL = True
except ImportError:
    SHAP_AVAIL = False
    print("shap not available â€” SHAP section will be skipped")

warnings.filterwarnings("ignore")

# â”€â”€ Global constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEED     = 42
N_SPLITS = 5
TARGET   = "demand"
np.random.seed(SEED)

DATA_DIR = pathlib.Path("data")
SUB_DIR  = pathlib.Path("submissions")
SUB_DIR.mkdir(exist_ok=True)

print(f"LGB  {lgb.__version__}")
print(f"CB   {cb.__version__}")
print(f"XGB  {xgb.__version__}")
print(f"Optuna {optuna.__version__}")
print(f"SHAP: {SHAP_AVAIL}   pygeohash: {GEOHASH_AVAIL}")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw  = pd.read_csv(DATA_DIR / "test.csv")

print(f"Train : {train_raw.shape}")
print(f"Test  : {test_raw.shape}")
print(f"\nColumns: {list(train_raw.columns)}")
print(f"\nMissing values (train):")
print(train_raw.isnull().sum()[train_raw.isnull().sum() > 0])
print(f"\nMissing values (test):")
print(test_raw.isnull().sum()[test_raw.isnull().sum() > 0])
print(f"\nTarget stats:")
print(train_raw[TARGET].describe())
print(f"\nDay unique values: {sorted(train_raw['day'].unique())}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Target distribution
axes[0,0].hist(train_raw[TARGET], bins=60, color="steelblue", alpha=0.8, edgecolor="white")
axes[0,0].set_title("Demand Distribution", fontsize=13)
axes[0,0].set_xlabel("demand")

# Demand by hour
tmp = train_raw.copy()
tmp["_hour"] = tmp["timestamp"].str.split(":").str[0].astype(int)
hour_means = tmp.groupby("_hour")[TARGET].mean()
axes[0,1].bar(hour_means.index, hour_means.values, color="steelblue", alpha=0.8)
axes[0,1].set_title("Avg Demand by Hour", fontsize=13)
axes[0,1].set_xlabel("Hour")

# Demand by day
day_means = train_raw.groupby("day")[TARGET].mean()
axes[0,2].bar(day_means.index, day_means.values, color="darkorange", alpha=0.8)
axes[0,2].set_title("Avg Demand by Day", fontsize=13)
axes[0,2].set_xlabel("Day")

# Temp vs Demand
axes[1,0].scatter(train_raw["Temperature"], train_raw[TARGET], alpha=0.03, s=1, c="steelblue")
axes[1,0].set_title("Temperature vs Demand", fontsize=13)

# Weather
wc = train_raw["Weather"].value_counts()
axes[1,1].bar(wc.index, wc.values, color="seagreen", alpha=0.8)
axes[1,1].set_title("Weather Distribution", fontsize=13)
plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=30)

# Road type
rc = train_raw["RoadType"].value_counts()
axes[1,2].barh(rc.index, rc.values, color="mediumpurple", alpha=0.8)
axes[1,2].set_title("RoadType Distribution", fontsize=13)

plt.suptitle("Gridlock 2.0 â€” EDA Overview", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(str(SUB_DIR / "eda_overview.png"), dpi=80, bbox_inches="tight")
plt.show()
print("EDA saved")

# â”€â”€ Geohash decoder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_DECODE_MAP = {c: i for i, c in enumerate(_BASE32)}

def _decode_geohash(gh):
    if GEOHASH_AVAIL:
        d = pgh.decode(gh)
        return d[0], d[1]
    lat_r, lon_r = [-90., 90.], [-180., 180.]
    is_lon = True
    for c in gh:
        bits = _DECODE_MAP[c]
        for shift in (4, 3, 2, 1, 0):
            bit = (bits >> shift) & 1
            r = lon_r if is_lon else lat_r
            mid = (r[0] + r[1]) / 2
            if bit: r[0] = mid
            else:   r[1] = mid
            is_lon = not is_lon
    return (lat_r[0]+lat_r[1])/2, (lon_r[0]+lon_r[1])/2

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(min(a, 1.0)))

LANDMARKS = [
    {"name": "KL_Intl_Airport",     "lat": 2.7456,  "lon": 101.7072, "type": "airport"},
    {"name": "Subang_Airport",      "lat": 3.1306,  "lon": 101.5493, "type": "airport"},
    {"name": "KLCC",                "lat": 3.1579,  "lon": 101.7116, "type": "commercial"},
    {"name": "Bukit_Bintang",       "lat": 3.1467,  "lon": 101.7108, "type": "commercial"},
    {"name": "Bukit_Jalil_Stadium", "lat": 3.0574,  "lon": 101.6921, "type": "stadium"},
    {"name": "HKL",                 "lat": 3.1730,  "lon": 101.7036, "type": "hospital"},
    {"name": "KL_Sentral",          "lat": 3.1342,  "lon": 101.6865, "type": "transit"},
    {"name": "Midvalley",           "lat": 3.1184,  "lon": 101.6769, "type": "mall"},
    {"name": "Putrajaya",           "lat": 2.9264,  "lon": 101.6964, "type": "government"},
    {"name": "Cyberjaya",           "lat": 2.9213,  "lon": 101.6559, "type": "tech"},
]

def build_geo_cache(df_train, df_test):
    all_gh = set(df_train["geohash"].unique()) | set(df_test["geohash"].unique())
    cache = {}
    for gh in all_gh:
        try:    cache[gh] = _decode_geohash(gh)
        except: cache[gh] = (3.1579, 101.7116)
    return cache

print("Geospatial utilities defined")

def parse_timestamp(ts):
    p = str(ts).split(":")
    return int(p[0]), (int(p[1]) if len(p) > 1 else 0)

def add_time_features(df):
    df = df.copy()
    hours, minutes = zip(*df["timestamp"].map(parse_timestamp))
    df["hour"]   = np.array(hours,   dtype=np.int8)
    df["minute"] = np.array(minutes, dtype=np.int8)
    df["time_of_day_min"] = (df["hour"] * 60 + df["minute"]).astype(np.int16)

    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["min_sin"]   = np.sin(2 * np.pi * df["minute"] / 60).astype(np.float32)
    df["min_cos"]   = np.cos(2 * np.pi * df["minute"] / 60).astype(np.float32)
    df["tod_sin"]   = np.sin(2 * np.pi * df["time_of_day_min"] / 1440).astype(np.float32)
    df["tod_cos"]   = np.cos(2 * np.pi * df["time_of_day_min"] / 1440).astype(np.float32)
    day_max = max(int(df["day"].max()), 7)
    df["day_sin"]   = np.sin(2 * np.pi * df["day"] / day_max).astype(np.float32)
    df["day_cos"]   = np.cos(2 * np.pi * df["day"] / day_max).astype(np.float32)

    df["is_morning_rush"] = ((df["hour"] >= 7)  & (df["hour"] <= 9)).astype(np.int8)
    df["is_evening_rush"] = ((df["hour"] >= 17) & (df["hour"] <= 19)).astype(np.int8)
    df["is_rush_hour"]    = (df["is_morning_rush"] | df["is_evening_rush"]).astype(np.int8)
    df["is_lunch_hour"]   = ((df["hour"] >= 12) & (df["hour"] <= 13)).astype(np.int8)
    df["is_night"]        = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(np.int8)
    df["is_business"]     = ((df["hour"] >= 8)  & (df["hour"] <= 18)).astype(np.int8)
    df["is_off_peak"]     = (~df["is_rush_hour"].astype(bool)
                              & ~df["is_night"].astype(bool)).astype(np.int8)

    df["time_shift"] = pd.cut(df["hour"], bins=[-1, 5, 11, 16, 23],
                               labels=[0, 1, 2, 3]).astype(np.int8)
    df["time_block"] = (df["hour"] * 4 + df["minute"] // 15).astype(np.int8)
    df["hour_sq"]    = (df["hour"] ** 2).astype(np.float32)
    df["hour_cub"]   = (df["hour"] ** 3).astype(np.float32)
    return df

print("Timestamp features defined")

def add_geo_features(df, geo_cache):
    df = df.copy()
    df["lat"] = df["geohash"].map(lambda g: geo_cache[g][0])
    df["lon"] = df["geohash"].map(lambda g: geo_cache[g][1])
    df["geo_prefix3"] = df["geohash"].str[:3]
    df["geo_prefix4"] = df["geohash"].str[:4]
    df["geo_prefix5"] = df["geohash"].str[:5]

    sigma = 2.0
    prox_scores  = np.zeros(len(df))
    min_dists    = np.full(len(df), 9999.0)
    nearest_type = ["none"] * len(df)
    lats, lons   = df["lat"].values, df["lon"].values

    for lm in LANDMARKS:
        dists = np.array([haversine_km(lats[i], lons[i], lm["lat"], lm["lon"])
                          for i in range(len(df))])
        prox_scores += np.exp(-(dists**2) / (2 * sigma**2))
        mask = dists < min_dists
        min_dists    = np.where(mask, dists, min_dists)
        nearest_type = [lm["type"] if mask[i] else nearest_type[i]
                        for i in range(len(df))]

    df["landmark_prox_score"]  = prox_scores.astype(np.float32)
    df["nearest_landmark_km"]  = min_dists.astype(np.float32)
    df["nearest_landmark_type"] = nearest_type
    df["log_landmark_km"]       = np.log1p(min_dists).astype(np.float32)

    klcc_lat, klcc_lon = 3.1579, 101.7116
    dist_cc = np.array([haversine_km(lats[i], lons[i], klcc_lat, klcc_lon)
                         for i in range(len(df))], dtype=np.float32)
    df["dist_city_centre_km"]   = dist_cc
    df["log_dist_city_centre"]  = np.log1p(dist_cc).astype(np.float32)
    return df

print("Geospatial features defined")

def add_weather_features(df):
    df = df.copy()
    df["is_sunny"]    = (df["Weather"] == "Sunny").astype(np.int8)
    df["is_rainy"]    = (df["Weather"] == "Rainy").astype(np.int8)
    df["is_foggy"]    = (df["Weather"] == "Foggy").astype(np.int8)
    df["is_snowy"]    = (df["Weather"] == "Snowy").astype(np.int8)
    df["bad_weather"] = (df["is_rainy"] | df["is_foggy"] | df["is_snowy"]).astype(np.int8)
    df["is_hot"]      = (df["Temperature"] > 35).astype(np.int8)
    df["is_cold"]     = (df["Temperature"] < 15).astype(np.int8)
    df["temp_bin"]    = pd.cut(df["Temperature"],
                                bins=[-np.inf, 5, 15, 25, 35, np.inf],
                                labels=[0, 1, 2, 3, 4]).astype("float32")
    mu = df["Temperature"].mean(); sigma = df["Temperature"].std() + 1e-6
    df["temp_norm"] = ((df["Temperature"] - mu) / sigma).astype(np.float32)
    return df

def add_road_features(df):
    df = df.copy()
    road_order = {"Residential": 1, "Local": 2, "Collector": 3,
                  "Arterial": 3, "Highway": 4, "Expressway": 4, "Unknown": 2}
    df["road_capacity_score"]    = df["RoadType"].map(road_order).fillna(2).astype(np.int8)
    df["road_capacity_proxy"]    = (df["road_capacity_score"]
                                    * df["NumberofLanes"].fillna(1)).astype(np.float32)
    df["has_landmarks"]          = (df["Landmarks"] == "Yes").astype(np.int8)
    df["large_vehicles_allowed"] = (df["LargeVehicles"] == "Allowed").astype(np.int8)
    df["road_complexity"]        = (
        df["road_capacity_score"] * df["NumberofLanes"].fillna(1)
        * (1 + df["has_landmarks"]) * (1 + df["large_vehicles_allowed"])
    ).astype(np.float32)
    return df

print("Weather & road features defined")

def add_global_stats(df_train, df_test):
    df_train = df_train.copy(); df_test = df_test.copy()
    gm = df_train["demand"].mean()
    hour_mean  = df_train.groupby("hour")["demand"].mean().to_dict()
    day_mean   = df_train.groupby("day")["demand"].mean().to_dict()
    block_mean = df_train.groupby("time_block")["demand"].mean().to_dict()
    for df in [df_train, df_test]:
        df["hour_global_mean"]  = df["hour"].map(hour_mean).fillna(gm).astype(np.float32)
        df["day_global_mean"]   = df["day"].map(day_mean).fillna(gm).astype(np.float32)
        df["block_global_mean"] = df["time_block"].map(block_mean).fillna(gm).astype(np.float32)
    return df_train, df_test


def add_geohash_hour_stats(df_train, df_test):
    df_train = df_train.copy(); df_test = df_test.copy()
    gm = df_train["demand"].mean()
    stats = df_train.groupby(["geohash", "hour"])["demand"].agg(
        ["mean", "std", "count",
         lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    ).reset_index()
    stats.columns = ["geohash", "hour", "gh_hour_mean", "gh_hour_std",
                     "gh_hour_cnt", "gh_hour_q25", "gh_hour_q75"]
    df_train = df_train.merge(stats, on=["geohash", "hour"], how="left")
    df_test  = df_test.merge(stats,  on=["geohash", "hour"], how="left")
    for col, fill in [("gh_hour_mean", gm), ("gh_hour_std", 0),
                       ("gh_hour_cnt", 0),  ("gh_hour_q25", gm), ("gh_hour_q75", gm)]:
        df_train[col] = df_train[col].fillna(fill)
        df_test[col]  = df_test[col].fillna(fill)
    return df_train, df_test


def add_geohash_day_stats(df_train, df_test):
    df_train = df_train.copy(); df_test = df_test.copy()
    gm = df_train["demand"].mean()
    stats = df_train.groupby(["geohash", "day"])["demand"].agg(["mean", "std"]).reset_index()
    stats.columns = ["geohash", "day", "gh_day_mean", "gh_day_std"]
    df_train = df_train.merge(stats, on=["geohash", "day"], how="left")
    df_test  = df_test.merge(stats,  on=["geohash", "day"], how="left")
    df_train["gh_day_mean"] = df_train["gh_day_mean"].fillna(gm)
    df_test["gh_day_mean"]  = df_test["gh_day_mean"].fillna(gm)
    df_train["gh_day_std"]  = df_train["gh_day_std"].fillna(0)
    df_test["gh_day_std"]   = df_test["gh_day_std"].fillna(0)
    return df_train, df_test


def add_triple_stats(df_train, df_test):
    """Triple interaction: geohash x day x hour â€” most granular temporal-spatial feature."""
    df_train = df_train.copy(); df_test = df_test.copy()
    stats = df_train.groupby(["geohash", "day", "hour"])["demand"].agg(
        ["mean", "count"]
    ).reset_index()
    stats.columns = ["geohash", "day", "hour", "gh_day_hour_mean", "gh_day_hour_cnt"]
    df_train = df_train.merge(stats, on=["geohash", "day", "hour"], how="left")
    df_test  = df_test.merge(stats,  on=["geohash", "day", "hour"], how="left")
    df_train["gh_day_hour_mean"] = df_train["gh_day_hour_mean"].fillna(df_train["gh_hour_mean"])
    df_test["gh_day_hour_mean"]  = df_test["gh_day_hour_mean"].fillna(df_test["gh_hour_mean"])
    df_train["gh_day_hour_cnt"]  = df_train["gh_day_hour_cnt"].fillna(0)
    df_test["gh_day_hour_cnt"]   = df_test["gh_day_hour_cnt"].fillna(0)
    return df_train, df_test


def add_prefix_stats(df_train, df_test):
    """geo_prefix3/4 x hour and x day demand means."""
    df_train = df_train.copy(); df_test = df_test.copy()
    gm = df_train["demand"].mean()
    for pcol in ["geo_prefix3", "geo_prefix4"]:
        for tcol, feat in [("hour", "hour_mean"), ("day", "day_mean")]:
            s = df_train.groupby([pcol, tcol])["demand"].mean().reset_index()
            s.columns = [pcol, tcol, f"{pcol}_{feat}"]
            df_train = df_train.merge(s, on=[pcol, tcol], how="left")
            df_test  = df_test.merge(s,  on=[pcol, tcol], how="left")
            df_train[f"{pcol}_{feat}"] = df_train[f"{pcol}_{feat}"].fillna(gm)
            df_test[f"{pcol}_{feat}"]  = df_test[f"{pcol}_{feat}"].fillna(gm)
    return df_train, df_test


def add_roadtype_hour_stats(df_train, df_test):
    df_train = df_train.copy(); df_test = df_test.copy()
    gm = df_train["demand"].mean()
    s = df_train.groupby(["RoadType", "hour"])["demand"].mean().reset_index()
    s.columns = ["RoadType", "hour", "roadtype_hour_mean"]
    df_train = df_train.merge(s, on=["RoadType", "hour"], how="left")
    df_test  = df_test.merge(s,  on=["RoadType", "hour"], how="left")
    df_train["roadtype_hour_mean"] = df_train["roadtype_hour_mean"].fillna(gm)
    df_test["roadtype_hour_mean"]  = df_test["roadtype_hour_mean"].fillna(gm)
    return df_train, df_test


def add_weather_hour_stats(df_train, df_test):
    df_train = df_train.copy(); df_test = df_test.copy()
    gm = df_train["demand"].mean()
    s = df_train.groupby(["Weather", "hour"])["demand"].mean().reset_index()
    s.columns = ["Weather", "hour", "weather_hour_mean"]
    df_train = df_train.merge(s, on=["Weather", "hour"], how="left")
    df_test  = df_test.merge(s,  on=["Weather", "hour"], how="left")
    df_train["weather_hour_mean"] = df_train["weather_hour_mean"].fillna(gm)
    df_test["weather_hour_mean"]  = df_test["weather_hour_mean"].fillna(gm)
    return df_train, df_test

print("Statistical aggregation functions defined")

def add_lag_rolling_features(df_train, df_test):
    """
    Lag and rolling demand features per geohash.
    NOTE: rows are sorted by (geohash, day, hour, minute) here.
    The INDEX FIX in full_pipeline() restores the original test.csv row order.
    """
    df_train = df_train.copy(); df_test = df_test.copy()
    df_test["demand"] = np.nan

    combined = pd.concat([df_train, df_test], ignore_index=True)
    combined["_sk"] = combined["day"] * 10000 + combined["hour"] * 100 + combined["minute"]
    combined = combined.sort_values(["geohash", "_sk"]).reset_index(drop=True)
    grp = combined.groupby("geohash")["demand"]

    for lag in [1, 2, 3, 6, 12, 24, 48]:
        combined[f"demand_lag_{lag}"] = grp.shift(lag)

    for w in [3, 6, 12, 24]:
        combined[f"demand_roll_mean_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
    for w in [3, 6, 24]:
        combined[f"demand_roll_std_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0)
        )
    combined["demand_roll_median_6"] = grp.transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).median()
    )
    combined["demand_roll_max_6"]  = grp.transform(lambda x: x.shift(1).rolling(6,  min_periods=1).max())
    combined["demand_roll_min_6"]  = grp.transform(lambda x: x.shift(1).rolling(6,  min_periods=1).min())
    combined["demand_roll_max_12"] = grp.transform(lambda x: x.shift(1).rolling(12, min_periods=1).max())
    combined["demand_cv_6"] = (
        combined["demand_roll_std_6"] / (combined["demand_roll_mean_6"].abs() + 1e-6)
    ).astype(np.float32)
    combined["demand_expanding_mean"] = grp.transform(
        lambda x: x.shift(1).expanding().mean()
    )
    combined.drop(columns=["_sk"], inplace=True)

    lag_cols = [c for c in combined.columns
                if any(t in c for t in ["lag", "roll", "expanding", "_cv_"])]

    is_test = combined["demand"].isna()
    df_tr_out = combined[~is_test].copy()
    df_te_out = combined[is_test].drop(columns=["demand"]).copy()

    for col in lag_cols:
        med = df_tr_out[col].median()
        df_tr_out[col] = df_tr_out[col].fillna(med)
        df_te_out[col] = df_te_out[col].fillna(med)

    return df_tr_out, df_te_out

print("Lag/rolling feature functions defined")

def add_frequency_encoding(df_train, df_test, col):
    freq = df_train[col].value_counts(normalize=True)
    df_train = df_train.copy(); df_test = df_test.copy()
    df_train[f"{col}_freq"] = df_train[col].map(freq).fillna(0).astype(np.float32)
    df_test[f"{col}_freq"]  = df_test[col].map(freq).fillna(0).astype(np.float32)
    return df_train, df_test


def add_oof_target_encoding(df_train, df_test, col, n_splits=5, smoothing=10, seed=42):
    df_train = df_train.copy(); df_test = df_test.copy()
    global_mean = df_train[TARGET].mean()
    new_col = f"{col}_target_enc"
    df_train[new_col] = np.nan

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(df_train):
        fold_tr = df_train.iloc[tr_idx]
        stats = fold_tr.groupby(col)[TARGET].agg(["mean", "count"])
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        df_train.loc[df_train.index[val_idx], new_col] = (
            df_train.iloc[val_idx][col].map(smoothed).fillna(global_mean).values
        )
    stats = df_train.groupby(col)[TARGET].agg(["mean", "count"])
    smoothed_full = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
    df_test[new_col]  = df_test[col].map(smoothed_full).fillna(global_mean).astype(np.float32)
    df_train[new_col] = df_train[new_col].astype(np.float32)
    return df_train, df_test


def add_interaction_features(df):
    df = df.copy()
    df["rain_x_rush"]     = (df["is_rainy"]    * df["is_rush_hour"]).astype(np.int8)
    df["fog_x_rush"]      = (df["is_foggy"]    * df["is_rush_hour"]).astype(np.int8)
    df["bad_wx_x_rush"]   = (df["bad_weather"] * df["is_rush_hour"]).astype(np.int8)
    df["bad_wx_x_night"]  = (df["bad_weather"] * df["is_night"]).astype(np.int8)
    temp_med = float(df["Temperature"].median())
    df["temp_x_hour"]     = (df["Temperature"].fillna(temp_med) * df["hour"]).astype(np.float32)
    df["temp_x_rush"]     = (df["Temperature"].fillna(temp_med) * df["is_rush_hour"]).astype(np.float32)
    df["capacity_x_rush"] = (df["road_capacity_proxy"] * df["is_rush_hour"]).astype(np.float32)
    df["lanes_x_rush"]    = (df["NumberofLanes"].fillna(1) * df["is_rush_hour"]).astype(np.float32)
    df["capacity_x_hour"] = (df["road_capacity_proxy"] * df["hour"]).astype(np.float32)
    df["prox_x_rush"]     = (df["landmark_prox_score"] * df["is_rush_hour"]).astype(np.float32)
    df["prox_x_hour"]     = (df["landmark_prox_score"] * df["hour"]).astype(np.float32)
    df["gh_hour_vs_global"] = (df["gh_hour_mean"] / (df["hour_global_mean"] + 1e-6)).astype(np.float32)
    df["gh_day_vs_global"]  = (df["gh_day_mean"]  / (df["day_global_mean"]  + 1e-6)).astype(np.float32)
    return df


def label_encode_objects(df_train, df_test, drop_cols):
    df_train = df_train.copy(); df_test = df_test.copy()
    for col in df_train.select_dtypes("object").columns:
        if col in drop_cols:
            continue
        le = LabelEncoder()
        le.fit(pd.concat([df_train[col], df_test[col]], axis=0).astype(str))
        df_train[col] = le.transform(df_train[col].astype(str))
        df_test[col]  = le.transform(df_test[col].astype(str))
    return df_train, df_test

print("Encoding & interaction functions defined")

def preprocess(df):
    df = df.copy()
    for col in ["RoadType", "LargeVehicles", "Landmarks", "Weather"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    return df


def full_pipeline(df_train_raw, df_test_raw):
    """
    Complete feature engineering pipeline.

    CRITICAL INDEX FIX at end:
      add_lag_rolling_features() sorts test rows by (geohash, time),
      so test_fe row order != test.csv order.
      We restore alignment: te.sort_values("Index") -> X_test row i = test.csv row i.
    """
    t0 = time.time()
    print("[1/11] Preprocessing...")
    tr = preprocess(df_train_raw)
    te = preprocess(df_test_raw)
    temp_med = tr["Temperature"].median()
    tr["Temperature"] = tr["Temperature"].fillna(temp_med)
    te["Temperature"] = te["Temperature"].fillna(temp_med)

    print("[2/11] Geo cache...")
    geo_cache = build_geo_cache(tr, te)

    print("[3/11] Time features...")
    tr = add_time_features(tr); te = add_time_features(te)

    print("[4/11] Geo features...")
    tr = add_geo_features(tr, geo_cache); te = add_geo_features(te, geo_cache)

    print("[5/11] Weather & road features...")
    tr = add_weather_features(tr); tr = add_road_features(tr)
    te = add_weather_features(te); te = add_road_features(te)

    print("[6/11] Global stats...")
    tr, te = add_global_stats(tr, te)

    print("[7/11] Geohash x hour and x day stats...")
    tr, te = add_geohash_hour_stats(tr, te)
    tr, te = add_geohash_day_stats(tr, te)

    print("[8/11] Triple interaction (geohash x day x hour)...")
    tr, te = add_triple_stats(tr, te)

    print("[9/11] Prefix, roadtype x hour, weather x hour...")
    tr, te = add_prefix_stats(tr, te)
    tr, te = add_roadtype_hour_stats(tr, te)
    tr, te = add_weather_hour_stats(tr, te)

    print("[10/11] Lag & rolling features...")
    tr, te = add_lag_rolling_features(tr, te)

    print("[11/11] Encodings, interactions, label encoding...")
    enc_cols = ["geohash", "geo_prefix3", "geo_prefix4", "geo_prefix5",
                "RoadType", "Weather", "nearest_landmark_type"]
    for col in enc_cols:
        tr, te = add_oof_target_encoding(tr, te, col)
        tr, te = add_frequency_encoding(tr, te, col)

    tr = add_interaction_features(tr)
    te = add_interaction_features(te)

    DROP_FOR_ENCODE = ["Index", TARGET, "timestamp", "geohash",
                       "geo_prefix3", "geo_prefix4", "geo_prefix5",
                       "RoadType", "Weather", "nearest_landmark_type",
                       "LargeVehicles", "Landmarks"]
    tr, te = label_encode_objects(tr, te, DROP_FOR_ENCODE)

    # â”€â”€ INDEX ALIGNMENT FIX â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Restores original test.csv row order so X_test[i] == test.csv row i
    te = te.sort_values("Index").reset_index(drop=True)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    print(f"\n  Train: {tr.shape}   Test: {te.shape}   ({time.time()-t0:.1f}s)")
    return tr, te, geo_cache

print("Full pipeline function defined")

train_fe, test_fe, geo_cache = full_pipeline(train_raw, test_raw)

DROP_COLS = ["Index", TARGET, "timestamp", "geohash",
             "geo_prefix3", "geo_prefix4", "geo_prefix5",
             "RoadType", "Weather", "nearest_landmark_type",
             "LargeVehicles", "Landmarks"]

FEATURE_COLS = [
    c for c in train_fe.columns
    if c not in DROP_COLS
    and c != TARGET
    and train_fe[c].dtype != "object"
]

X      = train_fe[FEATURE_COLS].values.astype(np.float32)
y      = train_fe[TARGET].values.astype(np.float32)
# test_fe is already sorted by Index (fixed in full_pipeline)
X_test = test_fe[FEATURE_COLS].values.astype(np.float32)

print(f"Features  : {len(FEATURE_COLS)}")
print(f"X         : {X.shape}")
print(f"y         : {y.shape}  min={y.min():.4f}  max={y.max():.4f}")
print(f"X_test    : {X_test.shape}")

seq_ok = list(test_fe["Index"]) == list(range(len(test_raw)))
print(f"\nIndex alignment fix active: {seq_ok}")

nan_x  = int(np.isnan(X).sum())
nan_xt = int(np.isnan(X_test).sum())
print(f"NaN in X      : {nan_x}  ({'OK' if nan_x==0 else 'WARNING'})")
print(f"NaN in X_test : {nan_xt}  ({'OK' if nan_xt==0 else 'WARNING'})")

print(f"\nFeature list ({len(FEATURE_COLS)} features):")
for i, f in enumerate(FEATURE_COLS):
    print(f"  {i+1:3d}. {f}")

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

def oof_r2(model_fn, X, y, X_test, kf):
    """
    N-fold OOF helper.
    model_fn(X_tr, y_tr, X_val, y_val) -> (val_preds, test_preds, model)
    Returns: oof_preds, test_preds_avg, fold_scores, models
    """
    oof   = np.zeros(len(y), dtype=np.float64)
    tests = np.zeros((N_SPLITS, len(X_test)), dtype=np.float64)
    scores, models = [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        val_p, test_p, mdl = model_fn(X_tr, y_tr, X_val, y_val)
        oof[val_idx] = val_p
        tests[fold-1] = test_p
        s = r2_score(y_val, val_p)
        scores.append(s)
        models.append(mdl)
        print(f"  Fold {fold}  R2={s:.5f}")

    print(f"  OOF R2={r2_score(y, oof):.5f}  mean-fold={np.mean(scores):.5f}")
    return oof, tests.mean(axis=0), scores, models

print(f"KFold: {N_SPLITS} splits, shuffle=True, seed={SEED}")

N_TRIALS_LGB = 50

def lgb_objective(trial):
    params = {
        "n_estimators"    : trial.suggest_int("n_estimators", 300, 2000),
        "learning_rate"   : trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves"      : trial.suggest_int("num_leaves", 31, 512),
        "max_depth"       : trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 150),
        "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain"  : trial.suggest_float("min_split_gain", 0.0, 1.0),
        "objective"       : "regression", "metric": "rmse",
        "verbosity"       : -1, "subsample_freq": 1,
        "seed": SEED, "n_jobs": -1,
    }
    scores = []
    for tr_idx, val_idx in kf.split(X):
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr_idx], y[tr_idx],
              eval_set=[(X[val_idx], y[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        scores.append(r2_score(y[val_idx], m.predict(X[val_idx])))
    return np.mean(scores)

print(f"Optimising LightGBM ({N_TRIALS_LGB} trials)...")
lgb_study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
lgb_study.optimize(lgb_objective, n_trials=N_TRIALS_LGB, show_progress_bar=True)
best_lgb_params = lgb_study.best_params
best_lgb_params.update({"objective": "regression", "metric": "rmse",
                         "verbosity": -1, "subsample_freq": 1,
                         "seed": SEED, "n_jobs": -1})
print(f"\nBest LGB R2={lgb_study.best_value:.5f}")
print(best_lgb_params)

N_TRIALS_CB = 50

def cb_objective(trial):
    params = {
        "iterations"        : trial.suggest_int("iterations", 300, 2500),
        "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth"             : trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg"       : trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength"   : trial.suggest_float("random_strength", 0.5, 5.0),
        "min_data_in_leaf"  : trial.suggest_int("min_data_in_leaf", 5, 50),
        "border_count"      : trial.suggest_categorical("border_count", [32, 64, 128, 254]),
        "od_type": "Iter", "od_wait": 50,
        "loss_function": "RMSE", "eval_metric": "R2",
        "random_seed": SEED, "verbose": False,
    }
    scores = []
    for tr_idx, val_idx in kf.split(X):
        m = CatBoostRegressor(**params)
        m.fit(X[tr_idx], y[tr_idx], eval_set=Pool(X[val_idx], y[val_idx]), verbose=False)
        scores.append(r2_score(y[val_idx], m.predict(X[val_idx])))
    return np.mean(scores)

print(f"Optimising CatBoost ({N_TRIALS_CB} trials)...")
cb_study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
cb_study.optimize(cb_objective, n_trials=N_TRIALS_CB, show_progress_bar=True)
best_cb_params = cb_study.best_params
best_cb_params.update({"od_type": "Iter", "od_wait": 50,
                        "loss_function": "RMSE", "eval_metric": "R2",
                        "random_seed": SEED, "verbose": False})
print(f"\nBest CB R2={cb_study.best_value:.5f}")
print(best_cb_params)

N_TRIALS_XGB = 40

def xgb_objective(trial):
    params = {
        "n_estimators"    : trial.suggest_int("n_estimators", 300, 1500),
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth"       : trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma"           : trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "objective"       : "reg:squarederror",
        "tree_method"     : "hist", "seed": SEED,
        "n_jobs": -1, "verbosity": 0,
    }
    scores = []
    for tr_idx, val_idx in kf.split(X):
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr_idx], y[tr_idx],
              eval_set=[(X[val_idx], y[val_idx])],
              early_stopping_rounds=50, verbose=False)
        scores.append(r2_score(y[val_idx], m.predict(X[val_idx])))
    return np.mean(scores)

print(f"Optimising XGBoost ({N_TRIALS_XGB} trials)...")
xgb_study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS_XGB, show_progress_bar=True)
best_xgb_params = xgb_study.best_params
best_xgb_params.update({"objective": "reg:squarederror", "tree_method": "hist",
                         "seed": SEED, "n_jobs": -1, "verbosity": 0})
print(f"\nBest XGB R2={xgb_study.best_value:.5f}")
print(best_xgb_params)

print("--- LightGBM OOF ---")
def lgb_fn(X_tr, y_tr, X_val, y_val):
    m = lgb.LGBMRegressor(**best_lgb_params)
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return m.predict(X_val), m.predict(X_test), m

lgb_oof, lgb_test, lgb_scores, lgb_models = oof_r2(lgb_fn, X, y, X_test, kf)

print("--- CatBoost OOF ---")
def cb_fn(X_tr, y_tr, X_val, y_val):
    m = CatBoostRegressor(**best_cb_params)
    m.fit(X_tr, y_tr, eval_set=Pool(X_val, y_val), verbose=False)
    return m.predict(X_val), m.predict(X_test), m

cb_oof, cb_test, cb_scores, cb_models = oof_r2(cb_fn, X, y, X_test, kf)

print("--- XGBoost OOF ---")
def xgb_fn(X_tr, y_tr, X_val, y_val):
    m = xgb.XGBRegressor(**best_xgb_params)
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=50, verbose=False)
    return m.predict(X_val), m.predict(X_test), m

xgb_oof, xgb_test, xgb_scores, xgb_models = oof_r2(xgb_fn, X, y, X_test, kf)

# â”€â”€ 1. Grid-search optimal weights â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
best_r2_w, best_w = -np.inf, (1/3, 1/3, 1/3)
for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        w3 = round(1.0 - w1 - w2, 6)
        if w3 < 0: continue
        s = r2_score(y, w1*lgb_oof + w2*cb_oof + w3*xgb_oof)
        if s > best_r2_w:
            best_r2_w, best_w = s, (w1, w2, w3)

w_lgb, w_cb, w_xgb = best_w
print(f"Optimal weights  LGB={w_lgb:.2f}  CB={w_cb:.2f}  XGB={w_xgb:.2f}")
print(f"Weighted-avg OOF R2={best_r2_w:.5f}")
weighted_oof  = w_lgb*lgb_oof  + w_cb*cb_oof  + w_xgb*xgb_oof
weighted_test = w_lgb*lgb_test + w_cb*cb_test + w_xgb*xgb_test

# â”€â”€ 2. Ridge meta-learner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
S_oof  = np.column_stack([lgb_oof,  cb_oof,  xgb_oof])
S_test = np.column_stack([lgb_test, cb_test, xgb_test])
ridge_oof_parts, ridge_test_parts, ridge_scores = [], [], []
for tr_idx, val_idx in kf.split(X):
    meta = Ridge(alpha=1.0)
    meta.fit(S_oof[tr_idx], y[tr_idx])
    ridge_oof_parts.append((val_idx, meta.predict(S_oof[val_idx])))
    ridge_test_parts.append(meta.predict(S_test))
    ridge_scores.append(r2_score(y[val_idx], ridge_oof_parts[-1][1]))
ridge_oof = np.zeros(len(y))
for idx, p in ridge_oof_parts: ridge_oof[idx] = p
ridge_test = np.array(ridge_test_parts).mean(axis=0)
print(f"Ridge stacking OOF R2={r2_score(y, ridge_oof):.5f}")

# â”€â”€ 3. LGB meta-learner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
meta_lgb_oof_parts, meta_lgb_test_parts, meta_lgb_scores = [], [], []
for tr_idx, val_idx in kf.split(X):
    meta = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                               random_state=SEED, verbosity=-1, n_jobs=-1)
    meta.fit(S_oof[tr_idx], y[tr_idx])
    meta_lgb_oof_parts.append((val_idx, meta.predict(S_oof[val_idx])))
    meta_lgb_test_parts.append(meta.predict(S_test))
    meta_lgb_scores.append(r2_score(y[val_idx], meta_lgb_oof_parts[-1][1]))
meta_lgb_oof = np.zeros(len(y))
for idx, p in meta_lgb_oof_parts: meta_lgb_oof[idx] = p
meta_lgb_test = np.array(meta_lgb_test_parts).mean(axis=0)
print(f"LGB-meta stacking OOF R2={r2_score(y, meta_lgb_oof):.5f}")

model_scores = {
    "Weighted-Avg"   : r2_score(y, weighted_oof),
    "Ridge-Stack"    : r2_score(y, ridge_oof),
    "LGB-Meta-Stack" : r2_score(y, meta_lgb_oof),
    "CatBoost"       : r2_score(y, cb_oof),
    "LightGBM"       : r2_score(y, lgb_oof),
    "XGBoost"        : r2_score(y, xgb_oof),
}
print("\n" + "="*55)
print("  OOF R2 Leaderboard")
print("="*55)
for name, score in sorted(model_scores.items(), key=lambda x: -x[1]):
    bar = "#" * int(score * 40)
    print(f"  {name:20s}  {score:.5f}  |{bar}")

best_model_name = max(model_scores, key=model_scores.get)
print(f"\n  Best: {best_model_name}  OOF R2={model_scores[best_model_name]:.5f}")
print(f"  Expected HackerEarth score: {model_scores[best_model_name]*100:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

imp_lgb = pd.Series(
    lgb_models[0].feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False).head(30)
axes[0].barh(imp_lgb.index[::-1], imp_lgb.values[::-1], color="steelblue", alpha=0.8)
axes[0].set_title("LightGBM â€” Top 30 Feature Importances", fontsize=13)
axes[0].set_xlabel("Importance (split count)")

imp_cb = pd.Series(
    cb_models[0].get_feature_importance(), index=FEATURE_COLS
).sort_values(ascending=False).head(30)
axes[1].barh(imp_cb.index[::-1], imp_cb.values[::-1], color="darkorange", alpha=0.8)
axes[1].set_title("CatBoost â€” Top 30 Feature Importances", fontsize=13)
axes[1].set_xlabel("Importance score")

plt.suptitle("Feature Importance Analysis", fontsize=15)
plt.tight_layout()
plt.savefig(str(SUB_DIR / "feature_importance.png"), dpi=80, bbox_inches="tight")
plt.show()
print("Feature importance saved")

if SHAP_AVAIL:
    print("Computing SHAP values (3000-row sample)...")
    sample_idx = np.random.choice(len(X), min(3000, len(X)), replace=False)
    X_sample   = X[sample_idx]
    explainer  = shap.TreeExplainer(lgb_models[0])
    shap_vals  = explainer.shap_values(X_sample)

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_vals, X_sample, feature_names=FEATURE_COLS,
                      max_display=25, show=False)
    plt.title("SHAP Summary â€” LightGBM (Top 25)", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(SUB_DIR / "shap_summary.png"), dpi=80, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals, X_sample, feature_names=FEATURE_COLS,
                      plot_type="bar", max_display=25, show=False)
    plt.title("SHAP Mean |value|", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(SUB_DIR / "shap_bar.png"), dpi=80, bbox_inches="tight")
    plt.show()
    print("SHAP plots saved")
else:
    print("SHAP not installed â€” run: pip install shap")

# â”€â”€ LightGBM full retrain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
full_lgb_params = best_lgb_params.copy()
fold_iters = [getattr(m, "best_iteration_", best_lgb_params["n_estimators"])
              for m in lgb_models]
full_lgb_params["n_estimators"] = max(int(np.mean(fold_iters) * 1.1),
                                       best_lgb_params["n_estimators"])
lgb_full = lgb.LGBMRegressor(**full_lgb_params)
lgb_full.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
lgb_full_test = lgb_full.predict(X_test)
print(f"LGB full: {lgb_full_test[:5]}")

# â”€â”€ CatBoost full retrain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
full_cb_params = {k: v for k, v in best_cb_params.items()
                  if k not in ("od_type", "od_wait")}
fold_iters_cb = [getattr(m, "best_iteration_", best_cb_params["iterations"])
                 for m in cb_models]
full_cb_params["iterations"] = max(int(np.mean(fold_iters_cb) * 1.1),
                                    best_cb_params["iterations"])
cb_full = CatBoostRegressor(**full_cb_params)
cb_full.fit(X, y, verbose=False)
cb_full_test = cb_full.predict(X_test)
print(f"CB  full: {cb_full_test[:5]}")

# â”€â”€ XGBoost full retrain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
xgb_full = xgb.XGBRegressor(**best_xgb_params)
xgb_full.fit(X, y, verbose=False)
xgb_full_test = xgb_full.predict(X_test)
print(f"XGB full: {xgb_full_test[:5]}")

PASS_S = "PASS"; FAIL_S = "FAIL"; WARN_S = "WARN"
print("=" * 55)
print("  DEBUG STEP 1 â€” Feature Matrix Diagnostics")
print("=" * 55)
shape_ok = X.shape[1] == X_test.shape[1]
print(f"  Feature count match : {PASS_S if shape_ok else FAIL_S}  ({X.shape[1]})")
nan_train = int(np.isnan(X).sum())
nan_test  = int(np.isnan(X_test).sum())
print(f"  NaN in X      : {nan_train}  {PASS_S if nan_train==0 else WARN_S}")
print(f"  NaN in X_test : {nan_test}   {PASS_S if nan_test==0 else FAIL_S}")
if nan_train > 0:
    npc = np.isnan(X).sum(axis=0)
    bad = [(FEATURE_COLS[i], int(npc[i])) for i in np.where(npc>0)[0]]
    for c, n in sorted(bad, key=lambda x:-x[1])[:10]:
        print(f"    {c:40s}: {n}")
inf_x  = int(np.isinf(X).sum())
inf_xt = int(np.isinf(X_test).sum())
print(f"  Inf in X      : {inf_x}   {PASS_S if inf_x==0 else FAIL_S}")
print(f"  Inf in X_test : {inf_xt}   {PASS_S if inf_xt==0 else FAIL_S}")
print(f"  y range       : [{y.min():.4f}, {y.max():.4f}]  "
      f"{PASS_S if y.min()>=0 and y.max()<=1 else FAIL_S}")
print("=" * 55)

print("=" * 55)
print("  DEBUG STEP 2 â€” Index Alignment Audit")
print("=" * 55)
fe_idx  = test_fe["Index"].values
raw_idx = test_raw["Index"].values
order_ok = np.array_equal(fe_idx, raw_idx)
seq_ok   = list(fe_idx) == list(range(len(test_raw)))
print(f"  Row count match   : {PASS_S if len(fe_idx)==len(raw_idx) else FAIL_S}")
print(f"  Order preserved   : {PASS_S if order_ok else FAIL_S}")
print(f"  Index sequential  : {PASS_S if seq_ok else FAIL_S}")
if not order_ok:
    n_mis = (fe_idx != raw_idx).sum()
    print(f"  *** {n_mis} rows in WRONG position ***")
else:
    print("  Row-to-prediction mapping is CORRECT.")
print("=" * 55)

final_preds = (
    w_lgb * lgb_full_test +
    w_cb  * cb_full_test  +
    w_xgb * xgb_full_test
)
final_preds = np.clip(final_preds, 0.0, 1.0)

print("=" * 55)
print("  DEBUG STEP 3 â€” Prediction Diagnostics")
print("=" * 55)
for name, preds in [("lgb_full", lgb_full_test), ("cb_full", cb_full_test),
                     ("xgb_full", xgb_full_test), ("ENSEMBLE", final_preds)]:
    nan_c = int(np.isnan(preds).sum())
    inf_c = int(np.isinf(preds).sum())
    ok = (nan_c == 0 and inf_c == 0)
    print(f"  [{name}]  NaN={nan_c}  Inf={inf_c}  "
          f"<0={(preds<0).sum()}  >1={(preds>1).sum()}  "
          f"{PASS_S if ok else FAIL_S}")
    print(f"    min={preds.min():.5f}  max={preds.max():.5f}  "
          f"mean={preds.mean():.5f}  std={preds.std():.5f}")
ks, p = scipy_stats.ks_2samp(y, final_preds)
print(f"\n  KS test vs train: stat={ks:.4f}  p={p:.4f}  "
      f"({PASS_S if p>0.01 else WARN_S})")
print("=" * 55)

submission = pd.DataFrame({
    "Index" : test_fe["Index"].values,   # aligned with X_test row-by-row
    "demand": final_preds
}).sort_values("Index").reset_index(drop=True)

# Hard assertions
assert len(submission) == len(test_raw),          f"Row mismatch: {len(submission)}"
assert list(submission["Index"]) == list(range(len(test_raw))), "Index not sequential!"
assert submission["demand"].isna().sum() == 0,    "NaN in predictions!"
assert np.isinf(submission["demand"].values).sum() == 0, "Inf in predictions!"
assert submission["demand"].min() >= 0.0,         "Negative predictions!"
assert submission["demand"].max() <= 1.0,         "Predictions > 1!"

out_path = SUB_DIR / "ultimate_submission.csv"
submission.to_csv(out_path, index=False)

print("=" * 55)
print("  FINAL SUBMISSION GENERATED")
print("=" * 55)
print(f"  File  : {out_path}")
print(f"  Shape : {submission.shape}")
print(f"  min   : {submission['demand'].min():.6f}")
print(f"  max   : {submission['demand'].max():.6f}")
print(f"  mean  : {submission['demand'].mean():.6f}")
print(f"  std   : {submission['demand'].std():.6f}")
print(f"\n  First 5 rows:")
print(submission.head(5).to_string(index=False))
print(f"\n  Upload  submissions/ultimate_submission.csv  to HackerEarth")
print("=" * 55)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, study, name in [(axes[0], lgb_study, "LightGBM"),
                         (axes[1], cb_study,  "CatBoost"),
                         (axes[2], xgb_study, "XGBoost")]:
    vals = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.maximum.accumulate(vals)
    ax.plot(vals, alpha=0.4, color="steelblue", label="trial")
    ax.plot(best_so_far, color="red", linewidth=2, label="best")
    ax.set_title(f"{name} Optuna History", fontsize=12)
    ax.set_xlabel("Trial"); ax.set_ylabel("OOF R2")
    ax.legend()
plt.suptitle("Hyperparameter Optimisation Progress", fontsize=14)
plt.tight_layout()
plt.savefig(str(SUB_DIR / "optuna_history.png"), dpi=80, bbox_inches="tight")
plt.show()

results = pd.DataFrame([
    {"Model": "Baseline (prev)",          "OOF_R2": None,    "LB_Score": 88.237},
    {"Model": "LightGBM (Optuna)",         "OOF_R2": r2_score(y, lgb_oof),       "LB_Score": None},
    {"Model": "CatBoost (Optuna)",         "OOF_R2": r2_score(y, cb_oof),        "LB_Score": None},
    {"Model": "XGBoost (Optuna)",          "OOF_R2": r2_score(y, xgb_oof),       "LB_Score": None},
    {"Model": "Weighted Ensemble",         "OOF_R2": r2_score(y, weighted_oof),  "LB_Score": None},
    {"Model": "Ridge Stacking",            "OOF_R2": r2_score(y, ridge_oof),     "LB_Score": None},
    {"Model": "LGB Meta Stacking",         "OOF_R2": r2_score(y, meta_lgb_oof),  "LB_Score": None},
    {"Model": "ULTIMATE (this notebook)", "OOF_R2": model_scores[best_model_name], "LB_Score": None},
])
results["OOF_R2"]   = results["OOF_R2"].map(lambda x: f"{x:.5f}" if x is not None else "-")
results["LB_Score"] = results["LB_Score"].map(lambda x: f"{x:.3f}" if x is not None else "pending")
results["HE_Score"] = results["OOF_R2"].map(lambda x: f"{float(x)*100:.2f}" if x != "-" else "-")

print("\n" + "="*65)
print("  LEADERBOARD TRACKER â€” fill LB_Score after submission")
print("="*65)
print(results.to_string(index=False))
print("="*65)
print(f"\n  Best OOF R2 : {max(model_scores.values()):.5f}")
print(f"  Expected HE : {max(model_scores.values())*100:.3f}")
print(f"\n  TIP: if OOF >> LB, increase regularization (reg_alpha, l2_leaf_reg)")

for name, preds in [("lgb_ultimate",  lgb_full_test),
                     ("cb_ultimate",   cb_full_test),
                     ("xgb_ultimate",  xgb_full_test)]:
    df = pd.DataFrame({
        "Index" : test_fe["Index"].values,
        "demand": np.clip(preds, 0.0, 1.0)
    }).sort_values("Index").reset_index(drop=True)
    df.to_csv(SUB_DIR / f"{name}_submission.csv", index=False)
    print(f"Saved: {name}_submission.csv")

print("\nAll submissions saved.")
print("Upload order: 1) ultimate_submission  2) weighted_ensemble  3) cb_ultimate")