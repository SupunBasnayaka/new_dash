import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "paddy_data.db")
MODEL_PATH = os.path.join(BASE_DIR, "paddy_ml_model.pkl")

IRR_TYPES = ["Major", "Minor", "Rainfed"]

# Inputs the model learns from (production is the target).
CAT_FEATURES = ["district", "province", "season", "irrigation_type"]
NUM_FEATURES = ["sown_extent_ha", "major_share", "minor_share", "rainfed_share"]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "production_mt"

# For the importance chart: columns that describe the same thing are added together.
FEATURE_GROUPS = {
    "district": ["district"],
    "province": ["province"],
    "season": ["season"],
    "sown_extent_ha": ["sown_extent_ha"],
    "irrigation": ["irrigation_type", "major_share", "minor_share", "rainfed_share"],
}

N_ESTIMATORS = 200
TEST_YEARS = 5          # newest complete years held out to measure accuracy honestly


def irrigation_features(major_ha, minor_ha, rainfed_ha):
    """Irrigation inputs -> model columns: total extent, the share of each irrigation type and
    the dominant type. Used both for training rows and for a user's prediction request."""
    ha = {"Major": float(major_ha), "Minor": float(minor_ha), "Rainfed": float(rainfed_ha)}
    total = sum(ha.values())
    if total <= 0:
        raise ValueError("Sown extent must be greater than zero.")
    return {
        "sown_extent_ha": total,
        "major_share": ha["Major"] / total,
        "minor_share": ha["Minor"] / total,
        "rainfed_share": ha["Rainfed"] / total,
        "irrigation_type": max(ha, key=ha.get),
    }


def load_training_data():
    """District-level rows with published production, plus the irrigation mix of that
    district-season.

    Production is only published for the all-irrigation ('Total') row, so that stays the target.
    The Major / Minor / Rainfed sown extents of the same district-season become the irrigation
    features. Rows are kept only when those three extents add up to the published total sown
    extent (a blank then means "no area of that type"); older rows with no irrigation split
    are dropped, never filled. National / scheme rows are excluded.
    """
    conn = sqlite3.connect(DB_PATH)
    tot = pd.read_sql(
        """SELECT district, province, season, year, sown_extent_ha, production_mt
           FROM paddy_stats
           WHERE area_type = 'district' AND irrigation_type = 'Total'
             AND sown_extent_ha IS NOT NULL AND production_mt IS NOT NULL
             AND province IS NOT NULL""",
        conn,
    )
    irr = pd.read_sql(
        """SELECT district, season, year, irrigation_type, sown_extent_ha AS ha
           FROM paddy_stats
           WHERE area_type = 'district' AND irrigation_type != 'Total'""",
        conn,
    )
    conn.close()

    wide = (irr.pivot_table(index=["district", "season", "year"], columns="irrigation_type",
                            values="ha", aggfunc="sum")
            .reindex(columns=IRR_TYPES).reset_index())
    df = tot.merge(wide, on=["district", "season", "year"], how="inner")

    split_sum = df[IRR_TYPES].sum(axis=1, skipna=True)
    df = df[(split_sum > 0) & ((split_sum / df["sown_extent_ha"] - 1).abs() < 0.02)].copy()
    df[IRR_TYPES] = df[IRR_TYPES].fillna(0.0)
    df["major_share"] = df["Major"] / split_sum[df.index]
    df["minor_share"] = df["Minor"] / split_sum[df.index]
    df["rainfed_share"] = df["Rainfed"] / split_sum[df.index]
    df["irrigation_type"] = df[IRR_TYPES].idxmax(axis=1)
    # "year" is kept in the table only to split train / test by time; it is NOT a model input.
    return df[["year"] + FEATURES + [TARGET]].reset_index(drop=True)


def _build_pipeline():
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", pre),
        ("regressor", RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1)),
    ])


def _grouped_importance(pipe):
    """Random-forest importance summed back onto the original inputs
    (one-hot columns of 'district' are added together, the irrigation columns are combined, ...)."""
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    raw = pd.Series(pipe.named_steps["regressor"].feature_importances_, index=names)
    stripped = [n.split("__", 1)[1] for n in names]

    def belongs(col, name):
        return name == col or (col in CAT_FEATURES and name.startswith(col + "_"))

    out = {}
    for group, cols in FEATURE_GROUPS.items():
        mask = [any(belongs(c, n) for c in cols) for n in stripped]
        out[group] = float(raw[mask].sum())
    return out


def _signature(df):
    return {"rows": int(len(df)), "max_year": int(df["year"].max()),
            "prod_sum": round(float(df[TARGET].sum()), 1), "features": list(FEATURES)}


def train_paddy_model():
    df = load_training_data()
    if df.empty:
        raise RuntimeError("No usable rows in paddy_data.db (need district rows with sown extent, "
                           "irrigation split and production).")

    # Honest accuracy: train on older years, test on the newest years.
    cutoff = df["year"].max() - TEST_YEARS
    train, test = df[df["year"] <= cutoff], df[df["year"] > cutoff]
    probe = _build_pipeline().fit(train[FEATURES], train[TARGET])
    pred = probe.predict(test[FEATURES])

    metrics = {
        "r2_test": float(r2_score(test[TARGET], pred)),
        "mae_test_mt": float(mean_absolute_error(test[TARGET], pred)),
        "test_years": f"{int(test['year'].min())}-{int(test['year'].max())}",
        "train_years": f"{int(train['year'].min())}-{int(train['year'].max())}",
    }

    # Final model uses every available row.
    final = _build_pipeline().fit(df[FEATURES], df[TARGET])
    bundle = {
        "pipeline": final,
        "metrics": metrics,
        "importance": _grouped_importance(final),
        "n_rows": int(len(df)),
        "n_estimators": N_ESTIMATORS,
        "features": FEATURES,
        "min_year": int(df["year"].min()),
        "max_year": int(df["year"].max()),
        "signature": _signature(df),
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_or_train_model():
    """Load the saved model, retraining whenever it is missing, unreadable or built from different data."""
    current = _signature(load_training_data())
    if os.path.exists(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)
            if isinstance(bundle, dict) and bundle.get("signature") == current:
                return bundle
        except Exception:                       # corrupt / incompatible pickle -> retrain
            pass
    return train_paddy_model()


if __name__ == "__main__":
    b = train_paddy_model()
    print(b["metrics"], b["n_rows"])
    print({k: round(v, 3) for k, v in b["importance"].items()})
