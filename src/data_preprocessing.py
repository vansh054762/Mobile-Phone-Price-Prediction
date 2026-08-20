"""
data_preprocessing.py
---------------------
Loads smartphones_specs_clean.csv (real dataset, 879 phones),
adds price_range labels, engineers features, and provides the
train/test split used by all other modules.

Price tiers (Indian market ₹):
  0  Budget       < ₹15,000
  1  Mid-Range    ₹15,000 – ₹29,999
  2  Upper Mid    ₹30,000 – ₹59,999
  3  Premium      ≥ ₹60,000
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATA_FILE    = "smartphones_specs_clean.csv"

PRICE_BINS   = [0, 15000, 30000, 60000, float("inf")]
PRICE_LABELS_INT = [0, 1, 2, 3]
PRICE_LABEL_NAMES = {
    0: "Budget        (< ₹15,000)",
    1: "Mid-Range     (₹15,000 – ₹30,000)",
    2: "Upper Mid     (₹30,000 – ₹60,000)",
    3: "Premium       (> ₹60,000)",
}


def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load real dataset and return cleaned DataFrame with price_range column."""
    df = pd.read_csv(filepath, index_col=0)

    # ── Rename to snake_case ──────────────────────────────────────────────────
    df = df.rename(columns={
        "Brand":                 "brand",
        "Ram Size(GB)":          "ram_gb",
        "Rom Size(GB)":          "storage_gb",
        "Battery Capacity":      "battery_mah",
        "Display Size(inches)":  "screen_size_in",
        "Rear Camera 1":         "camera_mp",
        "Processor Speed":       "processor_ghz",
        "5G":                    "five_g",
        "Display Refresh Rate":  "refresh_rate",
        "Resolution Width":      "res_width",
        "Resolution Height":     "res_height",
        "Front Camera":          "front_camera_mp",
        "NFC":                   "nfc",
        "IR Blaster":            "ir_blaster",
        "External Card Support": "ext_card",
        "Number of Sim":         "num_sim",
        "price":                 "price",
        "rating":                "rating",
    })

    # ── Keep useful columns ───────────────────────────────────────────────────
    keep = [
        "brand", "ram_gb", "storage_gb", "battery_mah",
        "screen_size_in", "camera_mp", "processor_ghz",
        "five_g", "refresh_rate", "front_camera_mp",
        "nfc", "ir_blaster", "ext_card",
        "res_width", "res_height", "rating", "price",
    ]
    df = df[keep].copy()

    # ── Coerce numerics ───────────────────────────────────────────────────────
    num_cols = [
        "ram_gb", "storage_gb", "battery_mah", "screen_size_in",
        "camera_mp", "processor_ghz", "res_width", "res_height",
        "front_camera_mp", "refresh_rate", "rating", "price",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.dropna(subset=["price", "ram_gb", "camera_mp", "processor_ghz"], inplace=True)
    df.drop_duplicates(inplace=True)

    # Remove junk prices
    df = df[(df["price"] >= 2000) & (df["price"] <= 700000)]

    # ── Add price_range label ─────────────────────────────────────────────────
    df["price_range"] = pd.cut(
        df["price"],
        bins=PRICE_BINS,
        labels=PRICE_LABELS_INT,
        right=False,
    ).astype(int)

    # Normalise brand
    df["brand"] = df["brand"].str.strip().str.title()

    return df.reset_index(drop=True)


def encode_and_scale(df: pd.DataFrame):
    """Feature engineering + label encoding + standard scaling."""
    df = df.copy()

    # Label-encode brand
    le = LabelEncoder()
    df["brand_enc"] = le.fit_transform(df["brand"])

    # ── Engineered features ───────────────────────────────────────────────────
    df["ppi"]              = (np.sqrt(df["res_width"]**2 + df["res_height"]**2)
                               / df["screen_size_in"])
    df["ram_x_storage"]    = df["ram_gb"]    * df["storage_gb"]
    df["cam_x_cpu"]        = df["camera_mp"] * df["processor_ghz"]
    df["battery_per_inch"] = df["battery_mah"] / df["screen_size_in"]
    df["total_camera"]     = df["camera_mp"] + df["front_camera_mp"]

    feature_cols = [
        # raw
        "brand_enc", "ram_gb", "storage_gb", "battery_mah",
        "screen_size_in", "camera_mp", "processor_ghz",
        "five_g", "refresh_rate", "front_camera_mp",
        "nfc", "ir_blaster", "ext_card", "rating",
        # engineered
        "ppi", "ram_x_storage", "cam_x_cpu",
        "battery_per_inch", "total_camera",
    ]

    X        = df[feature_cols].values
    y        = df["price_range"]
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols, le, scaler


def get_train_test_split(filepath: str):
    """Full pipeline: load → clean → encode → split."""
    df = load_and_clean(filepath)
    X, y, feature_cols, le, scaler = encode_and_scale(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, feature_cols, le, scaler


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = os.path.dirname(__file__)
    fp   = os.path.normpath(os.path.join(base, "..", "data", DATA_FILE))
    df   = load_and_clean(fp)

    print(f"✅  Dataset : {fp}")
    print(f"    Shape   : {df.shape[0]} phones × {df.shape[1]} columns")
    print(f"    Brands  : {sorted(df['brand'].unique())}\n")
    print("Price range distribution:")
    for k, v in df["price_range"].value_counts().sort_index().items():
        print(f"  [{k}] {PRICE_LABEL_NAMES[k]:<35} : {v:>3} phones")
    print(f"\n    Price   : ₹{df['price'].min():,.0f}  –  ₹{df['price'].max():,.0f}")
