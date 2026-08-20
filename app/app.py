"""
app.py  —  Flask Web App for Mobile Phone Price Prediction
Run: python app/app.py
Then open: http://localhost:5000
"""
import os, sys, json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import load_bundle, _build_features

BASE_DIR     = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "best_model.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "data", "smartphones_specs_clean.csv")

PRICE_TIERS = {
    0: {"label": "Budget",    "range": "< ₹15,000",          "color": "#2ecc71", "emoji": "🟢"},
    1: {"label": "Mid-Range", "range": "₹15,000 – ₹30,000",  "color": "#f39c12", "emoji": "🟡"},
    2: {"label": "Upper Mid", "range": "₹30,000 – ₹60,000",  "color": "#e67e22", "emoji": "🟠"},
    3: {"label": "Premium",   "range": "> ₹60,000",           "color": "#e74c3c", "emoji": "🔴"},
}

SIM_COLS = ["ram_gb", "storage_gb", "battery_mah", "screen_size_in",
            "camera_mp", "processor_ghz", "five_g", "refresh_rate",
            "front_camera_mp", "nfc"]

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Load once ─────────────────────────────────────────────────────────────────
bundle   = load_bundle(MODEL_PATH)
phone_df = None

def load_phone_df():
    global phone_df
    raw = pd.read_csv(DATASET_PATH, index_col=0)
    raw = raw.rename(columns={
        "Brand": "brand", "Ram Size(GB)": "ram_gb", "Rom Size(GB)": "storage_gb",
        "Battery Capacity": "battery_mah", "Display Size(inches)": "screen_size_in",
        "Rear Camera 1": "camera_mp", "Processor Speed": "processor_ghz",
        "5G": "five_g", "Display Refresh Rate": "refresh_rate",
        "Resolution Width": "res_width", "Resolution Height": "res_height",
        "Front Camera": "front_camera_mp", "NFC": "nfc",
        "IR Blaster": "ir_blaster", "External Card Support": "ext_card",
        "price": "price", "rating": "rating",
    })
    num = ["ram_gb","storage_gb","battery_mah","screen_size_in","camera_mp",
           "processor_ghz","res_width","res_height","front_camera_mp",
           "refresh_rate","rating","price","five_g","nfc","ir_blaster","ext_card"]
    for c in num:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw.dropna(subset=["price","ram_gb","camera_mp"], inplace=True)
    raw["brand"] = raw["brand"].str.strip().str.title()
    raw["model"] = raw["model"].str.strip()
    raw = raw[(raw["price"] >= 2000) & (raw["price"] <= 700000)]
    phone_df = raw.reset_index(drop=True)

load_phone_df()

# ── Similarity ────────────────────────────────────────────────────────────────
def find_similar(specs, top_n=5):
    df = phone_df.dropna(subset=SIM_COLS).copy()
    query = np.array([[
        specs["ram_gb"], specs["storage_gb"], specs["battery_mah"],
        specs["screen_size_in"], specs["camera_mp"], specs["processor_ghz"],
        specs["five_g"], specs["refresh_rate"], specs["front_camera_mp"], specs["nfc"],
    ]])
    matrix  = df[SIM_COLS].values
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(np.vstack([query, matrix]))
    dists   = euclidean_distances(scaled[[0]], scaled[1:])[0]
    top_idx = np.argsort(dists)[:top_n]
    result  = df.iloc[top_idx][
        ["model","brand","ram_gb","storage_gb","battery_mah",
         "camera_mp","processor_ghz","five_g","refresh_rate","rating","price"]
    ].copy()
    result["price"] = result["price"].astype(int)

    def tier(p):
        if p < 15000: return "Budget"
        if p < 30000: return "Mid-Range"
        if p < 60000: return "Upper Mid"
        return "Premium"

    result["tier"] = result["price"].apply(tier)
    return result.to_dict(orient="records")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    brands = sorted(phone_df["brand"].unique().tolist())
    return render_template("index.html", brands=brands,
                           total_phones=len(phone_df))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    specs = {
        "brand":          data["brand"],
        "ram_gb":         float(data["ram_gb"]),
        "storage_gb":     float(data["storage_gb"]),
        "battery_mah":    float(data["battery_mah"]),
        "screen_size_in": float(data["screen_size_in"]),
        "camera_mp":      float(data["camera_mp"]),
        "processor_ghz":  float(data["processor_ghz"]),
        "five_g":         int(data["five_g"]),
        "refresh_rate":   float(data["refresh_rate"]),
        "front_camera_mp":float(data["front_camera_mp"]),
        "nfc":            int(data["nfc"]),
        "ir_blaster":     int(data["ir_blaster"]),
        "ext_card":       int(data["ext_card"]),
        "rating":         float(data["rating"]),
        "res_width":      float(data.get("res_width", 1080)),
        "res_height":     float(data.get("res_height", 2400)),
    }
    X     = _build_features(specs, bundle["label_encoder"], bundle["feature_cols"])
    X_s   = bundle["scaler"].transform(X)
    pred  = int(bundle["model"].predict(X_s)[0])
    proba = bundle["model"].predict_proba(X_s)[0].tolist() \
            if hasattr(bundle["model"], "predict_proba") else [0.25]*4

    tier_info = PRICE_TIERS[pred]
    similar   = find_similar(specs, top_n=5)

    return jsonify({
        "prediction": pred,
        "tier":       tier_info,
        "proba":      proba,
        "model_name": bundle["model_name"],
        "similar":    similar,
    })

if __name__ == "__main__":
    print("\n🚀  Mobile Price Predictor — Flask")
    print("   Open → http://localhost:5000\n")
    app.run(debug=False, port=5000, host="localhost")
