"""
predict.py  —  Single phone price range prediction
Usage: python src/predict.py
"""
import os, sys, joblib
import numpy as np

BASE_DIR   = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

PRICE_LABELS = {
    0: "Budget        — under ₹15,000",
    1: "Mid-Range     — ₹15,000 to ₹30,000",
    2: "Upper Mid     — ₹30,000 to ₹60,000",
    3: "Premium       — above ₹60,000",
}


def load_bundle(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}\nRun: python src/model_training.py")
    return joblib.load(path)


def _build_features(specs: dict, le, feature_cols: list) -> np.ndarray:
    brand     = specs["brand"].strip().title()
    brand_enc = le.transform([brand])[0]

    ram_gb    = float(specs["ram_gb"])
    storage   = float(specs["storage_gb"])
    battery   = float(specs["battery_mah"])
    screen    = float(specs["screen_size_in"])
    camera    = float(specs["camera_mp"])
    cpu       = float(specs["processor_ghz"])
    five_g    = int(specs["five_g"])
    refresh   = float(specs.get("refresh_rate", 60))
    front_cam = float(specs.get("front_camera_mp", 16))
    nfc       = int(specs.get("nfc", 0))
    ir        = int(specs.get("ir_blaster", 0))
    ext_card  = int(specs.get("ext_card", 1))
    rating    = float(specs.get("rating", 75))

    # Engineered
    res_w = float(specs.get("res_width", 1080))
    res_h = float(specs.get("res_height", 2400))
    ppi              = np.sqrt(res_w**2 + res_h**2) / screen
    ram_x_storage    = ram_gb    * storage
    cam_x_cpu        = camera    * cpu
    battery_per_inch = battery   / screen
    total_camera     = camera    + front_cam

    row = [
        brand_enc, ram_gb, storage, battery, screen, camera, cpu,
        five_g, refresh, front_cam, nfc, ir, ext_card, rating,
        ppi, ram_x_storage, cam_x_cpu, battery_per_inch, total_camera,
    ]
    return np.array([row])


def predict_phone(specs: dict, bundle=None):
    if bundle is None:
        bundle = load_bundle()
    X = _build_features(specs, bundle["label_encoder"], bundle["feature_cols"])
    X_s  = bundle["scaler"].transform(X)
    pred = int(bundle["model"].predict(X_s)[0])
    return pred, PRICE_LABELS[pred]


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        {"name": "Realme C33 (Budget)",
         "brand":"Realme", "ram_gb":3, "storage_gb":32, "battery_mah":5000,
         "screen_size_in":6.5, "camera_mp":50, "processor_ghz":1.8,
         "five_g":0, "refresh_rate":60, "front_camera_mp":5,
         "nfc":0, "ir_blaster":0, "ext_card":1, "rating":64,
         "res_width":720, "res_height":1600},

        {"name": "OnePlus Nord 2T (Mid)",
         "brand":"Oneplus", "ram_gb":8, "storage_gb":128, "battery_mah":4500,
         "screen_size_in":6.43, "camera_mp":50, "processor_ghz":2.2,
         "five_g":1, "refresh_rate":90, "front_camera_mp":32,
         "nfc":1, "ir_blaster":0, "ext_card":1, "rating":84,
         "res_width":1080, "res_height":2400},

        {"name": "Apple iPhone 14 (Premium)",
         "brand":"Apple", "ram_gb":6, "storage_gb":128, "battery_mah":3279,
         "screen_size_in":6.1, "camera_mp":12, "processor_ghz":3.22,
         "five_g":1, "refresh_rate":60, "front_camera_mp":12,
         "nfc":1, "ir_blaster":0, "ext_card":0, "rating":81,
         "res_width":1170, "res_height":2532},
    ]

    bundle = load_bundle()
    print(f"\n🤖 Model: {bundle['model_name']}\n")
    print(f"{'Phone':<30} {'Range'}")
    print("─" * 65)
    for s in samples:
        name = s.pop("name")
        pred, label = predict_phone(s, bundle)
        print(f"{name:<30} [{pred}] {label}")
