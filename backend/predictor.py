import numpy as np
import pickle
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

BASE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")

print("Loading models...")
lstm_model        = tf.keras.models.load_model(os.path.join(MODELS_DIR, "lstm_model.h5"))
transformer_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "transformer_model.h5"))

with open(os.path.join(MODELS_DIR, "scaler.pkl"),       "rb") as f: scaler       = pickle.load(f)
with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "rb") as f: feature_cols = pickle.load(f)
with open(os.path.join(MODELS_DIR, "airline_map.pkl"),  "rb") as f: AIRLINE_MAP  = pickle.load(f)
with open(os.path.join(MODELS_DIR, "airport_map.pkl"),  "rb") as f: AIRPORT_MAP  = pickle.load(f)

TIMESTEPS  = lstm_model.input_shape[1]
N_FEATURES = len(feature_cols)
print(f"Model input: (batch, {TIMESTEPS}, {N_FEATURES})")
print(f"Feature cols: {feature_cols}")
print("✅ Models loaded!")

AIRPORT_COORDS = {
    "ATL":(33.64,-84.43),"LAX":(33.94,-118.40),"ORD":(41.97,-87.90),
    "DFW":(32.89,-97.04),"DEN":(39.85,-104.67),"JFK":(40.63,-73.77),
    "SFO":(37.62,-122.38),"SEA":(47.44,-122.31),"LAS":(36.08,-115.15),
    "MCO":(28.43,-81.31),"EWR":(40.69,-74.17),"CLT":(35.21,-80.94),
    "PHX":(33.43,-112.01),"MIA":(25.79,-80.29),"IAH":(29.98,-95.33),
    "BOS":(42.36,-71.01),"MSP":(44.88,-93.22),"FLL":(26.07,-80.15),
    "DTW":(42.21,-83.35),"PHL":(39.87,-75.24),"LGA":(40.77,-73.87),
    "BWI":(39.17,-76.67),"SLC":(40.78,-111.97),"SAN":(32.73,-117.19),
    "DCA":(38.85,-77.04),"MDW":(41.78,-87.74),"HNL":(21.32,-157.92),
    "TPA":(27.97,-82.53),"PDX":(45.58,-122.59),"STL":(38.74,-90.37),
}

def encode(val, mapping):
    return mapping.get(str(val).upper().strip(), len(mapping))

def build_feature_vector(data: dict) -> np.ndarray:
    row = []
    for col in feature_cols:
        if col == "AIRLINE_ENC":
            row.append(float(encode(data["AIRLINE_CODE"], AIRLINE_MAP)))
        elif col == "ORIGIN_ENC":
            row.append(float(encode(data["ORIGIN_AIRPORT"], AIRPORT_MAP)))
        elif col == "DEST_ENC":
            row.append(float(encode(data["DESTINATION_AIRPORT"], AIRPORT_MAP)))
        elif col == "ORIGIN_LAT":
            row.append(float(AIRPORT_COORDS.get(data["ORIGIN_AIRPORT"], (0,0))[0]))
        elif col == "ORIGIN_LON":
            row.append(float(AIRPORT_COORDS.get(data["ORIGIN_AIRPORT"], (0,0))[1]))
        else:
            row.append(float(data.get(col, 0)))

    arr        = np.array([row], dtype=np.float32)
    arr_scaled = scaler.transform(arr)
    return arr_scaled.reshape(1, TIMESTEPS, N_FEATURES)

def predict_delay(data: dict) -> dict:
    X = build_feature_vector(data)

    lstm_prob        = float(lstm_model.predict(X, verbose=0)[0][0])
    transformer_prob = float(transformer_model.predict(X, verbose=0)[0][0])
    ensemble_prob    = (lstm_prob + transformer_prob) / 2.0

    def label(p):
        if p >= 0.7: return "High risk"
        if p >= 0.4: return "Moderate risk"
        return "Low risk"

    return {
        "lstm_probability":        round(lstm_prob, 4),
        "transformer_probability": round(transformer_prob, 4),
        "ensemble_probability":    round(ensemble_prob, 4),
        "prediction":              "Delayed" if ensemble_prob >= 0.5 else "On-Time",
        "risk_label":              label(ensemble_prob),
        "confidence_pct":          round(ensemble_prob * 100, 1),
    }
