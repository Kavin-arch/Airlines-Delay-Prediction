"""
retrain.py - Retrains models on 2015 Kaggle flights.csv
Run: python retrain.py
Needs flights.csv in the same folder.
"""
import os, pickle, numpy as np, pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

BASE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")
CSV_PATH   = os.path.join(BASE, "flights.csv")

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

# ── 1. Load ───────────────────────────────────────────────────
print("Loading flights.csv ...")
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Raw rows: {len(df):,}")

# ── 2. Clean ──────────────────────────────────────────────────
df = df[df["CANCELLED"] == 0].copy()
df["ARRIVAL_DELAY"] = pd.to_numeric(df["ARRIVAL_DELAY"], errors="coerce")
df = df.dropna(subset=["ARRIVAL_DELAY"])
df["TARGET"] = (df["ARRIVAL_DELAY"] > 15).astype(int)
print(f"After clean: {len(df):,} | Delay rate: {df['TARGET'].mean():.2%}")

# Encode airline
airlines    = sorted(df["AIRLINE"].dropna().unique())
airline_map = {a: i for i, a in enumerate(airlines)}
df["AIRLINE_ENC"] = df["AIRLINE"].map(airline_map).fillna(0).astype(int)

# Encode airports
all_ap     = sorted(set(df["ORIGIN_AIRPORT"].dropna().astype(str)) |
                    set(df["DESTINATION_AIRPORT"].dropna().astype(str)))
airport_map = {a: i for i, a in enumerate(all_ap)}
df["ORIGIN_ENC"] = df["ORIGIN_AIRPORT"].astype(str).map(airport_map).fillna(0).astype(int)
df["DEST_ENC"]   = df["DESTINATION_AIRPORT"].astype(str).map(airport_map).fillna(0).astype(int)

# Lat/lon from origin
df["ORIGIN_LAT"] = df["ORIGIN_AIRPORT"].map(lambda x: AIRPORT_COORDS.get(str(x),(0,0))[0])
df["ORIGIN_LON"] = df["ORIGIN_AIRPORT"].map(lambda x: AIRPORT_COORDS.get(str(x),(0,0))[1])

# Dep/arr hour
df["DEP_HOUR"] = (pd.to_numeric(df["DEPARTURE_TIME"], errors="coerce").fillna(0).astype(int) // 100).clip(0,23)
df["ARR_HOUR"] = (pd.to_numeric(df["ARRIVAL_TIME"],   errors="coerce").fillna(0).astype(int) // 100).clip(0,23)
df["SCHEDULED_TIME"] = pd.to_numeric(df["SCHEDULED_TIME"], errors="coerce").fillna(120)
df["DISTANCE"]       = pd.to_numeric(df["DISTANCE"],       errors="coerce").fillna(500)
df["DIVERTED"]       = pd.to_numeric(df["DIVERTED"],       errors="coerce").fillna(0)

FEATURE_COLS = [
    "MONTH", "DAY", "DAY_OF_WEEK", "SCHEDULED_TIME",
    "AIRLINE_ENC", "ORIGIN_ENC", "DEST_ENC",
    "DISTANCE", "ORIGIN_LAT", "ORIGIN_LON",
    "DIVERTED", "DEP_HOUR", "ARR_HOUR"
]

df = df[FEATURE_COLS + ["TARGET"]].dropna()
print(f"Final rows: {len(df):,} | Delay rate: {df['TARGET'].mean():.2%}")

# ── 3. Balanced 50/50 sample ──────────────────────────────────
n_delayed = min(200_000, (df["TARGET"]==1).sum())
n_ontime  = min(200_000, (df["TARGET"]==0).sum())
delayed   = df[df["TARGET"]==1].sample(n_delayed, random_state=42)
on_time   = df[df["TARGET"]==0].sample(n_ontime,  random_state=42)
df = pd.concat([delayed, on_time]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Balanced sample: {len(df):,} | Delay rate: {df['TARGET'].mean():.2%}")

X_raw = df[FEATURE_COLS].values.astype(np.float32)
y     = df["TARGET"].values.astype(np.float32)

# ── 4. Scale & reshape to (samples, 1, 13) ───────────────────
scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)
X        = X_scaled.reshape(-1, 1, len(FEATURE_COLS))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
print(f"Train: {len(X_train):,} | Val: {len(X_val):,}")

# Class weights (even though balanced, helps training)
cw     = compute_class_weight("balanced", classes=np.array([0,1]), y=y_train)
cw_dict = {0: cw[0], 1: cw[1]}
print(f"Class weights: {cw_dict}")

# ── 5. Models ─────────────────────────────────────────────────
N = len(FEATURE_COLS)

def build_lstm(n):
    inp = keras.Input(shape=(1, n))
    x   = layers.LSTM(128, return_sequences=False)(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m   = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy", "AUC"])
    return m

def build_transformer(n):
    inp = keras.Input(shape=(1, n))
    x   = layers.MultiHeadAttention(num_heads=4, key_dim=32)(inp, inp)
    x   = layers.LayerNormalization()(x)
    x   = layers.Flatten()(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m   = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy", "AUC"])
    return m

cb = [
    keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_auc", mode="max"),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, monitor="val_loss")
]

# ── 6. Train ──────────────────────────────────────────────────
print("\nTraining LSTM...")
lstm_m = build_lstm(N)
lstm_m.fit(X_train, y_train, epochs=25, batch_size=512,
           validation_data=(X_val, y_val),
           class_weight=cw_dict, callbacks=cb, verbose=1)

print("\nTraining Transformer...")
trans_m = build_transformer(N)
trans_m.fit(X_train, y_train, epochs=25, batch_size=512,
            validation_data=(X_val, y_val),
            class_weight=cw_dict, callbacks=cb, verbose=1)

# ── 7. Quick sanity check ─────────────────────────────────────
test_input = X_val[:200]
lp = lstm_m.predict(test_input, verbose=0)
tp = trans_m.predict(test_input, verbose=0)
print(f"\nLSTM  preds  — min:{lp.min():.3f} max:{lp.max():.3f} mean:{lp.mean():.3f}")
print(f"Trans preds  — min:{tp.min():.3f} max:{tp.max():.3f} mean:{tp.mean():.3f}")

# ── 8. Save ───────────────────────────────────────────────────
for fname in ["lstm_model.h5", "transformer_model.h5"]:
    src = os.path.join(MODELS_DIR, fname)
    dst = src.replace(".h5", "_backup.h5")
    if os.path.exists(src):
        os.replace(src, dst)
        print(f"Backed up {fname}")

lstm_m.save(os.path.join(MODELS_DIR, "lstm_model.h5"))
trans_m.save(os.path.join(MODELS_DIR, "transformer_model.h5"))

with open(os.path.join(MODELS_DIR, "scaler.pkl"),       "wb") as f: pickle.dump(scaler,       f)
with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "wb") as f: pickle.dump(FEATURE_COLS, f)
with open(os.path.join(MODELS_DIR, "airline_map.pkl"),  "wb") as f: pickle.dump(airline_map,  f)
with open(os.path.join(MODELS_DIR, "airport_map.pkl"),  "wb") as f: pickle.dump(airport_map,  f)

print("\n✅ All saved!")
print(f"   LSTM input:        {lstm_m.input_shape}")
print(f"   Transformer input: {trans_m.input_shape}")
print("\nNow restart:  uvicorn main:app --reload --port 8000")
