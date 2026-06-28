import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Base directory where models are stored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class ModelLoader:
    """
    Loads and manages the LSTM and Transformer models, scaler, and feature columns
    for flight delay prediction.
    """

    def __init__(self, model_type: str = "lstm"):
        """
        Args:
            model_type: "lstm" or "transformer"
        """
        self.model_type = model_type.lower()
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self._load_all()

    def _load_all(self):
        """Load model, scaler, and feature columns."""
        self._load_feature_cols()
        self._load_scaler()
        self._load_model()

    def _load_feature_cols(self):
        """Load feature column names from pickle."""
        path = os.path.join(MODELS_DIR, "feature_cols.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"feature_cols.pkl not found at: {path}")
        with open(path, "rb") as f:
            self.feature_cols = pickle.load(f)
        print(f"[ModelLoader] Loaded {len(self.feature_cols)} feature columns.")

    def _load_scaler(self):
        """Load the fitted scaler from pickle."""
        path = os.path.join(MODELS_DIR, "scaler.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"scaler.pkl not found at: {path}")
        with open(path, "rb") as f:
            self.scaler = pickle.load(f)
        print("[ModelLoader] Scaler loaded.")

    def _load_model(self):
        """Load the selected Keras model (.h5)."""
        model_files = {
            "lstm": "lstm_model.h5",
            "transformer": "transformer_model.h5",
        }
        if self.model_type not in model_files:
            raise ValueError(
                f"Unknown model_type '{self.model_type}'. Choose 'lstm' or 'transformer'."
            )
        path = os.path.join(MODELS_DIR, model_files[self.model_type])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        self.model = tf.keras.models.load_model(path)
        print(f"[ModelLoader] '{self.model_type}' model loaded from {path}")

    def preprocess(self, input_data: dict) -> np.ndarray:
        """
        Preprocess a single input sample for prediction.

        Args:
            input_data: dict with keys matching feature_cols

        Returns:
            Scaled numpy array shaped (1, 1, num_features) for LSTM/Transformer
        """
        # Build feature vector in the correct order
        feature_vector = np.array(
            [input_data.get(col, 0.0) for col in self.feature_cols],
            dtype=np.float32,
        ).reshape(1, -1)

        # Scale features
        scaled = self.scaler.transform(feature_vector)

        # Reshape to (batch, timesteps, features) expected by LSTM/Transformer
        return scaled.reshape(1, 1, -1)

    def predict(self, input_data: dict) -> dict:
        """
        Run prediction on a single flight input.

        Args:
            input_data: dict with feature key-value pairs

        Returns:
            dict with 'delay_minutes' (float) and 'is_delayed' (bool)
        """
        processed = self.preprocess(input_data)
        prediction = self.model.predict(processed, verbose=0)

        # Assumes model outputs a single regression value (delay in minutes)
        delay_minutes = float(prediction[0][0])
        is_delayed = delay_minutes > 15  # common threshold: >15 min = delayed

        return {
            "delay_minutes": round(delay_minutes, 2),
            "is_delayed": is_delayed,
            "model_used": self.model_type,
        }

    def switch_model(self, model_type: str):
        """Switch between lstm and transformer at runtime."""
        self.model_type = model_type.lower()
        self._load_model()

    def get_feature_cols(self) -> list:
        """Return the list of expected feature column names."""
        return self.feature_cols

    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()


# --- Singleton pattern (optional, used by predictor.py) ---
_loader_instance: ModelLoader = None


def get_loader(model_type: str = "lstm") -> ModelLoader:
    """
    Return a cached ModelLoader instance.
    Call this from predictor.py to avoid reloading on every request.
    """
    global _loader_instance
    if _loader_instance is None or _loader_instance.model_type != model_type:
        _loader_instance = ModelLoader(model_type=model_type)
    return _loader_instance


# --- Quick test ---
if __name__ == "__main__":
    loader = get_loader("lstm")
    print("Feature columns:", loader.get_feature_cols())
    loader.summary()

    # Example dummy prediction
    sample = {col: 0.0 for col in loader.get_feature_cols()}
    result = loader.predict(sample)
    print("Sample prediction:", result)