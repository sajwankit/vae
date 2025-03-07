import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class FeaturesTransformer:
    def __init__(self, config, mode="train", scaler_dir="scalers"):
        """
        Initializes the feature engineering pipeline.

        Args:
            config (dict): Configuration containing feature categories.
        """
        self.config = config
        self.mode = mode
        self.scaler_dir = scaler_dir
        self.scalers = {}  # Dictionary to hold scalers dynamically

        # Ensure scaler directory exists
        if not os.path.exists(self.scaler_dir):
            os.makedirs(self.scaler_dir)

    def _get_features_by_type(self):
        """Extracts feature categories from the config dynamically."""
        return self.config["data"]["features"]

    def _get_scaler_path(self, feature_type):
        """Returns the path to save/load scalers."""
        return os.path.join(self.scaler_dir, f"{feature_type}_scaler.pkl")

    def _load_scalers(self, feature_types):
        """Loads saved scalers for test mode."""
        for feature_type in feature_types:
            scaler_path = self._get_scaler_path(feature_type)
            if os.path.exists(scaler_path):
                self.scalers[feature_type] = joblib.load(scaler_path)
            else:
                raise ValueError(f"Scaler for '{feature_type}' not found at {scaler_path}. Run training first.")

    def _save_scalers(self):
        """Saves the scalers for training mode."""
        for feature_type, scaler in self.scalers.items():
            joblib.dump(scaler, self._get_scaler_path(feature_type))

    def _apply_transformation(self, df, feature_type, transformation):
        """Applies transformation to a specific feature category."""
        feature_columns = self.config["data"]["features"].get(feature_type, [])
        if not feature_columns:
            return df  # Skip if no features exist in this category

        if feature_type not in self.scalers:
            if self.mode == "train":
                self.scalers[feature_type] = transformation.fit(df[feature_columns])
            else:  # Test mode, scaler should be loaded already
                self.scalers[feature_type] = self.scalers[feature_type]  # No fitting in test mode

        df[feature_columns] = self.scalers[feature_type].transform(df[feature_columns])
        return df

    def transform(self, df):
        """
        Transforms the given dataframe based on feature type.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        feature_types = self._get_features_by_type().keys()

        if self.mode == "test":
            self._load_scalers(feature_types)

        # Apply transformations based on feature type
        df = self._apply_transformation(df, "volume", MinMaxScaler())
        df = self._apply_transformation(df, "time-based", StandardScaler())  # Assuming log1p is not needed for all time features
        df = self._apply_transformation(df, "packet-size", MinMaxScaler())
        df = self._apply_transformation(df, "protocol-flags", MinMaxScaler())  # If needed
        df = self._apply_transformation(df, "connection-attributes", MinMaxScaler())

        if self.mode == "train":
            self._save_scalers()

        return df

# Example Usage
if __name__ == "__main__":
    config = {
        "data": {
            "features": {
                "volume": ["Tot Fwd Pkts", "Tot Bwd Pkts"],
                "time-based": ["Flow Duration", "Flow IAT Mean"],
                "packet-size": ["Fwd Pkt Len Max", "Bwd Pkt Len Min"],
                "protocol-flags": ["SYN Flag Cnt", "RST Flag Cnt"],
                "connection-attributes": ["Fwd Header Len", "Bwd Header Len"]
            }
        }
    }

    # Simulated dataset
    df_train = pd.DataFrame({
        "Tot Fwd Pkts": np.random.randint(1, 100, size=10),
        "Tot Bwd Pkts": np.random.randint(1, 100, size=10),
        "Flow Duration": np.random.randint(1000, 50000, size=10),
        "Flow IAT Mean": np.random.randint(50, 1000, size=10),
        "Fwd Pkt Len Max": np.random.randint(10, 200, size=10),
        "Bwd Pkt Len Min": np.random.randint(10, 200, size=10),
        "SYN Flag Cnt": np.random.randint(0, 2, size=10),
        "RST Flag Cnt": np.random.randint(0, 2, size=10),
        "Fwd Header Len": np.random.randint(20, 80, size=10),
        "Bwd Header Len": np.random.randint(20, 80, size=10)
    })

    # Training Mode
    pipeline = FeaturesTransformer(config, mode="train")
    df_train_transformed = pipeline.transform(df_train)
    print(df_train_transformed.head())

    # Testing Mode
    df_test = df_train.copy()  # Simulating new test data
    pipeline_test = FeaturesTransformer(config, mode="test")
    df_test_transformed = pipeline_test.transform(df_test)
    print(df_test_transformed.head())