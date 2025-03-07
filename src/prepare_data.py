import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from load_config import ConfigLoader

class PrepareData:
    def __init__(self, config):
        """
        Initializes the data preparation pipeline with the provided config.

        Parameters:
        - config (dict): Configuration dictionary containing feature categories, train-validation split, and other settings.
        """
        self.config = config
        self.feature_categories = self.config["data"]["features"]

    def handle_missing_data(self, df):
        """
        Handles missing and invalid values by applying median imputation for all feature categories.

        Parameters:
        - df (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: Cleaned DataFrame with missing values imputed.
        """
        # Get all feature lists dynamically from config
        all_features = []
        for category in self.feature_categories.values():
            all_features.extend(category)

        # Apply median imputation
        df[all_features] = df[all_features].fillna(df[all_features].median())

        return df

    def create_test_data(self, df, attack_label="DDoS attacks-LOIC-HTTP", benign_label="Benign", n=2):
        """
        Creates a test dataset by extracting attack traffic and sampling additional benign traffic.

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing attack and benign traffic.
        - attack_label (str): Label identifying attack traffic in the dataset.
        - benign_label (str): Label identifying benign traffic in the dataset.
        - n (int): Multiplier for selecting additional benign samples.

        Returns:
        - pd.DataFrame: Test dataset containing attack and benign traffic.
        """
        # Extract unique timestamps where attack traffic is observed
        attack_seconds = df[df["Label"] == attack_label]["Timestamp"].dt.floor("S").unique()
        x = len(attack_seconds)  # Number of unique attack seconds

        # Filter attack traffic
        test_df = df[df["Timestamp"].dt.floor("S").isin(attack_seconds)].copy()

        # Get remaining timestamps for benign traffic
        remaining_benign_seconds = df[(df["Label"] == benign_label) & ~df["Timestamp"].dt.floor("S").isin(attack_seconds)]["Timestamp"].dt.floor("S").unique()

        # Sample additional benign timestamps (n*x samples)
        benign_sample_seconds = np.random.choice(remaining_benign_seconds, size=min(n*x, len(remaining_benign_seconds)), replace=False)

        # Append benign samples to test set
        benign_samples = df[df["Timestamp"].dt.floor("S").isin(benign_sample_seconds)]
        test_df = pd.concat([test_df, benign_samples])

        return test_df

    def create_train_val_data(self, df, train_val_split=0.8, attack_label="DDoS attacks-LOIC-HTTP"):
        """
        Creates train and validation datasets by removing attack traffic and randomly splitting benign traffic.

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing both attack and benign traffic.
        - train_val_split (float): Proportion of benign data to use for training (default: 80% train, 20% validation).
        - attack_label (str): Label identifying attack traffic in the dataset.

        Returns:
        - pd.DataFrame: Training dataset (benign only).
        - pd.DataFrame: Validation dataset (benign only).
        """
        # Remove attack traffic
        benign_df = df[df["Label"] != attack_label].copy()

        # Randomly split benign data into train and validation sets
        train_df, val_df = train_test_split(benign_df, test_size=1 - train_val_split, random_state=42)

        return train_df, val_df

    def compress_and_save(self, df, filename="compressed_data.parquet"):
        """
        Compresses the DataFrame by optimizing data types and saves it as a Parquet file.

        Parameters:
        - df (pd.DataFrame): Input DataFrame to be compressed.
        - filename (str): Path to save the compressed Parquet file.
        """
        for col in df.columns:
            col_max = df[col].max()

            # Downcast integers
            if df[col].dtype == "int64":
                if col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

            # Downcast floats
            elif df[col].dtype == "float64":
                if col_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

        # Save as compressed Parquet file
        df.to_parquet(filename, compression="snappy")

        print(f"Compressed data saved to {filename}")
if __name__ == "__main__":
    config = ConfigLoader(config_path="/Users/ankitsajwan/tech/projects/vae/config/config.yaml").config

    # Load dataset
    df = pd.read_csv(config["data"]["train_path"])
    df = df[:int(0.5*len(df))]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # Ensure timestamp is in datetime format

    # Initialize PrepareData class
    data_prep = PrepareData(config)

    # Handle missing values
    df_cleaned = data_prep.handle_missing_data(df)

    # Create test dataset
    test_df = data_prep.create_test_data(df_cleaned)

    # Create train and validation datasets
    train_df, val_df = data_prep.create_train_val_data(df_cleaned, train_val_split=0.8)

    # Save compressed data
    data_prep.compress_and_save(train_df, "train_compressed.parquet")
    data_prep.compress_and_save(val_df, "val_compressed.parquet")
    data_prep.compress_and_save(test_df, "test_compressed.parquet")