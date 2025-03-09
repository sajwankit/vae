import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from load_config import ConfigLoader

from pathlib import Path

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
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df[all_features] = df[all_features].apply(lambda x: x.fillna(x.median()))
        return df

    def create_test_data(self, df, attack_label="DDoS attacks-LOIC-HTTP", benign_label="Benign", n=2):
        """
        Updates the DataFrame by setting the 'split' column to 'test' for identified rows.

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing attack and benign traffic.
        - attack_label (str): Label identifying attack traffic in the dataset.
        - benign_label (str): Label identifying benign traffic in the dataset.
        - n (int): Multiplier for selecting additional benign samples.

        Returns:
        - pd.DataFrame: Updated DataFrame with 'split' column set to 'test' for identified rows.
        """
        # Extract unique timestamps where attack traffic is observed
        attack_seconds = df[df["Label"] == attack_label]["Timestamp"].dt.floor("S").unique()
        x = len(attack_seconds)  # Number of unique attack seconds

        # Filter attack traffic
        test_indices = df[df["Timestamp"].dt.floor("S").isin(attack_seconds)].index

        # Get remaining timestamps for benign traffic
        remaining_benign_seconds = df[(df["Label"] == benign_label) & ~df["Timestamp"].dt.floor("S").isin(attack_seconds)]["Timestamp"].dt.floor("S").unique()

        # Sample additional benign timestamps (n*x samples)
        benign_sample_seconds = np.random.choice(remaining_benign_seconds, size=min(n*x, len(remaining_benign_seconds)), replace=False)

        # Append benign samples to test set
        benign_indices = df[df["Timestamp"].dt.floor("S").isin(benign_sample_seconds)].index

        # Update 'split' column to 'test' for identified rows
        df.loc[test_indices, "split"] = "test"
        df.loc[benign_indices, "split"] = "test"
            
        return df

    def create_train_val_data(self, df, train_val_split=0.8, attack_label="DDoS attacks-LOIC-HTTP", random_state=42):
        """
        Splits the benign data into training and validation sets based on the specified proportion.

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing both attack and benign traffic.
        - train_val_split (float): Proportion of benign data to use for training (default: 80% train, 20% validation).
        - attack_label (str): Label identifying attack traffic in the dataset.

        Returns:
        - pd.DataFrame: DataFrame with 'split' column updated to 'train' or 'val' for identified rows.
        """
        # Randomly split benign data into train and validation sets
        train_df, val_df = train_test_split(df[df["split"] != "test"], test_size=1 - train_val_split, random_state=random_state)
        # Update 'split' column
        df.loc[train_df.index, "split"] = "train"
        df.loc[val_df.index, "split"] = "val"

        return df

    def create_split(self, df, random_state=42):
        """
        Adds a 'split' column to the DataFrame indicating whether the row belongs to train, validation, or test set.

        Parameters:
        - df (pd.DataFrame): Original DataFrame.
        - test_df (pd.DataFrame): Test DataFrame.
        - train_df (pd.DataFrame): Train DataFrame.
        - val_df (pd.DataFrame): Validation DataFrame.

        Returns:
        - pd.DataFrame: DataFrame with 'split' column added.
        """        
        df["split"] = None
        df = self.create_test_data(df)
        df = self.create_train_val_data(df, random_state=random_state)
        return df
    
    def compress_df(self, df, filename="compressed_data.parquet"):
        """
        Compresses the DataFrame by optimizing data types and saves it as a Parquet file.

        Parameters:
        - df (pd.DataFrame): Input DataFrame to be compressed.
        - filename (str): Path to save the compressed Parquet file.
        """
        # Select only the required columns
        selected_columns = [col for category in self.feature_categories.values() for col in category]
        
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # Ensure timestamp is in datetime format

        for col in selected_columns:
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
        df = df[["Src IP", "Dst IP", "Timestamp", "split"] + selected_columns]
        return df


if __name__ == "__main__":
    config = ConfigLoader(config_path="/Users/ankitsajwan/tech/projects/vae/config/config.yaml").config
    data_dir = Path(config["data"]["base_dir"])
    # Load dataset
    df = pd.read_csv(data_dir / config["data"]["raw_data"])

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # Ensure timestamp is in datetime format
    unique_pairs = df.groupby(["Src IP", "Dst IP"]).ngroups
    print(f"Number of unique (src ip, dst ip) pairs: {unique_pairs}")
    if config["device"] == "mpu":
        # Filter to keep 1000 rows: 900 benign and 100 non-benign
        benign_df = df[df["Label"] == "Benign"].sample(900, random_state=config["seed"])
        
        # Ensure there are 50 unique attack seconds among the 100 non-benign rows
        non_benign_df = df[df["Label"] != "Benign"]
        attack_seconds = non_benign_df["Timestamp"].dt.floor("S").unique()
        selected_attack_seconds = np.random.choice(attack_seconds, size=50, replace=False)
        non_benign_df = non_benign_df[non_benign_df["Timestamp"].dt.floor("S").isin(selected_attack_seconds)].sample(100, random_state=42)
        
        df = pd.concat([benign_df, non_benign_df]).reset_index(drop=True)

    # Initialize PrepareData class
    data_prep = PrepareData(config)

    # Handle missing values
    df = data_prep.handle_missing_data(df)
    df = data_prep.create_split(df)
    df = data_prep.compress_df(df)

    # Create test dataset
    test_df = df[df["split"] == "test"].copy()
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    # Save compressed data
    test_df.to_parquet(data_dir / "test_compressed.parquet", compression="snappy")
    train_df.to_parquet(data_dir / "train_compressed.parquet", compression="snappy")
    val_df.to_parquet(data_dir / "val_compressed.parquet", compression="snappy")



