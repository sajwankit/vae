import pandas as pd
import numpy as np


class DataIngestionPipeline:
    def __init__(self, config):
        self.filepath = config["data"]["train_path"]
        self.df = None


    def load_data(self):
        """Loads the raw CSV dataset."""
        self.df = pd.read_csv(self.filepath)
        self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"])
        self.df["Time_Minute"] = self.df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        
    
    def preprocess_data(self):
        """Preprocesses the raw dataset
        - keep columns mentioned in the config
        - check for missing and nan values
        """
        

    def aggregate_per_minute(self):
        """Aggregates network flows per (Src IP, Dst IP, Minute)."""
        self.df_agg = self.df.groupby(["Src IP", "Dst IP", "Time_Minute"]).agg({
            "Tot Fwd Pkts": "sum",
            "Tot Bwd Pkts": "sum",
            "TotLen Fwd Pkts": "sum",
            "TotLen Bwd Pkts": "sum",
            "Flow Byts/s": ["mean", "max"],
            "Flow Pkts/s": ["mean", "max"],
            "Pkt Len Mean": "mean",
            "Pkt Len Std": "mean",
            "Pkt Len Max": "max",
            "Pkt Len Min": "min",
            "Flow IAT Mean": ["mean", "max"],
            "Active Mean": "mean",
            "Idle Mean": "mean",
            "SYN Flag Cnt": "sum",
            "RST Flag Cnt": "sum",
            "ACK Flag Cnt": "sum"
        }).reset_index()

        # Flatten column names
        self.df_agg.columns = ['_'.join(col).strip() if type(col) is tuple else col for col in self.df_agg.columns]

    def fill_missing_minutes(self):
        """Fills missing minutes with appropriate values."""
        self.df_agg["Time_Minute"] = pd.to_datetime(self.df_agg["Time_Minute"])
        time_range = pd.date_range(self.df_agg["Time_Minute"].min(), self.df_agg["Time_Minute"].max(), freq="T")

        all_combinations = pd.MultiIndex.from_product(
            [self.df_agg["Src IP"].unique(), self.df_agg["Dst IP"].unique(), time_range],
            names=["Src IP", "Dst IP", "Time_Minute"]
        )

        self.df_filled = self.df_agg.set_index(["Src IP", "Dst IP", "Time_Minute"]).reindex(all_combinations).reset_index()
        self.df_filled.fillna(0, inplace=True)  # Fill missing packet counts with 0
        self.df_filled.fillna(method="ffill", inplace=True)  # Forward fill for time-based averages

    def get_aggregated_data(self):
        """Returns the processed aggregated dataset."""
        return self.df_filled

# Example usage:
# pipeline = DataIngestionPipeline("cic_ids_2018.csv")
# pipeline.load_data()
# pipeline.aggregate_per_minute()
# pipeline.fill_missing_minutes()
# df_aggregated = pipeline.get_aggregated_data()