import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

from load_config import ConfigLoader
from feature_transformer import FeaturesTransformer

class VAEDataset(Dataset):
    def __init__(self, config, mode="train"):
        """
        Args:
            data (numpy.ndarray or torch.Tensor): The dataset to be used.
            mode (str): Mode of the dataset, one of 'train', 'val', or 'test'.
        """
        assert mode in ["train", "val", "test"], "mode should be one of 'train', 'val', or 'test'"

        data_dir = Path(config["data"]["base_dir"])

        self.feature_transformer = FeaturesTransformer(config, mode=mode, scaler_dir=data_dir)

        if mode == "train":
            data_path = data_dir / config["data"]["train"]
        elif mode == "val":
            data_path = data_dir / config["data"]["val"]
        elif mode == "test":
            data_path = data_dir / config["data"]["test"]
            
        self.data = pd.read_parquet(data_path)
        self.data = self.feature_transformer.transform(self.data)

        # Ensure data is in tensor format and normalized
        feature_categories = config["data"]["features"]
        selected_columns = [col for category in feature_categories.values() for col in category]
        selected_columns.sort()
        self.data = self.data[selected_columns]
        self.data = self.data.to_numpy()
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # VAE input = output (unsupervised learning)


if __name__ == "__main__":

    config = ConfigLoader(config_path="/Users/ankitsajwan/tech/projects/vae/config/config.yaml").config
    batch_size = config["dataloader"]["batch_size"]



    # train_dataset = VAEDataset(config=config, mode="train")
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # # Example usage
    # for batch in train_dataloader:
    #     print(batch.shape)  # Output: torch.Size([batch_size, feature_dim])
    #     break

    test_dataset = VAEDataset(config=config, mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Example usage
    for batch in test_dataloader:
        print(batch.shape)  # Output: torch.Size([batch_size, feature_dim])
        break
