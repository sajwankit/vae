from load_config import ConfigLoader
from dataloaders import VAEDataset
from models import VAE
import pandas as pd

from torch.utils.data import DataLoader

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.functional import F


class VAEDDoSTrainer:

    def __init__(self, config, train_loader, val_loader, test_loader=None):
        self.config = config
        self.model = VAE(
            config["model"]["input_dim"], 
            config["model"]["hidden_dims"], 
            config["model"]["latent_dim"]
        ).to(config["device"])
        self.optimizer = self._initialize_optimizer()
        self.device = config["device"]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def _load_model(self):
        input_dim = self.config["model"]["input_dim"]
        hidden_dims = self.config["model"]["hidden_dims"]
        hidden_dim = self.config["hidden_dim"]
        latent_dim = self.config["model"]["latent_dim"]
        self.model = VAE(input_dim, hidden_dims, latent_dim).to(self.device)

    def _initialize_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config["model"]["learning_rate"])

    def _train(self):
        self.model.train()
        train_loss = 0
        for x in self.train_loader:
            x = x.to(self.device)
            self.optimizer.zero_grad()
            x_reconstructed, mu, logvar = self.model(x)
            reconstruction_loss, kl_divergence  = self.vae_loss(x, x_reconstructed, mu, logvar)     
            loss = reconstruction_loss + kl_divergence
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for x in self.val_loader:
                x = x.to(self.device)
                x_reconstructed, mu, logvar = self.model(x)
                reconstruction_loss, kl_divergence = self.vae_loss(x, x_reconstructed, mu, logvar)
                loss = reconstruction_loss + kl_divergence
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def test(self):
        test_losses = []
        self.model.eval()
        with torch.no_grad():
            for i, x in enumerate(self.test_loader):
                x = x.to(self.device)
                x_reconstructed, mu, logvar = self.model(x)
                reconstruction_loss, _ = self.vae_loss(x, x_reconstructed, mu, logvar, reduction='none')
                loss = reconstruction_loss.mean(dim=-1).tolist()
                test_losses.extend(loss)
        return test_losses

    # Define the loss function
    def vae_loss(self, x, x_reconstructed, mu, logvar, reduction='mean'):
        # 1. Reconstruction Loss (MSE)
        reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction=reduction)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss, kl_divergence

    def train(self, epochs=10):
        for epoch in range(epochs):
            st = time.time()
            train_loss = self._train()
            val_loss = self._validate()
            print(f"Epoch {epoch + 1}/{epochs}, Train loss: {train_loss :.4f}, Val loss: {val_loss :.4f}, Time: {time.time() - st:.4f}")
        print("Training complete!")


if __name__ == "__main__":
    
    import os

    # Get current working directory
    current_dir = os.getcwd()

    # Move one directory up
    parent_dir = Path(os.path.dirname(current_dir))
    config_path = parent_dir / "vae/config/config.yaml"
    config = ConfigLoader(config_path).config


    train_dataset = VAEDataset(config=config, mode="train")

    train_loader = DataLoader(train_dataset, batch_size=config["train_loader"]["batch_size"], shuffle=True)
    val_dataset = VAEDataset(config=config, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=config["val_loader"]["batch_size"], shuffle=False)

    test_dataset = VAEDataset(config=config, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=64*32, shuffle=False)

    trainer = VAEDDoSTrainer(config, train_loader, val_loader, test_loader)
    trainer.train(epochs=config["epochs"])

    test_df = pd.read_parquet(config["data"]["test"])
    test_losses = trainer.test()
    # test_losses = []
    test_df["test_loss"] = test_losses
    test_df.to_parquet(config["data"]["test"], index=False)

    
    

