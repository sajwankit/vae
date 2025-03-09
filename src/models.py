import torch
import torch.nn as nn

# Define the encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                self.fc.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                self.fc.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        for fc_layer in self.fc:
            x = torch.relu(fc_layer(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Define the decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                self.fc.append(nn.Linear(latent_dim, hidden_dims[i]))
            else:
                self.fc.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))

        self.fc_output = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z):
        for fc_layer in self.fc:
            z = torch.relu(fc_layer(z))
        x_reconstructed = self.fc_output(z)
        return x_reconstructed

# Define the VAE
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar