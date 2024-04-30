import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class Encoder(nn.Module):
    def __init__(self, n_channel, latent_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channel, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(8192, latent_dim)
        self.fc_logvar = nn.Linear(8192, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, n_channel):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 8, 8)
        x = self.conv(x)
        return x

class VAE(nn.Module):
    def __init__(self,  config):
        super(VAE, self).__init__()
        self.config = config
        n_channel = config['n_channel']
        n_stack = config['history_length']
        latent_dim = config['latent_dim']
        self.encoder = Encoder(n_channel * n_stack, latent_dim)
        self.decoder = Decoder(latent_dim, n_channel)

    def encode_and_sample(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def decode(self, latent):
        return self.decoder(latent.squeeze(1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_encoding(self, x):
        mu, logvar = self.encoder(x)
        return mu


if __name__ == '__main__':
    config = {
        'lr': 1e-3,
        'n_epochs': 10,
        'batch_size': 32,
        'n_channel': 3,
        'latent_dim': 16,
        'history_length': 1
    }
    model = VAE(config)

    x = torch.randn(1, 3*config['history_length'], 64, 64)
    recon_x, mu, logvar = model(x)
    print("Input shape:", x.shape)
    print("Mu shape:", mu.shape)
    print("Reconstructed shape:", recon_x.shape)