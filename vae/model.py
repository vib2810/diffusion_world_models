import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Encoder(nn.Module):
    def __init__(self, n_channel, latent_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channel, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

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
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_channel, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 8, 8)
        x = self.conv(x)
        return x

class VAE(pl.LightningModule):
    def __init__(self,  config):
        super(VAE, self).__init__()
        self.config = config
        n_channel = config['n_channel']
        n_stack = config['history_length']
        latent_dim = config['latent_dim']
    
        self.save_hyperparameters(config)
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def training_step(self, batch, batch_idx):
        obs, action, next_obs = batch
        x_recon, mu, logvar = self(obs)
        # print(x_recon.shape, obs[:, -4:-1].shape) # reconstruct last observation
        recon_loss = F.mse_loss(x_recon, obs[:, -4:-1], reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_divergence
        self.log_dict(
            {
                "train/loss": loss,
                "train/kl_divergence": kl_divergence,
                "train/recon_loss": recon_loss,
            }, prog_bar=True, on_epoch=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        obs, action, next_obs = batch
        x_recon, mu, logvar = self(obs)
        recon_loss = F.mse_loss(x_recon, obs[:, -4:-1], reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_divergence
        self.log_dict(
            {
                "val/loss": loss,
                "val/recon_loss": recon_loss,
                "val/kl_divergence": kl_divergence
            }, prog_bar=True, on_epoch=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['lr'])


if __name__ == '__main__':
    n_stack = 3
    model = VAE(n_stack=n_stack)

    x = torch.randn(1, 3*n_stack, 64, 64)
    recon_x, mu, logvar = model(x)
    print("Input shape:", x.shape)
    print("Reconstructed shape:", recon_x.shape)