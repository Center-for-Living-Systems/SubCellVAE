
# Step 1: Define a standard Autoencoder (AE)
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from skimage import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os
import joblib
import torch.nn.functional as F


class AE(torch.nn.Module):
    def __init__(self, latent_dim=8, input_ps=32, no_ch = 1,BN_flag=False, dropout_flag=False):
        super(AE, self).__init__()

        # Compute the spatial size after 3 conv layers
        final_size = input_ps // (2 ** 3)  # each Conv2d has stride=2

        def maybe_bn(n): return torch.nn.BatchNorm2d(n) if BN_flag else torch.nn.Identity()
        def maybe_dropout(): return torch.nn.Dropout(p=0.3) if dropout_flag else torch.nn.Identity()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(no_ch, 32, 3, stride=2, padding=1),
            maybe_bn(32),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            maybe_bn(64),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            maybe_bn(128),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Flatten()
        )

        self.encoder_fc = torch.nn.Sequential(
            torch.nn.Linear(128 * final_size * final_size, 1024),
            torch.nn.LeakyReLU(0.01),
            maybe_dropout(),
            torch.nn.Linear(1024, latent_dim)
        )

        self.decoder_fc = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 1024),
            torch.nn.LeakyReLU(0.01),
            maybe_dropout(),
            torch.nn.Linear(1024, 128 * final_size * final_size)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            maybe_bn(64),
            torch.nn.LeakyReLU(0.01),
            torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            maybe_bn(32),
            torch.nn.LeakyReLU(0.01),
            torch.nn.ConvTranspose2d(32, no_ch, 3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )

        self.final_size = final_size  # store for decoding

    def encode(self, x):
        x = self.encoder(x)
        return self.encoder_fc(x)

    def decode(self, z):
        x = self.decoder_fc(z).view(-1, 128, self.final_size, self.final_size)
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z



def plot_reconstruction_progress(model, dataloader, device, epoch):
    """Plot original vs reconstructed images during training"""
    model.eval()
    with torch.no_grad():
        for x, _,_ in dataloader:
            x = x.to(device)
            recon, _ = model(x)
            break  # only one batch

    x = x.cpu()
    recon = recon.cpu()
    print(x.shape)

    print(f"Input stats — min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
    print(f"Reconstruction stats — min: {recon.min().item():.4f}, max: {recon.max().item():.4f}, mean: {recon.mean().item():.4f}, std: {recon.std().item():.4f}")

    n = min(8, x.size(0))
    fig, axes = plt.subplots(2, n, figsize=(n * 1, 2))
    for i in range(n):
        axes[0, i].imshow(x[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    plt.suptitle(f"Reconstruction at Epoch {epoch}")
    plt.tight_layout()
    return fig
    


def normalized_mse(x_hat, x):
    mse_loss = nn.MSELoss(reduction='mean')
    mse = mse_loss(x_hat, x)
    norm = torch.mean(x ** 2)  # average signal power
    return mse / norm
    
# Step 2: Train Autoencoder and Extract Latent Representations
def train_ae(model, train_loader, val_loader, device, epochs, lr, loss_norm_flag,result_dir):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if loss_norm_flag:
        loss_fn = normalized_mse
    else:
        loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    # print the training loss during the process
    error_print_period = max(1,int(epochs/50))
    recon_view_period = max(1,int(epochs/10))

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, _, _ in train_loader:
            x = x.to(device)
            recon, _ = model(x)
            loss = loss_fn(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ , _ in val_loader:
                x = x.to(device)
                recon, _ = model(x)
                loss = loss_fn(recon, x)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % error_print_period == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if (epoch + 1) % recon_view_period == 0:
            fig = plot_reconstruction_progress(model, val_loader, device, epoch + 1)
            fig.savefig(os.path.join(result_dir, 'plot_reconstruction_progress_'+str(epoch)+'.png'))       
            torch.save(model, os.path.join(result_dir, 'ae_model_ep'+str(epoch)+'.pt'))

        # # Save best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model, os.path.join(result_dir, 'ae_model_at_min_val_loss.pt'))

    joblib.dump(train_losses, os.path.join(result_dir, 'train_losses_ep'+ str(epoch)+'.pkl'))
    joblib.dump(val_losses, os.path.join(result_dir, 'val_losses_ep'+ str(epoch)+'.pkl'))

    # Plot training and validation loss
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    fig.savefig(os.path.join(result_dir, 'train_val_losses.png'))

    return model, train_losses, val_losses





##############################

# -------------------------
# VAE (standard) for 32x32
# -------------------------
class VAE32(nn.Module):
    """
    Standard VAE for 32x32 patches (beta=1 by default in loss).
    Same architecture as your BetaVAE32.
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 12, out_activation: str = "sigmoid"):
        super().__init__()
        assert latent_dim > 0
        assert out_activation in ("sigmoid", "identity")

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.out_activation = out_activation

        # Encoder: 32 -> 16 -> 8 -> 4
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(32),
            nn.SiLU(),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),           # 8x8
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),          # 4x4
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )

        self.enc_out_dim = 128 * 4 * 4
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, self.enc_out_dim)
        self.dec_core = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16x16
            nn.BatchNorm2d(32),
            nn.SiLU(),

            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),  # 32x32
        )

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 128, 4, 4)
        xhat = self.dec_core(h)
        if self.out_activation == "sigmoid":
            xhat = torch.sigmoid(xhat)
        return xhat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar, z


# -------------------------
# Loss for VAE / Beta-VAE
# -------------------------
def vae_loss(x, xhat, mu, logvar, beta: float = 1.0, recon: str = "mse"):
    """
    beta=1.0 -> standard VAE
    beta>1.0 -> beta-VAE
    """
    if recon == "mse":
        recon_loss = F.mse_loss(xhat, x, reduction="mean")
    elif recon == "bce":
        recon_loss = F.binary_cross_entropy(xhat, x, reduction="mean")
    else:
        raise ValueError("recon must be 'mse' or 'bce'")

    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_per_sample.mean()

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


# -------------------------
# Training loop (VAE/BetaVAE)
# -------------------------
def train_vae(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    beta,
    recon_type,
    result_dir,
    loss_norm_flag=False,  # optional; usually False for VAE
):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_recon_losses, train_kl_losses = [], []
    val_recon_losses, val_kl_losses = [], []

    error_print_period = max(1, int(epochs / 50))
    recon_view_period = max(1, int(epochs / 10))

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for x, _, _ in train_loader:
            x = x.to(device)
            xhat, mu, logvar, z = model(x)

            loss, recon_l, kl_l = vae_loss(x, xhat, mu, logvar, beta=beta, recon=recon_type)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_l.item()
            total_kl += kl_l.item()

        total_loss /= len(train_loader)
        total_recon /= len(train_loader)
        total_kl /= len(train_loader)

        train_losses.append(total_loss)
        train_recon_losses.append(total_recon)
        train_kl_losses.append(total_kl)

        # ---- val ----
        model.eval()
        v_loss = 0.0
        v_recon = 0.0
        v_kl = 0.0

        with torch.no_grad():
            for x, _, _ in val_loader:
                x = x.to(device)
                xhat, mu, logvar, z = model(x)
                loss, recon_l, kl_l = vae_loss(x, xhat, mu, logvar, beta=beta, recon=recon_type)

                v_loss += loss.item()
                v_recon += recon_l.item()
                v_kl += kl_l.item()

        v_loss /= len(val_loader)
        v_recon /= len(val_loader)
        v_kl /= len(val_loader)

        val_losses.append(v_loss)
        val_recon_losses.append(v_recon)
        val_kl_losses.append(v_kl)

        if (epoch + 1) % error_print_period == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train: total={total_loss:.4f} recon={total_recon:.4f} kl={total_kl:.4f} | "
                f"Val: total={v_loss:.4f} recon={v_recon:.4f} kl={v_kl:.4f}"
            )

        if (epoch + 1) % recon_view_period == 0:
            # save recon preview
            fig = plt.figure(figsize=(8, 3))
            model.eval()
            with torch.no_grad():
                for x, _, _ in val_loader:
                    x = x.to(device)
                    xhat, *_ = model(x)
                    x = x.cpu()
                    xhat = xhat.cpu()
                    break

            n = min(8, x.size(0))
            for i in range(n):
                plt.subplot(2, n, i + 1)
                plt.imshow(x[i].squeeze(), cmap="gray", vmin=0, vmax=1)
                plt.axis("off")
                plt.subplot(2, n, n + i + 1)
                plt.imshow(xhat[i].squeeze(), cmap="gray", vmin=0, vmax=1)
                plt.axis("off")
            plt.suptitle(f"Recon @ epoch {epoch+1}")
            plt.tight_layout()
            fig.savefig(os.path.join(result_dir, f"vae_recon_epoch{epoch+1}.png"))
            plt.close(fig)

            # save model checkpoint
            torch.save(model, os.path.join(result_dir, f"vae_model_ep{epoch+1}.pt"))

    # save curves
    joblib.dump(train_losses, os.path.join(result_dir, "vae_train_total.pkl"))
    joblib.dump(val_losses, os.path.join(result_dir, "vae_val_total.pkl"))
    joblib.dump(train_recon_losses, os.path.join(result_dir, "vae_train_recon.pkl"))
    joblib.dump(val_recon_losses, os.path.join(result_dir, "vae_val_recon.pkl"))
    joblib.dump(train_kl_losses, os.path.join(result_dir, "vae_train_kl.pkl"))
    joblib.dump(val_kl_losses, os.path.join(result_dir, "vae_val_kl.pkl"))

    # plot total loss
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), train_losses, label="Train Total")
    plt.plot(range(epochs), val_losses, label="Val Total")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("VAE Total Loss")
    fig.savefig(os.path.join(result_dir, "vae_train_val_total.png"))
    plt.close(fig)

    return model, train_losses, val_losses
