"""
autoencoder.py
==============
Four autoencoder variants:
  1. AE          – standard convolutional autoencoder
  2. VAE32       – variational AE (standard VAE or beta-VAE via beta parameter)
  3. SemiSupAE   – semi-supervised AE with auxiliary classification head
  4. ContrastiveAE – contrastive AE using salt-and-pepper noise augmentation

All models share the same encoder/decoder backbone dimensions and expect
square input patches (default 32×32).
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import joblib


# =============================================================================
# Shared utilities
# =============================================================================

def normalized_mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """MSE normalised by average signal power (dimensionless loss)."""
    mse = F.mse_loss(x_hat, x, reduction="mean")
    norm = torch.mean(x ** 2).clamp(min=1e-8)
    return mse / norm


def salt_and_pepper_noise(x: torch.Tensor, noise_prob: float = 0.05) -> torch.Tensor:
    """
    Apply salt-and-pepper noise to a batch of images.

    Parameters
    ----------
    x          : (B, C, H, W) tensor, values in [0, 1]
    noise_prob : probability that any given pixel is corrupted
    """
    noisy = x.clone()
    mask = torch.rand_like(x)
    noisy[mask < noise_prob / 2] = 0.0          # pepper
    noisy[(mask >= noise_prob / 2) & (mask < noise_prob)] = 1.0  # salt
    return noisy


def plot_reconstruction_progress(model, dataloader, device, epoch, vae_mode=False):
    """Show original vs. reconstructed patches for one batch."""
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            if vae_mode:
                recon = model(x)[0]        # xhat, mu, logvar, z
            else:
                recon = model(x)[0]        # recon, z  (or recon, z for SemiSup)
            break

    x, recon = x.cpu(), recon.cpu()
    print(f"[epoch {epoch}] input  — min {x.min():.3f}  max {x.max():.3f}  mean {x.mean():.3f}")
    print(f"[epoch {epoch}] recon  — min {recon.min():.3f}  max {recon.max():.3f}  mean {recon.mean():.3f}")

    n = min(8, x.size(0))
    fig, axes = plt.subplots(2, n, figsize=(n, 2))
    for i in range(n):
        axes[0, i].imshow(x[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    plt.suptitle(f"Reconstruction @ epoch {epoch}")
    plt.tight_layout()
    return fig


# =============================================================================
# 1. Standard convolutional Autoencoder (AE)
# =============================================================================

class AE(nn.Module):
    """
    Convolutional autoencoder.

    Parameters
    ----------
    latent_dim    : dimension of the bottleneck vector
    input_ps      : spatial size of the (square) input patch
    no_ch         : number of input/output channels
    BN_flag       : whether to use BatchNorm2d in conv layers
    dropout_flag  : whether to add Dropout(0.3) in FC layers
    """

    def __init__(
        self,
        latent_dim: int = 8,
        input_ps: int = 32,
        no_ch: int = 1,
        BN_flag: bool = False,
        dropout_flag: bool = False,
    ):
        super().__init__()

        final_size = input_ps // (2 ** 3)   # 3 stride-2 conv layers

        def maybe_bn(n):       return nn.BatchNorm2d(n) if BN_flag       else nn.Identity()
        def maybe_dropout():   return nn.Dropout(0.3)   if dropout_flag  else nn.Identity()

        self.encoder = nn.Sequential(
            nn.Conv2d(no_ch, 32, 3, stride=2, padding=1),
            maybe_bn(32), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            maybe_bn(64), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            maybe_bn(128), nn.LeakyReLU(0.01),
            nn.Flatten(),
        )

        flat = 128 * final_size * final_size
        self.encoder_fc = nn.Sequential(
            nn.Linear(flat, 1024), nn.LeakyReLU(0.01),
            maybe_dropout(),
            nn.Linear(1024, latent_dim),
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.LeakyReLU(0.01),
            maybe_dropout(),
            nn.Linear(1024, flat),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64,   3, stride=2, padding=1, output_padding=1),
            maybe_bn(64), nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32,    3, stride=2, padding=1, output_padding=1),
            maybe_bn(32), nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, no_ch, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

        self.final_size = final_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_fc(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(-1, 128, self.final_size, self.final_size)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z


def train_ae(model, train_loader, val_loader, device, epochs, lr,
             loss_norm_flag, result_dir):
    """Training loop for the standard AE."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = normalized_mse if loss_norm_flag else nn.MSELoss()

    train_losses, val_losses = [], []
    error_print_period = max(1, epochs // 50)
    recon_view_period  = max(1, epochs // 10)

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for x, *_ in train_loader:
            x = x.to(device)
            recon, _ = model(x)
            loss = loss_fn(recon, x)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)
        train_losses.append(t_loss)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x, *_ in val_loader:
                x = x.to(device)
                recon, _ = model(x)
                v_loss += loss_fn(recon, x).item()
        v_loss /= len(val_loader)
        val_losses.append(v_loss)

        if (epoch + 1) % error_print_period == 0:
            print(f"AE  epoch {epoch+1}/{epochs}  train={t_loss:.4f}  val={v_loss:.4f}")

        if (epoch + 1) % recon_view_period == 0:
            fig = plot_reconstruction_progress(model, val_loader, device, epoch + 1)
            fig.savefig(os.path.join(result_dir, f"ae_recon_ep{epoch+1}.png"))
            plt.close(fig)
            torch.save(model, os.path.join(result_dir, f"ae_model_ep{epoch+1}.pt"))

    _save_loss_curves(train_losses, val_losses, epochs,
                      "AE Total Loss", result_dir, "ae")
    return model, train_losses, val_losses


# =============================================================================
# 2. VAE / Beta-VAE
# =============================================================================

class VAE32(nn.Module):
    """
    Variational Autoencoder for 32×32 patches.
    Set beta=1 in the loss for a standard VAE.
    Set beta>1 (e.g. 4, 8) for a beta-VAE that encourages disentanglement.

    Parameters
    ----------
    in_channels    : number of input channels
    latent_dim     : dimension of the latent space
    out_activation : 'sigmoid' (pixel values in [0,1]) or 'identity'
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 12,
        out_activation: str = "sigmoid",
    ):
        super().__init__()
        assert latent_dim > 0
        assert out_activation in ("sigmoid", "identity")

        self.in_channels   = in_channels
        self.latent_dim    = latent_dim
        self.out_activation = out_activation

        # Encoder: 32 → 16 → 8 → 4
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32,  4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64,           4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128,          4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.SiLU(),
        )

        self.enc_out_dim = 128 * 4 * 4
        self.fc_mu     = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # Decoder
        self.fc_dec   = nn.Linear(latent_dim, self.enc_out_dim)
        self.dec_core = nn.Sequential(
            nn.ConvTranspose2d(128, 64,        4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32,         4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor):
        h = self.enc(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(z.size(0), 128, 4, 4)
        xhat = self.dec_core(h)
        if self.out_activation == "sigmoid":
            xhat = torch.sigmoid(xhat)
        return xhat

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z    = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar, z


def vae_loss(
    x: torch.Tensor,
    xhat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    recon: str = "mse",
):
    """
    ELBO loss for VAE / beta-VAE.

    Returns
    -------
    total, recon_loss, kl_loss
    """
    if recon == "mse":
        recon_loss = F.mse_loss(xhat, x, reduction="mean")
    elif recon == "bce":
        recon_loss = F.binary_cross_entropy(xhat, x, reduction="mean")
    else:
        raise ValueError("recon must be 'mse' or 'bce'")

    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


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
    beta_anneal: bool = False,   # linearly warm up beta from 0 → beta
):
    """
    Training loop for VAE32 (standard VAE or beta-VAE).

    Parameters
    ----------
    beta_anneal : if True, linearly increase beta from 0 to `beta`
                  over the first half of training (KL warm-up).
                  Helps avoid posterior collapse.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_recon, train_kl    = [], []
    val_recon,   val_kl      = [], []

    error_print_period = max(1, epochs // 50)
    recon_view_period  = max(1, epochs // 10)

    for epoch in range(epochs):
        # KL warm-up schedule
        if beta_anneal:
            warmup_epochs = epochs // 2
            current_beta = beta * min(1.0, (epoch + 1) / warmup_epochs)
        else:
            current_beta = beta

        # --- train ---
        model.train()
        tl = tr = tk = 0.0
        for x, *_ in train_loader:
            x = x.to(device)
            xhat, mu, logvar, _ = model(x)
            loss, rl, kl = vae_loss(x, xhat, mu, logvar,
                                    beta=current_beta, recon=recon_type)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tl += loss.item(); tr += rl.item(); tk += kl.item()
        n = len(train_loader)
        tl /= n; tr /= n; tk /= n
        train_losses.append(tl); train_recon.append(tr); train_kl.append(tk)

        # --- val ---
        model.eval()
        vl = vr = vk = 0.0
        with torch.no_grad():
            for x, *_ in val_loader:
                x = x.to(device)
                xhat, mu, logvar, _ = model(x)
                loss, rl, kl = vae_loss(x, xhat, mu, logvar,
                                        beta=current_beta, recon=recon_type)
                vl += loss.item(); vr += rl.item(); vk += kl.item()
        n = len(val_loader)
        vl /= n; vr /= n; vk /= n
        val_losses.append(vl); val_recon.append(vr); val_kl.append(vk)

        if (epoch + 1) % error_print_period == 0:
            print(
                f"VAE  epoch {epoch+1}/{epochs}  β={current_beta:.2f} | "
                f"train total={tl:.4f} recon={tr:.4f} kl={tk:.4f} | "
                f"val   total={vl:.4f} recon={vr:.4f} kl={vk:.4f}"
            )

        if (epoch + 1) % recon_view_period == 0:
            fig, axes = plt.subplots(2, 8, figsize=(8, 2))
            model.eval()
            with torch.no_grad():
                for x, *_ in val_loader:
                    x = x.to(device)
                    xhat, *_ = model(x)
                    x = x.cpu(); xhat = xhat.cpu()
                    break
            n_show = min(8, x.size(0))
            for i in range(n_show):
                axes[0, i].imshow(x[i].squeeze(),    cmap="gray", vmin=0, vmax=1)
                axes[0, i].axis("off")
                axes[1, i].imshow(xhat[i].squeeze(), cmap="gray", vmin=0, vmax=1)
                axes[1, i].axis("off")
            plt.suptitle(f"VAE recon @ epoch {epoch+1}  β={current_beta:.2f}")
            plt.tight_layout()
            fig.savefig(os.path.join(result_dir, f"vae_recon_ep{epoch+1}.png"))
            plt.close(fig)
            torch.save(model, os.path.join(result_dir, f"vae_model_ep{epoch+1}.pt"))

    # Save loss curves
    for name, arr in [
        ("vae_train_total", train_losses), ("vae_val_total",   val_losses),
        ("vae_train_recon", train_recon),  ("vae_val_recon",   val_recon),
        ("vae_train_kl",    train_kl),     ("vae_val_kl",      val_kl),
    ]:
        joblib.dump(arr, os.path.join(result_dir, f"{name}.pkl"))

    _save_loss_curves(train_losses, val_losses, epochs,
                      f"VAE Total Loss (β={beta})", result_dir, "vae")
    return model, train_losses, val_losses


# =============================================================================
# 3. Semi-supervised Autoencoder (SemiSupAE)
# =============================================================================

class SemiSupAE(nn.Module):
    """
    Semi-supervised Autoencoder.

    The encoder maps an image to a latent vector z.
    A lightweight classification head maps z → class logits.
    Loss = λ_recon * MSE(recon, x)  +  λ_cls * CE(logits, y)
    where the CE term is applied only on labelled samples.

    Parameters
    ----------
    num_classes   : number of target classes
    latent_dim    : bottleneck dimension
    input_ps      : spatial size of (square) input patch
    no_ch         : number of input channels
    BN_flag       : BatchNorm in conv layers
    dropout_flag  : Dropout in FC layers
    """

    def __init__(
        self,
        num_classes: int,
        latent_dim: int = 16,
        input_ps: int = 32,
        no_ch: int = 1,
        BN_flag: bool = True,
        dropout_flag: bool = False,
    ):
        super().__init__()

        final_size = input_ps // (2 ** 3)
        flat = 128 * final_size * final_size

        def maybe_bn(n):     return nn.BatchNorm2d(n) if BN_flag      else nn.Identity()
        def maybe_dropout(): return nn.Dropout(0.3)   if dropout_flag else nn.Identity()

        # ---------- encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(no_ch, 32, 3, stride=2, padding=1),
            maybe_bn(32), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            maybe_bn(64), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            maybe_bn(128), nn.LeakyReLU(0.01),
            nn.Flatten(),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(flat, 1024), nn.LeakyReLU(0.01),
            maybe_dropout(),
            nn.Linear(1024, latent_dim),
        )

        # ---------- decoder ----------
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.LeakyReLU(0.01),
            maybe_dropout(),
            nn.Linear(1024, flat),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64,   3, stride=2, padding=1, output_padding=1),
            maybe_bn(64), nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32,    3, stride=2, padding=1, output_padding=1),
            maybe_bn(32), nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, no_ch, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

        # ---------- classification head ----------
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        self.final_size  = final_size
        self.num_classes = num_classes

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_fc(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(-1, 128, self.final_size, self.final_size)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        z      = self.encode(x)
        recon  = self.decode(z)
        logits = self.classifier(z)
        return recon, z, logits


def semisup_ae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,    # -1 means unlabelled
    lambda_recon: float = 1.0,
    lambda_cls: float   = 1.0,
):
    """
    Combined reconstruction + classification loss.

    Labelled samples (label >= 0) contribute to both terms.
    Unlabelled samples (label == -1) contribute only to the recon term.

    Returns
    -------
    total_loss, recon_loss, cls_loss
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")

    labelled_mask = labels >= 0
    if labelled_mask.any():
        cls_loss = F.cross_entropy(
            logits[labelled_mask], labels[labelled_mask], reduction="mean"
        )
    else:
        cls_loss = torch.tensor(0.0, device=x.device)

    total = lambda_recon * recon_loss + lambda_cls * cls_loss
    return total, recon_loss, cls_loss


def train_semisup_ae(
    model,
    train_loader,    # yields (x, label, ...)   label=-1 for unlabelled
    val_loader,
    device,
    epochs,
    lr,
    lambda_recon,
    lambda_cls,
    result_dir,
):
    """
    Training loop for SemiSupAE.

    The dataloader must return (x, label, ...) where label is an integer
    class index, or -1 for unlabelled samples.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses   = [], []
    train_recon, train_cls     = [], []
    val_recon,   val_cls       = [], []

    error_print_period = max(1, epochs // 50)
    recon_view_period  = max(1, epochs // 10)

    for epoch in range(epochs):
        model.train()
        tl = tr = tc = 0.0
        for batch in train_loader:
            x      = batch[0].to(device)
            # batch[1] = condition, batch[2] = annotation_label (-1 = unlabelled)
            labels = batch[2].to(device) if len(batch) > 2 else batch[1].to(device)

            recon, _, logits = model(x)
            loss, rl, cl = semisup_ae_loss(
                x, recon, logits, labels,
                lambda_recon=lambda_recon, lambda_cls=lambda_cls,
            )
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tl += loss.item(); tr += rl.item(); tc += cl.item()

        n = len(train_loader)
        tl /= n; tr /= n; tc /= n
        train_losses.append(tl); train_recon.append(tr); train_cls.append(tc)

        model.eval()
        vl = vr = vc = 0.0
        correct = total_labelled = 0
        with torch.no_grad():
            for batch in val_loader:
                x      = batch[0].to(device)
                labels = batch[2].to(device) if len(batch) > 2 else batch[1].to(device)

                recon, _, logits = model(x)
                loss, rl, cl = semisup_ae_loss(
                    x, recon, logits, labels,
                    lambda_recon=lambda_recon, lambda_cls=lambda_cls,
                )
                vl += loss.item(); vr += rl.item(); vc += cl.item()

                # classification accuracy on labelled val samples
                mask = labels >= 0
                if mask.any():
                    preds = logits[mask].argmax(dim=1)
                    correct        += (preds == labels[mask]).sum().item()
                    total_labelled += mask.sum().item()

        n = len(val_loader)
        vl /= n; vr /= n; vc /= n
        val_losses.append(vl); val_recon.append(vr); val_cls.append(vc)

        acc_str = ""
        if total_labelled > 0:
            acc = 100.0 * correct / total_labelled
            acc_str = f"  val_acc={acc:.1f}%"

        if (epoch + 1) % error_print_period == 0:
            print(
                f"SemiSup  epoch {epoch+1}/{epochs} | "
                f"train total={tl:.4f} recon={tr:.4f} cls={tc:.4f} | "
                f"val   total={vl:.4f} recon={vr:.4f} cls={vc:.4f}{acc_str}"
            )

        if (epoch + 1) % recon_view_period == 0:
            fig = plot_reconstruction_progress(model, val_loader, device, epoch + 1)
            fig.savefig(os.path.join(result_dir, f"semisup_recon_ep{epoch+1}.png"))
            plt.close(fig)
            torch.save(model, os.path.join(result_dir, f"semisup_model_ep{epoch+1}.pt"))

    for name, arr in [
        ("ss_train_total", train_losses), ("ss_val_total", val_losses),
        ("ss_train_recon", train_recon),  ("ss_val_recon", val_recon),
        ("ss_train_cls",   train_cls),    ("ss_val_cls",   val_cls),
    ]:
        joblib.dump(arr, os.path.join(result_dir, f"{name}.pkl"))

    _save_loss_curves(train_losses, val_losses, epochs,
                      "SemiSup AE Total Loss", result_dir, "semisup")
    return model, train_losses, val_losses


# =============================================================================
# 4. Contrastive Autoencoder (ContrastiveAE)
# =============================================================================

class ContrastiveAE(nn.Module):
    """
    Contrastive Autoencoder.

    The model uses two views of each image:
      - view 1 : the original (clean) patch
      - view 2 : salt-and-pepper corrupted patch (nuisance variation)

    The encoder is trained to produce embeddings that are *invariant* to the
    noise corruption via an NT-Xent (SimCLR-style) contrastive loss on a
    projection head, while the decoder is still trained to reconstruct the
    *original* image from the clean-view embedding.

    Combined loss:
        L = λ_recon * MSE(decode(z_clean), x_clean)
          + λ_contrast * NT-Xent(proj(z_clean), proj(z_noisy))

    Parameters
    ----------
    latent_dim   : bottleneck dimension
    proj_dim     : output dimension of the projection head (for contrastive loss)
    input_ps     : spatial size of (square) input patch
    no_ch        : number of input channels
    noise_prob   : salt-and-pepper noise probability for the second view
    BN_flag      : BatchNorm in conv layers
    """

    def __init__(
        self,
        latent_dim: int  = 16,
        proj_dim: int    = 64,
        input_ps: int    = 32,
        no_ch: int       = 1,
        noise_prob: float = 0.05,
        BN_flag: bool    = True,
    ):
        super().__init__()

        final_size = input_ps // (2 ** 3)
        flat = 128 * final_size * final_size
        self.noise_prob = noise_prob

        def maybe_bn(n): return nn.BatchNorm2d(n) if BN_flag else nn.Identity()

        # ---------- shared encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(no_ch, 32, 3, stride=2, padding=1),
            maybe_bn(32), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            maybe_bn(64), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            maybe_bn(128), nn.LeakyReLU(0.01),
            nn.Flatten(),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(flat, 1024), nn.LeakyReLU(0.01),
            nn.Linear(1024, latent_dim),
        )

        # ---------- decoder (reconstruction from clean view) ----------
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.LeakyReLU(0.01),
            nn.Linear(1024, flat),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64,   3, stride=2, padding=1, output_padding=1),
            maybe_bn(64), nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32,    3, stride=2, padding=1, output_padding=1),
            maybe_bn(32), nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, no_ch, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

        # ---------- projection head (for contrastive loss only) ----------
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, proj_dim),
        )

        self.final_size = final_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_fc(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(-1, 128, self.final_size, self.final_size)
        return self.decoder(h)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return self.projector(z)

    def forward(self, x: torch.Tensor):
        """
        Forward pass used at inference time (clean image → recon + latent).
        """
        z    = self.encode(x)
        recon = self.decode(z)
        return recon, z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent (Normalised Temperature-scaled Cross Entropy) loss, SimCLR-style.

    z1, z2 : (B, D) projection vectors for the two views of each sample.
    Positive pair: (z1[i], z2[i]).
    All other pairs within the batch are negatives.
    """
    B = z1.size(0)

    # L2-normalise
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate: shape (2B, D)
    z = torch.cat([z1, z2], dim=0)

    # Pairwise cosine similarity (2B × 2B)
    sim = torch.matmul(z, z.T) / temperature

    # Mask out self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B,     device=z.device),
    ])  # shape (2B,)

    loss = F.cross_entropy(sim, labels)
    return loss


def contrastive_ae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    proj_clean: torch.Tensor,
    proj_noisy: torch.Tensor,
    lambda_recon: float    = 1.0,
    lambda_contrast: float = 0.5,
    temperature: float     = 0.5,
):
    """
    Combined reconstruction + NT-Xent contrastive loss.

    Returns
    -------
    total_loss, recon_loss, contrast_loss
    """
    recon_loss    = F.mse_loss(recon, x, reduction="mean")
    contrast_loss = nt_xent_loss(proj_clean, proj_noisy, temperature)
    total         = lambda_recon * recon_loss + lambda_contrast * contrast_loss
    return total, recon_loss, contrast_loss


def train_contrastive_ae(
    model,
    train_loader,    # yields (x, label, ...)   label not required
    val_loader,
    device,
    epochs,
    lr,
    lambda_recon,
    lambda_contrast,
    result_dir,
    noise_prob: float  = 0.05,
    temperature: float = 0.5,
):
    """
    Training loop for ContrastiveAE.

    For each batch the function creates a noisy view on-the-fly using
    salt-and-pepper noise.  The contrastive loss encourages the encoder
    to produce noise-invariant embeddings.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses     = [], []
    train_recon, train_contrast  = [], []
    val_recon,   val_contrast    = [], []

    error_print_period = max(1, epochs // 50)
    recon_view_period  = max(1, epochs // 10)

    for epoch in range(epochs):
        model.train()
        tl = tr = tc = 0.0

        for batch in train_loader:
            x = batch[0].to(device)

            # --- two views ---
            x_noisy = salt_and_pepper_noise(x, noise_prob=noise_prob)

            z_clean = model.encode(x)
            z_noisy = model.encode(x_noisy)

            recon      = model.decode(z_clean)
            proj_clean = model.project(z_clean)
            proj_noisy = model.project(z_noisy)

            loss, rl, cl = contrastive_ae_loss(
                x, recon, proj_clean, proj_noisy,
                lambda_recon=lambda_recon,
                lambda_contrast=lambda_contrast,
                temperature=temperature,
            )
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tl += loss.item(); tr += rl.item(); tc += cl.item()

        n = len(train_loader)
        tl /= n; tr /= n; tc /= n
        train_losses.append(tl); train_recon.append(tr); train_contrast.append(tc)

        model.eval()
        vl = vr = vc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                x_noisy = salt_and_pepper_noise(x, noise_prob=noise_prob)

                z_clean = model.encode(x)
                z_noisy = model.encode(x_noisy)

                recon      = model.decode(z_clean)
                proj_clean = model.project(z_clean)
                proj_noisy = model.project(z_noisy)

                loss, rl, cl = contrastive_ae_loss(
                    x, recon, proj_clean, proj_noisy,
                    lambda_recon=lambda_recon,
                    lambda_contrast=lambda_contrast,
                    temperature=temperature,
                )
                vl += loss.item(); vr += rl.item(); vc += cl.item()

        n = len(val_loader)
        vl /= n; vr /= n; vc /= n
        val_losses.append(vl); val_recon.append(vr); val_contrast.append(vc)

        if (epoch + 1) % error_print_period == 0:
            print(
                f"Contrastive  epoch {epoch+1}/{epochs} | "
                f"train total={tl:.4f} recon={tr:.4f} contrast={tc:.4f} | "
                f"val   total={vl:.4f} recon={vr:.4f} contrast={vc:.4f}"
            )

        if (epoch + 1) % recon_view_period == 0:
            fig = plot_reconstruction_progress(model, val_loader, device, epoch + 1)
            fig.savefig(os.path.join(result_dir, f"contrastive_recon_ep{epoch+1}.png"))
            plt.close(fig)

            # Also visualise clean vs noisy vs recon side-by-side
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    x_noisy = salt_and_pepper_noise(x, noise_prob=noise_prob)
                    recon, _ = model(x)
                    x = x.cpu(); x_noisy = x_noisy.cpu(); recon = recon.cpu()
                    break

            n_show = min(6, x.size(0))
            fig2, axes = plt.subplots(3, n_show, figsize=(n_show, 3))
            for i in range(n_show):
                axes[0, i].imshow(x[i].squeeze(),       cmap="gray", vmin=0, vmax=1)
                axes[0, i].axis("off")
                axes[1, i].imshow(x_noisy[i].squeeze(), cmap="gray", vmin=0, vmax=1)
                axes[1, i].axis("off")
                axes[2, i].imshow(recon[i].squeeze(),   cmap="gray", vmin=0, vmax=1)
                axes[2, i].axis("off")
            axes[0, 0].set_ylabel("clean",  fontsize=7)
            axes[1, 0].set_ylabel("noisy",  fontsize=7)
            axes[2, 0].set_ylabel("recon",  fontsize=7)
            plt.suptitle(f"Contrastive AE @ epoch {epoch+1}")
            plt.tight_layout()
            fig2.savefig(os.path.join(result_dir, f"contrastive_views_ep{epoch+1}.png"))
            plt.close(fig2)
            torch.save(model, os.path.join(result_dir, f"contrastive_model_ep{epoch+1}.pt"))

    for name, arr in [
        ("ct_train_total",    train_losses),   ("ct_val_total",    val_losses),
        ("ct_train_recon",    train_recon),    ("ct_val_recon",    val_recon),
        ("ct_train_contrast", train_contrast), ("ct_val_contrast", val_contrast),
    ]:
        joblib.dump(arr, os.path.join(result_dir, f"{name}.pkl"))

    _save_loss_curves(train_losses, val_losses, epochs,
                      "Contrastive AE Total Loss", result_dir, "contrastive")
    return model, train_losses, val_losses


# =============================================================================
# Internal helper
# =============================================================================

def _save_loss_curves(train_losses, val_losses, epochs, title, result_dir, prefix):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), train_losses, label="Train")
    plt.plot(range(epochs), val_losses,   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(title); plt.legend()
    fig.savefig(os.path.join(result_dir, f"{prefix}_train_val_loss.png"))
    plt.close(fig)