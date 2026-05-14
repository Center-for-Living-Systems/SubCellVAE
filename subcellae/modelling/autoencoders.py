"""
autoencoder.py
==============
Four autoencoder variants:
  1. AE          – standard convolutional autoencoder
  2. VAE32       – variational AE (standard VAE or beta-VAE via beta parameter)
  3. SemiSupAE   – semi-supervised AE with auxiliary classification head
  4. ContrastiveAE – contrastive AE with rotation × flip geometric augmentation

All models share the same encoder/decoder backbone dimensions and expect
square input patches (default 32×32).

Multi-channel support
---------------------
All models natively support multi-channel inputs — no separate class is needed.
Pass ``no_ch=C`` (AE / SemiSupAE / ContrastiveAE) or ``in_channels=C`` (VAE32)
where C is the number of input channels (e.g. 4 for a 4-channel image).
The first conv layer and last transposed-conv layer automatically adapt.
Use :class:`~subcellae.modelling.dataset.MultiChannelPatchDataset` to load
stacked ``(C, H, W)`` tensors from per-channel patch directories.
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


def intensity_scale(x: torch.Tensor, scale_range: tuple = (0.8, 1.2)) -> torch.Tensor:
    """
    Randomly scale the intensity of each image in a batch by an independent
    factor drawn uniformly from ``scale_range``, then clamp to [0, 1].

    Parameters
    ----------
    x           : (B, C, H, W) tensor, values in [0, 1]
    scale_range : (low, high) multiplicative range; default (0.8, 1.2)
    """
    lo, hi = scale_range
    # One scale factor per image, broadcast over C, H, W
    scale = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(lo, hi)
    return (x * scale).clamp(0.0, 1.0)


def augment_contrastive_view(
    x: torch.Tensor,
    rotation_mode: str = "rot90",
    use_flip: bool = False,
    noise_prob: float = 0.0,
    intensity_scale_range: tuple | None = None,
) -> torch.Tensor:
    """
    Create an augmented view for contrastive learning.

    Geometric augmentation pool (per image, uniformly sampled):
      - rotation_mode="rot90", use_flip=False : 4 transforms
        {0°, 90°, 180°, 270°}
      - rotation_mode="rot90", use_flip=True  : 8 transforms
        all elements of the dihedral group D4 (symmetries of the square):
        {0°, 90°, 180°, 270°} × {no flip, horizontal flip}
        Note: adding a vertical flip option would produce duplicates —
        e.g. R180+H == V flip, R180+V == H flip, R270+H == R90+V, etc.
        Sampling 4 rotations × {none, H flip} covers all 8 D4 elements
        exactly once with no redundancy.

    Optionally followed by intensity scaling and salt-and-pepper noise.

    Parameters
    ----------
    x : (B, C, H, W) clean patches, values in [0, 1]
    rotation_mode :
        "rot90" : random k*90° rotation per image, k in {0,1,2,3} (default)
        "none"  : skip rotation
    use_flip : if True, each image independently gets {no flip, H flip}
               after rotation, covering all 8 D4 symmetries uniformly
    noise_prob : fraction of pixels corrupted by salt-and-pepper noise;
                 0.0 disables noise entirely
    intensity_scale_range : optional (low, high) multiplicative intensity
                            range applied per image after geometry ops
    """
    if rotation_mode not in {"none", "rot90"}:
        raise ValueError(f"Unsupported rotation_mode: {rotation_mode}")

    out = x.clone()
    bsz = out.size(0)

    # ── Rotation: 0°, 90°, 180°, 270° ────────────────────────────────────────
    if rotation_mode == "rot90":
        ks = torch.randint(0, 4, (bsz,), device=out.device)
        rotated = out.clone()
        for k in range(4):
            mask = ks == k
            if mask.any():
                rotated[mask] = torch.rot90(out[mask], k=int(k), dims=(-2, -1))
        out = rotated

    # ── Flip: {no flip, horizontal flip} — covers all 8 D4 elements with ─────
    # ── rotation, with no duplicates                                       ─────
    if use_flip:
        h_mask = torch.rand(bsz, device=out.device) > 0.5
        if h_mask.any():
            out[h_mask] = out[h_mask].flip(-1)

    # ── Intensity scaling (optional) ──────────────────────────────────────────
    if intensity_scale_range is not None:
        out = intensity_scale(out, scale_range=intensity_scale_range)

    # ── Salt-and-pepper noise (optional) ─────────────────────────────────────
    if noise_prob > 0:
        out = salt_and_pepper_noise(out, noise_prob=noise_prob)

    return out


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

    def _to_display(t):
        """Return a 2-D (H, W) array for imshow; picks channel 0 if multi-channel."""
        return t[0] if t.shape[0] > 1 else t.squeeze()

    n = min(16, x.size(0))
    idx = torch.randperm(x.size(0))[:n]
    fig, axes = plt.subplots(2, n, figsize=(n, 2))
    for col, i in enumerate(idx):
        axes[0, col].imshow(_to_display(x[i]), cmap="gray", vmin=0, vmax=1)
        axes[0, col].axis("off")
        axes[1, col].imshow(_to_display(recon[i]), cmap="gray", vmin=0, vmax=1)
        axes[1, col].axis("off")
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
            nn.Sigmoid(),  # output in (0,1); always has non-zero gradient (avoids Hardtanh dead zone on sparse patches)
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


# ---------------------------------------------------------------------------
# LR scheduler helpers (shared by train_ae, train_vae, etc.)
# ---------------------------------------------------------------------------

def _make_scheduler(optimizer, mode, epochs, patience, factor, lr_min):
    """Return a scheduler instance or None when mode is 'none'."""
    if mode == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, min_lr=lr_min,
        )
    if mode == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr_min,
        )
    return None  # "none"


def _step_scheduler(scheduler, mode, val_loss):
    """Step the scheduler; plateau needs the metric, cosine does not."""
    if scheduler is None:
        return
    if mode == "plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()


def train_ae(model, train_loader, val_loader, device, epochs, lr,
             loss_norm_flag, result_dir,
             weight_decay=0.0,
             lr_scheduler="none", lr_scheduler_patience=20,
             lr_scheduler_factor=0.5, lr_min=1e-6):
    """Training loop for the standard AE."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = normalized_mse if loss_norm_flag else nn.MSELoss()

    scheduler = _make_scheduler(optimizer, lr_scheduler, epochs,
                                lr_scheduler_patience, lr_scheduler_factor, lr_min)

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

        _step_scheduler(scheduler, lr_scheduler, v_loss)

        if (epoch + 1) % error_print_period == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"AE  epoch {epoch+1}/{epochs}  train={t_loss:.4f}  val={v_loss:.4f}  lr={current_lr:.2e}")

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
    lr_scheduler="none", lr_scheduler_patience=20,
    lr_scheduler_factor=0.5, lr_min=1e-6,
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
    scheduler = _make_scheduler(optimizer, lr_scheduler, epochs,
                                lr_scheduler_patience, lr_scheduler_factor, lr_min)

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

        _step_scheduler(scheduler, lr_scheduler, vl)

        if (epoch + 1) % error_print_period == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"VAE  epoch {epoch+1}/{epochs}  β={current_beta:.2f} | "
                f"train total={tl:.4f} recon={tr:.4f} kl={tk:.4f} | "
                f"val   total={vl:.4f} recon={vr:.4f} kl={vk:.4f}  lr={current_lr:.2e}"
            )

        if (epoch + 1) % recon_view_period == 0:
            model.eval()
            with torch.no_grad():
                for x, *_ in val_loader:
                    x = x.to(device)
                    xhat, *_ = model(x)
                    x = x.cpu(); xhat = xhat.cpu()
                    break
            n_show = min(16, x.size(0))
            idx = torch.randperm(x.size(0))[:n_show]
            fig, axes = plt.subplots(2, n_show, figsize=(n_show, 2))
            for col, i in enumerate(idx):
                axes[0, col].imshow(x[i].squeeze(),    cmap="gray", vmin=0, vmax=1)
                axes[0, col].axis("off")
                axes[1, col].imshow(xhat[i].squeeze(), cmap="gray", vmin=0, vmax=1)
                axes[1, col].axis("off")
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
        num_classes_2: int = 0,
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
            nn.Sigmoid(),  # output in (0,1); always has non-zero gradient (avoids Hardtanh dead zone on sparse patches)
        )

        # ---------- classification head (primary) ----------
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        # ---------- classification head (secondary, optional) ----------
        self.num_classes_2 = num_classes_2
        if num_classes_2 > 0:
            self.classifier2 = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes_2),
            )
        else:
            self.classifier2 = None

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

    def forward_dual(self, x: torch.Tensor):
        """Forward pass returning logits for both classifier heads.

        Only valid when the model was constructed with ``num_classes_2 > 0``.
        Returns ``(recon, z, logits, logits2)``.
        """
        z       = self.encode(x)
        recon   = self.decode(z)
        logits  = self.classifier(z)
        logits2 = self.classifier2(z)
        return recon, z, logits, logits2


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


def semisup_ae_loss_dual(
    x: torch.Tensor,
    recon: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,    # -1 = unlabelled
    logits2: torch.Tensor,
    labels2: torch.Tensor,   # -1 = unlabelled
    lambda_recon: float = 1.0,
    lambda_cls: float   = 1.0,
    lambda_cls2: float  = 1.0,
):
    """Combined reconstruction + dual classification loss.

    Identical to :func:`semisup_ae_loss` but adds a second CE term for a
    second set of labels (e.g. Position).  Each CE term is computed only on
    the patches that carry that label.

    Returns
    -------
    total_loss, recon_loss, cls_loss, cls_loss2
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")

    mask1 = labels >= 0
    cls_loss = (
        F.cross_entropy(logits[mask1], labels[mask1], reduction="mean")
        if mask1.any() else torch.tensor(0.0, device=x.device)
    )

    mask2 = labels2 >= 0
    cls_loss2 = (
        F.cross_entropy(logits2[mask2], labels2[mask2], reduction="mean")
        if mask2.any() else torch.tensor(0.0, device=x.device)
    )

    total = lambda_recon * recon_loss + lambda_cls * cls_loss + lambda_cls2 * cls_loss2
    return total, recon_loss, cls_loss, cls_loss2


def train_semisup_ae(
    model,
    train_loader,    # yields (x, condition, label, label2, path)  label=-1 = unlabelled
    val_loader,
    device,
    epochs,
    lr,
    lambda_recon,
    lambda_cls,
    result_dir,
    lambda_cls2: float = 0.0,            # >0 activates the second classification head
    weight_decay: float = 1e-4,          # L2 regularisation on all weights
    early_stopping_patience: int = 0,    # 0 = disabled; stop when val doesn't improve
    min_epochs_for_best: int = 200,      # ignore best-checkpoint tracking before this epoch
    warmup_epochs: int = 200,            # epochs of recon-only before adding cls loss
):
    """
    Training loop for SemiSupAE.

    The dataloader must return ``(x, condition, label, label2, path)`` where
    each label is an integer class index, or ``-1`` for unlabelled samples.

    When ``lambda_cls2 > 0`` and the model has a second head (``model.classifier2``),
    the dual loss :func:`semisup_ae_loss_dual` is used, incorporating both
    FA-type and Position labels simultaneously.

    Regularisation / stability
    --------------------------
    weight_decay : float
        Adam L2 weight decay.  Helps prevent encoder/decoder overfitting.
    early_stopping_patience : int
        Stop training when val loss has not improved for this many consecutive
        epochs, then restore the weights from the best epoch.  Set to 0 to
        disable early stopping (full ``epochs`` are always run).
    min_epochs_for_best : int
        Best-checkpoint tracking does not start until this epoch is reached.
        Prevents saving a degenerate early model as the "best".
        Set to 0 to track from the very first epoch.
    warmup_epochs : int
        Number of epochs to train with reconstruction loss only (classification
        weights forced to 0).  After warmup, the configured ``lambda_cls`` /
        ``lambda_cls2`` values are restored.  Default 200.
    A ``ReduceLROnPlateau`` scheduler (factor 0.5, patience 10) is active
    during phase 2 only.  It is skipped during warmup to prevent the
    reconstruction plateau from draining the LR before classification starts.
    At the warmup→phase-2 transition the LR is reset to its original value
    and the scheduler is restarted fresh.
    """
    dual_mode = lambda_cls2 > 0 and getattr(model, "classifier2", None) is not None

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
    )

    train_losses, val_losses   = [], []
    train_recon, train_cls     = [], []
    val_recon,   val_cls       = [], []
    train_cls2,  val_cls2      = [], []

    best_val_loss = float("inf")
    best_state    = None
    best_epoch    = 0
    no_improve    = 0

    error_print_period = max(1, epochs // 50)
    recon_view_period  = max(1, epochs // 10)

    for epoch in range(epochs):
        # Two-phase warmup: reconstruction-only for the first warmup_epochs epochs
        in_warmup = warmup_epochs > 0 and epoch < warmup_epochs
        eff_lambda_cls  = 0.0 if in_warmup else lambda_cls
        eff_lambda_cls2 = 0.0 if in_warmup else lambda_cls2

        model.train()
        tl = tr = tc = tc2 = 0.0
        for batch in train_loader:
            x      = batch[0].to(device)
            labels = batch[2].to(device)

            if dual_mode:
                labels2 = batch[3].to(device)
                recon, _, logits, logits2 = model.forward_dual(x)
                loss, rl, cl, cl2 = semisup_ae_loss_dual(
                    x, recon, logits, labels, logits2, labels2,
                    lambda_recon=lambda_recon,
                    lambda_cls=eff_lambda_cls,
                    lambda_cls2=eff_lambda_cls2,
                )
            else:
                recon, _, logits = model(x)
                loss, rl, cl = semisup_ae_loss(
                    x, recon, logits, labels,
                    lambda_recon=lambda_recon, lambda_cls=eff_lambda_cls,
                )
                cl2 = 0.0

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tl += loss.item(); tr += rl.item(); tc += cl.item() if hasattr(cl, "item") else cl
            tc2 += cl2.item() if hasattr(cl2, "item") else cl2

        n = len(train_loader)
        tl /= n; tr /= n; tc /= n; tc2 /= n
        train_losses.append(tl); train_recon.append(tr)
        train_cls.append(tc);    train_cls2.append(tc2)

        model.eval()
        vl = vr = vc = vc2 = 0.0
        correct = total_labelled = 0
        correct2 = total_labelled2 = 0
        with torch.no_grad():
            for batch in val_loader:
                x      = batch[0].to(device)
                labels = batch[2].to(device)

                if dual_mode:
                    labels2 = batch[3].to(device)
                    recon, _, logits, logits2 = model.forward_dual(x)
                    loss, rl, cl, cl2 = semisup_ae_loss_dual(
                        x, recon, logits, labels, logits2, labels2,
                        lambda_recon=lambda_recon,
                        lambda_cls=eff_lambda_cls,
                        lambda_cls2=eff_lambda_cls2,
                    )
                    mask2 = labels2 >= 0
                    if mask2.any():
                        correct2        += (logits2[mask2].argmax(1) == labels2[mask2]).sum().item()
                        total_labelled2 += mask2.sum().item()
                else:
                    recon, _, logits = model(x)
                    loss, rl, cl = semisup_ae_loss(
                        x, recon, logits, labels,
                        lambda_recon=lambda_recon, lambda_cls=eff_lambda_cls,
                    )
                    cl2 = 0.0

                vl += loss.item(); vr += rl.item(); vc += cl.item() if hasattr(cl, "item") else cl
                vc2 += cl2.item() if hasattr(cl2, "item") else cl2

                mask = labels >= 0
                if mask.any():
                    correct        += (logits[mask].argmax(1) == labels[mask]).sum().item()
                    total_labelled += mask.sum().item()

        n = len(val_loader)
        vl /= n; vr /= n; vc /= n; vc2 /= n
        val_losses.append(vl); val_recon.append(vr)
        val_cls.append(vc);    val_cls2.append(vc2)

        acc_str = ""
        if total_labelled > 0:
            acc_str += f"  val_acc1={100.*correct/total_labelled:.1f}%"
        if total_labelled2 > 0:
            acc_str += f"  val_acc2={100.*correct2/total_labelled2:.1f}%"

        # LR scheduler: skip during warmup so recon plateau doesn't drain the LR.
        # At the warmup→phase-2 transition, reset LR to the original value so the
        # classifier head can actually learn.
        if not in_warmup:
            scheduler.step(vl)
        elif epoch + 1 == warmup_epochs:
            # First epoch of phase 2: reset LR and scheduler state
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
            )
            print(f"SemiSup  warmup complete — LR reset to {lr:.2e}, scheduler restarted")
        current_lr = optimizer.param_groups[0]["lr"]

        # Best-checkpoint tracking (only after min_epochs_for_best) and early stopping
        if (epoch + 1) >= min_epochs_for_best and vl < best_val_loss:
            best_val_loss = vl
            best_epoch    = epoch + 1
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if (epoch + 1) % error_print_period == 0:
            cls2_str = f" cls2={tc2:.4f}/{vc2:.4f}" if dual_mode else ""
            phase_str = "[warmup] " if in_warmup else ""
            print(
                f"SemiSup  {phase_str}epoch {epoch+1}/{epochs} | "
                f"train total={tl:.4f} recon={tr:.4f} cls={tc:.4f}{cls2_str.replace('/', ' | val cls2=')} | "
                f"val   total={vl:.4f} recon={vr:.4f} cls={vc:.4f}{acc_str} | "
                f"lr={current_lr:.2e} best_ep={best_epoch}"
            )

        if early_stopping_patience > 0 and no_improve >= early_stopping_patience:
            print(
                f"SemiSup  early stopping at epoch {epoch+1} "
                f"(no val improvement for {early_stopping_patience} epochs; "
                f"best val={best_val_loss:.6f} at epoch {best_epoch})"
            )
            break

        if (epoch + 1) % recon_view_period == 0:
            fig = plot_reconstruction_progress(model, val_loader, device, epoch + 1)
            fig.savefig(os.path.join(result_dir, f"semisup_recon_ep{epoch+1}.png"))
            plt.close(fig)
            torch.save(model, os.path.join(result_dir, f"semisup_model_ep{epoch+1}.pt"))

    # Restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"SemiSup  restored best model from epoch {best_epoch} "
              f"(val loss {best_val_loss:.6f})")

    save_arrays = [
        ("ss_train_total", train_losses), ("ss_val_total", val_losses),
        ("ss_train_recon", train_recon),  ("ss_val_recon", val_recon),
        ("ss_train_cls",   train_cls),    ("ss_val_cls",   val_cls),
    ]
    if dual_mode:
        save_arrays += [("ss_train_cls2", train_cls2), ("ss_val_cls2", val_cls2)]
    for name, arr in save_arrays:
        joblib.dump(arr, os.path.join(result_dir, f"{name}.pkl"))

    actual_epochs = len(train_losses)
    _save_loss_curves(train_losses, val_losses, actual_epochs,
                      "SemiSup AE Total Loss", result_dir, "semisup")
    _save_semisup_component_curves(
        train_recon, val_recon,
        train_cls,   val_cls,
        train_cls2,  val_cls2,
        result_dir,
        dual_mode=dual_mode,
    )
    return model, train_losses, val_losses


# =============================================================================
# 4. Contrastive Autoencoder (ContrastiveAE)
# =============================================================================

class ContrastiveAE(nn.Module):
    """
    Contrastive Autoencoder.

    The model uses two views of each image:
      - view 1 : the original (clean) patch
      - view 2 : an augmented patch, typically a rotated version of the same patch

    The encoder is trained to produce embeddings that are tolerant to
    orientation changes via an NT-Xent (SimCLR-style) contrastive loss on a
    projection head, while the decoder is still trained to reconstruct the
    original image from the clean-view embedding.

    Combined loss:
        L = λ_recon * MSE(decode(z_clean), x_clean)
          + λ_contrast * NT-Xent(proj(z_clean), proj(z_aug))

    Parameters
    ----------
    latent_dim   : bottleneck dimension
    proj_dim     : output dimension of the projection head (for contrastive loss)
    input_ps     : spatial size of (square) input patch
    no_ch        : number of input channels
    noise_prob   : legacy attribute kept for compatibility; augmentation is
                   configured in the training loop
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
            nn.Sigmoid(),  # output in (0,1); always has non-zero gradient (avoids Hardtanh dead zone on sparse patches)
        )

        # ---------- projection head (for contrastive loss only) ----------
        proj_hidden = latent_dim * 4  # scales with representation size
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, proj_dim),
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


def supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    Hybrid Supervised Contrastive loss (SupCon-style).

    Expects a 2N-sized batch formed by stacking the clean and augmented views:
        z      = cat([z_clean, z_aug], dim=0)   shape (2N, D)
        labels = cat([orig_labels, orig_labels]) shape (2N,)

    For **labeled** anchors (label >= 0):
        positives = all other samples in the batch sharing the same label
        (this automatically includes the paired augmented view).
    For **unlabeled** anchors (label == -1):
        positives = only the paired augmented view (equivalent to NT-Xent).

    Parameters
    ----------
    z      : (2N, D) projection vectors (will be L2-normalised internally)
    labels : (2N,) integer class IDs; -1 marks unlabeled
    """
    N2 = z.size(0)      # = 2 * batch_size
    B  = N2 // 2
    z  = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature    # (2N, 2N)

    # Mask self-similarities so they are excluded from the denominator
    self_mask = torch.eye(N2, dtype=torch.bool, device=z.device)
    sim_no_self = sim.masked_fill(self_mask, float("-inf"))

    # --- Build positive mask ---
    # Self-supervised fallback: (i, i+B) and (i+B, i) for each sample
    ss_mask = torch.zeros(N2, N2, dtype=torch.bool, device=z.device)
    idx = torch.arange(B, device=z.device)
    ss_mask[idx, idx + B] = True
    ss_mask[idx + B, idx] = True

    # Supervised: same class, both labeled, not self
    lab = labels.unsqueeze(1)                              # (2N, 1)
    both_labeled = (labels >= 0).unsqueeze(1) & (labels >= 0).unsqueeze(0)
    sup_mask = (lab == lab.T) & ~self_mask & both_labeled  # (2N, 2N)

    # Select: labeled anchors use sup_mask, unlabeled use ss_mask
    labeled_anchor = (labels >= 0).unsqueeze(1).expand(N2, N2)
    pos_mask = torch.where(labeled_anchor, sup_mask, ss_mask)

    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return z.sum() * 0.0   # safe zero that keeps the computation graph

    # SupCon loss:  L_i = -1/|P_i| * sum_{p in P_i}(sim_ip) + log_denom_i
    log_denom   = torch.logsumexp(sim_no_self, dim=1)               # (2N,)
    n_pos       = pos_mask.float().sum(dim=1).clamp(min=1)          # (2N,)
    pos_sim_sum = (sim * pos_mask.float()).sum(dim=1)                # (2N,)
    loss_per    = -pos_sim_sum / n_pos + log_denom                   # (2N,)

    return loss_per[has_pos].mean()


def contrastive_ae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    proj_clean: torch.Tensor,
    proj_aug: torch.Tensor,
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
    contrast_loss = nt_xent_loss(proj_clean, proj_aug, temperature)
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
    rotation_mode: str           = "rot90",
    noise_prob: float            = 0.0,
    temperature: float           = 0.5,
    use_flip: bool               = False,
    intensity_scale_range: tuple | None = None,
):
    """
    Training loop for ContrastiveAE (self-supervised NT-Xent).

    For each batch two views are created on-the-fly:
      - View 1 (clean)     : the original patch
      - View 2 (augmented) : typically a rotated version of the same patch

    The contrastive loss (NT-Xent) encourages the encoder to produce
    embeddings that keep morphologically similar patches close even when the
    same structure appears at different orientations.

    Parameters
    ----------
    rotation_mode         : augmentation used to define positive pairs;
                            default "rot90" for random 90° rotations
    noise_prob            : optional salt-and-pepper corruption probability
    intensity_scale_range : optional per-image intensity scaling range
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
            x_aug = augment_contrastive_view(x, rotation_mode=rotation_mode, use_flip=use_flip, noise_prob=noise_prob, intensity_scale_range=intensity_scale_range)

            z_clean = model.encode(x)
            z_aug = model.encode(x_aug)

            recon      = model.decode(z_clean)
            proj_clean = model.project(z_clean)
            proj_aug = model.project(z_aug)

            loss, rl, cl = contrastive_ae_loss(
                x, recon, proj_clean, proj_aug,
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
                x_aug = augment_contrastive_view(x, rotation_mode=rotation_mode, use_flip=use_flip, noise_prob=noise_prob, intensity_scale_range=intensity_scale_range)

                z_clean = model.encode(x)
                z_aug = model.encode(x_aug)

                recon      = model.decode(z_clean)
                proj_clean = model.project(z_clean)
                proj_aug = model.project(z_aug)

                loss, rl, cl = contrastive_ae_loss(
                    x, recon, proj_clean, proj_aug,
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

            # Also visualise clean vs augmented vs recon side-by-side
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    x_aug = augment_contrastive_view(x, rotation_mode=rotation_mode, use_flip=use_flip, noise_prob=noise_prob, intensity_scale_range=intensity_scale_range)
                    recon, _ = model(x)
                    x = x.cpu(); x_aug = x_aug.cpu(); recon = recon.cpu()
                    break

            n_show = min(16, x.size(0))
            idx = torch.randperm(x.size(0))[:n_show]
            fig2, axes = plt.subplots(3, n_show, figsize=(n_show, 3))
            for col, i in enumerate(idx):
                axes[0, col].imshow(x[i].squeeze(),     cmap="gray", vmin=0, vmax=1)
                axes[0, col].axis("off")
                axes[1, col].imshow(x_aug[i].squeeze(), cmap="gray", vmin=0, vmax=1)
                axes[1, col].axis("off")
                axes[2, col].imshow(recon[i].squeeze(), cmap="gray", vmin=0, vmax=1)
                axes[2, col].axis("off")
            axes[0, 0].set_ylabel("clean",  fontsize=7)
            axes[1, 0].set_ylabel("aug",    fontsize=7)
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


def train_supervised_contrastive_ae(
    model,
    train_loader,   # yields (x, condition, annotation_label, path)
    val_loader,
    device,
    epochs,
    lr,
    lambda_recon,
    lambda_contrast,
    result_dir,
    rotation_mode: str = "rot90",
    noise_prob: float  = 0.0,
    temperature: float = 0.5,
    use_flip: bool     = False,
    intensity_scale_range: tuple | None = None,
):
    """
    Training loop for ContrastiveAE with Supervised Contrastive loss (SupCon).

    Two views per image are created on-the-fly (typically random 90° rotations,
    optionally with flips / noise / intensity scaling). For each mini-batch the
    2N-sized projection set is built by
    concatenating both views.  The SupCon loss pulls together:
      - same-class patches from different images (when labels are available)
      - each image with its own augmented view (fallback for unlabeled patches)

    Reconstruction loss is computed on the clean view only.

    Parameters
    ----------
    noise_prob   : salt-and-pepper corruption probability for the augmented view
    temperature  : softmax temperature for the SupCon loss
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses    = [], []
    train_recon, train_contrast = [], []
    val_recon,   val_contrast   = [], []

    error_print_period = max(1, epochs // 50)
    recon_view_period  = max(1, epochs // 10)

    for epoch in range(epochs):
        model.train()
        tl = tr = tc = 0.0

        for batch in train_loader:
            x      = batch[0].to(device)
            labels = batch[2].to(device)   # annotation_label; -1 if unlabeled

            x_aug = augment_contrastive_view(x, rotation_mode=rotation_mode, use_flip=use_flip, noise_prob=noise_prob, intensity_scale_range=intensity_scale_range)

            z1 = model.encode(x)
            z2 = model.encode(x_aug)

            recon      = model.decode(z1)
            proj1      = model.project(z1)
            proj2      = model.project(z2)

            # 2N-sized batch for SupCon loss
            proj_all   = torch.cat([proj1, proj2], dim=0)              # (2N, D)
            labels_all = torch.cat([labels, labels], dim=0)            # (2N,)

            rl = F.mse_loss(recon, x, reduction="mean")
            cl = supcon_loss(proj_all, labels_all, temperature=temperature)
            loss = lambda_recon * rl + lambda_contrast * cl

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tl += loss.item(); tr += rl.item(); tc += cl.item()

        n = len(train_loader)
        tl /= n; tr /= n; tc /= n
        train_losses.append(tl); train_recon.append(tr); train_contrast.append(tc)

        model.eval()
        vl = vr = vc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x      = batch[0].to(device)
                labels = batch[2].to(device)
                x_aug  = augment_contrastive_view(x, rotation_mode=rotation_mode, use_flip=use_flip, noise_prob=noise_prob, intensity_scale_range=intensity_scale_range)

                z1 = model.encode(x)
                z2 = model.encode(x_aug)

                recon      = model.decode(z1)
                proj1      = model.project(z1)
                proj2      = model.project(z2)

                proj_all   = torch.cat([proj1, proj2], dim=0)
                labels_all = torch.cat([labels, labels], dim=0)

                rl = F.mse_loss(recon, x, reduction="mean")
                cl = supcon_loss(proj_all, labels_all, temperature=temperature)
                loss = lambda_recon * rl + lambda_contrast * cl

                vl += loss.item(); vr += rl.item(); vc += cl.item()

        n = len(val_loader)
        vl /= n; vr /= n; vc /= n
        val_losses.append(vl); val_recon.append(vr); val_contrast.append(vc)

        if (epoch + 1) % error_print_period == 0:
            print(
                f"SupCon AE  epoch {epoch+1}/{epochs} | "
                f"train total={tl:.4f} recon={tr:.4f} contrast={tc:.4f} | "
                f"val   total={vl:.4f} recon={vr:.4f} contrast={vc:.4f}"
            )

        if (epoch + 1) % recon_view_period == 0:
            fig = plot_reconstruction_progress(model, val_loader, device, epoch + 1)
            fig.savefig(os.path.join(result_dir, f"supcon_recon_ep{epoch+1}.png"))
            plt.close(fig)

            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x     = batch[0].to(device)
                    x_aug = augment_contrastive_view(x, rotation_mode=rotation_mode, use_flip=use_flip, noise_prob=noise_prob, intensity_scale_range=intensity_scale_range)
                    recon, _ = model(x)
                    x = x.cpu(); x_aug = x_aug.cpu(); recon = recon.cpu()
                    break

            n_show = min(16, x.size(0))
            idx = torch.randperm(x.size(0))[:n_show]
            fig2, axes = plt.subplots(3, n_show, figsize=(n_show, 3))
            for col, i in enumerate(idx):
                axes[0, col].imshow(x[i].squeeze(),     cmap="gray", vmin=0, vmax=1)
                axes[0, col].axis("off")
                axes[1, col].imshow(x_aug[i].squeeze(), cmap="gray", vmin=0, vmax=1)
                axes[1, col].axis("off")
                axes[2, col].imshow(recon[i].squeeze(), cmap="gray", vmin=0, vmax=1)
                axes[2, col].axis("off")
            axes[0, 0].set_ylabel("clean", fontsize=7)
            axes[1, 0].set_ylabel("aug",   fontsize=7)
            axes[2, 0].set_ylabel("recon", fontsize=7)
            plt.suptitle(f"SupCon AE @ epoch {epoch+1}")
            plt.tight_layout()
            fig2.savefig(os.path.join(result_dir, f"supcon_views_ep{epoch+1}.png"))
            plt.close(fig2)
            torch.save(model, os.path.join(result_dir, f"supcon_model_ep{epoch+1}.pt"))

    for name, arr in [
        ("sc_train_total",    train_losses),   ("sc_val_total",    val_losses),
        ("sc_train_recon",    train_recon),    ("sc_val_recon",    val_recon),
        ("sc_train_contrast", train_contrast), ("sc_val_contrast", val_contrast),
    ]:
        joblib.dump(arr, os.path.join(result_dir, f"{name}.pkl"))

    _save_loss_curves(train_losses, val_losses, epochs,
                      "SupCon AE Total Loss", result_dir, "supcon")
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


def _save_semisup_component_curves(
    train_recon, val_recon,
    train_cls,   val_cls,
    train_cls2,  val_cls2,
    result_dir,
    dual_mode: bool = False,
):
    """Save per-component loss curves (recon / cls1 / cls2) for SemiSupAE.

    Produces a single figure with one subplot per component so the relative
    magnitudes and dynamics are easy to compare side-by-side.
    """
    epochs = len(train_recon)
    xs = range(epochs)

    n_panels = 3 if dual_mode else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=False)

    # --- Reconstruction ---
    ax = axes[0]
    ax.plot(xs, train_recon, label="Train")
    ax.plot(xs, val_recon,   label="Val")
    ax.set_title("Reconstruction loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    # --- Classification head 1 ---
    ax = axes[1]
    ax.plot(xs, train_cls, label="Train")
    ax.plot(xs, val_cls,   label="Val")
    ax.set_title("CLS-1 loss (FA type)" if dual_mode else "CLS loss")
    ax.set_xlabel("Epoch")
    ax.legend()

    # --- Classification head 2 (dual mode only) ---
    if dual_mode:
        ax = axes[2]
        ax.plot(xs, train_cls2, label="Train")
        ax.plot(xs, val_cls2,   label="Val")
        ax.set_title("CLS-2 loss (Position)")
        ax.set_xlabel("Epoch")
        ax.legend()

    fig.suptitle("SemiSup AE – component losses", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "semisup_component_losses.png"), dpi=150)
    plt.close(fig)