import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class EFBE(nn.Module):
    """
    Explicit Frequency Band Enhancement (EFBE) mechanism.
    Partitions the amplitude spectrum into N concentric rings and applies
    adaptive power-law transformations.
    """

    def __init__(self, num_rings=4):
        super(EFBE, self).__init__()
        self.num_rings = num_rings
        # MLP to dynamically predict adaptive weights 'w' based on frequency distribution
        self.mlp = nn.Sequential(
            nn.Linear(num_rings, num_rings * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_rings * 2, num_rings),
            nn.Softplus()  # Ensures weights are positive for power-law transformation
        )

    def get_ring_masks(self, h, w, device):
        """Generates N concentric ring masks for frequency partitioning."""
        # Create coordinate grid
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        mesh_y, mesh_x = torch.meshgrid(y, x, indexing='ij')
        # Calculate Euclidean distance from center (0, 0)
        dist = torch.sqrt(mesh_x ** 2 + mesh_y ** 2)

        masks = []
        # Divide into N concentric regions based on distance from center
        thresholds = torch.linspace(0, dist.max(), self.num_rings + 1)
        for i in range(self.num_rings):
            mask = (dist >= thresholds[i]) & (dist < thresholds[i + 1])
            masks.append(mask.float())
        return torch.stack(masks)  # [N, H, W]

    def forward(self, amplitude):
        b, c, h, w = amplitude.size()
        masks = self.get_ring_masks(h, w, amplitude.device)  # [N, H, W]

        # 1. Ring-wise Global Average Pooling (GAP)
        # amplitude: [B, 1, H, W], masks: [N, H, W]
        ring_energies = []
        for i in range(self.num_rings):
            # Energy in current frequency band
            energy = (amplitude * masks[i]).sum(dim=(2, 3)) / (masks[i].sum() + 1e-6)
            ring_energies.append(energy)

        ring_energies = torch.cat(ring_energies, dim=1)  # [B, N]

        # 2. Dynamic Weight Prediction via MLP
        weights = self.mlp(ring_energies)  # [B, N]

        # 3. Band-wise Power-law Transformation
        enhanced_amplitude = torch.zeros_like(amplitude)
        epsilon = 1e-6  # Numerical stability constant
        for i in range(self.num_rings):
            # Apply: A' = (A_ring + eps) ^ w_i
            w_i = weights[:, i].view(b, 1, 1, 1)
            band_enhanced = torch.pow(amplitude * masks[i] + epsilon, w_i)
            enhanced_amplitude += band_enhanced * masks[i]

        return enhanced_amplitude


class AFER_Module(nn.Module):
    """
    Adaptive Frequency Enhancement and Reconstruction (AFER) Module.
    Extracts high-frequency priors and propagates them across stages.
    """

    def __init__(self, in_channels, num_rings=4):
        super(AFER_Module, self).__init__()
        # Compact spatial feature map extraction via pooling
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        # EFBE for adaptive spectral modulation
        self.efbe = EFBE(num_rings=num_rings)

        # Refinement after attention
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Cross-stage alignment conv
        self.align_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, prev_afer_feat=None):
        """
        Args:
            x: Input spatial feature Fi from current stage.
            prev_afer_feat: Feature from the preceding AFER module (Ffi-1).
        """
        identity = x
        b, c, h, w = x.size()

        # --- Stage 1: Spectrum Extraction ---
        # Generate compact spatial map Fg via channel pooling logic
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        fg = avg_out + max_out  # [B, 1, H, W]

        # 2D FFT and centering shift
        spectrum = torch.fft.fft2(fg)
        spectrum_shifted = torch.fft.fftshift(spectrum)

        # Decouple Amplitude (A) and Phase (P)
        amplitude = torch.abs(spectrum_shifted)
        phase = torch.angle(spectrum_shifted)

        # --- Stage 2: Explicit Frequency Band Enhancement (EFBE) ---
        enhanced_amplitude = self.efbe(amplitude)

        # Recombine to form reconstructed complex spectrum S'
        # Following Euler's formula: S' = A' * exp(j * P)
        recon_spectrum_shifted = enhanced_amplitude * torch.exp(1j * phase)

        # Inverse shift and 2D IFFT back to spatial domain
        recon_spectrum = torch.fft.ifftshift(recon_spectrum_shifted)
        spatial_response = torch.fft.ifft2(recon_spectrum).real

        # Frequency-guided spatial attention mask Mi
        mask = torch.sigmoid(spatial_response)

        # --- Stage 3: Enhanced Reconstruction ---
        # Feature accentuation and residual refinement
        x_en = identity * mask
        x_en = self.refine(x_en) + identity

        # Cross-stage Propagation: Ffi = Conv(Ffi-1) + Fen
        if prev_afer_feat is not None:
            # Spatially align preceding features if dimensions differ (handled by architecture)
            out = self.align_conv(prev_afer_feat) + x_en
        else:
            out = x_en

        return out