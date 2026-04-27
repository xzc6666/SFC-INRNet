import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MFPE(nn.Module):
    """
    Multi-Frequency Positional Encoding (MFPE).
    Maps continuous 2D spatial coordinates into a high-dimensional frequency space
    to overcome the spectral bias of standard MLPs.
    """

    def __init__(self, L=10):
        super(MFPE, self).__init__()
        self.L = L

    def forward(self, coords):
        """
        coords: [B, 2, H, W] representing (x, y) coordinates normalized to [-1, 1].
        """
        b, _, h, w = coords.size()
        coords_flat = coords.view(b, 2, -1)  # [B, 2, N], where N = H * W

        encoded_feats = []
        # \Psi(x) = [sin(2^0 \pi x), cos(2^0 \pi x), ..., sin(2^{L-1} \pi x), cos(2^{L-1} \pi x)]
        for i in range(self.L):
            freq = (2 ** i) * math.pi
            encoded_feats.append(torch.sin(freq * coords_flat))
            encoded_feats.append(torch.cos(freq * coords_flat))

        # Concatenate along the channel dimension -> 2 (x,y) * 2 (sin,cos) * L
        encoded_seq = torch.cat(encoded_feats, dim=1)  # [B, 4L, N]
        return encoded_seq


class FAID_Stage(nn.Module):
    """
    Frequency-Aware Implicit Decoder Stage.
    Reformulates discrete upsampling as continuous function regression.
    """

    def __init__(self, in_channels, out_channels, L=10):
        super(FAID_Stage, self).__init__()
        self.mfpe = MFPE(L=L)

        # MLP for point-wise continuous mapping: f(x, y) -> target response
        # Input dim: Feature channels (in_channels) + Positional encoding (4 * L)
        mlp_in_dim = in_channels + 4 * L
        self.mlp = nn.Sequential(
            nn.Conv1d(mlp_in_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, out_channels, 1)
        )

    def gen_coordinates(self, h, w, device):
        """Generates continuous normalized coordinates in [-1, 1]."""
        y_steps = torch.linspace(-1, 1, h, device=device)
        x_steps = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(y_steps, x_steps, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]
        return coords

    def forward(self, m_i, prev_out=None):
        b, c, h, w = m_i.size()

        # 1. Feature Aggregation and Nearest-Neighbor Upsampling (NNUP)
        if prev_out is not None:
            # Upsample preceding decoded feature to current resolution
            d_up = F.interpolate(prev_out, size=(h, w), mode='nearest')
            f_in = m_i + d_up
        else:
            f_in = m_i

        # We assume f_in is the target upscaled feature F_up as per Fig 5.
        f_up = f_in
        f_seq = f_up.view(b, c, -1)  # Flatten: [B, C, N]

        # 2. Continuous Coordinate Grid and MFPE
        coords = self.gen_coordinates(h, w, m_i.device).expand(b, -1, -1, -1)
        gamma_seq = self.mfpe(coords)  # [B, 4L, N]

        # 3. Feature Concatenation
        joint_feat = torch.cat([f_seq, gamma_seq], dim=1)  # [B, C + 4L, N]

        # 4. Implicit MLP Decoding
        out_seq = self.mlp(joint_feat)  # [B, out_channels, N]

        # 5. Reshape back to native spatial geometry
        out_spatial = out_seq.view(b, -1, h, w)
        return out_spatial