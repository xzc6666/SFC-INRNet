import torch
import torch.nn as nn
import torch.nn.functional as F


class SFC_AFGC(nn.Module):
    """
    Asymmetric Frequency-Guided Cross-Attention (AFGC).
    Applied to shallow layers to efficiently align high-resolution spatial details with frequency priors.
    (Strictly implements Fig. 3 in the SFC-INRNet manuscript)
    """

    def __init__(self, channels, reduction_ratio=4):
        super(SFC_AFGC, self).__init__()
        # Using GroupNorm(1, C) as an exact mathematical equivalent to LayerNorm for [B, C, H, W] tensors
        self.ln_s = nn.GroupNorm(1, channels)
        self.ln_f = nn.GroupNorm(1, channels)

        # Linear projections mapped to 1x1 convolutions
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # Spatial reduction for asymmetric attention (drastically reduces computation)
        self.spatial_reduction = nn.AvgPool2d(reduction_ratio, stride=reduction_ratio)
        self.scale = channels ** -0.5

        # Refinement convolution
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, f_s, f_f):
        b, c, h, w = f_s.size()

        # 1. Query from Spatial features (Full Resolution)
        norm_s = self.ln_s(f_s)
        q = self.q_proj(norm_s).view(b, c, -1).permute(0, 2, 1)  # [B, HW, C]

        # 2. Key and Value from Frequency features (Reduced Resolution)
        norm_f = self.ln_f(f_f)
        reduced_f = self.spatial_reduction(norm_f)
        k = self.k_proj(reduced_f).view(b, c, -1)  # [B, C, HW/r^2]
        v = self.v_proj(reduced_f).view(b, c, -1).permute(0, 2, 1)  # [B, HW/r^2, C]

        # 3. Scaled Dot-Product Cross-Attention
        attn = (q @ k) * self.scale  # [B, HW, HW/r^2]
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # [B, HW, C]
        out = out.permute(0, 2, 1).view(b, c, h, w)  # Rearrange back to 2D

        # 4. Dense Residual Mechanism
        f_hat = f_s + out  # Residual 1
        f_c = self.conv3x3(f_hat)  # Local Refinement

        # Final output aggregation
        return f_s + f_hat + f_c


class SFC_CGTF(nn.Module):
    """
    Cross-Guided Top-K Fusion Module (CGTF).
    Applied to deep layers to prune background redundancy and amplify discriminative target semantics.
    (Strictly implements Fig. 4 in the SFC-INRNet manuscript)
    """

    def __init__(self, channels):
        super(SFC_CGTF, self).__init__()
        self.k = channels // 2  # Discards the less informative half (topk_ratio = 0.5)

        # Cross Guidance MLPs
        self.mlp_s = nn.Sequential(
            nn.Linear(self.k, self.k // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.k // 2, self.k)
        )
        self.mlp_f = nn.Sequential(
            nn.Linear(self.k, self.k // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.k // 2, self.k)
        )

        # Multi-scale parallel convolutions
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)

        # Final compression to restore original channel dimension
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, f_s, f_f):
        b, c, h, w = f_s.size()

        # 1. Top-K Ranking and Filtering based on Spatial Intensity
        intensity_s = f_s.mean(dim=(2, 3))  # Global Average Pooling for response intensity
        _, idx_s = torch.topk(intensity_s, self.k, dim=1)
        idx_s_gather = idx_s.view(b, self.k, 1, 1).expand(b, self.k, h, w)
        topk_s = torch.gather(f_s, 1, idx_s_gather)  # Retain top 50% spatial channels

        intensity_f = f_f.mean(dim=(2, 3))
        _, idx_f = torch.topk(intensity_f, self.k, dim=1)
        idx_f_gather = idx_f.view(b, self.k, 1, 1).expand(b, self.k, h, w)
        topk_f = torch.gather(f_f, 1, idx_f_gather)  # Retain top 50% frequency channels

        # 2. Cross-domain Mask Generation
        gap_s = topk_s.mean(dim=(2, 3))  # [B, K]
        gap_f = topk_f.mean(dim=(2, 3))  # [B, K]

        mask_s = torch.sigmoid(self.mlp_s(gap_s)).view(b, self.k, 1, 1)
        mask_f = torch.sigmoid(self.mlp_f(gap_f)).view(b, self.k, 1, 1)

        # 3. Cross Modulation (Spatial semantics suppress Freq noise, Freq priors enhance Spatial targets)
        mod_s = topk_s * mask_f
        mod_f = topk_f * mask_s

        # 4. Multi-scale Aggregation
        concat_feat = torch.cat([mod_s, mod_f], dim=1)  # Concatenated to original 'C' channels

        feat_3 = self.conv3x3(concat_feat)
        feat_5 = self.conv5x5(concat_feat)
        aggregated = feat_3 + feat_5

        # 5. Output compression and residual skip connection
        out = self.conv1x1(aggregated)
        return f_s + out


class SFC_LFA(nn.Module):
    """
    Lightweight Feature Aggregation (LFA).
    Applied to intermediate layers to ensure smooth dimensional transitions between domains.
    """

    def __init__(self, channels):
        super(SFC_LFA, self).__init__()
        self.compress = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.smooth = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_s, f_f):
        concat_feat = torch.cat([f_s, f_f], dim=1)
        out = self.compress(concat_feat)
        out = self.relu(self.bn(self.smooth(out)))
        return out


class SFC_MSHN_Neck(nn.Module):
    """
    The Complete Multi-Scale Heterogeneous Neck.
    Orchestrates AFGC, LFA, and CGTF modules across 5 architectural stages.
    """

    def __init__(self, channel_list):
        super(SFC_MSHN_Neck, self).__init__()
        # Assuming channel_list contains the channel dimensions for the 5 ResNet stages
        # e.g., [64, 64, 128, 256, 512] for a standard modified ResNet-34

        # Shallow Layer: Asymmetric Frequency-Guided Cross-Attention
        self.afgc = SFC_AFGC(channel_list[0])

        # Intermediate Layers: Lightweight Feature Aggregation
        self.lfa2 = SFC_LFA(channel_list[1])
        self.lfa3 = SFC_LFA(channel_list[2])
        self.lfa4 = SFC_LFA(channel_list[3])

        # Deep Layer: Cross-Guided Top-K Fusion
        self.cgtf = SFC_CGTF(channel_list[4])

    def forward(self, spatial_feats, freq_feats):
        """
        spatial_feats: List of features [F1, F2, F3, F4, F5] from ResNet backbone.
        freq_feats: List of features [Ff1, Ff2, Ff3, Ff4, Ff5] from AFER branch.
        Returns the multi-scale fused features for the implicit decoder.
        """
        out1 = self.afgc(spatial_feats[0], freq_feats[0])
        out2 = self.lfa2(spatial_feats[1], freq_feats[1])
        out3 = self.lfa3(spatial_feats[2], freq_feats[2])
        out4 = self.lfa4(spatial_feats[3], freq_feats[3])
        out5 = self.cgtf(spatial_feats[4], freq_feats[4])

        return [out1, out2, out3, out4, out5]