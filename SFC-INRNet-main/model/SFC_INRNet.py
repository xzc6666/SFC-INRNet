import torch
import torch.nn as nn
import torchvision.models as models

# 导入我们已经分块写好的三大核心创新模块
from .AFER import AFER_Module
from .MSHN import SFC_MSHN_Neck
from .FAID import FAID_Stage


class SFC_INRNet(nn.Module):
    """
    Spatial-Frequency Collaborative Implicit Representation Network (SFC-INRNet)
    Official Implementation.
    """

    def __init__(self, L=10):
        super(SFC_INRNet, self).__init__()

        # ==========================================
        # 1. Dual-Domain Collaborative Encoder
        # ==========================================
        # 1.1 Spatial Backbone (Using standard ResNet-34)
        resnet = models.resnet34(pretrained=True)
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # Stage 1 [64]
        self.layer1 = resnet.layer1  # Stage 2 [64]
        self.layer2 = resnet.layer2  # Stage 3 [128]
        self.layer3 = resnet.layer3  # Stage 4 [256]
        self.layer4 = resnet.layer4  # Stage 5 [512]

        # 1.2 Adaptive Frequency Representation Branch (AFERs)
        self.afer1 = AFER_Module(64)
        self.afer2 = AFER_Module(64)
        self.afer3 = AFER_Module(128)
        self.afer4 = AFER_Module(256)
        self.afer5 = AFER_Module(512)

        # ==========================================
        # 2. Multi-Scale Heterogeneous Neck (MSHN)
        # ==========================================
        channel_list = [64, 64, 128, 256, 512]
        self.mshn = SFC_MSHN_Neck(channel_list)

        # ==========================================
        # 3. Frequency-Aware Implicit Decoder (FAID)
        # ==========================================
        # Cascaded FAID stages to progressively decode and reconstruct target
        self.faid5 = FAID_Stage(in_channels=512, out_channels=256, L=L)
        self.faid4 = FAID_Stage(in_channels=256, out_channels=128, L=L)
        self.faid3 = FAID_Stage(in_channels=128, out_channels=64, L=L)
        self.faid2 = FAID_Stage(in_channels=64, out_channels=64, L=L)
        self.faid1 = FAID_Stage(in_channels=64, out_channels=32, L=L)

        # Final projection to prediction mask (1 channel)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # ------------------------------------------
        # Step 1: Spatial-Frequency Feature Extraction
        # ------------------------------------------
        # Stage 1
        s1 = self.conv1(x)
        f1 = self.afer1(s1)

        # Stage 2
        s2 = self.layer1(s1)
        f2 = self.afer2(s2, prev_afer_feat=f1)

        # Stage 3
        s3 = self.layer2(s2)
        f3 = self.afer3(s3, prev_afer_feat=f2)

        # Stage 4
        s4 = self.layer3(s3)
        f4 = self.afer4(s4, prev_afer_feat=f3)

        # Stage 5
        s5 = self.layer4(s4)
        f5 = self.afer5(s5, prev_afer_feat=f4)

        # ------------------------------------------
        # Step 2: Multi-Scale Heterogeneous Fusion
        # ------------------------------------------
        spatial_feats = [s1, s2, s3, s4, s5]
        freq_feats = [f1, f2, f3, f4, f5]

        m1, m2, m3, m4, m5 = self.mshn(spatial_feats, freq_feats)

        # ------------------------------------------
        # Step 3: Continuous Implicit Decoding
        # ------------------------------------------
        d5 = self.faid5(m5)  # Decode from deepest semantic layer
        d4 = self.faid4(m4, prev_out=d5)  # Progressive upsampling and fusion
        d3 = self.faid3(m3, prev_out=d4)
        d2 = self.faid2(m2, prev_out=d3)
        d1 = self.faid1(m1, prev_out=d2)

        # ------------------------------------------
        # Step 4: Final Prediction
        # ------------------------------------------
        # Upsample the final feature back to the original image input resolution
        d0 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=True)
        pred_mask = self.final_conv(d0)

        return pred_mask