from .modules import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv11(nn.Module):
    def __init__(self, num_classes: int = 80):
        super(YOLOv11, self).__init__()
        # Backbone
        self.layer0 = Conv(3, 16, 3, 2, 1)
        self.layer1 = Conv(16, 32, 3, 2, 1)
        self.layer2 = C3k2(32, 64, n=1, c3k=False)
        self.layer3 = Conv(64, 64, 3, 2, 1)
        self.layer4 = C3k2(64, 128, n=1, c3k=False)
        self.layer5 = Conv(128, 128, 3, 2, 1)
        self.layer6 = C3k2(128, 128, n=1, c3k=True)
        self.layer7 = Conv(128, 256, 3, 2, 1)
        self.layer8 = C3k2(256, 256, n=1, c3k=True)
        self.layer9 = SPPF(256, 256)
        self.layer10 = C2PSA(256, 256, n=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer13 = C3k2(384, 128, n=1, c3k=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer16 = C3k2(256, 64, n=1, c3k=False)
        self.layer17 = Conv(64, 64, 3, 2, 1)
        self.layer19 = C3k2(192, 128, n=1, c3k=False)
        self.layer20 = Conv(128, 128, 3, 2, 1)
        self.layer22 = C3k2(384, 256, n=1, c3k=True)

        self.detect = Detect(nc=num_classes, ch=(64, 128, 256))

    def forward(self, x: torch.Tensor):
        # Backbone
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        x9 = self.layer9(x8)
        x10 = self.layer10(x9)

        # Neck up path
        u1 = self.upsample1(x10)
        c1 = torch.cat([u1, x6], dim=1)
        x13 = self.layer13(c1)

        u2 = self.upsample2(x13)
        u2 = self._match_hw(u2, x4)
        c2 = torch.cat([u2, x4], dim=1)         # 128 + 128 = 256
        x16 = self.layer16(c2)                  # /8 64ch (P3)

        # Neck down path
        d1 = self.layer17(x16)                  # /16 64ch
        c3 = torch.cat([d1, x13], dim=1)        # 64 + 128 = 192
        x19 = self.layer19(c3)                  # /16 128ch (P4)

        d2 = self.layer20(x19)                  # /32 128ch
        c4 = torch.cat([d2, x10], dim=1)        # 128 + 256 = 384
        x22 = self.layer22(c4)                  # /32 256ch (P5)

        # Detect head expects list of feature maps [small, medium, large]
        return self.detect([x16, x19, x22])

    @staticmethod
    def _match_hw(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode='nearest')
        return x