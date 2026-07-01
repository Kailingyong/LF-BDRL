import torch
from torch import nn
from moco.builder import MoCo
from einops import rearrange



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out


class DLnet(nn.Module):
    def __init__(self, args):
        super(DLnet, self).__init__()
        # Encoder
        self.E = MoCo(base_encoder=Encoder)

    def forward(self, x_query, x_key):

        b, u, v, c, h, w =  x_query.shape
        x_query = rearrange( x_query, 'b u v c h w -> (b u v) c h w')
        x_key = rearrange(x_key, 'b u v c h w -> (b u v) c h w')

        if self.training:
            # degradation-aware represenetion learning
            fea, logits, labels = self.E(x_query,  x_key)
            return fea, logits, labels
        else:
            fea = self.E(x_query, x_key)
            return fea
