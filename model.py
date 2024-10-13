import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.models.attention import GNNAttention
from lib.models.encoder import Encoder
from lib.models.decoder import Decoder


class Displace_Strain(nn.Module):
    def __init__(self):
        super(Displace_Strain, self).__init__()
        self.encoder = Encoder()
        self.gnn = GNNAttention(dim=128, k=32)
        self.decoder = Decoder(num_class=3)

    def forward(self, src, tgt):
        B, N, C = src.shape

        xyz, f = self.encoder(tgt)
        xyz0, xyz1 = xyz[0::2], xyz[1::2]
        f0, f1 = f[0::2], f[1::2]
        npts = xyz0.size(1)

        f0, f1 = self.gnn(xyz0, xyz1, f0, f1)

        f0 = f0/(torch.linalg.norm(f0, dim=-1).unsqueeze(-1)+1e-6)
        f1 = f1/(torch.linalg.norm(f1, dim=-1).unsqueeze(-1)+1e-6)

        displacelment = self.decoder(xyz0, f0, xyz1, f1)
        strain = self.decoder(xyz1, f1, xyz0, f0)

        return displacelment, strain