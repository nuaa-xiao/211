import torch.nn as nn
import torch
import torch.nn.functional as F
from ..pointnet2.pointnet2_modules import PointNetFeaturePropagation

class Decoder(nn.Module):
    def __init__(self, num_class):
        super(Decoder, self).__init__()
        self.num_class = num_class
        #feature propagation

        self.fp1 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=320, mlp=[256, 128])
        self.fp3 = PointNetFeaturePropagation(in_channel=192, mlp=[128, 128])
        self.fp4 = PointNetFeaturePropagation(in_channel=128+additional_channel, mlp=[128, 64])

        self.conv1 = nn.Conv1d(64, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(32, num_class, 1)

    def forward(self, xyz, xyz1, f1, xyz2, f2, xyz3, f3, xyz4, f4):
        #pass through SA modules
        f0 =None
        ff3 =self.fp1(xyz3,xyz4,f3,f4)
        ff2 =self.fp2(xyz2,xyz3,f2,ff3)
        ff1 =self.fp3(xyz1,xyz2,f1,ff2)
        ff0 =self.fp4(xyz,xyz1, f0, ff1)

        # FC layers
        x = self.drop1(F.relu(self.bn1(self.conv1(ff0))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, f4

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss
