import torch
from lib.pointnet2.pointnet2_modules import PointnetSAModuleMSG, PointnetFPModule

class Encoder(torch.nn.Module):
    '''Extracts point-wise features from point cloud'''
    def __init__(self):
        super().__init__()

        #for SA modules, the first element of the mlp should be the number of input features (ignoring the xyz, which is added automatically). For the following SA modules, the number of input features is computed as the sum (from concatenation) of the features from the previous level (last index)
        self.sa1 = PointnetSAModuleMSG(npoint=1024, radii=[1.6], nsamples=[48], mlps=[[0,16,32,64]]) #Output features dim: 64
        self.sa2 = PointnetSAModuleMSG(npoint=512, radii=[3.2], nsamples=[32], mlps=[[64,64,64]])     #Output features dim: 64
        self.sa3 = PointnetSAModuleMSG(npoint=256, radii=[6.4], nsamples=[32], mlps=[[64,128,128]]) #Output features dim: 128
        self.sa4 = PointnetSAModuleMSG(npoint=256, radii=[10], nsamples=[8], mlps=[[128,128,256,256]]) #Output features dim: 256


    def forward(self, pts):

        '''
        :param pts: input points with dims [B,N,F], F>=3
        :return:
            xyz: sampled coordinates of input [B,npoint[-1],3]
            features: extracted features per point [B, nfeat[-1], npoint[-1]]. nfeat[-1] is the sum of last mlps idx from sa3
        '''
        assert len(pts.shape) == 3, 'pts must have shape [B,N,F]'
        assert pts.size(2) >= 3, 'pts must have shape [B,N,F] with F >= 3'

        #pass through SA modules
        f = None
        xyz = pts.contiguous()
        xyz1, f1 = self.sa1(xyz, f)
        xyz2, f2 = self.sa2(xyz1, f1)
        xyz3, f3 = self.sa3(xyz2, f2)
        xyz4, f4 = self.sa4(xyz3, f3)


        return xyz, xyz1, f1, xyz2, f2, xyz3, f3, xyz4, f4
