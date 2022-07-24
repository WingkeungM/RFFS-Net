import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rffsnet_util import *


class RFFSNet(nn.Module):
    def __init__(self, input_channels=4, use_xyz=True, num_classes=5, dropout=0.6):
        super(RFFSNet, self).__init__()
        if use_xyz:
            C_in = input_channels + 3
        self.num_classes = num_classes
        self.sa1 = DensitySetAbstraction(npoint=1024, nsample=32, in_channel=C_in,
                                         mlp=[64, 64, 128], bandwidth=1, group_all=False)
        self.sa2 = DensitySetAbstraction(npoint=256, nsample=32, in_channel=128 + 3,
                                         mlp=[128, 128, 256], bandwidth=2, group_all=False)
        self.sa3 = DensitySetAbstraction(npoint=128, nsample=32, in_channel=256 + 3,
                                         mlp=[256, 256, 512], bandwidth=4, group_all=False)

        self.up3 = FPModule(mlp=[512 + 256, 512])
        self.up2 = FPModule(mlp=[512 + 128, 384])
        self.up1 = FPModule(mlp=[384 + input_channels, 128])

        self.dagfusion = DAGFusion(512, 512, 16, 16, [1, 2, 4, 8])
        self.multidecoder = MultiDecoder(inchannel=[512, 512, 384], num_classes=num_classes,
                                         dropout=dropout, mid_channel=[64, 128])

        self.fc1 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Conv1d(128, 64, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Conv1d(64, num_classes, kernel_size=1, bias=True)

    def forward(self, points, labels):
        B, _, _ = points.shape
        xyz, features = self._break_up_pc(points)

        l1_xyz, l1_points, l1_labels = self.sa1(xyz, features, labels)
        l2_xyz, l2_points, l2_labels = self.sa2(l1_xyz, l1_points, l1_labels)
        l3_xyz, l3_points, l3_labels = self.sa3(l2_xyz, l2_points, l2_labels)
        decoder_labels = [l1_labels, l2_labels, l3_labels]

        l3_points = self.dagfusion(l3_xyz.permute(0, 2, 1).contiguous(), l3_points.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()

        l2_points = self.up3(l2_xyz.transpose(1, 2).contiguous(), l3_xyz.transpose(1, 2).contiguous(),
                             l2_points.contiguous(), l3_points.contiguous())
        l1_points = self.up2(l1_xyz.transpose(1, 2).contiguous(), l2_xyz.transpose(1, 2).contiguous(),
                             l1_points.contiguous(), l2_points.contiguous())
        up_features = self.up1(xyz.transpose(1, 2).contiguous(), l1_xyz.transpose(1, 2).contiguous(),
                               features.contiguous(), l1_points.contiguous())

        decoder_out = self.multidecoder([l3_points, l2_points, l1_points])

        x = self.drop1(F.relu(self.bn1(self.fc1(up_features))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x.transpose(1, 2).contiguous(), decoder_out, decoder_labels

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].transpose(1, 2).contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    num_classes = 9
    inputs = torch.rand(2, 4096, 8)
    inputs = inputs.cuda()
    model = RFFSNet(num_classes=num_classes)
    model = model.cuda()

    out = model(inputs)
    print(out.size())
    print("Done!")
