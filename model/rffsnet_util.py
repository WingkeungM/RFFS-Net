import math
import torch

import torch.nn as nn
import torch.nn.functional as F
import pointnet2_ops_lib.pointnet2_ops.pointnet2_utils as pointnet2_utils


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    :param src: source points, [B, N, C]
    :param dst: target points, [B, M, C]
    :return: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    :param points: input points data, [B, N, C]
    :param idx: sample index data, [B, S]
    :return: indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    :param xyz: pointcloud data, [B, N, C]
    :param npoint: number of samples
    :return: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    :param radius: local region radius
    :param nsample: max sample number in local region
    :param xyz: all points, [B, N, C]
    :param new_xyz: query points, [B, S, C]
    :return: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    :param nsample: max sample number in local region
    :param xyz: all points, [B, N, C]
    :param new_xyz:query points, [B, S, C]
    :return: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)

    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def sample_and_group(npoint, nsample, xyz, points, density_scale=None, labels=None):
    """

    :param npoint: number of sampling points
    :param nsample: number of KNN points
    :param xyz: input points position data, [B, N, C]
    :param points: features, [B, N, D]
    :param density_scale:
    :return:
        new_xyz: xyz after fps, [B, 1, C]
        new_points: features of KNN POINTS, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)

    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)   # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if labels is not None:
        new_labels = pointnet2_utils.gather_operation(labels.unsqueeze(dim=1).float(), fps_idx.int())

    if density_scale is None:
        if labels is not None:
            return new_xyz, new_points, grouped_xyz_norm, idx, new_labels
        else:
            return new_xyz, new_points, grouped_xyz_norm, idx
    else:
        grouped_density = index_points(density_scale, idx)
        if labels is not None:
            return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density, new_labels
        else:
            return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density


def sample_and_group_all(xyz, points, density_scale=None):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    new_xyz = xyz.mean(dim=1, keepdim=True)
    grouped_xyz = xyz.view(B, 1, N, C) - new_xyz.view(B, 1, 1, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz
    else:
        grouped_density = density_scale.view(B, 1, N, 1)
        return new_xyz, new_points, grouped_xyz, grouped_density


def group(nsample, xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)     # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)    # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm


def compute_density(xyz, bandwidth):
    """
    :param xyz: input points position data, [B, N, C]
    :param bandwidth: standard deviation
    :return:
    """
    sqrdists = square_distance(xyz, xyz)
    gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = gaussion_density.mean(dim=-1)

    return xyz_density


class DensityNet(nn.Module):
    def __init__(self, hidden_unit=[8, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList() 

        self.mlp_convs.append(nn.Conv1d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv1d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv1d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm1d(1))

    def forward(self, xyz_density):
        density_scale = xyz_density.unsqueeze(1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            density_scale = bn(conv(density_scale))

            if i == len(self.mlp_convs):
                density_scale = F.sigmoid(density_scale) + 0.5
            else:
                density_scale = F.relu(density_scale)
        
        return density_scale


class WeightNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights = F.relu(bn(conv(weights)))

        return weights


class DensitySetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all):
        super(DensitySetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.bandwidth = bandwidth
        self.group_all = group_all
        self.densitynet = DensityNet()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])

    def forward(self, xyz, points, labels):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        xyz_density = compute_density(xyz, self.bandwidth)
        density_scale = self.densitynet(xyz_density)

        if self.group_all:
            new_xyz, new_points, grouped_xyz_norm, grouped_density = \
                sample_and_group_all(xyz, points, density_scale.view(B, N, 1))
        else:
            new_xyz, new_points, grouped_xyz_norm, _, grouped_density, new_labels = \
                sample_and_group(self.npoint, self.nsample, xyz, points, density_scale.view(B, N, 1), labels)

        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        grouped_xyz = grouped_xyz * grouped_density.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2),
                                  other=weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = F.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points, new_labels.squeeze(dim=1).long()


class GraphDilatedKNN(nn.Module):
    def __init__(self, sample, dilated_rate, step=4):
        super(GraphDilatedKNN, self).__init__()
        self.sample = sample
        self.step = step
        self.dilated_rate = dilated_rate
        self.sample_num = int(self.sample // self.step) * (self.dilated_rate - 1 + self.step) \
                          + math.ceil((self.sample // self.step - int(self.sample // self.step))
                                      * (self.dilated_rate - 1 + self.step))

    def forward(self, xyz, feature=None):
        xyz_knn_ids = pointnet2_utils.ball_query(16, self.sample_num, xyz, xyz)  # b,n,sample*rate
        xyz_dilated_ids = torch.empty((xyz_knn_ids.shape[0], xyz_knn_ids.shape[1], 0)).int().cuda()
        for i in range(math.ceil(self.sample_num // (self.dilated_rate - 1 + self.step))):
            idx_temp = xyz_knn_ids[:, :,
                       ((i + 1) * (self.dilated_rate - 1) + i * self.step):(i + 1) * (
                               self.dilated_rate - 1 + self.step)]
            if i == math.ceil(self.sample_num // (self.dilated_rate - 1 + self.step)) - 1:
                idx_temp = xyz_knn_ids[:, :, ((i + 1) * (self.dilated_rate - 1) + i * self.step):]
            xyz_dilated_ids = torch.cat((xyz_dilated_ids, idx_temp), dim=-1)

        dilated_xyz = pointnet2_utils.grouping_operation(xyz.permute(0, 2, 1).contiguous(), xyz_dilated_ids)

        if feature is not None:
            dilated_feature = pointnet2_utils.grouping_operation(feature.permute(0, 2, 1).contiguous(), xyz_dilated_ids)
            return dilated_xyz, dilated_feature
        else:
            return dilated_xyz


class AnnularDilatedKNN(nn.Module):
    def __init__(self, sample, dilated_rate):
        super(AnnularDilatedKNN, self).__init__()
        self.sample = sample
        self.dilated_rate = dilated_rate

    def forward(self, xyz, feature=None):
        xyz_knn_ids = pointnet2_utils.ball_query(16, self.sample * self.dilated_rate, xyz, xyz)  # b,n,sample*rate
        xyz_dilated_ids_center = xyz_knn_ids[:, :, 0].unsqueeze(-1)
        if self.dilated_rate == 1:
            xyz_dilated_ids = xyz_knn_ids[:, :, 0:self.sample]
        else:
            xyz_dilated_ids_region = xyz_knn_ids[:, :,
                                     (self.dilated_rate - 1) * self.sample: self.dilated_rate * self.sample - 1]
            xyz_dilated_ids = torch.cat((xyz_dilated_ids_center, xyz_dilated_ids_region), dim=-1)
        dilated_xyz = pointnet2_utils.grouping_operation(xyz.permute(0, 2, 1).contiguous(), xyz_dilated_ids)
        if feature is not None:
            dilated_feature = pointnet2_utils.grouping_operation(feature.permute(0, 2, 1).contiguous(), xyz_dilated_ids)
            return dilated_xyz, dilated_feature
        else:
            return dilated_xyz


class DilatedGraphConv(nn.Module):
    def __init__(self, inchannel, outchannel, k, dilated_rate=1, step=4):
        super(DilatedGraphConv, self).__init__()
        self.dilated_rate = dilated_rate
        self.sample = k
        self.step = step
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.dilated_knn = GraphDilatedKNN(self.sample, self.dilated_rate)
        self.conv = nn.Conv2d(self.inchannel, self.outchannel, 1)
        self.bn = nn.BatchNorm2d(self.outchannel)

    def forward(self, xyz, features):
        grouped_dilated_xyz, grouped_dilated_features = self.dilated_knn(xyz, features)
        grouped_dilated_features -= features.permute(0, 2, 1).unsqueeze(dim=-1)

        grouped_dilated_features = F.relu(self.bn(self.conv(grouped_dilated_features)))
        features = torch.max(grouped_dilated_features, dim=-1)[0].permute(0, 2, 1)
        return features


class AnnularDilatedConv(nn.Module):
    def __init__(self, inchannel, outchannel, k, dilated_rate=1):
        super(AnnularDilatedConv, self).__init__()
        self.dilated_rate = dilated_rate
        self.sample = k
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.dilated_knn = AnnularDilatedKNN(self.sample, self.dilated_rate)
        self.conv = nn.Conv2d(self.inchannel, self.outchannel, 1)
        self.bn = nn.BatchNorm2d(self.outchannel)

    def forward(self, xyz, features):
        grouped_dilated_xyz, grouped_dilated_features = self.dilated_knn(xyz, features)
        grouped_dilated_features -= features.permute(0, 2, 1).unsqueeze(dim=-1)

        grouped_dilated_features = F.relu(self.bn(self.conv(grouped_dilated_features)))
        features = torch.max(grouped_dilated_features, dim=-1)[0].permute(0, 2, 1)
        return features


class DAGFusion(nn.Module):
    def __init__(self, inchannel, outchannel, k1, k2, dilated_rate_list):
        super(DAGFusion, self).__init__()
        self.sample1 = k1
        self.sample2 = k2
        self.dilated_list = dilated_rate_list

        self.dgconv_list = nn.ModuleList([
            DilatedGraphConv(inchannel + i * (outchannel // 4), outchannel // 4, self.sample1, self.dilated_list[i], step=4)
            for i in range(len(self.dilated_list))
        ])

        self.adconv_list = nn.ModuleList([
            AnnularDilatedConv(inchannel + j * (outchannel // 4), outchannel // 4, self.sample2, self.dilated_list[j])
            for j in range(len(self.dilated_list))
        ])

        self.conv = nn.Conv1d(outchannel * 2, outchannel, 1)
        self.bn = nn.BatchNorm1d(outchannel)

    def forward(self, xyz, features):
        feat_g = features
        feat_graph_list = []
        for dgconv in self.dgconv_list:
            feat_graph = dgconv(xyz, feat_g)
            feat_g = torch.cat((feat_g, feat_graph), dim=-1)
            feat_graph_list.append(feat_graph)
        feat_graph = torch.cat(feat_graph_list, dim=-1)

        feat_a = features
        feat_ann_list = []
        for adconv in self.adconv_list:
            feat_annular = adconv(xyz, feat_a)
            feat_a = torch.cat((feat_a, feat_annular), dim=-1)
            feat_ann_list.append(feat_annular)
        feat_ann = torch.cat(feat_ann_list, dim=-1)

        feat_fusion = torch.cat((feat_graph, feat_ann), dim=-1).permute(0, 2, 1)
        feat_fusion = F.relu(self.bn(self.conv(feat_fusion))).permute(0, 2, 1)
        return feat_fusion


class Decoder(nn.Module):
    def __init__(self, inchannel, num_classes, dropout=0.6, mid_channel=[64, 128]):
        super(Decoder, self).__init__()
        self.inchannel = inchannel
        self.mid_channel = mid_channel
        self.num_classes = num_classes

        self.drop = nn.Dropout(dropout)

        self.conv1 = nn.Sequential(nn.Conv1d(self.inchannel, self.mid_channel[0], 1),
                                   nn.BatchNorm1d(self.mid_channel[0]))
        self.conv2 = nn.Sequential(nn.Conv1d(self.inchannel, mid_channel[1], 1),
                                   nn.BatchNorm1d(self.mid_channel[1]))
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.mid_channel[0] + self.mid_channel[1], self.num_classes, 1))

    def forward(self, features):
        conv1_out = F.relu(self.conv1(features))
        conv2_out = F.relu((self.conv2(features)))
        conv_out = torch.cat((conv1_out, conv2_out), dim=1)
        conv_out = self.drop(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = F.log_softmax(conv_out, dim=1).permute(0, 2, 1)
        return conv_out


class MultiDecoder(nn.Module):
    def __init__(self, inchannel=[512, 512, 384], num_classes=6, dropout=0.6, mid_channel=[64, 128]):
        super(MultiDecoder, self).__init__()
        self.inchannel = inchannel
        self.mid_channel = mid_channel
        self.num_classes = num_classes

        self.decoder_list = nn.ModuleList([
            Decoder(self.inchannel[i], self.num_classes, dropout=dropout, mid_channel=mid_channel)
            for i in range(len(self.inchannel))
        ])

    def forward(self, points_list):
        decoder_out_list = []
        for i, decoder in enumerate(self.decoder_list):
            decoder_out = decoder(points_list[i])
            decoder_out_list.append(decoder_out)
        return decoder_out_list


class FPModule(nn.Module):
    def __init__(self, mlp):
        super(FPModule, self).__init__()

        self.mlp = nn.Sequential()
        for i in range(len(mlp) - 1):
            self.mlp.add_module("conv_" + str(i), nn.Conv2d(mlp[i], mlp[i + 1], kernel_size=1, bias=False))
            self.mlp.add_module("bn_" + str(i), nn.BatchNorm2d(mlp[i + 1]))
            self.mlp.add_module("relu_" + str(i), nn.ReLU(inplace=True))

    def forward(self, unknown, known, unknow_feats, known_feats):
        """

        :param unknown: target xyz，(batch, n, 3)
        :param known: input xyz，(batch, m, 3)
        :param unknow_feats: target features，(batch, C1, n)
        :param known_feats: input features，(batch, C2, m)
        :return: (batch, mlp[-1], n)
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*(known_feats.size()[0:2] + [unknown.size(1)]))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)
