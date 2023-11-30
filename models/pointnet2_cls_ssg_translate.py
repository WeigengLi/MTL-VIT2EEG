import torch.nn as nn
import torch.nn.functional as F

# PointNet2模型的主要架构
class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super(get_model, self).__init__()
        # 根据是否使用法线通道确定输入通道数
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        # 定义三个点集抽象层
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        # 定义三个全连接层
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    # 前向传播函数
    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        # 逐层应用点集抽象层和全连接层
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points

# 定义模型的损失函数类
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    # 前向传播函数
    def forward(self, pred, target, trans_feat):
        # 使用负对数似然损失计算预测值和目标值之间的差异
        total_loss = F.nll_loss(pred, target)
        return total_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

# 计时函数，用于测量代码段的执行时间
def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

# 点云规范化函数，将点云规范化到单位球内
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

# 计算点之间的欧几里得距离的平方
def square_distance(src, dst):
    """
    计算每对点之间的欧几里得距离。

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    输入:
        src: 源点集，[B, N, C]
        dst: 目标点集，[B, M, C]
    输出:
        dist: 每个点对的平方距离，[B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

# 点索引函数，用于根据索引选择点集中的点
def index_points(points, idx):
    """
    输入:
        points: 输入点数据，[B, N, C]
        idx: 采样索引数据，[B, S]
    返回:
        new_points: 索引后的点数据，[B, S, C]
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

# 最远点采样函数，用于从点云中选择一组离得最远的点
def farthest_point_sample(xyz, npoint):
    """
    输入:
        xyz: 点云数据，[B, N, 3]
        npoint: 采样点数
    返回:
        centroids: 采样点云索引，[B, npoint]
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

# 查询球形区域内的点
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    输入:
        radius: 局部区域半径
        nsample: 局部区域中的最大采样点数
        xyz: 所有点，[B, N, 3]
        new_xyz: 查询点，[B, S, 3]
    返回:
        group_idx: 分组点索引，[B, S, nsample]
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

# 采样和分组函数
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    输入:
        npoint: 采样点数
        radius: 局部区域半径
        nsample: 局部区域中的最大采样点数
        xyz: 输入点的位置数据，[B, N, 3]
        points: 输入点的数据，[B, N, D]
    返回:
        new_xyz: 采样点的位置数据，[B, npoint, nsample, 3]
        new_points: 采样点的数据，[B, npoint, nsample, 3+D]
    """
    B, N, C =
