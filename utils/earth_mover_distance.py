import torch
import numpy as np
import ot


def earth_mover_distance(source, target, transpose=True):
    """
    source: Tensor, [B, N, D] - 点云 1
    target: Tensor, [B, M, D] - 点云 2
    transpose: bool, 是否需要转置
    """
    if transpose:
        source = source.permute(0, 2, 1)
        target = target.permute(0, 2, 1)

    # 计算 pair-wise 的距离矩阵
    cost_matrix = torch.cdist(source, target, p=2)  # 欧几里得距离
    assignment = torch.argmin(cost_matrix, dim=2)  # 简单分配策略

    # 计算 EMD
    emd_loss = cost_matrix.gather(2, assignment.unsqueeze(-1)).mean()
    return emd_loss


def wasserstein_distance_3d_batched(point_clouds1, point_clouds2, transpose=False):
    distances = []

    for i in range(point_clouds1.shape[0]):
        # Convert PyTorch tensors to numpy arrays
        X = point_clouds1[i].numpy()
        Y = point_clouds2[i].numpy()

        if transpose:
            X = np.transpose(X)
            Y = np.transpose(Y)

        # Compute the Wasserstein distance using the POT library
        # Define the cost matrix as the Euclidean distance matrix
        cost_matrix = ot.dist(X, Y, metric='euclidean')

        # Compute the Wasserstein distance using the EMD method
        wasserstein_distance = ot.emd2([], [], cost_matrix)
        distances.append(wasserstein_distance)

    return distances

