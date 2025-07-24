import torch

def chamfer_distance(pc1, pc2):
    """
    Compute Chamfer Distance between two point clouds.
    Args:
        pc1: Tensor of shape (N, P1, D), where N=batch size, P1=number of points, D=dimensions.
        pc2: Tensor of shape (N, P2, D), where N=batch size, P2=number of points, D=dimensions.
    Returns:
        dist: Chamfer Distance
    """
    dist1 = torch.cdist(pc1, pc2).min(dim=2)[0]
    dist2 = torch.cdist(pc2, pc1).min(dim=2)[0]
    return dist1.mean() + dist2.mean()

