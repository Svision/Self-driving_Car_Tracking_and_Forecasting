import torch


def compute_ADE(centroids: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculates the average displacement (L2) error between a set of centroids and a set of labels
    over all timesteps

    Args:
        centroids: N x T x 2 tensor of predicted centroids
        labels: N x T x 2 tensor of actual gt centroids
    """
    valid = ~torch.isnan(labels).any(-1)
    # print(centroids[:, 0:10, :].shape)
    # print(labels.shape)
    # print(torch.norm(centroids[:, 0:10, :] - labels, dim=-1).shape)
    # print(valid.shape)
    # print(torch.norm(centroids[:, 0:10, :] - labels, dim=-1)[valid].shape)
    return (torch.norm(centroids[:, 0:10, :] - labels, dim=-1)[valid]).mean().item()


def compute_FDE(centroids: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculates the final (last timestep) displacement error between a set of centroids and a set of labels

    Args:
        centroids: N x T x 2 tensor of predicted centroids
        labels: N x T x 2 tensor of actual gt centroids
    """
    final_centroids = centroids[:, -1]
    final_labels = labels[:, -1]
    valid = ~torch.isnan(final_labels).any(-1)
    return (torch.norm(final_centroids - final_labels, dim=-1)[valid]).mean().item()


def compute_per_frame_err(
    centroids: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Calculates the per_frame_error

    Args:
        centroids: N x T x 2 tensor of predicted centroids
        labels: N x T x 2 tensor of actual gt centroids
    """
    valid = ~torch.isnan(labels).any(-1).any(-1)
    return (torch.norm(centroids[:, 0:10, :] - labels, dim=-1)[valid]).mean(0)
