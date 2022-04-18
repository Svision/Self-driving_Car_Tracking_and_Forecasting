from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor, nn


def compute_l1_loss(targets: Tensor, predictions: Tensor) -> Tensor:
    """Compute the mean absolute error (MAE)/L1 loss between `predictions` and `targets`.

    Specifically, the l1-weighted MSE loss can be computed as follows:
    1. Compute a binary mask of the `targets` that are not NaN, and apply it to the `targets` and `predictions`
    2. Compute the MAE loss between `predictions` and `targets`.
        This should give us a [batch_size * num_actors x T x 2] tensor `l1_loss`.
    3. Compute the mean of `l1_loss`. This gives us our final scalar loss.

    Args:
        targets: A [batch_size * num_actors x T x 2] tensor, containing the ground truth targets.
        predictions: A [batch_size * num_actors x T x 2] tensor, containing the predictions.

    Returns:
        A scalar MAE loss between `predictions` and `targets`
    """
    # TODO: Implement.
    # mask = torch.ones(targets.shape)
    # mask[targets.isnan()] = 0
    # l1_loss = abs(predictions * mask - targets * mask)
    # return torch.nanmean(l1_loss)

    filtered_targets = targets[~torch.any(targets.isnan(), dim=2)]
    filtered_predictions = predictions[~torch.any(targets.isnan(), dim=2)]
    l1_loss = abs(filtered_targets - filtered_predictions)
    return torch.mean(l1_loss)

  
def compute_nll_loss(targets: Tensor, predictions: Tensor) -> Tensor:
    """Compute the negative likelihood loss between `predictions` and `targets`.

    Specifically, the l1-weighted MSE loss can be computed as follows:
    1. filter out nan targets and corresponding predictions
    2. generate the covariance matrix

    Args:
        targets: A [batch_size * num_actors x T x 2] tensor, containing the ground truth targets.
        predictions: A [batch_size * num_actors x T x 5] tensor, containing the predictions.
                    (mu_x, mu_y, rho, sd_x, sd_y)

    Returns:
        A scalar NLL loss between `predictions` and `targets`
    """
    BN, T, _ = targets.shape
    filtered_targets = torch.where(targets.isnan(), torch.tensor(0, dtype=targets.dtype), targets)
    # generate cov matrix
    sd_x = predictions[:, :, 3:4]
    sd_y = predictions[:, :, 4:5]
    rho = predictions[:, :, 2:3] / torch.max(abs(predictions[:, :, 2:3]))  # normalize between [-1, 1]

    cov_row1 = torch.cat((sd_x ** 2, rho * sd_x * sd_y), dim=2)
    cov_row2 = torch.cat((rho * sd_x * sd_y, sd_y ** 2), dim=2)
    cov = torch.stack((cov_row1, cov_row2), dim=2)  # B*N x T x 2 x 2
    var = torch.linalg.det(cov).clamp(min=0)
    nll_fn = nn.GaussianNLLLoss(reduction='mean')
    loss = nll_fn(predictions[..., 0:2], filtered_targets, var)

    return loss


@dataclass
class PredictionLossConfig:
    """Prediction loss function configuration.

    Attributes:
        l1_loss_weight: The multiplicative weight of the L1 loss
    """

    l1_loss_weight: float
    nll_loss_weight: float


@dataclass
class PredictionLossMetadata:
    """Detailed breakdown of the Prediction loss."""

    total_loss: torch.Tensor
    # l1_loss: torch.Tensor
    nll_loss: torch.Tensor


class PredictionLossFunction(torch.nn.Module):
    """A loss function to train a Prediction model."""

    def __init__(self, config: PredictionLossConfig) -> None:
        super(PredictionLossFunction, self).__init__()
        self._l1_loss_weight = config.l1_loss_weight
        self._nll_loss_weight = config.nll_loss_weight

    def forward(
        self, predictions: List[Tensor], targets: List[Tensor]
    ) -> Tuple[torch.Tensor, PredictionLossMetadata]:
        """Compute the loss between the predicted Predictions and target labels.

        Args:
            predictions: A list of batch_size x [num_actors x T x 2] tensor containing the outputs of
                `PredictionModel`.
            targets:  A list of batch_size x [num_actors x T x 2] tensor containing the ground truth output.

        Returns:
            The scalar tensor containing the weighted loss between `predictions` and `targets`.
        """
        predictions_tensor = torch.cat(predictions)
        targets_tensor = torch.cat(targets)

        # 1. Unpack the targets tensor.
        target_centroids = targets_tensor[..., :2]  # [batch_size * num_actors x T x 2]

        # # 2. Unpack the predictions tensor.
        # predicted_centroids = predictions_tensor[
        #     ..., :2
        # ]  # [batch_size * num_actors x T x 2]

        # # 3. Compute individual loss terms for l1
        # l1_loss = compute_l1_loss(target_centroids, predicted_centroids)
        #
        # # 4. Aggregate losses using the configured weights.
        # total_loss = l1_loss * self._l1_loss_weight

        # 2. Unpack the predictions tensor.
        predicted_centroids = predictions_tensor[
                              ..., :5
        ]  # [batch_size * num_actors x T x 5]

        # 3. Compute individual loss terms for l1
        nll_loss = compute_nll_loss(target_centroids, predicted_centroids)

        # 4. Aggregate losses using the configured weights.
        total_loss = nll_loss * self._nll_loss_weight

        loss_metadata = PredictionLossMetadata(total_loss, nll_loss)
        return total_loss, loss_metadata
