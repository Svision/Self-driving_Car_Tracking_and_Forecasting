from turtle import forward
from typing import List, Tuple

import torch
from torch import nn, Tensor

from prediction.model import PredictionModel, PredictionModelConfig
from prediction.utils.reshape import flatten, unflatten_batch
from prediction.utils.transform import transform_using_actor_frame


class InteractiveEncoder(nn.Module):
    ## write the code for the new model here
    def __init__(self, config) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.num_label_timesteps * 3, nhead=8)
        self._net = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, x):
        return self._net(x)


class InteractiveModel(PredictionModel):
    def __init__(self, config: PredictionModelConfig) -> None:
        super().__init__(config)
        self._encoder = InteractiveEncoder(config)

    @staticmethod
    def _preprocess(x_batches: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Preprocess the inputs

        1. Flatten batch and actor dimensions
        2. Transform each actor's history so that its position at the latest timestep is (0, 0) with 0 rad of yaw
            (i.e. it is in actor frame)
        3. Pad nans with zero
        4. Remove the bounding box size from the inputs
        5. Flatten the time and feature dimensions

        Args:
            x_batches (List[Tensor]): List of length batch_size of [N x T x 5] trajectories

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - preprocessed input trajectories [batch_size * N x T * 3]
                - id of each actor's batch in the flattened list [batch_size * N]
                - original position and yaw of each actor at the latest timestep in SDV frame [batch_size * N, 3]
        """
        x, batch_ids = flatten(x_batches)  # [batch_size * N x T x 5]
        original_x_pose = torch.clone(x[:, -1, :3])

        # Move positions to actor frame
        transformed_positions = transform_using_actor_frame(
            x[..., :2], x[:, -1, :3], translate_to=True
        )
        x[..., :2] = transformed_positions
        # Move yaw to actor frame
        x[..., 2] = x[..., 2] - x[:, -1:, 2]

        # Pad nans
        x[x.isnan()] = 0

        # Remove box size
        x = x[..., :3]

        x = x.flatten(1, 2)  # [batch_size * N x T * 3]

        return x, batch_ids, original_x_pose

    def forward(self, x_batches: List[Tensor]) -> List[Tensor]:
        """Perform a forward pass of the model's neural network.

        Args:
            x_batches: A [batch_size x N x T_in x 5] tensor, representing the input history
                centroid, yaw and size in a bird's eye view voxel representation.

        Returns:
            A [batch_size x N x T_out x 2] tensor, representing the future trajectory
                centroid outputs.
        """
        # currently the data is preprocessed to be in the actor frame. This
        # means that the relative position of each agent is lost. You may have
        # to change how to the preprocessing works to account for this
        x, batch_ids, original_x_pose = self._preprocess(x_batches)
        out = self._decoder(self._encoder(x))
        out_batches = self._postprocess(out, batch_ids, original_x_pose)
        return out_batches

