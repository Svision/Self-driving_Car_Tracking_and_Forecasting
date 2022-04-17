from typing import List

import torch
from torch import nn, Tensor
from typing import List, Tuple
from prediction.model import PredictionModel, PredictionModelConfig
from prediction.types import Trajectories
from prediction.utils.reshape import unflatten_batch
from prediction.utils.transform import transform_using_actor_frame


class ProbabilisticDecoder(nn.Module):
    ## write the code for the new model here

    def __init__(self, config) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(128, config.num_label_timesteps * 5),
        )

    def forward(self, x):
        return self._net(x)


class ProbabilisticModel(PredictionModel):
    def __init__(self, config: PredictionModelConfig) -> None:
        super().__init__(config)
        self._decoder = ProbabilisticDecoder(config)

    @staticmethod
    def _postprocess(
            out: Tensor, batch_ids: Tensor, original_x_pose: Tensor
    ) -> List[Tensor]:
        """Postprocess predictions

        1. Unflatten time and position dimensions
        2. Transform predictions back into SDV frame
        3. Unflatten batch and actor dimension

        Args:
            out (Tensor): predicted input trajectories [batch_size * N x T * 5]
            batch_ids (Tensor): id of each actor's batch in the flattened list [batch_size * N]
            original_x_pose (Tensor): original position and yaw of each actor at the latest timestep in SDV frame
                [batch_size * N, 3]

        Returns:
            List[Tensor]: List of length batch_size of output predicted trajectories in SDV frame [N x T x 2]
        """
        num_actors = len(batch_ids)
        out = out.reshape(num_actors, -1, 5)  # [batch_size * N x T x 5]

        # Transform from actor frame, to make the prediction problem easier
        out[..., 0:2] = transform_using_actor_frame(
            out[..., 0:2], original_x_pose, translate_to=False
        )

        # Translate so that latest timestep for each actor is the origin
        out_batches = unflatten_batch(out, batch_ids)
        return out_batches

    @torch.no_grad()
    def inference(self, history: Tensor) -> Trajectories:
        """Predict a set of 2d future trajectory predictions from the detection history

        Args:
            history: A [batch_size x N x T x 5] tensor, representing the input history
                centroid, yaw and size in a bird's eye view voxel representation.

        Returns:
            A set of 2D future trajectory centroid predictions.
        """
        self.eval()
        pred = self.forward([history])[0]  # shape: B * N x T x 5
        num_timesteps, num_coords = pred.shape[-2:]

        # Add dummy values for yaws and boxes here because we will fill them in from the ground truth
        return Trajectories(
            pred[:, :, 0:2],
            torch.zeros(pred.shape[0], num_timesteps),
            pred[:, :, 3:5],
        )

