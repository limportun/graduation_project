import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchrl.networks.init as init


class STTRL_DVO_Encoder(nn.Module):
  def __init__(
      self,
      _state_dim,
      _lidar_dim,
      time_step,
      token_dim
  ):
    super(STTRL_DVO_Encoder, self).__init__()
    self._state_dim = _state_dim
    self._lidar_dim = _lidar_dim
    self.time_step = time_step
    self.token_dim = token_dim

    layer_list = [
      nn.Conv1d(in_channels=self.time_step, out_channels=4 * self.time_step,
                kernel_size=5, stride=3, padding=1, groups=self.time_step), nn.ReLU(),
      nn.Conv1d(in_channels=4 * self.time_step, out_channels=16 * self.time_step,
                kernel_size=5, stride=3, padding=1, groups=self.time_step), nn.ReLU(),
      nn.Conv1d(in_channels=16 * self.time_step, out_channels=self.token_dim * self.time_step,
                kernel_size=3, stride=2, padding=1, groups=self.time_step), nn.ReLU(),
    ]
    self.lidar_cnn_layer = nn.Sequential(*layer_list)
    lidar_patches = _lidar_dim
    for kernel_size, stride, padding in ((5, 3, 1), (5, 3, 1), (3, 2, 1)):
      lidar_patches = (
        (lidar_patches + 2 * padding - kernel_size) // stride + 1
      )
    if lidar_patches <= 0:
      raise ValueError(f"lidar_dim is too small for the CNN encoder: {_lidar_dim}")
    self.lidar_patches = lidar_patches

    mlp_layer_list = [
      nn.Linear(self._state_dim, self.token_dim),
      nn.ReLU(),
    ]
    self.state_linear = nn.Sequential(*mlp_layer_list)

  def forward(self, lidar_x, state_x):
    batch_size = lidar_x.shape[0]
    if lidar_x.shape[-1] != self._lidar_dim:
      raise ValueError(
        f"Expected {self._lidar_dim} LiDAR values, got {lidar_x.shape[-1]}"
      )
    if state_x.shape[-1] != self._state_dim:
      raise ValueError(
        f"Expected {self._state_dim} state values, got {state_x.shape[-1]}"
      )
    lidar_out = self.lidar_cnn_layer(lidar_x)
    lidar_out = lidar_out.view(
      batch_size, self.time_step, self.token_dim, self.lidar_patches
    )
    lidar_out = lidar_out.reshape(
      batch_size, self.time_step * self.lidar_patches, self.token_dim
    )

    state_x = state_x.reshape(-1, self._state_dim)  # torch.Size([bs*ts, 7])
    state_out = self.state_linear(state_x).unsqueeze(1) # torch.Size([bs*ts, tdim])
    state_out = state_out.view(batch_size, self.time_step, self.token_dim)  # torch.Size([bs, ts, tdim])

    lidar_out = lidar_out.view(
      batch_size, self.time_step, self.lidar_patches, self.token_dim
    )
    state_out = state_out.unsqueeze(2)
    out = torch.cat([lidar_out, state_out], dim=2)
    out = out.view(batch_size, -1, self.token_dim)

    return out
