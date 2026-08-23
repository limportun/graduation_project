import math
import torch
import torch.nn as nn
import torchrl.networks.init as init


class STTRL_DVO_Backbone(nn.Module):
  def __init__(
      self,
      encoder,
      time_step,
      token_dim,
      state_input_shape,
      lidar_input_shape,
      transformer_params=(),
      append_hidden_shapes=(),
      append_hidden_init_func=init.basic_init,
      net_last_init_func=init.uniform_init,
      activation_func=nn.ReLU,
      using_causal=True,
      capture_attention=False,
      **kwargs):
    super().__init__()
    self.encoder = encoder
    self.attention_weights = []

    self.time_step = time_step
    self.state_input_shape = state_input_shape
    self.observation_shape = state_input_shape + lidar_input_shape
    self.token_dim = token_dim
    self.activation_func = activation_func
    self.using_causal = using_causal
    self.capture_attention = capture_attention

    # self-attention
    self.lidar_append_layers = nn.ModuleList()
    for n_head, dim_feedforward in transformer_params:
      lidar_att_layer = CustomTransformerEncoderLayer( # nn.TransformerEncoderLayer(
        token_dim, n_head, dim_feedforward,
        dropout=0, batch_first=True,
        capture_attention=capture_attention,
      )
      self.lidar_append_layers.append(lidar_att_layer)

    # MLP
    sequence_length = 20
    self.final_mlp_layers = []
    for input_shape, next_shape in append_hidden_shapes:
      lidar_fc = nn.Linear(input_shape, next_shape)
      append_hidden_init_func(lidar_fc)
      self.final_mlp_layers.append(lidar_fc)
      self.final_mlp_layers.append(self.activation_func())

    self.final_mlp = nn.Sequential(*self.final_mlp_layers)

    # Causal Transformer Mask
    tokens_per_time_step = self.encoder.lidar_patches + 1
    sequence_length = self.time_step * tokens_per_time_step

    token_timesteps = torch.arange(sequence_length) // tokens_per_time_step
    allowed = token_timesteps.unsqueeze(1) >= token_timesteps.unsqueeze(0)
    causal_mask = torch.zeros((sequence_length, sequence_length))
    causal_mask.masked_fill_(~allowed, float('-inf'))
    # Deterministically rebuilt from architecture settings. Keeping this
    # non-persistent preserves compatibility with checkpoints created before
    # the mask was registered as a buffer.
    self.register_buffer('causal_mask', causal_mask, persistent=False)

    position = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
      torch.arange(0, self.token_dim, 2, dtype=torch.float32)
      * (-math.log(10000.0) / self.token_dim)
    )
    pe_width = self.token_dim if self.token_dim % 2 == 0 else self.token_dim + 1
    positional_encoding = torch.zeros(sequence_length, pe_width)
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer(
      'positional_encoding', positional_encoding[:, :self.token_dim].unsqueeze(0),
      persistent=False,
    )

  def set_attention_capture(self, enabled):
    """Enable detached attention snapshots for evaluation visualizations."""
    self.capture_attention = bool(enabled)
    for layer in self.lidar_append_layers:
      layer.capture_attention = self.capture_attention
      if not self.capture_attention:
        layer.attn_weights = None
    if not self.capture_attention:
      self.attention_weights = []

  def forward(self, x):
    state_input = []
    lidar_input = []

    for t in range(self.time_step):
      start_idx = t * self.observation_shape
      end_idx = (t + 1) * self.observation_shape

      state_input_t = x[:,start_idx:start_idx + self.state_input_shape]
      lidar_input_t = x[:,start_idx + self.state_input_shape:end_idx]

      state_input.append(state_input_t)
      lidar_input.append(lidar_input_t)

    # state_input:(ts,7)  lidar_input:(ts,360)
    state_input = torch.stack(state_input, dim = 0) # torch.Size([ts, bs, 7])
    state_input = state_input.permute(1,0,2) # torch.Size([bs, ts, 7])
    lidar_input = torch.stack(lidar_input, dim = 0) # torch.Size([ts, bs, 360])
    lidar_input = lidar_input.permute(1,0,2) # torch.Size([bs, ts, 360])

    out = self.encoder( # (bs, ts*21, tdim)
      lidar_input, state_input
    )

    batch_size = out.shape[0] # bs
    sequence_length = out.shape[1] # ts*21

    out = out + self.positional_encoding[:, :sequence_length].to(dtype=out.dtype)

    self.attention_weights = []
    for att_layer in self.lidar_append_layers:
      if self.using_causal:
        out = att_layer(out, src_mask=self.causal_mask)
      else:
        out = att_layer(out)
      if att_layer.attn_weights is not None:
        self.attention_weights.append(att_layer.attn_weights)

    # Mixture Pooling along Lidar tokens
    out = out.view(batch_size, self.time_step, int(sequence_length/self.time_step), self.token_dim) # [-1, ts, 21, tdim]
    mean_lidar_feature = torch.mean(out[:, :, :-1, :], dim=2) # [-1, ts, tdim]
    min_lidar_feature = torch.min(out[:, :, :-1, :], dim=2).values # [-1, ts, tdim]
    max_lidar_feature = torch.max(out[:, :, :-1, :], dim=2).values # [-1, ts, tdim]
    state_feature = out[:, :, -1, :] # [-1, ts, tdim]
    out = torch.cat((mean_lidar_feature, min_lidar_feature, max_lidar_feature, state_feature), dim=1) # [-1, ts*4, tdim]

    # out after MLP
    out = out.flatten(start_dim=1) # (-1, ts*4*tdim)
    out = self.final_mlp(out)

    # out [bs, latent_dim_pi]
    return out


# Custom Transformer Encoder Layer to save attention weights for visualization
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, capture_attention=False, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__(*args, **kwargs)
        self.capture_attention = capture_attention
        self.attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None,
                is_causal=False):
        """Transformer encoder forward pass with optional attention capture."""
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=self.capture_attention,
            average_attn_weights=True,
            is_causal=is_causal,
        )
        self.attn_weights = (
            attn_weights.detach() if attn_weights is not None else None
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
