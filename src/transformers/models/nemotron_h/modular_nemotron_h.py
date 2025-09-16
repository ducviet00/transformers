# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional, Union

import torch
from torch import nn
from ..jamba.configuration_jamba import JambaConfig
from ..jamba.modeling_jamba import (
    HybridMambaAttentionDynamicCache,
    JambaAttention,
    JambaAttentionDecoderLayer,
    JambaFlashAttention2,
    JambaForCausalLM,
    JambaForSequenceClassification,
    JambaMambaDecoderLayer,
    JambaMambaMixer,
    JambaMLP,
    JambaModel,
    JambaPreTrainedModel,
    JambaRMSNorm,
    JambaSdpaAttention,
    JambaSparseMoeBlock,
)

from ..mamba2.modeling_mamba2 import Mamba2Mixer

from ..nemotron.modeling_nemotron import (
    NemotronMLP,
)

from ..llama.modeling_llama import LlamaRMSNorm


if is_mamba_2_ssm_available():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
else:
    mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined, selective_state_update = None, None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all(
    (
        selective_state_update,
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
        causal_conv1d_fn,
        causal_conv1d_update,
    )
)


class NemotronHConfig(JambaConfig):
    pass


class NemotronHRMSNorm(LlamaRMSNorm):
    pass


class HybridMambaAttentionDynamicCache(HybridMambaAttentionDynamicCache):
    pass


class NemotronHAttention(JambaAttention):
    pass


class NemotronHFlashAttention2(JambaFlashAttention2):
    pass


class NemotronHSdpaAttention(JambaSdpaAttention):
    pass


class NemotronHMamba2Mixer(Mamba2Mixer):
    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        return super().cuda_kernels_forward(
            hidden_states,
            cache_params,
            cache_position,
            attention_mask,
        )

    def torch_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        return super().torch_forward(
            hidden_states,
            cache_params,
            cache_position,
            attention_mask,
        )

    def forward(
        self,
        hidden_states,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if is_fast_path_available and "cuda" in self.in_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)
        dtype = hidden_states.dtype
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)


class NemotronHMLP(NemotronMLP):
    pass


NEMOTRON_H_ATTENTION_CLASSES = {
    "eager": NemotronHAttention,
    "flash_attention_2": NemotronHFlashAttention2,
    "sdpa": NemotronHSdpaAttention,
}


class NemotronHBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # M: Mamba2, *: Attention, -: MLP
        self.block_type = config.layers_block_type[layer_idx]
        if self.block_type == "mamba":
            self.mixer = NemotronHMamba2Mixer(config, layer_idx=layer_idx)
        elif self.block_type == "attention":
            self.mixer = NEMOTRON_H_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)
        elif self.block_type == "mlp":
            self.mixer = NemotronHMLP(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Invalid layer pattern {config.hybrid_override_pattern[layer_idx]}")

    def forward(
        self,
        hidden_states,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):

        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        if self.block_type == "mamba":
            hidden_states = self.mixer(
                hidden_states, cache_params=cache_params, cache_position=cache_position
            )
        elif self.block_type == "attention":
            hidden_states = self.mixer(
                hidden_states, cache_position=cache_position
            )
            hidden_states = hidden_states[0]
        elif self.block_type == "mlp":
            hidden_states = self.mixer(
                hidden_states
            )
        else:
            raise ValueError(f"Invalid block_type: {self.block_type}")

        hidden_states = residual + hidden_states
        return hidden_states


class NemotronHPreTrainedModel(JambaPreTrainedModel):
    pass


class NemotronHModel(JambaModel):
    pass


class NemotronHForCausalLM(JambaForCausalLM):
    pass


class NemotronHForSequenceClassification(JambaForSequenceClassification):
    pass


__all__ = [
    "NemotronHConfig",
    "NemotronHForCausalLM",
    "NemotronHForSequenceClassification",
    "NemotronHModel",
    "NemotronHPreTrainedModel",
]
