# Copyright 2023-present the HuggingFace Inc. team.
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
from __future__ import annotations

import math
import os
import warnings
import logging

logger = logging.getLogger(__name__)

from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose
from peft.metrics import record_hydralora_metrics

from .config import HydraLoraConfig

import sys


def debug(msg: str):
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)

_GLOBAL_DEBUG_ROUTING_EVERY = int(os.environ.get("HYDRA_DEBUG_ROUTING_EVERY", "0") or 0)


LANGUAGE_PAD_ID = -1


class HydraLoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_route")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.lora_num = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.lora_route = nn.ModuleDict({})

        # Hierarchical / MoE bookkeeping
        self._hydra_expert_parent: dict[str, str] = {}
        self._hydra_parent_children: dict[str, list[str]] = {}

        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        setattr(self, "_active_adapters", [])
        expert_lora_nums_override = kwargs.pop("hydralora_expert_lora_nums", None)
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

        # self.W_V = None
        # self.W_S = None
        # self.Uh = None

        # hierarchical design addition
        self.use_hydralora_experts = kwargs.pop("use_hydralora_experts", False)
        self.num_experts = kwargs.pop("hydralora_num_experts", 1)
        self.top_k = kwargs.pop("hydralora_top_k", 1)
        self.hydralora_debug = kwargs.pop("hydralora_debug", False)
        self.language_list = kwargs.pop("language_list", None)
        self.family_list = kwargs.pop("family_list", None)
        self.language_to_family_ids = kwargs.pop("language_to_family_ids", None)
        self.language_to_subgroup_ids = kwargs.pop("language_to_subgroup_ids", None)
        self.language_router_mode = kwargs.pop("language_router_mode", "learned")
        self.language_bias_value = kwargs.pop("language_bias_value", 0.0)
        self.language_head_router_mode = kwargs.pop("language_head_router_mode", None) or self.language_router_mode
        self.language_head_bias_value = kwargs.pop("language_head_bias_value", None)
        if self.language_head_bias_value is None:
            self.language_head_bias_value = self.language_bias_value
        self.language_column = kwargs.pop("language_column", None)
        self.language_guidance_scope = kwargs.pop("language_guidance_scope", "all")
        if self.language_guidance_scope not in {"all", "expert_only", "none"}:
            raise ValueError(f"Unknown language_guidance_scope '{self.language_guidance_scope}'.")
        self._language_to_idx = {lang: idx for idx, lang in enumerate(self.language_list)} if self.language_list else {}
        if self.language_list:
            if self.language_to_family_ids is not None:
                expert_mapping = torch.tensor(self.language_to_family_ids, dtype=torch.long)
            else:
                expert_mapping = torch.arange(len(self.language_list), dtype=torch.long)
        else:
            expert_mapping = torch.empty(0, dtype=torch.long)
        self.register_buffer("language_id_to_expert", expert_mapping, persistent=False)
        family_mapping = (
            torch.tensor(self.language_to_family_ids, dtype=torch.long)
            if self.language_to_family_ids is not None
            else torch.empty(0, dtype=torch.long)
        )
        self.register_buffer("language_id_to_family", family_mapping, persistent=False)
        subgroup_mapping = (
            torch.tensor(self.language_to_subgroup_ids, dtype=torch.long)
            if self.language_list is not None and self.language_to_subgroup_ids is not None
            else torch.empty(0, dtype=torch.long)
        )
        self.register_buffer("language_id_to_subgroup", subgroup_mapping, persistent=False)

        if self.use_hydralora_experts:
            if self.family_list:
                self.num_experts = len(self.family_list)
            elif self.language_list:
                self.num_experts = len(self.language_list)
            if self.top_k > self.num_experts:
                self.top_k = self.num_experts
            self.router = nn.Linear(self.in_features, self.num_experts, bias=False)
            self._move_router_to_device_of_base_layer()
            if self.hydralora_debug:
                debug(
                    f"[HYDRA DEBUG] Initialized router in {self.__class__.__name__} "
                    f"(in={self.in_features}, experts={self.num_experts}, top_k={self.top_k})"
                )
        self._missing_language_warning_emitted: set[str] = set()
        self._expert_lora_nums = self._normalize_expert_counts(expert_lora_nums_override)
        self._debug_forward_calls = 0

    def _should_debug_routing(self) -> bool:
        if not getattr(self, "hydralora_debug", False):
            return False
        every = _GLOBAL_DEBUG_ROUTING_EVERY
        if every <= 0:
            return True
        self._debug_forward_calls += 1
        return (self._debug_forward_calls % every) == 0

    def _debug_print_hydra_setup(self, parent: str) -> None:
        if not getattr(self, "hydralora_debug", False):
            return
        if not getattr(self, "use_hydralora_experts", False):
            return
        debug(f"[HYDRA DEBUG] Setup(parent='{parent}'): experts={self.num_experts}, top_k={self.top_k}")
        for e in range(self.num_experts):
            name = f"expert_{e}"
            head_cnt = int(self.lora_num.get(name, 0) or 0)
            langs: list[str] = []
            if self.language_list:
                if self.family_list and self.language_to_family_ids is not None:
                    for lang_idx, fam_idx in enumerate(self.language_to_family_ids):
                        if int(fam_idx) == e and lang_idx < len(self.language_list):
                            langs.append(self.language_list[lang_idx])
                else:
                    if e < len(self.language_list):
                        langs.append(self.language_list[e])
            lang_preview = ",".join(langs[:5])
            suffix = f", langs=[{lang_preview}{'...' if len(langs) > 5 else ''}]" if langs else ""
            debug(f"[HYDRA DEBUG]   {name}: lora_num={head_cnt}{suffix}")
        debug("=" * 60)

    def update_layer(
            self, adapter_name, r, lora_alpha, lora_dropout, lora_num, init_lora_weights
    ):
        # Hierarchical HydraLoRA
        if self.use_hydralora_experts and adapter_name == getattr(self, "_active_adapter", adapter_name):
            self._active_adapters = []
            adapter_children: list[str] = []

            for e in range(self.num_experts):
                name = f"expert_{e}"
                adapter_children.append(name)
                self._hydra_expert_parent[name] = adapter_name

                expert_lora_num = self._expert_lora_nums[e] if self._expert_lora_nums else lora_num
                self.r[name] = r
                self.lora_alpha[name] = lora_alpha
                self.lora_num[name] = expert_lora_num
                self.lora_dropout[name] = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
                # Core Hydra concept: one A, multiple B per expert
                self.lora_A[name] = nn.Linear(self.in_features, r, bias=False)
                self.lora_B[name] = nn.ModuleList(
                    [nn.Linear(r, self.out_features, bias=False) for _ in range(expert_lora_num)]
                )
                self.lora_route[name] = nn.Linear(self.in_features, expert_lora_num, bias=False)
                self.scaling[name] = lora_alpha / r

                self.reset_lora_parameters(name, init_lora_weights)

                self._move_adapter_to_device_of_base_layer(name)
                self._active_adapters.append(name)

            self._hydra_parent_children[adapter_name] = adapter_children

            if self.hydralora_debug:
                debug(
                    f"[HYDRA DEBUG] Created {self.num_experts} HydraLoRA experts for parent '{adapter_name}' "
                    f"(r={r}, lora_num={lora_num}, top_k={self.top_k})"
                )
                self._debug_print_hydra_setup(adapter_name)
                self._verify_hydralora_expert_init()

            # Use the *parent* name for external API; children are stored in _hydra_parent_children
            self.set_adapter(adapter_name)
            return

        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.lora_num[adapter_name] = lora_num
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.ModuleList(
            [nn.Linear(r, self.out_features, bias=False) for _ in range(lora_num)])
        self.lora_route[adapter_name] = nn.Linear(self.in_features, lora_num, bias=False)

        self.scaling[adapter_name] = lora_alpha / r

        self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        self._active_adapters = [adapter_name]
        self.set_adapter(self._active_adapters)

    def set_adapter(self, adapter_names: Union[str, list[str]]) -> None:
        """
        Override BaseTunerLayer.set_adapter to support hierarchical Hydra experts:
        - if a parent adapter is given, expand to its expert children.
        - optionally enable/disable router grad based on whether any Hydra parent is active.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        expanded: list[str] = []
        any_hydra_parent_active = False

        for name in adapter_names:
            # Parent: expand to its children
            if self.use_hydralora_experts and name in self._hydra_parent_children:
                expanded.extend(self._hydra_parent_children[name])
                any_hydra_parent_active = True
            else:
                expanded.append(name)
                # Also mark if an individual expert belonging to a Hydra parent is used
                if self.use_hydralora_experts and name in self._hydra_expert_parent:
                    any_hydra_parent_active = True

        # Router grads only when Hydra hierarchy is actually in use
        if self.use_hydralora_experts and hasattr(self, "router"):
            self._move_router_to_device_of_base_layer()
            self.router.requires_grad_(any_hydra_parent_active)

        # Defer to BaseTunerLayer for actual bookkeeping
        super().set_adapter(expanded)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            for layer in self.lora_B[adapter_name]:
                nn.init.zeros_(layer.weight)

            nn.init.kaiming_uniform_(self.lora_route[adapter_name].weight, a=math.sqrt(5))
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key, None)
        return value

    def _move_router_to_device_of_base_layer(self) -> None:
        if not hasattr(self, "router") or self.router is None:
            return
        base_layer = self.get_base_layer()
        weight = getattr(base_layer, "weight", None)
        if weight is None:
            return
        self.router.to(weight.device, dtype=weight.dtype)

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self._active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self._active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

    def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result

    def _verify_hydralora_expert_init(self) -> None:
        """Debug helper similar to CoLA's _verify_cola_expert_init, but for HydraLoRA."""
        if not getattr(self, "hydralora_debug", False):
            return

        debug(f"[HYDRA DEBUG] Verifying HydraLoRA experts in {self.__class__.__name__} ({self.num_experts} experts)")
        for e in range(self.num_experts):
            name = f"expert_{e}"
            if name not in self.lora_A:
                debug(f"  Expert {e} not found in lora_A; skipping")
                continue
            A = self.lora_A[name]
            Bs = self.lora_B[name]
            route = self.lora_route[name]
            scale = self.scaling[name]

            with torch.no_grad():
                a_mean = A.weight.mean().item()
                a_std = A.weight.std().item()
                b_mean = torch.stack([b.weight.mean() for b in Bs]).mean().item()
                b_std = torch.stack([b.weight.std() for b in Bs]).mean().item()
                route_mean = route.weight.mean().item()
                route_std = route.weight.std().item()

            debug(
                f"  Expert {e}: "
                f"A.shape={tuple(A.weight.shape)}, "
                f"num_B={len(Bs)}, "
                f"B[0].shape={tuple(Bs[0].weight.shape)}, "
                f"route.shape={tuple(route.weight.shape)}, "
                f"scale={scale:.4f}"
            )
        debug(
            f"    stats: A(mean={a_mean:.4e}, std={a_std:.4e}), "
            f"B(mean={b_mean:.4e}, std={b_std:.4e}), "
            f"route(mean={route_mean:.4e}, std={route_std:.4e})"
        )
        debug("[HYDRA DEBUG] Expert verification complete\n" + "=" * 60)

    def _normalize_expert_counts(self, counts: Optional[list[int]]) -> Optional[list[int]]:
        if not self.use_hydralora_experts or counts is None:
            return None
        values = list(counts)
        if not values:
            return None
        if len(values) != self.num_experts:
            raise ValueError(f"hydralora_expert_lora_nums must provide {self.num_experts} entries, got {len(values)}.")
        if any(v <= 0 for v in values):
            raise ValueError("hydralora_expert_lora_nums entries must be positive integers.")
        return values

    def _language_head_targets(
            self, language_ids: Optional[torch.Tensor], adapter_name: str
    ) -> Optional[torch.Tensor]:
        if (language_ids is None or self.language_list is None or self.language_guidance_scope != "all"):
            return None
        head_count = self.lora_num.get(adapter_name)
        if not head_count:
            return None
        mapping = getattr(self, "language_id_to_subgroup", None)
        if mapping is not None and torch.is_tensor(mapping) and mapping.numel() > 0:
            mapping = mapping.to(language_ids.device)
            head_ids = torch.full_like(language_ids, LANGUAGE_PAD_ID)
            valid = (language_ids >= 0) & (language_ids < mapping.numel())
            if valid.any():
                candidate = mapping[language_ids[valid]]
                # Guard against malformed mappings that exceed available heads
                candidate = torch.where(candidate < head_count, candidate, torch.full_like(candidate, LANGUAGE_PAD_ID))
                head_ids[valid] = candidate
            return head_ids
        # Fallback: simple modulo by head count
        head_ids = torch.full_like(language_ids, LANGUAGE_PAD_ID)
        valid = (language_ids >= 0) & (language_ids < len(self.language_list))
        if valid.any():
            head_ids[valid] = language_ids[valid] % head_count
        return head_ids

    def _language_expert_targets(self, language_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if (language_ids is None or self.language_id_to_expert.numel() == 0 or self.language_guidance_scope == "none"):
            return None
        mapping = self.language_id_to_expert.to(language_ids.device)
        expert_ids = torch.full_like(language_ids, LANGUAGE_PAD_ID)
        valid = (language_ids >= 0) & (language_ids < mapping.numel())
        if valid.any():
            expert_ids[valid] = mapping[language_ids[valid]]
        return expert_ids

    def _log_missing_language_targets(self, prefix: str, reason: str) -> None:
        key = f"{prefix}:{reason}"
        if key in self._missing_language_warning_emitted:
            return
        column = self.language_column or "<unset>"
        logger.warning(
            "HydraLoRA layer '%s' missing %s routing metadata (%s). Verify dataset column '%s' and language_map.",
            self.__class__.__name__,
            prefix,
            reason,
            column,
        )
        self._missing_language_warning_emitted.add(key)

    def _append_target_metrics(
        self,
        metrics: dict[str, float],
        metrics_weight: float,
        prefix: str,
        target_tensor: Optional[torch.Tensor],
        selection: torch.Tensor,
        probs: torch.Tensor,
        language_ids: Optional[torch.Tensor],
        expect_targets: bool,
    ) -> float:
        if target_tensor is not None and torch.is_tensor(target_tensor):
            valid_batch = target_tensor >= 0
            seq_len = probs.size(1)
            if valid_batch.any():
                expanded_targets = target_tensor[valid_batch].view(-1, 1).expand(-1, seq_len)
                top1 = selection[valid_batch]
                target_hits = (top1 == expanded_targets).float()
                target_probs = probs[valid_batch].gather(
                    -1,
                    target_tensor[valid_batch].view(-1, 1, 1).expand(-1, seq_len, 1),
                ).squeeze(-1)
                valid_tokens = target_probs.numel()
                target_entropy = float((-torch.log(target_probs + 1e-8)).mean().item())
                metrics.update(
                    {
                        f"{prefix}_target_hit_rate": float(target_hits.mean().item()),
                        f"{prefix}_target_prob_mean": float(target_probs.mean().item()),
                        f"{prefix}_target_neglogp": target_entropy,
                        f"{prefix}_target_token_frac": float(
                            valid_tokens / max(seq_len * valid_batch.sum().item(), 1)
                        ),
                    }
                )
                return float(valid_tokens if valid_tokens > 0 else metrics_weight)

            metrics.update(
                {
                    f"{prefix}_target_hit_rate": 0.0,
                    f"{prefix}_target_prob_mean": 0.0,
                    f"{prefix}_target_neglogp": 0.0,
                    f"{prefix}_target_token_frac": 0.0,
                }
            )
            if expect_targets:
                self._log_missing_language_targets(prefix, "targets were all pad ids")
            return metrics_weight

        if expect_targets:
            metrics.update(
                {
                    f"{prefix}_target_hit_rate": 0.0,
                    f"{prefix}_target_prob_mean": 0.0,
                    f"{prefix}_target_neglogp": 0.0,
                    f"{prefix}_target_token_frac": 0.0,
                }
            )
            if language_ids is None:
                reason = "no language_ids tensor provided"
            elif torch.is_tensor(language_ids) and (language_ids >= 0).any():
                reason = "language_ids could not be mapped to targets"
            else:
                reason = "language_ids contained only pad ids"
            self._log_missing_language_targets(prefix, reason)
        return metrics_weight

    def _apply_language_bias_heads(
            self, logits: torch.Tensor, head_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if head_ids is None or self.language_head_router_mode != "bias":
            return logits
        valid = head_ids >= 0
        if not valid.any():
            return logits
        bias = torch.zeros(logits.size(0), logits.size(-1), device=logits.device, dtype=logits.dtype)
        bias[valid, head_ids[valid]] = self.language_head_bias_value
        return logits + bias.unsqueeze(1)

    def _enforce_language_heads(
            self, weights: torch.Tensor, head_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if head_ids is None or self.language_head_router_mode != "hard":
            return weights
        valid = head_ids >= 0
        if not valid.any():
            return weights
        weights = weights.clone()
        weights[valid] = 0
        weights[valid, :, head_ids[valid]] = 1
        return weights

    def _apply_language_bias_experts(
            self, logits: torch.Tensor, expert_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if expert_ids is None or self.language_router_mode != "bias":
            return logits
        valid = expert_ids >= 0
        if not valid.any():
            return logits
        bias = torch.zeros(logits.size(0), logits.size(-1), device=logits.device, dtype=logits.dtype)
        bias[valid, expert_ids[valid]] = self.language_bias_value
        return logits + bias.unsqueeze(1)

    def _enforce_language_experts(
            self, topi: torch.Tensor, weights: torch.Tensor, expert_ids: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if expert_ids is None or self.language_router_mode != "hard":
            return topi, weights
        valid = expert_ids >= 0
        if not valid.any():
            return topi, weights
        seq_len = topi.size(1)
        replacement = expert_ids[valid].view(-1, 1, 1).expand(-1, seq_len, self.top_k)
        topi = topi.clone()
        weights = weights.clone()
        topi[valid] = replacement
        weights[valid] = 0
        weights[valid, :, 0] = 1
        return topi, weights

    def _cache_router_state(
            self, logits: torch.Tensor, language_ids: Optional[torch.Tensor], prefix: str,
            targets: Optional[torch.Tensor]
    ) -> None:
        self._cache_store(f"{prefix}_router_logits", logits)
        if language_ids is not None and torch.is_tensor(language_ids):
            self._cache_store(f"{prefix}_router_language_ids", language_ids)
        if targets is not None and torch.is_tensor(targets):
            self._cache_store(f"{prefix}_router_targets", targets)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, HydraLoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            base_layer,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            lora_num: int = 1,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            is_target_conv_1d_layer: bool = False,
            init_lora_weights: Union[bool, str] = True,
            **kwargs,
    ) -> None:
        super().__init__()
        HydraLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_num=lora_num,
            init_lora_weights=init_lora_weights,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        language_ids = kwargs.pop("language_ids", None)
        if isinstance(language_ids, torch.Tensor):
            language_ids = language_ids.to(x.device).long()
            if language_ids.dim() > 1:
                language_ids = language_ids.view(language_ids.size(0))
        else:
            language_ids = None
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            if not getattr(self, "use_hydralora_experts", False):
                for active_adapter in self._active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    lora_route = self.lora_route[active_adapter]

                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]

                    x = x.to(lora_A.weight.dtype)
                    route_logits = lora_route(x.to(torch.float32)).to(result.dtype)
                    use_head_guidance = self.language_guidance_scope == "all"
                    head_targets = (self._language_head_targets(language_ids, active_adapter) if use_head_guidance else None)
                    if use_head_guidance:
                        self._cache_router_state(route_logits, language_ids, f"hydra_head_{active_adapter}", head_targets)
                    route_logits = self._apply_language_bias_heads(route_logits, head_targets)
                    route_weight = nn.functional.softmax(route_logits, dim=-1, dtype=torch.float32).to(result.dtype)
                    route_weight = self._enforce_language_heads(route_weight, head_targets)
                    head_assign = torch.argmax(route_weight, dim=-1, keepdim=True)
                    with torch.no_grad():
                        token_count = route_weight.numel() // route_weight.size(-1)
                        if token_count > 0:
                            flat_assign = head_assign.reshape(-1)
                            head_counts = torch.bincount(
                                flat_assign,
                                minlength=self.lora_num[active_adapter],
                            ).to(torch.float32)
                            mean_head = head_counts.mean().item()
                            if mean_head > 0:
                                head_cv = float(
                                    (head_counts.std(unbiased=False) / (mean_head + 1e-6)).item()
                                )
                            else:
                                head_cv = 0.0
                            head_active = float((head_counts > 0).float().mean().item())
                            head_entropy = float(
                                (-route_weight * torch.log(route_weight + 1e-8)).sum(dim=-1).mean().item()
                            )
                            total_head_assign = head_counts.sum()
                            if total_head_assign > 0:
                                head_frac = head_counts / total_head_assign
                                head_max = float(head_frac.max().item())
                                head_min = float(head_frac.min().item())
                            else:
                                head_max = 0.0
                                head_min = 0.0
                            metrics = {
                                "head_load_cv": head_cv,
                                "head_active_frac": head_active,
                                "head_router_entropy": head_entropy,
                                "head_load_max_frac": head_max,
                                "head_load_min_frac": head_min,
                            }
                            metrics_weight = float(token_count)
                            metrics_weight = self._append_target_metrics(
                                metrics=metrics,
                                metrics_weight=metrics_weight,
                                prefix="head",
                                target_tensor=head_targets,
                                selection=head_assign.squeeze(-1),
                                probs=route_weight,
                                language_ids=language_ids,
                                expect_targets=use_head_guidance and self.language_list is not None,
                            )
                            record_hydralora_metrics(metrics, weight=metrics_weight)

                    for i in range(self.lora_num[active_adapter]):
                        result = result + torch.unsqueeze(route_weight[:, :, i], -1) * lora_B[i](
                            (lora_A(dropout(x)))) * scaling

                result = result.to(torch_result_dtype)
            else:
                router_dtype = getattr(self.router.weight, "dtype", torch.float32)
                logits = self.router(x.to(router_dtype)).to(x.dtype)
                
                use_expert_guidance = self.language_guidance_scope in {"all", "expert_only"}
                expert_targets = (self._language_expert_targets(language_ids) if use_expert_guidance else None)
                if use_expert_guidance:
                    self._cache_router_state(logits, language_ids, "hydra_expert", expert_targets)
                logits = self._apply_language_bias_experts(logits, expert_targets)
                
                topv, topi = torch.topk(logits, self.top_k, dim=-1)
                weights = torch.softmax(topv.to(torch.float32), dim=-1).to(x.dtype)
                topi, weights = self._enforce_language_experts(topi, weights, expert_targets)

                if self._should_debug_routing():
                    with torch.no_grad():
                        debug(
                            f"[HYDRA DEBUG] Router in {self.__class__.__name__}: "
                            f"logits_mean={logits.mean().item():.4e}, "
                            f"logits_std={logits.std().item():.4e}"
                        )
                        sample_b, sample_t = 0, 0
                        lang_id = int(language_ids[sample_b].item()) if language_ids is not None else LANGUAGE_PAD_ID
                        tgt_ex = (
                            int(expert_targets[sample_b].item())
                            if expert_targets is not None and torch.is_tensor(expert_targets)
                            else LANGUAGE_PAD_ID
                        )
                        sample_w = weights[sample_b, sample_t].detach().cpu().tolist()
                        sample_i = topi[sample_b, sample_t].detach().cpu().tolist()
                        debug(
                            f"[HYDRA DEBUG] Sample(b={sample_b},t={sample_t}) lang_id={lang_id} "
                            f"expert_target={tgt_ex} top_experts={sample_i}, weights={sample_w}"
                        )
                        if tgt_ex >= 0:
                            ex_name = f"expert_{tgt_ex}"
                            head_router = self.lora_route.get(ex_name)
                            head_count = int(self.lora_num.get(ex_name, 0) or 0)
                            if head_router is not None and head_count > 1:
                                head_logits = head_router(x.to(torch.float32)).to(x.dtype)
                                head_targets = (
                                    self._language_head_targets(language_ids, ex_name)
                                    if self.language_guidance_scope == "all"
                                    else None
                                )
                                head_logits = self._apply_language_bias_heads(head_logits, head_targets)
                                head_probs = torch.softmax(head_logits.to(torch.float32), dim=-1)
                                head_probs_tok = head_probs[sample_b, sample_t].detach().cpu()
                                top_h = torch.topk(head_probs_tok, k=min(3, head_count), dim=-1)
                                debug(
                                    f"[HYDRA DEBUG]   head_target="
                                    f"{int(head_targets[sample_b].item()) if head_targets is not None else LANGUAGE_PAD_ID} "
                                    f"top_heads={top_h.indices.tolist()} probs={top_h.values.tolist()}"
                                )

                with torch.no_grad():
                    token_count = topi.numel()
                    if token_count > 0:
                        flat_indices = topi.reshape(-1)
                        counts = torch.bincount(flat_indices, minlength=self.num_experts).to(torch.float32)
                        active_frac = float((counts > 0).float().mean().item())
                        mean_load = counts.mean().item()
                        if mean_load > 0:
                            load_cv = float((counts.std(unbiased=False) / (mean_load + 1e-6)).item())
                        else:
                            load_cv = 0.0
                        router_probs = torch.softmax(logits.to(torch.float32), dim=-1)
                        entropy = float((-router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1).mean().item())
                        weight_mean = float(weights.mean().item())
                        metrics = {
                            "expert_load_cv": load_cv,
                            "expert_active_frac": active_frac,
                            "expert_router_entropy": entropy,
                            "expert_topk_weight_mean": weight_mean,
                        }
                        total_assign = counts.sum()
                        if total_assign > 0:
                            frac = counts / total_assign
                            metrics["expert_load_max_frac"] = float(frac.max().item())
                            metrics["expert_load_min_frac"] = float(frac.min().item())
                        else:
                            metrics["expert_load_max_frac"] = 0.0
                            metrics["expert_load_min_frac"] = 0.0
                        metrics_weight = float(token_count)
                        metrics_weight = self._append_target_metrics(
                            metrics=metrics,
                            metrics_weight=metrics_weight,
                            prefix="expert",
                            target_tensor=expert_targets,
                            selection=topi[:, :, 0],
                            probs=router_probs,
                            language_ids=language_ids,
                            expect_targets=use_expert_guidance and self.language_list is not None,
                        )
                        record_hydralora_metrics(metrics, weight=metrics_weight)

                # Memory-friendly mixing: avoid materializing [B, S, D_out, E] for all experts.
                # We compute each expert delta once and only keep the final mixed output.
                moe_out = torch.zeros_like(result, dtype=result.dtype)
                for e in range(self.num_experts):
                    expert_delta = self._adapter_delta(
                        x,
                        f"expert_{e}",
                        language_ids=language_ids,
                        expert_id=e,
                        expert_targets=expert_targets,
                    ).to(moe_out.dtype)
                    for k in range(self.top_k):
                        mask = topi[:, :, k].eq(e)
                        if not mask.any():
                            continue
                        moe_out = moe_out + expert_delta * (weights[:, :, k] * mask.to(weights.dtype)).unsqueeze(-1)

                result = result + moe_out
                result = result.to(torch_result_dtype)

        return result

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights += delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter][0].weight.device
        dtype = self.lora_B[adapter][0].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        # weight_B = self.lora_B[adapter].weight
        weight_B = torch.sum(torch.stack([layer.weight for layer in self.lora_B[adapter]]), dim=0)

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            # self.lora_B[adapter].weight.data = weight_B.to(dtype)
            for i in range(self.num_B[adapter]):
                self.lora_B[adapter][i].weight.data = self.lora_B[adapter][i].weight.to(dtype)

        return output_tensor

    def _adapter_delta(
        self,
        x: torch.Tensor,
        name: str,
        *,
        language_ids: Optional[torch.Tensor] = None,
        expert_id: Optional[int] = None,
        expert_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the HydraLoRA delta for a single adapter/expert.

        - Flat Hydra: routed mixture over B heads (handled in Linear.forward, not here).
        - Expert Hydra: we optionally apply the same head routing per expert when lora_route[name] exists.
        """
        A = self.lora_A[name]  # nn.Linear
        B_list = self.lora_B[name]  # ModuleList[nn.Linear]
        drop = self.lora_dropout[name]
        scale = self.scaling[name]

        if not B_list:
            return torch.zeros_like(x, dtype=self.get_base_layer().weight.dtype)

        intermediate = drop(x.to(A.weight.dtype))  # [B, S, D_in]
        a_dot_x = A(intermediate)  # [B, S, r]

        lora_route = self.lora_route[name] if name in self.lora_route else None
        if lora_route is None or len(B_list) == 1:
            out = sum(B(a_dot_x) for B in B_list)
            return out * scale

        route_logits = lora_route(x.to(torch.float32)).to(x.dtype)  # [B, S, num_heads]
        head_targets: Optional[torch.Tensor] = None
        use_head_guidance = self.language_guidance_scope == "all"
        if use_head_guidance and language_ids is not None:
            head_targets = self._language_head_targets(language_ids, name)
            if (
                head_targets is not None
                and expert_id is not None
                and expert_targets is not None
                and torch.is_tensor(expert_targets)
            ):
                mismatch = expert_targets != int(expert_id)
                if mismatch.any():
                    head_targets = head_targets.clone()
                    head_targets[mismatch] = LANGUAGE_PAD_ID

            self._cache_router_state(route_logits, language_ids, f"hydra_head_{name}", head_targets)

        route_logits = self._apply_language_bias_heads(route_logits, head_targets)
        route_weight = nn.functional.softmax(route_logits, dim=-1, dtype=torch.float32).to(x.dtype)  # [B, S, H]
        route_weight = self._enforce_language_heads(route_weight, head_targets)

        out = 0
        for i, B in enumerate(B_list):
            out = out + torch.unsqueeze(route_weight[:, :, i], -1) * B(a_dot_x)

        return out * scale

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

    def pop_language_router_cache(self) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        caches: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        # Head routers for standard Hydra adapters
        head_keys = [
            key for key in list(self._caches.keys()) if key.startswith("hydra_head_") and key.endswith("_router_logits")
        ]
        for key in head_keys:
            logits = self._cache_pop(key)
            targets = self._cache_pop(key.replace("_router_logits", "_router_targets"))
            if logits is not None and targets is not None:
                caches.append(("hydra_head", logits, targets))
            self._cache_pop(key.replace("_router_logits", "_router_language_ids"))
        # Global expert router
        logits = self._cache_pop("hydra_expert_router_logits")
        targets = self._cache_pop("hydra_expert_router_targets")
        if logits is not None and targets is not None:
            caches.append(("hydra_expert", logits, targets))
        self._cache_pop("hydra_expert_router_language_ids")
        return caches


class Embedding(nn.Module, HydraLoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
            self,
            base_layer: nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: Union[bool, str] = True,
            **kwargs,
    ) -> None:
        super().__init__()
        HydraLoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)
        self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self._active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_embedding_A.keys():
                continue

            embedding_A = self.lora_embedding_A[active_adapter].T
            embedding_B = self.lora_embedding_B[active_adapter].T
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]]
            after_A = self._embed(sub_batch, embedding_A)
            result[sub_batch_indices_list[i]] += (after_A @ embedding_B) * scaling

        return result

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        kwargs.pop("language_ids", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self._active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]
                after_A = self._embed(x, embedding_A)
                result = result + (after_A @ embedding_B) * scaling
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Conv2d(nn.Module, HydraLoraLayer):
    # Lora implemented in a conv2d layer
    def __init__(
            self,
            base_layer: nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: Union[bool, str] = True,
            **kwargs,
    ) -> None:
        super().__init__()
        HydraLoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        self.lora_A[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
        self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        self.set_adapter(self._active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)

                    orig_weights += delta_weight
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            # conv2d 3x3
            output_tensor = (
                    F.conv2d(
                        weight_A.permute(1, 0, 2, 3),
                        weight_B,
                    ).permute(1, 0, 2, 3)
                    * self.scaling[adapter]
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self._active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                result = result + lora_B(lora_A(dropout(x))) * scaling

            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_default(
        target: torch.nn.Module,
        adapter_name: str,
        lora_config: HydraLoraConfig,
        **kwargs,
) -> Optional[torch.nn.Module]:
    """
    target: Linear(in_features=3072, out_features=3072, bias=False)
    adapter_name: default
    lora_config: HydraLoraConfig(peft_type=<PeftType.HYDRALORA: 'HYDRALORA'>,
      auto_mapping=None, 
      base_model_name_or_path='/data/develop/smallz/Llama-3.2-3B', 
      revision=None, 
      task_type=<TaskType.CAUSAL_LM: 'CAUSAL_LM'>, 
      inference_mode=False, 
      r=8, 
      target_modules={'v_proj', 'o_proj', 'gate_proj', 'q_proj', 'down_proj', 'k_proj', 'up_proj'}, 
      lora_alpha=16, 
      lora_dropout=0.0, 
      fan_in_fan_out=False, 
      bias='none', 
      use_rslora=False, 
      modules_to_save=None, 
      init_lora_weights=True, 
      layers_to_transform=None, 
      layers_pattern=None, 
      rank_pattern={}, 
      alpha_pattern={}, 
      megatron_config=None, 
      megatron_core='megatron.core', 
      loftq_config={}, 
      use_dora=False, 
      layer_replication=None, 
      runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=False))
    kwargs: {'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.0, 'fan_in_fan_out': False, 'init_lora_weights': True, 'use_rslora': False, 'use_dora': False, 'ephemeral_gpu_offload': False, 'loaded_in_8bit': False, 'loaded_in_4bit': False}
    """
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
