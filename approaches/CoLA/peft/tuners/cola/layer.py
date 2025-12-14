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
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose
from peft.metrics import record_cola_metrics

from .config import ColaConfig
import sys

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except ImportError:
    FSDP = None  # FSDP not always available (e.g. single-GPU runs)


_GLOBAL_DEBUG = os.environ.get("COLA_DEBUG", "").lower() in {"1", "true", "yes", "on"}
_GLOBAL_DEBUG_SUPPRESS = os.environ.get("COLA_SUPPRESS_DEBUG", "").lower() in {"1", "true", "yes", "on"}
_GLOBAL_DEBUG_ROUTING_EVERY = int(os.environ.get("COLA_DEBUG_ROUTING_EVERY", "0") or 0)


def debug(msg: str, enabled: bool = False):
    if _GLOBAL_DEBUG_SUPPRESS:
        return
    if not (enabled or _GLOBAL_DEBUG):
        return
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)


LANGUAGE_PAD_ID = -1
VALID_COLA_STRATEGIES = {"fully", "random_ab", "random_ba", "heuristic"}


class ColaLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.num_A = {}
        self.num_B = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self._cola_expert_parent: dict[str, str] = {}
        self._cola_parent_children: dict[str, list[str]] = {}
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        setattr(self, "_active_adapters", [])
        expert_num_A_override = kwargs.pop("expert_num_A", None)
        expert_num_B_override = kwargs.pop("expert_num_B", None)
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

        # hierarchical design addition
        self.use_cola_experts = kwargs.pop("use_cola_experts", False)
        self.num_experts = kwargs.pop("cola_num_experts", 1)
        self.cola_debug = kwargs.pop("cola_debug", False)
        self.top_k = kwargs.pop("cola_top_k", 1)
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
        self.cola_strategy = kwargs.pop("cola_strategy", "fully")
        self.language_guidance_scope = kwargs.pop("language_guidance_scope", "all")
        if self.language_guidance_scope not in {"all", "expert_only", "none"}:
            raise ValueError(f"Unknown language_guidance_scope '{self.language_guidance_scope}'.")
        if self.cola_strategy not in VALID_COLA_STRATEGIES:
            raise ValueError(f"Unknown CoLA collaboration strategy '{self.cola_strategy}'.")
        self._language_to_idx = {lang: idx for idx, lang in enumerate(self.language_list)} if self.language_list else {}
        self._family_to_idx = {fam: idx for idx, fam in enumerate(self.family_list)} if self.family_list else {}
        self._family_a_modules: dict[int, nn.ModuleList] = {}
        self._family_members: dict[int, list[int]] = {}
        self._expert_language_idx: dict[str, Optional[int]] = {}
        if self.language_list and self.language_to_family_ids is not None:
            for lang_idx, fam_idx in enumerate(self.language_to_family_ids):
                self._family_members.setdefault(int(fam_idx), []).append(lang_idx)
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

        if self.use_cola_experts:
            if self.family_list:
                self.num_experts = len(self.family_list)
            elif self.language_list:
                self.num_experts = len(self.language_list)
            if self.top_k > self.num_experts:
                self.top_k = self.num_experts
            self.router = nn.Linear(self.in_features, self.num_experts, bias=False)
            self._move_router_to_device_of_base_layer()
        self._missing_language_warning_emitted = False
        self._expert_num_A = self._normalize_expert_counts(expert_num_A_override, "expert_num_A")
        self._expert_num_B = self._normalize_expert_counts(expert_num_B_override, "expert_num_B")
        self._debug_forward_calls = 0


    def _fsdp_summon_is_active(self) -> bool:
        """Check whether we can safely summon full params (FSDP world + initialized dist) whoich we need for sharded weights."""
        return FSDP is not None and dist.is_available() and dist.is_initialized()

    @contextmanager
    def _summon_base_weight(self, writeback: bool = False):
        """Temporarily materialize the full base weight tensor if it is FSDP-sharded for shared pissa init"""
        base_layer = self.get_base_layer()
        if self._fsdp_summon_is_active():
            with FSDP.summon_full_params(base_layer, recurse=False, writeback=writeback):
                yield base_layer.weight
                return
        yield base_layer.weight

    def _clone_base_weight(self) -> tuple[torch.Tensor, torch.dtype]:
        """Return a detached clone of the (possibly sharded) base weight and its original dtype."""
        rank, world_size = self._distributed_rank_world()
        with self._summon_base_weight(writeback=False) as weight_param:
            dtype = weight_param.dtype
            orig_device = weight_param.device
            bcast_device = self._preferred_svd_device(weight_param.data)
            if rank == 0:
                weight = weight_param.data.detach().clone().to(bcast_device)
            else:
                weight = torch.empty(weight_param.data.shape, device=bcast_device, dtype=weight_param.dtype)

        if world_size > 1:
            dist.broadcast(weight, src=0)

        weight = weight.to(orig_device)

        return weight, dtype

    def _write_base_weight(self, updated_weight: torch.Tensor, dtype: torch.dtype) -> None:
        """Write an updated dense weight tensor back into the sharded parameter."""
        with self._summon_base_weight(writeback=True) as weight_param:
            weight_param.data.copy_(updated_weight.to(dtype).to(weight_param.device))

    def _distributed_rank_world(self) -> tuple[int, int]:
        """Utility to fetch (rank, world_size) even when torch.distributed is unused."""
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _preferred_svd_device(self, tensor: torch.Tensor) -> torch.device:
        """
        Determine which device to run SVD/broadcast buffers on.
        Prefer the tensor's current device; otherwise fall back to the local GPU if available.
        """
        device = tensor.device
        if device.type == "cuda":
            return device
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
            except (TypeError, ValueError):
                local_rank = 0
            local_rank = local_rank % torch.cuda.device_count()
            return torch.device("cuda", local_rank)
        return device

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, num_A, num_B, init_lora_weights):
        use_pissa = self._should_use_pissa(init_lora_weights)
        if self.use_cola_experts and adapter_name == self._active_adapter:
            self._active_adapters = []
            adapter_names = []

            language_order = self.language_list or []
            use_family_experts = bool(self.family_list)
            for e in range(self.num_experts):
                name = f"expert_{e}"
                adapter_names.append(name)
                self._cola_expert_parent[name] = adapter_name
                if use_family_experts:
                    language_idx = None
                    family_idx = e
                else:
                    language_idx = e if self.language_list and e < len(language_order) else None
                    family_idx = None
                    if language_idx is not None and self.language_to_family_ids is not None:
                        if language_idx < len(self.language_to_family_ids):
                            family_idx = self.language_to_family_ids[language_idx]
                self._expert_language_idx[name] = language_idx
                expert_num_A = self._expert_num_A[e] if self._expert_num_A else num_A
                expert_num_B = self._expert_num_B[e] if self._expert_num_B else num_B

                self.r[name] = r
                self.lora_alpha[name] = lora_alpha
                self.num_A[name] = expert_num_A
                self.num_B[name] = expert_num_B
                self.lora_dropout[name] = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
                if family_idx is not None:
                    shared = self._family_a_modules.get(family_idx)
                    if shared is None:
                        shared = nn.ModuleList([nn.Linear(self.in_features, r, bias=False) for _ in range(expert_num_A)])
                        self._family_a_modules[family_idx] = shared
                    elif len(shared) != expert_num_A:
                        raise ValueError(
                            f"Family {family_idx} expects {len(shared)} shared A stacks, but expert {e} requested {expert_num_A}."
                        )
                    module_list = nn.ModuleList()
                    for shared_layer in shared:
                        wrapper = nn.Linear(self.in_features, r, bias=False)
                        wrapper.weight = shared_layer.weight
                        module_list.append(wrapper)
                    self.lora_A[name] = module_list
                else:
                    self.lora_A[name] = nn.ModuleList([nn.Linear(self.in_features, r, bias=False) for _ in range(expert_num_A)])
                self.lora_B[name] = nn.ModuleList([nn.Linear(r, self.out_features, bias=False) for _ in range(expert_num_B)])
                self.scaling[name] = lora_alpha / r

                self._move_adapter_to_device_of_base_layer(name)
                self._active_adapters.append(name)

            for name in adapter_names:
                for a in self.lora_A[name]:
                    for p in a.parameters():
                        p.requires_grad = True
                for b in self.lora_B[name]:
                    for p in b.parameters():
                        p.requires_grad = True

            if use_pissa:
                self.shared_pissa_init(adapter_names)
            elif init_lora_weights:
                for name in adapter_names:
                    self.reset_lora_parameters(name, init_lora_weights)

            self._verify_cola_expert_init()
            self._cola_parent_children[adapter_name] = adapter_names
            self.set_adapter(adapter_name)
            debug(
                f"[MoE-COLA] Created {self.num_experts} CoLA experts (r={r}, num_A={num_A}, num_B={num_B})",
                enabled=self.cola_debug,
            )
            self._debug_print_cola_setup()
            return

        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.num_A[adapter_name] = num_A
        self.num_B[adapter_name] = num_B
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        self.lora_A[adapter_name] = nn.ModuleList([nn.Linear(self.in_features, r, bias=False) for _ in range(num_A)])
        self.lora_B[adapter_name] = nn.ModuleList([nn.Linear(r, self.out_features, bias=False) for _ in range(num_B)])
        self.scaling[adapter_name] = lora_alpha / r

        if use_pissa:
            self.pissa_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        # Ensure the newly added adapter becomes the only active adapter.
        self._active_adapters = [adapter_name]
        self.set_adapter(self._active_adapters)

    def set_adapter(self, adapter_names: str | list[str]) -> None:
        """Override to keep MoE experts in sync with their parent adapter selection."""
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        expanded: list[str] = []
        cola_parent_active = False
        for name in adapter_names:
            if self.use_cola_experts and name in self._cola_parent_children:
                expanded.extend(self._cola_parent_children[name])
                cola_parent_active = True
            else:
                expanded.append(name)
                if self.use_cola_experts and name in self._cola_expert_parent:
                    cola_parent_active = True

        if self.use_cola_experts and hasattr(self, "router"):
            self._move_router_to_device_of_base_layer()
            self.router.requires_grad_(cola_parent_active)

        super().set_adapter(expanded)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                # nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
                for layer in self.lora_A[adapter_name]:
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "gaussian":
                for layer in self.lora_A[adapter_name]:
                    nn.init.normal_(layer.weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            for layer in self.lora_B[adapter_name]:
                nn.init.zeros_(layer.weight)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def _should_use_pissa(self, init_lora_weights: Union[bool, str]) -> bool:
        return isinstance(init_lora_weights, str) and init_lora_weights.lower().startswith("pissa")

    def pissa_init(self, adapter_name):
        weight, dtype = self._clone_base_weight()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        rank, world_size = self._distributed_rank_world()
        nan_count = torch.isnan(weight).sum().item()
        inf_count = torch.isinf(weight).sum().item()
        debug(
            f"[PiSSA-DEBUG] rank={rank}/{world_size} layer={self.get_base_layer().__class__.__name__} "
            f"shape={tuple(weight.shape)} nan={nan_count} inf={inf_count}",
            enabled=self.cola_debug,
        )
        if not torch.isfinite(weight).all():
            raise ValueError(
                "Encountered non-finite values in base weight while running PiSSA init. "
                "Ensure FSDP/ZeRO shards are materialized before initialization."
            )
        weight_fp32 = weight.to(torch.float32)  # run SVD under higher precision for stability
        compute_device = self._preferred_svd_device(weight_fp32)
        weight_fp32 = weight_fp32.to(compute_device)

        r = self.r[adapter_name]
        out_features, in_features = weight_fp32.shape
        lora_A = torch.empty((r, in_features), dtype=torch.float32, device=compute_device)
        lora_B = torch.empty((out_features, r), dtype=torch.float32, device=compute_device)
        updated_weight = torch.empty_like(weight_fp32, device=compute_device)

        if rank == 0:
            V, S, Uh = torch.linalg.svd(weight_fp32, full_matrices=False)

            Vr = V[:, :r]
            Sr = S[:r]
            Sr /= self.scaling[adapter_name]
            Uhr = Uh[:r]

            lora_A_tmp = (torch.diag(torch.sqrt(Sr)) @ Uhr) / self.num_A[adapter_name]
            lora_B_tmp = (Vr @ torch.diag(torch.sqrt(Sr))) / self.num_B[adapter_name]
            updated_weight_tmp = weight_fp32 - self.scaling[adapter_name] * (
                    (lora_B_tmp * self.num_B[adapter_name]) @ (lora_A_tmp * self.num_A[adapter_name])
            )

            lora_A.copy_(lora_A_tmp)
            lora_B.copy_(lora_B_tmp)
            updated_weight.copy_(updated_weight_tmp)

        if world_size > 1:
            dist.broadcast(lora_A, src=0)
            dist.broadcast(lora_B, src=0)
            dist.broadcast(updated_weight, src=0)

        for i in range(self.num_A[adapter_name]):
            target = self.lora_A[adapter_name][i].weight
            target.data.copy_(lora_A.to(dtype=target.dtype, device=target.device))

        for i in range(self.num_B[adapter_name]):
            target = self.lora_B[adapter_name][i].weight
            target.data.copy_(lora_B.to(dtype=target.dtype, device=target.device))

        self._write_base_weight(updated_weight, dtype)

    def shared_pissa_init(self, adapter_names):
        weight, dtype = self._clone_base_weight()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        rank, world_size = self._distributed_rank_world()
        nan_count = torch.isnan(weight).sum().item()
        inf_count = torch.isinf(weight).sum().item()
        debug(
            f"[PiSSA-DEBUG] rank={rank}/{world_size} layer={self.get_base_layer().__class__.__name__} "
            f"shape={tuple(weight.shape)} nan={nan_count} inf={inf_count}",
            enabled=self.cola_debug,
        )
        if not torch.isfinite(weight).all():
            raise ValueError(
                "Encountered non-finite values in base weight while running shared PiSSA init. "
                "Ensure FSDP/ZeRO shards are materialized before initialization."
            )
        weight_fp32 = weight.to(torch.float32)  # work on dense fp32 copy undependant of original dtype
        compute_device = self._preferred_svd_device(weight_fp32)
        weight_fp32 = weight_fp32.to(compute_device)

        ref = adapter_names[0]

        r = self.r[ref]
        out_features, in_features = weight_fp32.shape
        lora_A = torch.empty((r, in_features), dtype=torch.float32, device=compute_device)
        lora_B = torch.empty((out_features, r), dtype=torch.float32, device=compute_device)
        updated_weight = torch.empty_like(weight_fp32, device=compute_device)

        if rank == 0:
            V, S, Uh = torch.linalg.svd(weight_fp32, full_matrices=False)

            Vr = V[:, :r]
            Sr = S[:r]
            Sr /= self.scaling[ref]
            Uhr = Uh[:r]

            lora_A_tmp = (torch.diag(torch.sqrt(Sr)) @ Uhr) / self.num_A[ref]
            lora_B_tmp = (Vr @ torch.diag(torch.sqrt(Sr))) / self.num_B[ref]
            updated_weight_tmp = weight_fp32 - self.scaling[ref] * (
                    (lora_B_tmp * self.num_B[ref]) @ (lora_A_tmp * self.num_A[ref])
            )

            lora_A.copy_(lora_A_tmp)
            lora_B.copy_(lora_B_tmp)
            updated_weight.copy_(updated_weight_tmp)

        if world_size > 1:
            dist.broadcast(lora_A, src=0)
            dist.broadcast(lora_B, src=0)
            dist.broadcast(updated_weight, src=0)

        # assign same init to all experts
        for name in adapter_names:
            for i in range(self.num_A[name]):
                target = self.lora_A[name][i].weight
                target.data.copy_(lora_A.to(dtype=target.dtype, device=target.device))
            for i in range(self.num_B[name]):
                target = self.lora_B[name][i].weight
                target.data.copy_(lora_B.to(dtype=target.dtype, device=target.device))

        # residualize base weight only once
        self._write_base_weight(updated_weight, dtype)

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

    def _verify_cola_expert_init(self):
        """Verify CoLA expert initialization if cola_debug flag is active."""
        if not getattr(self, "cola_debug", False):
            return

        debug(
            f"[COLA DEBUG] Verifying CoLA experts in {self.__class__.__name__} ({self.num_experts} experts)",
            enabled=self.cola_debug,
        )
        for i in range(self.num_experts):
            debug(
                f"  Expert {i}: "
                f"A_shapes={[a.weight.shape for a in self.lora_A[f'expert_{i}']]}, "
                f"B_shapes={[b.weight.shape for b in self.lora_B[f'expert_{i}']]}, "
                f"scaling={self.scaling[f'expert_{i}']}",
                enabled=self.cola_debug,
            )
        for i in range(self.num_experts - 1):
            a_equal = torch.allclose(
                self.lora_A[f'expert_{i}'][0].weight,
                self.lora_A[f'expert_{i + 1}'][0].weight
            )
            b_equal = torch.allclose(
                self.lora_B[f'expert_{i}'][0].weight,
                self.lora_B[f'expert_{i + 1}'][0].weight
            )
            debug(
                f"[COLA DEBUG] Experts {i} vs {i + 1}: A identical={a_equal}, B identical={b_equal}",
                enabled=self.cola_debug,
            )
        debug("=" * 60, enabled=self.cola_debug)

    def _debug_print_cola_setup(self) -> None:
        if not getattr(self, "cola_debug", False):
            return
        if not getattr(self, "use_cola_experts", False):
            return
        debug(f"[COLA DEBUG] Setup: experts={self.num_experts}, top_k={self.top_k}", enabled=self.cola_debug)
        for e in range(self.num_experts):
            name = f"expert_{e}"
            a_cnt = len(self.lora_A.get(name, []))
            b_cnt = len(self.lora_B.get(name, []))
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
            debug(f"[COLA DEBUG]   {name}: num_A={a_cnt}, num_B={b_cnt}{suffix}", enabled=self.cola_debug)
        debug("=" * 60, enabled=self.cola_debug)

    def _should_debug_routing(self) -> bool:
        if not getattr(self, "cola_debug", False):
            return False
        every = _GLOBAL_DEBUG_ROUTING_EVERY
        if every <= 0:
            return False
        self._debug_forward_calls += 1
        return (self._debug_forward_calls % every) == 0

    def _normalize_expert_counts(self, counts: Optional[list[int]], label: str) -> Optional[list[int]]:
        if not self.use_cola_experts or counts is None:
            return None
        values = list(counts)
        if not values:
            return None
        if len(values) != self.num_experts:
            raise ValueError(f"{label} must provide {self.num_experts} entries, got {len(values)}.")
        if any(v <= 0 for v in values):
            raise ValueError(f"{label} entries must be positive integers.")
        return values

    def _language_expert_targets(self, language_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if (language_ids is None or self.language_id_to_expert.numel() == 0 or self.language_guidance_scope == "none"):
            return None
        if not torch.is_tensor(language_ids):
            return None
        mapping = self.language_id_to_expert.to(language_ids.device)
        expert_ids = torch.full_like(language_ids, LANGUAGE_PAD_ID)
        valid = (language_ids >= 0) & (language_ids < mapping.numel())
        if valid.any():
            expert_ids[valid] = mapping[language_ids[valid]]
        return expert_ids

    def _language_head_targets(self, language_ids: Optional[torch.Tensor], adapter_name: str) -> Optional[torch.Tensor]:
        if (language_ids is None or self.language_list is None or self.language_guidance_scope != "all"):
            return None
        if not torch.is_tensor(language_ids):
            return None
        head_count = self.num_B.get(adapter_name)
        if not head_count:
            return None
        mapping = getattr(self, "language_id_to_subgroup", None)
        if mapping is not None and torch.is_tensor(mapping) and mapping.numel() > 0:
            mapping = mapping.to(language_ids.device)
            head_ids = torch.full_like(language_ids, LANGUAGE_PAD_ID)
            valid = (language_ids >= 0) & (language_ids < mapping.numel())
            if valid.any():
                candidate = mapping[language_ids[valid]]
                candidate = torch.where(candidate < head_count, candidate, torch.full_like(candidate, LANGUAGE_PAD_ID))
                head_ids[valid] = candidate
            return head_ids

        head_ids = torch.full_like(language_ids, LANGUAGE_PAD_ID)
        valid = (language_ids >= 0) & (language_ids < len(self.language_list))
        if valid.any():
            head_ids[valid] = language_ids[valid] % head_count
        return head_ids

    def _language_head_weights(
        self, head_ids: Optional[torch.Tensor], head_count: int, *, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        if head_ids is None or head_count <= 0:
            return None
        if not torch.is_tensor(head_ids):
            return None
        # In "learned" mode, CoLA head weighting is disabled entirely so we preserve paper-faithful
        # fully-collaborative summation (no implicit 1/head_count scaling).
        if self.language_head_router_mode == "learned":
            return None
        batch = head_ids.size(0)
        valid = (head_ids >= 0) & (head_ids < head_count)

        if self.language_head_router_mode == "hard":
            weights = torch.full((batch, head_count), 0.0, device=device, dtype=dtype)
            if valid.any():
                weights[valid, head_ids[valid]] = 1.0
            if (~valid).any():
                weights[~valid] = 1.0 / head_count
            return weights

        if self.language_head_router_mode == "bias":
            logits = torch.zeros((batch, head_count), device=device, dtype=torch.float32)
            if valid.any():
                logits[valid, head_ids[valid]] = float(self.language_head_bias_value)
            return torch.softmax(logits, dim=-1).to(dtype)

        return torch.full((batch, head_count), 1.0 / head_count, device=device, dtype=dtype)

    def _log_missing_language_metadata(self, reason: str) -> None:
        if self._missing_language_warning_emitted:
            return
        column = self.language_column or "<unset>"
        logger.warning(
            "CoLA layer '%s' is missing language routing metadata (%s). Check dataset column '%s' and language_map.",
            self.__class__.__name__,
            reason,
            column,
        )
        self._missing_language_warning_emitted = True

    def _append_language_target_metrics(
        self,
        metrics: dict[str, float],
        metrics_weight: float,
        language_targets: Optional[torch.Tensor],
        language_ids: Optional[torch.Tensor],
        router_probs: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> float:
        if language_targets is not None and torch.is_tensor(language_targets):
            valid_batch = language_targets >= 0
            seq_len = top_indices.size(1)
            if valid_batch.any():
                expanded_targets = language_targets[valid_batch].view(-1, 1).expand(-1, seq_len)
                top1 = top_indices[valid_batch, :, 0]
                target_hits = (top1 == expanded_targets).float()
                target_probs = router_probs[valid_batch].gather(
                    -1,
                    language_targets[valid_batch].view(-1, 1, 1).expand(-1, seq_len, 1),
                ).squeeze(-1)
                valid_tokens = target_probs.numel()
                target_entropy = float((-torch.log(target_probs + 1e-8)).mean().item())
                metrics.update(
                    {
                        "language_target_hit_rate": float(target_hits.mean().item()),
                        "language_target_prob_mean": float(target_probs.mean().item()),
                        "language_target_neglogp": target_entropy,
                        "language_target_token_frac": float(
                            valid_tokens / max(seq_len * valid_batch.sum().item(), 1)
                        ),
                    }
                )
                return float(valid_tokens if valid_tokens > 0 else metrics_weight)

            metrics.update(
                {
                    "language_target_hit_rate": 0.0,
                    "language_target_prob_mean": 0.0,
                    "language_target_neglogp": 0.0,
                    "language_target_token_frac": 0.0,
                }
            )
            if self.language_list:
                self._log_missing_language_metadata("language targets were all pad ids")
            return metrics_weight

        if self.language_list:
            metrics.update(
                {
                    "language_target_hit_rate": 0.0,
                    "language_target_prob_mean": 0.0,
                    "language_target_neglogp": 0.0,
                    "language_target_token_frac": 0.0,
                }
            )
            if language_ids is None:
                reason = "no language_ids tensor provided"
            elif torch.is_tensor(language_ids) and (language_ids >= 0).any():
                reason = "language_ids could not be mapped to experts"
            else:
                reason = "language_ids contained only pad ids"
            self._log_missing_language_metadata(reason)
        return metrics_weight

    def _apply_language_bias(self, logits: torch.Tensor, expert_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if expert_ids is None or self.language_router_mode != "bias":
            return logits
        valid = expert_ids >= 0
        if not valid.any():
            return logits
        bias = torch.zeros(logits.size(0), logits.size(-1), device=logits.device, dtype=logits.dtype)
        bias[valid, expert_ids[valid]] = self.language_bias_value
        return logits + bias.unsqueeze(1)

    def _enforce_language_routing(
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
        self, logits: torch.Tensor, language_ids: Optional[torch.Tensor], expert_ids: Optional[torch.Tensor]
    ) -> None:
        self._cache_store("cola_router_logits", logits)
        if language_ids is not None and torch.is_tensor(language_ids):
            self._cache_store("cola_router_language_ids", language_ids)
        if expert_ids is not None and torch.is_tensor(expert_ids):
            self._cache_store("cola_router_targets", expert_ids)


"""
# TODO: think about reintroducing later
class ColaExpert(nn.Module):
    def __init__(self, in_features, out_features, num_A, num_B, r, lora_alpha, dropout=0.0):
        super().__init__()
        self.As = nn.ModuleList([nn.Linear(in_features, r, bias=False) for _ in range(num_A)])
        self.Bs = nn.ModuleList([nn.Linear(r, out_features, bias=False) for _ in range(num_B)])

        self.dropout = nn.Dropout(dropout)
        self.scale = lora_alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = sum(A(self.dropout(x)) for A in self.As) / len(self.As)
        out = sum(B(h) for B in self.Bs) / len(self.Bs)
        return out * self.scale
"""


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, ColaLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            base_layer,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            num_A: int = 1,
            num_B: int = 1,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            is_target_conv_1d_layer: bool = False,
            init_lora_weights: Union[bool, str] = True,
            **kwargs,
    ) -> None:
        super().__init__()

        ColaLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_A=num_A,
            num_B=num_B,
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
        # currently unused but popped to avoid leaking into base layer

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            return self.base_layer(x, *args, **kwargs)
        else:
            # base_dtype = self.base_layer.weight.dtype   # get dtype of base layers weight
            # x = x.to(base_dtype)                        # match dtype due to mixed precision errors
            result = self.base_layer(x, *args, **kwargs)  # base output
            torch_result_dtype = result.dtype

            if not getattr(self, "use_cola_experts", False):
                for active_adapter in self._active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]

                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    adapter_input = dropout(x.to(lora_A[0].weight.dtype))
                    a_outputs = [layer(adapter_input) for layer in lora_A]
                    if not a_outputs or len(lora_B) == 0:
                        continue

                    if self.cola_strategy == "fully":
                        for a_out in a_outputs:
                            for B_layer in lora_B:
                                result = result + B_layer(a_out) * scaling
                    elif self.cola_strategy == "random_ab":
                        if len(lora_B) == 0:
                            continue
                        for a_out in a_outputs:
                            idx = torch.randint(len(lora_B), (1,), device=a_out.device).item()
                            result = result + lora_B[idx](a_out) * scaling
                    elif self.cola_strategy == "random_ba":
                        if len(a_outputs) == 0:
                            continue
                        for B_layer in lora_B:
                            idx = torch.randint(len(a_outputs), (1,), device=a_outputs[0].device).item()
                            result = result + B_layer(a_outputs[idx]) * scaling
                    elif self.cola_strategy == "heuristic":
                        num_a = len(a_outputs)
                        num_b = len(lora_B)
                        if num_b < num_a or num_a == 0:
                            # fall back to fully-collaborative if not enough B heads
                            for a_out in a_outputs:
                                for B_layer in lora_B:
                                    result = result + B_layer(a_out) * scaling
                            continue
                        ratio = max(1, num_b // num_a)
                        b_idx = 0
                        for a_out in a_outputs:
                            assigned = 0
                            while assigned < ratio and b_idx < num_b:
                                result = result + lora_B[b_idx](a_out) * scaling
                                b_idx += 1
                                assigned += 1
                        while b_idx < num_b:
                            target_idx = b_idx % num_a
                            result = result + lora_B[b_idx](a_outputs[target_idx]) * scaling
                            b_idx += 1
                    else:
                        raise ValueError(f"Unsupported CoLA strategy '{self.cola_strategy}'.")

            else:
                router_dtype = self.router.weight.dtype
                router_inp = x.to(router_dtype)
                logits = self.router(router_inp)
                language_targets = self._language_expert_targets(language_ids)
                if self.language_guidance_scope != "none":
                    self._cache_router_state(logits, language_ids, language_targets)
                logits = self._apply_language_bias(logits, language_targets)
                topv, topi = torch.topk(logits, self.top_k, dim=-1)
                weights = torch.softmax(topv.to(torch.float32), dim=-1).to(x.dtype)
                topi, weights = self._enforce_language_routing(topi, weights, language_targets)
                if self._should_debug_routing():
                    with torch.no_grad():
                        sample_b, sample_t = 0, 0
                        lang_id = int(language_ids[sample_b].item()) if language_ids is not None else LANGUAGE_PAD_ID
                        tgt_ex = (
                            int(language_targets[sample_b].item())
                            if language_targets is not None and torch.is_tensor(language_targets)
                            else LANGUAGE_PAD_ID
                        )
                        sel_ex = [int(v) for v in topi[sample_b, sample_t].detach().cpu().tolist()]
                        sel_w = [float(v) for v in weights[sample_b, sample_t].detach().cpu().tolist()]
                        debug(
                            f"[COLA DEBUG] Sample(b={sample_b},t={sample_t}) lang_id={lang_id} "
                            f"expert_target={tgt_ex} top_experts={sel_ex} weights={sel_w}",
                            enabled=True,
                        )
                        if tgt_ex >= 0:
                            expert_name = f"expert_{tgt_ex}"
                            head_ids = self._language_head_targets(language_ids, expert_name)
                            head_id = int(head_ids[sample_b].item()) if head_ids is not None else LANGUAGE_PAD_ID
                            head_w = self._language_head_weights(
                                torch.tensor([head_id], device=x.device),
                                int(self.num_B.get(expert_name, 0) or 0),
                                device=x.device,
                                dtype=torch.float32,
                            )
                            if head_w is not None:
                                debug(
                                    f"[COLA DEBUG]   head_target={head_id} head_weights={head_w[0].detach().cpu().tolist()}",
                                    enabled=True,
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

                        total_assign = counts.sum()
                        if total_assign > 0:
                            load_frac = counts / total_assign
                            max_frac = float(load_frac.max().item())
                            min_frac = float(load_frac.min().item())
                        else:
                            max_frac = 0.0
                            min_frac = 0.0

                        metrics = {
                            "expert_load_cv": load_cv,
                            "active_expert_frac": active_frac,
                            "router_entropy": entropy,
                            "topk_weight_mean": weight_mean,
                            "expert_load_max_frac": max_frac,
                            "expert_load_min_frac": min_frac,
                        }
                        metrics_weight = float(token_count)
                        metrics_weight = self._append_language_target_metrics(
                            metrics=metrics,
                            metrics_weight=metrics_weight,
                            language_targets=language_targets,
                            language_ids=language_ids,
                            router_probs=router_probs,
                            top_indices=topi,
                        )
                        record_cola_metrics(metrics, weight=metrics_weight)

                expert_outs = [
                    self._adapter_delta(
                        x,
                        f"expert_{e}",
                        language_ids=language_ids,
                        expert_id=e,
                        expert_targets=language_targets,
                    )
                    for e in range(self.num_experts)
                ]
                expert_outs = torch.stack(expert_outs, dim=-1)  # [B, S, D_out, E]

                topi_expanded = topi.unsqueeze(2).expand(-1, -1, expert_outs.size(2), -1)
                gathered = torch.gather(expert_outs, dim=3, index=topi_expanded)
                weights_expanded = weights.unsqueeze(2)
                moe_out = (gathered * weights_expanded).sum(dim=-1)

                result = result + moe_out
            # return result.to(self.base_layer.weight.dtype)
            return result.to(torch_result_dtype)

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

        weight_A = torch.sum(torch.stack([layer.weight for layer in self.lora_A[adapter]]), dim=0)
        weight_B = torch.sum(torch.stack([layer.weight for layer in self.lora_B[adapter]]), dim=0)

        # weight_A = self.lora_A[adapter].weight
        # weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            for i in range(self.num_A[adapter]):
                self.lora_A[adapter][i].weight.data = self.lora_A[adapter][i].weight.to(dtype)
            for i in range(self.num_B[adapter]):
                self.lora_B[adapter][i].weight.data = self.lora_B[adapter][i].weight.to(dtype)

            # self.lora_A[adapter].weight.data = weight_A.to(dtype)
            # self.lora_B[adapter].weight.data = weight_B.to(dtype)

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
        A_list = self.lora_A[name]
        B_list = self.lora_B[name]
        drop = self.lora_dropout[name]
        scale = self.scaling[name]

        if not A_list or not B_list:
            return torch.zeros_like(x, dtype=self.get_base_layer().weight.dtype)

        intermediate = drop(x.to(A_list[0].weight.dtype))
        a_outputs = [A(intermediate) for A in A_list]
        num_a = len(a_outputs)
        num_b = len(B_list)
        strategy = getattr(self, "cola_strategy", "fully")
        if strategy == "heuristic" and (num_b < num_a or num_a == 0):
            strategy = "fully"
        if strategy.startswith("random") and num_b == 0:
            strategy = "fully"

        out = 0
        if strategy == "fully":
            head_weights = None
            head_targets = self._language_head_targets(language_ids, name)
            if head_targets is not None and expert_targets is not None and expert_id is not None:
                head_targets = torch.where(expert_targets == int(expert_id), head_targets, LANGUAGE_PAD_ID)
            if head_targets is not None:
                head_weights = self._language_head_weights(
                    head_targets, num_b, device=intermediate.device, dtype=intermediate.dtype
                )
            if head_weights is None:
                for a_out in a_outputs:
                    for b_layer in B_list:
                        out = out + b_layer(a_out)
            else:
                head_weights = head_weights.view(-1, 1, num_b)
                for b_idx, b_layer in enumerate(B_list):
                    b_sum = 0
                    for a_out in a_outputs:
                        b_sum = b_sum + b_layer(a_out)
                    out = out + b_sum * head_weights[:, :, b_idx].to(b_sum.dtype).unsqueeze(-1)
        elif strategy == "random" or strategy == "random_ab":
            if num_b == 0:
                return torch.zeros_like(a_outputs[0])
            for a_out in a_outputs:
                idx = torch.randint(0, num_b, (1,), device=a_out.device).item()
                out = out + B_list[idx](a_out)
        elif strategy == "random_ba":
            if num_a == 0:
                return torch.zeros_like(B_list[0](intermediate))
            for b_layer in B_list:
                idx = torch.randint(0, num_a, (1,), device=a_outputs[0].device).item()
                out = out + b_layer(a_outputs[idx])
        elif strategy == "heuristic":
            if num_a == 1:
                shared = a_outputs[0]
                for b_layer in B_list:
                    out = out + b_layer(shared)
            else:
                limit = min(num_a - 1, num_b)
                for i in range(limit):
                    out = out + B_list[i](a_outputs[i])
                shared = a_outputs[-1]
                for j in range(limit, num_b):
                    out = out + B_list[j](shared)
        else:
            raise ValueError(f"Unsupported CoLA strategy '{strategy}'.")

        return out * scale

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

    def pop_language_router_cache(self) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        caches: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        logits = self._cache_pop("cola_router_logits")
        targets = self._cache_pop("cola_router_targets")
        if logits is not None and targets is not None:
            caches.append(("cola_expert", logits, targets))
        # ensure we clear stale metadata
        self._cache_pop("cola_router_language_ids")
        return caches


class Embedding(nn.Module, ColaLayer):
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
        ColaLayer.__init__(self, base_layer)

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


class Conv2d(nn.Module, ColaLayer):
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
        ColaLayer.__init__(self, base_layer)

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
        lora_config: ColaConfig,
        **kwargs,
) -> Optional[torch.nn.Module]:
    """
    target: Linear(in_features=3072, out_features=3072, bias=False)
    adapter_name: default
    lora_config: ColaConfig(peft_type=<PeftType.COLA: 'COLA'>,
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
