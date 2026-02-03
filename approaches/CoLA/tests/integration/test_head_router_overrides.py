from __future__ import annotations

import torch
import pytest

from peft.tuners.cola.layer import Linear as ColaLinear
from peft.tuners.hydralora.layer import Linear as HydraLinear


def test_cola_head_router_mode_override_disables_weights_in_learned() -> None:
    base = torch.nn.Linear(4, 4, bias=False)
    layer = ColaLinear(
        base_layer=base,
        adapter_name="default",
        r=2,
        lora_alpha=4,
        num_A=1,
        num_B=3,
        use_cola_experts=True,
        cola_num_experts=1,
        family_list=["g0"],
        language_list=["l0"],
        language_to_family_ids=[0],
        language_to_subgroup_ids=[0],
        language_router_mode="learned",
        language_head_router_mode="learned",
        language_guidance_scope="all",
    )

    head_ids = torch.tensor([0], dtype=torch.long)
    assert layer._language_head_weights(head_ids, head_count=3, device=head_ids.device, dtype=torch.float32) is None


def test_cola_head_router_mode_override_enables_bias_weights() -> None:
    base = torch.nn.Linear(4, 4, bias=False)
    layer = ColaLinear(
        base_layer=base,
        adapter_name="default",
        r=2,
        lora_alpha=4,
        num_A=1,
        num_B=3,
        use_cola_experts=True,
        cola_num_experts=1,
        family_list=["g0"],
        language_list=["l0"],
        language_to_family_ids=[0],
        language_to_subgroup_ids=[1],
        language_router_mode="learned",
        language_head_router_mode="bias",
        language_bias_value=0.0,
        language_head_bias_value=2.0,
        language_guidance_scope="all",
    )

    head_ids = torch.tensor([1], dtype=torch.long)
    weights = layer._language_head_weights(head_ids, head_count=3, device=head_ids.device, dtype=torch.float32)
    assert weights is not None
    assert weights.shape == (1, 3)
    assert float(weights.sum().item()) == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert int(weights.argmax(dim=-1).item()) == 1


def test_hydra_head_router_override_bias_is_separate_from_expert_router() -> None:
    base = torch.nn.Linear(4, 4, bias=False)
    layer = HydraLinear(
        base_layer=base,
        adapter_name="default",
        r=2,
        lora_alpha=4,
        lora_num=3,
        use_hydralora_experts=True,
        hydralora_num_experts=1,
        family_list=["g0"],
        language_list=["l0"],
        language_to_family_ids=[0],
        language_to_subgroup_ids=[2],
        language_router_mode="learned",
        language_head_router_mode="bias",
        language_bias_value=0.0,
        language_head_bias_value=2.0,
        language_guidance_scope="all",
    )

    assert layer.language_router_mode == "learned"
    assert layer.language_head_router_mode == "bias"


def test_cola_hard_expert_routing_forces_target() -> None:
    base = torch.nn.Linear(4, 4, bias=False)
    layer = ColaLinear(
        base_layer=base,
        adapter_name="default",
        r=2,
        lora_alpha=4,
        num_A=1,
        num_B=2,
        use_cola_experts=True,
        cola_num_experts=3,
        cola_top_k=2,
        family_list=["g0", "g1", "g2"],
        language_list=["l0", "l1", "l2"],
        language_to_family_ids=[0, 1, 2],
        language_router_mode="hard",
        language_guidance_scope="all",
    )

    topi = torch.tensor([[[0, 2], [1, 0]], [[2, 0], [0, 2]]], dtype=torch.long)
    weights = torch.full_like(topi, 0.5, dtype=torch.float32)
    expert_targets = torch.tensor([2, 1], dtype=torch.long)

    enforced_topi, enforced_weights = layer._enforce_language_routing(topi, weights, expert_targets)

    assert torch.all(enforced_topi[0] == 2)
    assert torch.all(enforced_topi[1] == 1)
    assert torch.all(enforced_weights[0, :, 0] == 1)
    assert torch.all(enforced_weights[0, :, 1] == 0)
    assert torch.all(enforced_weights[1, :, 0] == 1)
    assert torch.all(enforced_weights[1, :, 1] == 0)


def test_hydra_hard_head_routing_forces_target() -> None:
    base = torch.nn.Linear(4, 4, bias=False)
    layer = HydraLinear(
        base_layer=base,
        adapter_name="default",
        r=2,
        lora_alpha=4,
        lora_num=3,
        use_hydralora_experts=True,
        hydralora_num_experts=2,
        family_list=["g0", "g1"],
        language_list=["l0", "l1"],
        language_to_family_ids=[0, 1],
        language_to_subgroup_ids=[0, 2],
        language_router_mode="learned",
        language_head_router_mode="hard",
        language_guidance_scope="all",
    )

    weights = torch.full((2, 2, 3), 1.0 / 3, dtype=torch.float32)
    head_targets = torch.tensor([0, 2], dtype=torch.long)
    enforced = layer._enforce_language_heads(weights, head_targets)

    assert torch.all(enforced[0, :, 0] == 1)
    assert torch.all(enforced[0, :, 1:] == 0)
    assert torch.all(enforced[1, :, 2] == 1)
    assert torch.all(enforced[1, :, :2] == 0)
