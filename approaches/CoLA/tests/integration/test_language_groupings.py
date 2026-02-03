from __future__ import annotations

import json
from pathlib import Path

import torch

from llamafactory.extras.language import load_language_groupings
from peft.tuners.cola.layer import Linear as ColaLinear
from peft.tuners.hydralora.layer import Linear as HydraLinear, LANGUAGE_PAD_ID


def test_load_language_groupings_subgroups(tmp_path: Path) -> None:
    payload = {
        "0": {
            "languages": ["a", "b"],
            "subgroups": {"B0": ["a"], "B1": ["c"]},
            "metadata": {"d": {"family": "foo"}},
        },
        "1": {"group": "fam1", "languages": ["e"]},
    }
    path = tmp_path / "tier.json"
    path.write_text(json.dumps(payload))

    language_map, families, subgroup_sizes, language_to_subgroup = load_language_groupings(str(path))

    assert language_map["a"] == "0"
    assert language_map["e"] == "fam1"
    assert families == ["0", "fam1"]
    assert subgroup_sizes == [2, 0]  # two subgroups under family "0", none under "fam1"
    assert language_to_subgroup["a"] == 0
    assert language_to_subgroup["c"] == 1
    assert "b" not in language_to_subgroup  # languages without subgroup stay unmapped


def test_cola_expert_b_counts_follow_subgroup_sizes() -> None:
    base = torch.nn.Linear(4, 4, bias=False)
    subgroup_sizes = [2, 1]  # per-expert B counts
    layer = ColaLinear(
        base_layer=base,
        adapter_name="default",
        r=2,
        lora_alpha=4,
        num_A=1,
        num_B=1,
        use_cola_experts=True,
        cola_num_experts=2,
        family_list=["g0", "g1"],
        language_list=["l0", "l1"],
        language_to_family_ids=[0, 1],
        expert_num_B=subgroup_sizes,
    )

    assert len(layer.lora_B["expert_0"]) == 2
    assert len(layer.lora_B["expert_1"]) == 1


def test_hydra_expert_head_counts_and_targets_from_subgroups() -> None:
    base = torch.nn.Linear(4, 4, bias=False)
    subgroup_sizes = [1, 3]
    language_to_subgroup_ids = [0, -1]  # only first language has subgroup mapping
    layer = HydraLinear(
        base_layer=base,
        adapter_name="default",
        r=2,
        lora_alpha=4,
        lora_num=1,  # default, will be overridden per expert
        use_hydralora_experts=True,
        hydralora_num_experts=2,
        family_list=["g0", "g1"],
        language_list=["l0", "l1"],
        language_to_family_ids=[0, 1],
        hydralora_expert_lora_nums=subgroup_sizes,
        language_to_subgroup_ids=language_to_subgroup_ids,
    )

    assert len(layer.lora_B["expert_0"]) == 1
    assert len(layer.lora_B["expert_1"]) == 3

    language_ids = torch.tensor([0, 1])
    head_targets = layer._language_head_targets(language_ids, "expert_0")
    assert head_targets.tolist() == [0, LANGUAGE_PAD_ID]
