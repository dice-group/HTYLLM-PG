from __future__ import annotations

from pathlib import Path

from llamafactory.extras.language import load_language_groupings

from scripts.comparison import router_setup
from scripts.comparison.ablation_specs import (
    default_ablation_script_path,
    default_ground_truth_path,
    parse_cola_variants,
    parse_ground_truth,
    parse_hydra_variants,
    parse_lora_variants,
)


LABEL_TO_VARIANT = {
    "lora-baseline": "A0",
    "colaflat": "C0",
    "colaexp-lpr": "C1",
    "colaexp-headbias": "C2",
    "colaexp-lpr-expert-only": "C1b",
    "hydra-flat": "H0",
    "hydra-exp-lpr": "H1",
    "hydra-exp-lpr-expert-only": "H1b",
}


def _get_value(config: dict[str, str], key: str, default: str) -> str:
    value = config.get(key)
    if value is None or value == "":
        return default
    return value


def test_ground_truth_doc_has_expected_variants() -> None:
    ground_truth = parse_ground_truth(default_ground_truth_path())
    expected = {"A0", "C0", "C1", "C2", "H0", "H1", "C1b", "H1b"}
    missing = expected - ground_truth.keys()
    assert not missing, f"Missing variants in ground truth doc: {sorted(missing)}"
    for key in expected:
        assert "finetuning_type" in ground_truth[key]


def test_script_variants_match_ground_truth() -> None:
    ground_truth = parse_ground_truth(default_ground_truth_path())
    script_path = default_ablation_script_path()

    cola_variants = parse_cola_variants(script_path, include_commented=True)
    hydra_variants = parse_hydra_variants(script_path, include_commented=True)
    lora_variants = parse_lora_variants(script_path, include_commented=True)

    labels = {v.label for v in cola_variants + hydra_variants + lora_variants}
    expected_labels = set(LABEL_TO_VARIANT.keys())
    missing_labels = expected_labels - labels
    assert not missing_labels, f"Missing labels in run script: {sorted(missing_labels)}"

    for variant in cola_variants:
        variant_id = LABEL_TO_VARIANT.get(variant.label)
        assert variant_id is not None
        cfg = ground_truth[variant_id]
        assert cfg.get("finetuning_type") == "cola"
        assert variant.use_experts is (_get_value(cfg, "USE_COLA_EXPERTS", "False") == "True")
        assert variant.num_A == int(_get_value(cfg, "NUM_A", str(variant.num_A)))
        assert variant.num_B == int(_get_value(cfg, "NUM_B", str(variant.num_B)))
        assert variant.strategy == _get_value(cfg, "COLA_STRATEGY", variant.strategy)
        assert variant.top_k == int(_get_value(cfg, "COLA_TOP_K", str(variant.top_k)))
        assert variant.router_mode == _get_value(cfg, "LANGUAGE_ROUTER_MODE", variant.router_mode)
        assert variant.guidance_scope == _get_value(cfg, "LANGUAGE_GUIDANCE_SCOPE", variant.guidance_scope)
        assert variant.prior_weight == float(_get_value(cfg, "LANGUAGE_PRIOR_WEIGHT", str(variant.prior_weight)))
        assert variant.bias_value == float(_get_value(cfg, "LANGUAGE_BIAS_VALUE", str(variant.bias_value)))
        expected_head_mode = _get_value(cfg, "LANGUAGE_HEAD_ROUTER_MODE", variant.router_mode)
        expected_head_bias = _get_value(cfg, "LANGUAGE_HEAD_BIAS_VALUE", str(variant.bias_value))
        assert variant.head_router_mode == expected_head_mode
        assert variant.head_bias_value == float(expected_head_bias)

    for variant in hydra_variants:
        variant_id = LABEL_TO_VARIANT.get(variant.label)
        assert variant_id is not None
        cfg = ground_truth[variant_id]
        assert cfg.get("finetuning_type") == "hydralora"
        assert variant.use_experts is (_get_value(cfg, "USE_HYDRALORA_EXPERTS", "False") == "True")
        assert variant.lora_num == int(_get_value(cfg, "LORA_NUM", str(variant.lora_num)))
        assert variant.top_k == int(_get_value(cfg, "HYDRALORA_TOP_K", str(variant.top_k)))
        assert variant.router_mode == _get_value(cfg, "LANGUAGE_ROUTER_MODE", variant.router_mode)
        assert variant.guidance_scope == _get_value(cfg, "LANGUAGE_GUIDANCE_SCOPE", variant.guidance_scope)
        assert variant.prior_weight == float(_get_value(cfg, "LANGUAGE_PRIOR_WEIGHT", str(variant.prior_weight)))
        assert variant.bias_value == float(_get_value(cfg, "LANGUAGE_BIAS_VALUE", str(variant.bias_value)))
        expected_head_mode = _get_value(cfg, "LANGUAGE_HEAD_ROUTER_MODE", variant.router_mode)
        expected_head_bias = _get_value(cfg, "LANGUAGE_HEAD_BIAS_VALUE", str(variant.bias_value))
        assert variant.head_router_mode == expected_head_mode
        assert variant.head_bias_value == float(expected_head_bias)

    for variant in lora_variants:
        variant_id = LABEL_TO_VARIANT.get(variant.label)
        assert variant_id is not None
        cfg = ground_truth[variant_id]
        assert cfg.get("finetuning_type") == "lora"
        assert _get_value(cfg, "LANGUAGE_GUIDANCE_SCOPE", "none") == "none"
        assert float(_get_value(cfg, "LANGUAGE_PRIOR_WEIGHT", "0.0")) == 0.0


def test_router_setup_validates_ground_truth_variants() -> None:
    tier_path = Path("tools/two_stage_clustering/12_tier_language_groupings.json")
    assert tier_path.exists()
    _, families, _, _ = load_language_groupings(str(tier_path))
    expected_experts = len(families or [])

    script_path = default_ablation_script_path()
    cola_variants = parse_cola_variants(script_path, include_commented=True)
    hydra_variants = parse_hydra_variants(script_path, include_commented=True)

    for variant in cola_variants:
        env = {
            "LANGUAGE_MAP": str(tier_path),
            "MODEL_VARIANT": variant.label,
            "LANGUAGE_TIER": "tier12",
            "USE_COLA_EXPERTS": str(variant.use_experts),
            "COLA_NUM_EXPERTS": str(expected_experts if variant.use_experts else 1),
            "COLA_TOP_K": str(variant.top_k),
            "COLA_NUM_B": str(variant.num_B),
            "LANGUAGE_ROUTER_MODE": variant.router_mode,
            "LANGUAGE_HEAD_ROUTER_MODE": variant.head_router_mode,
            "LANGUAGE_GUIDANCE_SCOPE": variant.guidance_scope,
            "LANGUAGE_PRIOR_WEIGHT": str(variant.prior_weight),
            "LANGUAGE_BIAS_VALUE": str(variant.bias_value),
            "LANGUAGE_HEAD_BIAS_VALUE": str(variant.head_bias_value),
        }
        config = router_setup.build_router_config("cola", env)
        router_setup.validate_router_config(config)

    for variant in hydra_variants:
        env = {
            "LANGUAGE_MAP": str(tier_path),
            "MODEL_VARIANT": variant.label,
            "LANGUAGE_TIER": "tier12",
            "USE_HYDRALORA_EXPERTS": str(variant.use_experts),
            "HYDRALORA_NUM_EXPERTS": str(expected_experts if variant.use_experts else 1),
            "HYDRALORA_TOP_K": str(variant.top_k),
            "LORA_NUM": str(variant.lora_num),
            "LANGUAGE_ROUTER_MODE": variant.router_mode,
            "LANGUAGE_HEAD_ROUTER_MODE": variant.head_router_mode,
            "LANGUAGE_GUIDANCE_SCOPE": variant.guidance_scope,
            "LANGUAGE_PRIOR_WEIGHT": str(variant.prior_weight),
            "LANGUAGE_BIAS_VALUE": str(variant.bias_value),
            "LANGUAGE_HEAD_BIAS_VALUE": str(variant.head_bias_value),
        }
        config = router_setup.build_router_config("hydra", env)
        router_setup.validate_router_config(config)
