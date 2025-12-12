from __future__ import annotations

import pytest

from scripts.comparison import router_setup


def test_validate_hydra_router_config(tmp_path) -> None:
    language_map = tmp_path / "map.json"
    language_map.write_text("{}")
    env = {
        "LANGUAGE_MAP": str(language_map),
        "MODEL_VARIANT": "hydra-variant",
        "LANGUAGE_TIER": "tier12",
        "USE_HYDRALORA_EXPERTS": "True",
        "HYDRALORA_NUM_EXPERTS": "3",
        "HYDRALORA_TOP_K": "1",
        "LORA_NUM": "4",
        "LANGUAGE_ROUTER_MODE": "learned",
        "LANGUAGE_GUIDANCE_SCOPE": "all",
        "LANGUAGE_PRIOR_WEIGHT": "0.1",
        "LANGUAGE_BIAS_VALUE": "0.0",
    }
    config = router_setup.build_router_config("hydra", env)
    router_setup.validate_router_config(config)


def test_validate_cola_router_requires_language_map(tmp_path) -> None:
    env = {
        "LANGUAGE_MAP": "",
        "MODEL_VARIANT": "cola-flat",
        "LANGUAGE_TIER": "tier12",
        "USE_COLA_EXPERTS": "True",
        "COLA_NUM_EXPERTS": "2",
        "COLA_TOP_K": "1",
        "LANGUAGE_ROUTER_MODE": "learned",
        "LANGUAGE_GUIDANCE_SCOPE": "none",
        "LANGUAGE_PRIOR_WEIGHT": "0.0",
        "LANGUAGE_BIAS_VALUE": "0.0",
    }
    config = router_setup.build_router_config("cola", env)
    with pytest.raises(ValueError):
        router_setup.validate_router_config(config)
