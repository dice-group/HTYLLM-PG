from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from transformers import Seq2SeqTrainingArguments

from llamafactory.hparams import FinetuningArguments
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from peft.metrics import pop_tracked_metrics
from peft.tuners.cola.layer import Linear as ColaLinear
from peft.tuners.hydralora.layer import Linear as HydraLinear


class DummyRouterModel(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, input_ids: torch.Tensor, language_ids: torch.Tensor | None = None, **_: object):
        output = self.layer(input_ids, language_ids=language_ids)
        return {"loss": output.mean()}


def _build_trainer(tmp_path, model: torch.nn.Module, finetuning_args: FinetuningArguments) -> CustomSeq2SeqTrainer:
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        report_to=[],
        do_train=False,
    )
    return CustomSeq2SeqTrainer(
        model=model,
        args=args,
        tokenizer=None,
        processor=None,
        finetuning_args=finetuning_args,
        data_collator=lambda x: x,
    )


def test_cola_lpr_loss_and_metrics(tmp_path) -> None:
    base = torch.nn.Linear(4, 4, bias=False)
    layer = ColaLinear(
        base_layer=base,
        adapter_name="default",
        r=2,
        lora_alpha=4,
        num_A=1,
        num_B=2,
        use_cola_experts=True,
        cola_num_experts=2,
        cola_top_k=1,
        family_list=["g0", "g1"],
        language_list=["l0", "l1"],
        language_to_family_ids=[0, 1],
        language_router_mode="hard",
        language_guidance_scope="all",
        language_prior_weight=0.1,
    )
    model = DummyRouterModel(layer)
    finetuning_args = replace(FinetuningArguments(), finetuning_type="cola", language_prior_weight=0.1)
    trainer = _build_trainer(tmp_path, model, finetuning_args)

    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.randn(2, 3, 4, device=device),
        "language_ids": torch.tensor([0, 1], device=device),
    }
    model(**inputs)

    loss = trainer._compute_language_prior_loss()
    assert loss is not None
    assert loss.item() > 0
    assert any("language_prior_loss" in entry for entry in trainer.state.log_history)

    metrics = pop_tracked_metrics()
    assert "cola/expert_load_cv" in metrics
    assert metrics["cola/language_target_hit_rate"] == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_hydra_lpr_loss_and_metrics(tmp_path) -> None:
    base = torch.nn.Linear(4, 4, bias=False)
    layer = HydraLinear(
        base_layer=base,
        adapter_name="default",
        r=2,
        lora_alpha=4,
        lora_num=2,
        use_hydralora_experts=True,
        hydralora_num_experts=2,
        hydralora_top_k=1,
        family_list=["g0", "g1"],
        language_list=["l0", "l1"],
        language_to_family_ids=[0, 1],
        language_to_subgroup_ids=[0, 1],
        language_router_mode="hard",
        language_guidance_scope="all",
        language_prior_weight=0.1,
    )
    model = DummyRouterModel(layer)
    finetuning_args = replace(FinetuningArguments(), finetuning_type="hydralora", language_prior_weight=0.1)
    trainer = _build_trainer(tmp_path, model, finetuning_args)

    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.randn(2, 3, 4, device=device),
        "language_ids": torch.tensor([0, 1], device=device),
    }
    model(**inputs)

    loss = trainer._compute_language_prior_loss()
    assert loss is not None
    assert loss.item() > 0
    assert any("language_prior_loss" in entry for entry in trainer.state.log_history)

    metrics = pop_tracked_metrics()
    assert "hydralora/expert_load_cv" in metrics
    assert metrics["hydralora/expert_target_hit_rate"] == pytest.approx(1.0, rel=1e-5, abs=1e-5)
