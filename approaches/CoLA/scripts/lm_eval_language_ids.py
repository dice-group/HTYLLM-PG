import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch

logger = logging.getLogger(__name__)


_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _log_adapter_stats(model: torch.nn.Module) -> None:
    try:
        names = [name for name, _ in model.named_parameters()]
    except Exception:  # noqa: BLE001
        return
    router_count = sum(".router." in name for name in names)
    expert_count = sum(".expert_" in name for name in names)
    lora_count = sum(".lora_" in name for name in names)
    logger.info(
        "Adapter params: lora=%d router=%d expert=%d",
        lora_count,
        router_count,
        expert_count,
    )
    cfg = getattr(model, "peft_config", None)
    if cfg:
        for key, val in cfg.items():
            num_experts = getattr(val, "num_experts", None)
            heads_per_expert = getattr(val, "heads_per_expert", None)
            router_mode = getattr(val, "router_mode", None)
            if num_experts or heads_per_expert or router_mode:
                logger.info(
                    "Adapter config[%s]: num_experts=%s heads_per_expert=%s router_mode=%s",
                    key,
                    num_experts,
                    heads_per_expert,
                    router_mode,
                )


def _parse_torch_dtype(value: Optional[str]):
    if value is None:
        return None
    val = str(value).strip().lower()
    if val in ("", "none", "null"):
        return None
    if val == "auto":
        return "auto"
    if val in ("bf16", "bfloat16"):
        return torch.bfloat16
    if val in ("fp16", "float16", "half"):
        return torch.float16
    if val in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def _normalize_device_map(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    val = str(value).strip()
    if not val or val.lower() in ("none", "null"):
        return None
    return val


def _resolve_adapter_dir(path: Path) -> Path:
    if (path / "adapter_config.json").exists():
        return path
    alt = Path(f"{path}_adapter")
    if (alt / "adapter_config.json").exists():
        return alt
    return path


def _load_language_list(adapter_dir: Path) -> list[str]:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    cfg = json.loads(cfg_path.read_text())
    lang_list = cfg.get("language_list")
    if not lang_list:
        raise ValueError("adapter_config.json missing language_list")
    return list(lang_list)


def _parse_tasks(tasks: str) -> list[str]:
    if os.path.isfile(tasks):
        return [line.strip() for line in Path(tasks).read_text().splitlines() if line.strip()]
    return [t.strip() for t in tasks.split(",") if t.strip()]


def _task_to_lang_code(task: str) -> Optional[str]:
    # take suffix after first underscore, e.g. belebele_eng_Latn -> eng_Latn
    if "_" not in task:
        return None
    return task.split("_", 1)[1]


def _build_lang_id_map(lang_list: Iterable[str]) -> dict[str, int]:
    return {lang: idx for idx, lang in enumerate(lang_list)}


def _build_hflm_with_lang_class(HFLM):
    """Create a subclass of HFLM that injects language_ids."""

    class HFLMWithLang(HFLM):
        def __init__(self, language_id: Optional[int] = None, **kwargs):
            self._language_id = language_id
            super().__init__(**kwargs)

        def _inject_lang_ids(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
            if self._language_id is None:
                return None
            return torch.full(
                (input_ids.size(0),),
                int(self._language_id),
                device=input_ids.device,
                dtype=torch.long,
            )

        def _model_call(self, inps, attn_mask=None, labels=None):
            # Mirror lm_eval.models.huggingface.HFLM._model_call, injecting language_ids.
            import transformers

            lang_ids = self._inject_lang_ids(inps)
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=self.mixed_precision_dtype,
                    enabled=self.mixed_precision_dtype is not None,
                ),
            ):
                if attn_mask is not None or labels is not None:
                    assert attn_mask is not None and labels is not None
                    assert transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS
                    return self.model(
                        input_ids=inps,
                        attention_mask=attn_mask,
                        labels=labels,
                        **({"language_ids": lang_ids} if lang_ids is not None else {}),
                    ).logits

                assert self.AUTO_MODEL_CLASS in (
                    transformers.AutoModelForCausalLM,
                    transformers.AutoModelForVision2Seq,
                )
                if lang_ids is not None:
                    return self.model(input_ids=inps, language_ids=lang_ids).logits
                return self.model(inps).logits

        def _model_generate(self, context, max_length, stop, **generation_kwargs):
            lang_ids = self._inject_lang_ids(context)
            if lang_ids is not None:
                generation_kwargs["language_ids"] = lang_ids
            return super()._model_generate(context, max_length, stop, **generation_kwargs)

    return HFLMWithLang


def _run_eval(
    *,
    pretrained: str,
    peft: str,
    tokenizer: str,
    tasks: list[str],
    batch_size: str,
    output_path: Path,
    device: str,
    language_id: Optional[int],
    limit: Optional[float],
    wandb_args: Optional[str],
    wandb_config_args: Optional[str],
    log_samples: bool,
    log_router_metrics: bool,
    torch_dtype,
    device_map: Optional[str],
):
    # Ensure repo-local PEFT (with CoLA/Hydra) is used inside lm_eval.
    repo_root = Path(__file__).resolve().parents[1]
    peft_path = repo_root / "LLaMA-Factory" / "src"
    if str(peft_path) not in sys.path:
        sys.path.insert(0, str(peft_path))
    for name in list(sys.modules.keys()):
        if name == "peft" or name.startswith("peft."):
            del sys.modules[name]

    # Force-load repo-local peft before lm_eval so HFLM uses it.
    import importlib.util

    peft_pkg = peft_path / "peft"
    peft_init = peft_pkg / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "peft", peft_init, submodule_search_locations=[str(peft_pkg)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load local peft from {peft_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["peft"] = module
    spec.loader.exec_module(module)

    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
        from lm_eval.loggers.wandb_logger import WandbLogger
        from lm_eval.utils import simple_parse_args_string
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("lm_eval/transformers is not installed in this environment") from exc

    # Load PEFT locally to support CoLA/Hydra even if site-packages peft lacks them.
    from peft import PeftModel  # repo-local due to sys.path override above

    pop_tracked_metrics = None
    if log_router_metrics:
        try:
            from peft.metrics import pop_tracked_metrics as _pop_tracked_metrics

            pop_tracked_metrics = _pop_tracked_metrics
            pop_tracked_metrics()
        except Exception:  # noqa: BLE001
            pop_tracked_metrics = None

    model_key = (pretrained, peft)
    if model_key not in _MODEL_CACHE:
        load_kwargs = {"low_cpu_mem_usage": True}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        base_model = AutoModelForCausalLM.from_pretrained(pretrained, **load_kwargs)
        logger.info("Base model device: %s", next(base_model.parameters()).device)
        peft_model = PeftModel.from_pretrained(base_model, peft, is_trainable=False)
        _log_adapter_stats(peft_model)
        _MODEL_CACHE[model_key] = peft_model

    HFLMWithLang = _build_hflm_with_lang_class(HFLM)
    model = HFLMWithLang(
        language_id=language_id,
        pretrained=_MODEL_CACHE[model_key],
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )

    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        batch_size=batch_size,
        device=device,
        limit=limit,
        log_samples=log_samples,
    )

    if results is None:
        return

    if log_router_metrics and pop_tracked_metrics is not None:
        router_metrics = pop_tracked_metrics()
        if router_metrics:
            results.setdefault("router_metrics", {}).update(router_metrics)
            results.setdefault("results", {})["_router_metrics"] = router_metrics

    samples = None
    if log_samples and "samples" in results:
        samples = results.pop("samples")

    if wandb_args:
        try:
            wandb_args_dict = simple_parse_args_string(wandb_args)
            wandb_config_args_dict = simple_parse_args_string(wandb_config_args)
            wandb_logger = WandbLogger(wandb_args_dict, wandb_config_args_dict)
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()
            if log_samples and samples:
                wandb_logger.log_eval_samples(samples)
            wandb_logger.run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] W&B logging failed: {exc}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=str))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Adapter checkpoint dir (or base checkpoint) ")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer (base model) path")
    parser.add_argument("--tasks", required=True, help="Comma list or file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=float, default=None, help="lm_eval limit per task")
    parser.add_argument("--mode", choices=["with_ids", "no_ids", "both"], default="both")
    parser.add_argument(
        "--torch-dtype",
        default=os.environ.get("LM_EVAL_TORCH_DTYPE"),
        help="Torch dtype for model load (auto|bf16|fp16|fp32).",
    )
    parser.add_argument(
        "--device-map",
        default=os.environ.get("LM_EVAL_DEVICE_MAP"),
        help="Optional device_map for model load (e.g. auto).",
    )
    parser.add_argument("--wandb-args", default=None, help="Comma args for wandb.init, e.g. project=lm-eval,job_type=eval")
    parser.add_argument("--wandb-config-args", default=None, help="Comma args for wandb.config.update")
    parser.add_argument("--log-samples", action="store_true", help="Log lm_eval samples in results/W&B")
    parser.add_argument("--log-router-metrics", action="store_true", help="Log CoLA/Hydra router metrics from PEFT")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s|%(asctime)s|%(name)s:%(lineno)d >> %(message)s",
    )

    ckpt = _resolve_adapter_dir(Path(args.checkpoint))
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    tasks = _parse_tasks(args.tasks)
    outdir = Path(args.output_dir)

    lang_list = _load_language_list(ckpt)
    lang_map = _build_lang_id_map(lang_list)
    torch_dtype = _parse_torch_dtype(args.torch_dtype)
    if torch_dtype is None and args.device.startswith("cuda"):
        torch_dtype = "auto"
    device_map = _normalize_device_map(args.device_map)

    # Build HF model_args for lm_eval
    base_model = args.tokenizer

    if args.dry_run:
        for task in tasks:
            code = _task_to_lang_code(task)
            lang_id = lang_map.get(code) if code else None
            print(f"{task}: lang_code={code} lang_id={lang_id}")
        return 0

    if args.mode in ("no_ids", "both"):
        _run_eval(
            pretrained=base_model,
            peft=str(ckpt),
            tokenizer=base_model,
            tasks=tasks,
            batch_size=args.batch_size,
            output_path=outdir / "no_language_ids.json",
            device=args.device,
            language_id=None,
            limit=args.limit,
            wandb_args=args.wandb_args,
            wandb_config_args=args.wandb_config_args,
            log_samples=args.log_samples,
            log_router_metrics=args.log_router_metrics,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    if args.mode in ("with_ids", "both"):
        # Run per-task so each task gets a fixed language ID
        for task in tasks:
            code = _task_to_lang_code(task)
            if code is None:
                continue
            lang_id = lang_map.get(code)
            if lang_id is None:
                continue
            _run_eval(
                pretrained=base_model,
                peft=str(ckpt),
                tokenizer=base_model,
                tasks=[task],
                batch_size=args.batch_size,
                output_path=outdir / f"with_language_ids_{task}.json",
                device=args.device,
                language_id=lang_id,
                limit=args.limit,
                wandb_args=args.wandb_args,
                wandb_config_args=args.wandb_config_args,
                log_samples=args.log_samples,
                log_router_metrics=args.log_router_metrics,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
