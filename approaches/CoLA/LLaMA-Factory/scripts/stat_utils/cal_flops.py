# Copyright 2024 Microsoft Corporation and the LlamaFactory team.
#
# This code is inspired by the Microsoft's DeepSpeed library.
# https://www.deepspeed.ai/tutorials/flops-profiler/
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

import fire
import torch
from deepspeed.accelerator import get_accelerator  # type: ignore
from deepspeed.profiling.flops_profiler import get_model_profile  # type: ignore

from llamafactory.chat import ChatModel


def calculate_flops(
    model_name_or_path: str,
    batch_size: int = 1,
    seq_length: int = 512,
    flash_attn: str = "auto",
    finetuning_type: str = "lora",
    use_cola_experts: bool = False,
    cola_num_experts: int = 1,
    cola_top_k: int = 1,
    cola_strategy: str = "fully",
    num_a: int = 1,
    num_b: int = 1,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    use_cola_pissa_init: bool = True,
    cola_init_lora_weights: str = "",
):
    r"""
    Calculates the flops of pre-trained models.
    Usage: python cal_flops.py --model_name_or_path path_to_model --batch_size 1 --seq_length 512
    """
    with get_accelerator().device(0):
        infer_args = dict(
            model_name_or_path=model_name_or_path,
            template="empty",
            flash_attn=flash_attn,
            finetuning_type=finetuning_type,
            use_cola_experts=use_cola_experts,
            cola_num_experts=cola_num_experts,
            cola_top_k=cola_top_k,
            cola_strategy=cola_strategy,
            num_A=num_a,
            num_B=num_b,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            use_cola_pissa_init=use_cola_pissa_init,
        )
        if cola_init_lora_weights:
            infer_args["cola_init_lora_weights"] = cola_init_lora_weights

        chat_model = ChatModel(infer_args)
        fake_input = torch.ones((batch_size, seq_length), dtype=torch.long, device=chat_model.engine.model.device)
        input_dict = {"input_ids": fake_input, "labels": fake_input.clone()}
        flops, macs, params = get_model_profile(
            chat_model.engine.model, kwargs=input_dict, print_profile=True, detailed=True
        )
        print("FLOPs:", flops)
        print("MACs:", macs)
        print("Params:", params)


if __name__ == "__main__":
    fire.Fire(calculate_flops)
