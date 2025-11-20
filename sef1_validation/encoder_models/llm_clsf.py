from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from legm import from_namespace

from sef1_validation.encoder_models.trainer import SubjectiveClassifierTrainer


class LLMClassifier(nn.Module):

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return dict(
            model_name_or_path=dict(
                type=str,
                required=True,
                help="model name or path to load tokenizer and model from",
                metadata=dict(name=True, name_priority=2),
                searchable=True,
            ),
            load_in_4bit=dict(
                action="store_true",
                help="whether to load the model in 4-bit precision",
            ),
            load_in_8bit=dict(
                action="store_true",
                help="whether to load the model in 8-bit precision",
            ),
            torch_dtype=dict(
                type=str,
                default="float16",
                choices=["float16", "bfloat16", "float32"],
                help="the torch dtype to use when loading the model",
            ),
            trust_remote_code=dict(
                action="store_true",
                help="whether to trust remote code when loading the model",
            ),
            device=dict(
                type=str,
                help="the device to load the model on (e.g., 'cpu', 'cuda', 'cuda:0')",
            ),
            lora_r=dict(
                type=int,
                default=2,
                help="the rank of the LoRA adapters",
                searchable=True,
                metadata=dict(name=True, name_priority=1),
            ),
            lora_alpha=dict(
                type=int,
                default=4,
                help="the alpha of the LoRA adapters",
                searchable=True,
            ),
            lora_dropout=dict(
                type=float,
                default=0.1,
                help="the dropout of the LoRA adapters",
                searchable=True,
            ),
        )

    @from_namespace
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device: str = None,
        torch_dtype: str = "float",
        trust_remote_code: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_classes, torch_dtype=torch_dtype
        )
        # Ensure a valid pad token id for decoder-only models (e.g., Llama).
        # Some HF heads (sequence classification) require pad_token_id to be set
        # when batch size > 1 to identify the last non-pad token per sequence.
        if (
            getattr(self.config, "pad_token_id", None) is None
            and getattr(self.config, "eos_token_id", None) is not None
        ):
            eos_id = self.config.eos_token_id
            if isinstance(eos_id, (list, tuple)):
                eos_id = int(eos_id[0]) if len(eos_id) > 0 else None
            if eos_id is not None:
                self.config.pad_token_id = eos_id
        self.qconfig = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=self.config,
            quantization_config=self.qconfig,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )

        # Double-check on the instantiated model's configs as well.
        if (
            getattr(model.config, "pad_token_id", None) is None
            and getattr(model.config, "eos_token_id", None) is not None
        ):
            eos_id = model.config.eos_token_id
            if isinstance(eos_id, (list, tuple)):
                eos_id = int(eos_id[0]) if len(eos_id) > 0 else None
            if eos_id is not None:
                model.config.pad_token_id = eos_id
        if hasattr(model, "generation_config"):
            gen = model.generation_config
            if (
                getattr(gen, "pad_token_id", None) is None
                and getattr(gen, "eos_token_id", None) is not None
            ):
                eos_id = gen.eos_token_id
                if isinstance(eos_id, (list, tuple)):
                    eos_id = int(eos_id[0]) if len(eos_id) > 0 else None
                if eos_id is not None:
                    gen.pad_token_id = eos_id

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

        lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
            ],
        )

        self.lm = get_peft_model(model, lora)
        self.lm.print_trainable_parameters()

    def forward(self, *args, **kwargs):
        return self.lm(*args, **kwargs)


class LLMClassifierTrainer(SubjectiveClassifierTrainer):
    @staticmethod
    def get_model_state_dict(model: nn.Module) -> dict[str, nn.Parameter]:
        # get only lora weights
        model_state_dict = model.lm.state_dict()
        lora_state_dict = {
            k: v for k, v in model_state_dict.items() if "lora_" in k
        }
        return lora_state_dict

    @staticmethod
    def load_model_state_dict(model, state_dict):
        model.lm.load_state_dict(state_dict, strict=False)
