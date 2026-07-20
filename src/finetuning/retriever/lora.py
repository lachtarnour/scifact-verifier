"""LoRA-specific model setup for the SPECTER2 retriever experiment."""

import torch

from .config import LORA_CONFIG, RETRIEVER_CONFIG, LoraExperimentConfig, RetrieverFinetuningConfig
from .encoder import load_model, set_role_adapters


def load_lora_model(
    retriever_settings: RetrieverFinetuningConfig = RETRIEVER_CONFIG,
    lora_settings: LoraExperimentConfig = LORA_CONFIG,
    device: torch.device | None = None,
):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("peft is required. Run: pip install -r requirements.txt") from exc

    tokenizer, model, device = load_model(
        base_model=retriever_settings.base_model,
        query_adapter=retriever_settings.query_adapter,
        document_adapter=retriever_settings.document_adapter,
        device=device,
    )
    query_adapter_name = getattr(model, "_specter2_query_adapter", None)
    document_adapter_name = getattr(model, "_specter2_document_adapter", None)

    lora_config = LoraConfig(
        r=lora_settings.r,
        lora_alpha=lora_settings.alpha,
        lora_dropout=lora_settings.dropout,
        target_modules=lora_settings.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    set_role_adapters(model, query_adapter_name, document_adapter_name)
    model.to(device)
    return tokenizer, model, device
