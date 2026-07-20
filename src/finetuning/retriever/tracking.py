"""Optional Weights & Biases logging for retriever fine-tuning."""

from typing import Any

import torch


def init_wandb(
    enabled: bool,
    project: str,
    run_name: str | None,
    mode: str,
    config: dict[str, Any],
):
    if not enabled:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is required for W&B logging. Run: pip install wandb"
        ) from exc

    return wandb.init(
        project=project,
        name=run_name,
        mode=mode,
        config=config,
        settings=wandb.Settings(_disable_stats=True),
    )


def gpu_memory_stats(device) -> dict[str, float]:
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return {}

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    gb = 1024**3
    return {
        "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated(device_index) / gb,
    }
