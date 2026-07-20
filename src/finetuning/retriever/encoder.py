"""SPECTER2 encoder helpers used by retriever training and evaluation."""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


QUERY_ROLE = "query"
DOCUMENT_ROLE = "document"


def format_document(title: str, abstract: str, sep_token: str) -> str:
    title = title.strip()
    abstract = abstract.strip()
    if not title:
        return abstract
    return f"{title} {sep_token} {abstract}"


def set_role_adapters(
    model,
    query_adapter_name: str | None,
    document_adapter_name: str | None,
) -> None:
    model._specter2_query_adapter = query_adapter_name
    model._specter2_document_adapter = document_adapter_name


def _adapter_model(model):
    candidates = [
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "set_active_adapters"):
            return candidate
    return None


def activate_role_adapter(model, role: str) -> None:
    if role == QUERY_ROLE:
        adapter_name = getattr(model, "_specter2_query_adapter", None)
    elif role == DOCUMENT_ROLE:
        adapter_name = getattr(model, "_specter2_document_adapter", None)
    else:
        raise ValueError(f"Unknown encoder role: {role}")

    if not adapter_name:
        return

    adapter_model = _adapter_model(model)
    if adapter_model is None:
        raise RuntimeError("No adapter-aware model found for SPECTER2 role switching.")

    adapter_model.set_active_adapters(adapter_name)


def _load_adapter(model, adapter_id: str | None, load_as: str) -> str | None:
    if not adapter_id:
        return None
    loaded_name = model.load_adapter(
        adapter_id,
        source="hf",
        load_as=load_as,
        set_active=False,
    )
    return loaded_name or load_as


def load_model(
    base_model: str,
    query_adapter: str | None = None,
    document_adapter: str | None = None,
    lora_adapter: str | None = None,
    device: torch.device | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if query_adapter or document_adapter:
        try:
            from adapters import AutoAdapterModel
        except ImportError as exc:
            raise RuntimeError("adapters is required for SPECTER2 task adapters") from exc

        model = AutoAdapterModel.from_pretrained(base_model)
        query_adapter_name = _load_adapter(model, query_adapter, QUERY_ROLE)
        document_adapter_name = _load_adapter(model, document_adapter, DOCUMENT_ROLE)
    else:
        model = AutoModel.from_pretrained(base_model)
        query_adapter_name = None
        document_adapter_name = None

    if lora_adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("peft is required to load a LoRA adapter") from exc
        model = PeftModel.from_pretrained(model, lora_adapter)

    set_role_adapters(model, query_adapter_name, document_adapter_name)

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    model.to(device)
    return tokenizer, model, device


class Specter2Encoder:
    def __init__(
        self,
        base_model: str,
        query_adapter: str | None = None,
        document_adapter: str | None = None,
        lora_adapter: str | None = None,
        device: torch.device | None = None,
        max_length: int = 512,
    ):
        self.tokenizer, self.model, self.device = load_model(
            base_model=base_model,
            query_adapter=query_adapter,
            document_adapter=document_adapter,
            lora_adapter=lora_adapter,
            device=device,
        )
        self.max_length = max_length
        self.sep_token = self.tokenizer.sep_token or "[SEP]"

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        role: str = DOCUMENT_ROLE,
        show_progress: bool = False,
    ):
        embeddings = encode_texts(
            texts=texts,
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.device,
            batch_size=batch_size,
            max_length=self.max_length,
            role=role,
            show_progress=show_progress,
            description=f"Encoding {role}s",
        )
        return embeddings.numpy()


def embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    role: str = DOCUMENT_ROLE,
) -> torch.Tensor:
    activate_role_adapter(model, role)
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    output = model(**inputs)
    embeddings = output.last_hidden_state[:, 0, :]
    return F.normalize(embeddings, p=2, dim=1)


def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    role: str = DOCUMENT_ROLE,
    show_progress: bool = False,
    description: str = "Encoding",
) -> torch.Tensor:
    model.eval()
    batches = []
    starts = range(0, len(texts), batch_size)
    if show_progress:
        starts = tqdm(starts, desc=description, unit="batch")

    with torch.no_grad():
        for start in starts:
            batch = texts[start : start + batch_size]
            batches.append(embed_texts(batch, tokenizer, model, device, max_length, role=role).cpu())
    return torch.cat(batches, dim=0)
