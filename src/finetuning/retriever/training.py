"""Shared training helpers for dense retriever fine-tuning experiments."""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.evaluation.metrics import compute_all_metrics

from .config import RetrieverFinetuningConfig
from .encoder import DOCUMENT_ROLE, QUERY_ROLE, embed_texts, encode_texts, format_document
from .tracking import gpu_memory_stats


class RetrieverDataset(Dataset):
    def __init__(self, path: Path, corpus: dict[str, dict[str, str]], sep_token: str):
        self.rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
        self.corpus = corpus
        self.sep_token = sep_token

    def __len__(self) -> int:
        return len(self.rows)

    def document_text(self, doc_id: str) -> str:
        doc = self.corpus[doc_id]
        return format_document(doc.get("title", ""), doc.get("text", ""), self.sep_token)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {
            "query": row["query"],
            "positive": self.document_text(row["positive_doc_id"]),
            "negatives": [self.document_text(doc_id) for doc_id in row["negative_doc_ids"]],
        }


def collate_batch(rows: list[dict]) -> dict:
    return {
        "queries": [row["query"] for row in rows],
        "positives": [row["positive"] for row in rows],
        "negatives": [row["negatives"] for row in rows],
    }


def build_dataloader(
    train_file: Path,
    corpus: dict[str, dict[str, str]],
    sep_token: str,
    batch_size: int,
) -> DataLoader:
    dataset = RetrieverDataset(train_file, corpus, sep_token)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)


def load_eval_examples(path: Path) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            query_id = row["query_id"]
            queries[query_id] = row["query"]
            qrels.setdefault(query_id, {})[row["positive_doc_id"]] = 1

    return queries, qrels


def contrastive_loss(
    query_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    positive_scores = (query_embeddings * positive_embeddings).sum(dim=1, keepdim=True)
    negative_scores = torch.einsum("bd,bnd->bn", query_embeddings, negative_embeddings)
    scores = torch.cat([positive_scores, negative_scores], dim=1) / temperature
    labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, labels)


def compute_contrastive_loss(
    batch: dict,
    tokenizer,
    model,
    device,
    settings: RetrieverFinetuningConfig,
) -> torch.Tensor:
    queries = batch["queries"]
    positives = batch["positives"]
    negatives = batch["negatives"]
    num_negatives = len(negatives[0])

    query_embeddings = embed_texts(
        queries,
        tokenizer,
        model,
        device,
        settings.max_length,
        role=QUERY_ROLE,
    )
    positive_embeddings = embed_texts(
        positives,
        tokenizer,
        model,
        device,
        settings.max_length,
        role=DOCUMENT_ROLE,
    )

    flat_negatives = [text for group in negatives for text in group]
    negative_embeddings = embed_texts(
        flat_negatives,
        tokenizer,
        model,
        device,
        settings.max_length,
        role=DOCUMENT_ROLE,
    )
    negative_embeddings = negative_embeddings.view(len(queries), num_negatives, -1)

    return contrastive_loss(
        query_embeddings=query_embeddings,
        positive_embeddings=positive_embeddings,
        negative_embeddings=negative_embeddings,
        temperature=settings.temperature,
    )


def evaluate_training_split(
    model,
    tokenizer,
    corpus: dict[str, dict[str, str]],
    eval_file: Path,
    device,
    settings: RetrieverFinetuningConfig,
    batch_size: int,
) -> dict[str, float]:
    queries, qrels = load_eval_examples(eval_file)
    sep_token = tokenizer.sep_token or "[SEP]"

    doc_ids = list(corpus)
    doc_texts = [
        format_document(corpus[doc_id].get("title", ""), corpus[doc_id].get("text", ""), sep_token)
        for doc_id in doc_ids
    ]
    query_ids = list(qrels)
    query_texts = [queries[query_id] for query_id in query_ids]

    doc_embeddings = encode_texts(
        doc_texts,
        tokenizer,
        model,
        device,
        batch_size,
        settings.max_length,
        role=DOCUMENT_ROLE,
    )
    query_embeddings = encode_texts(
        query_texts,
        tokenizer,
        model,
        device,
        batch_size,
        settings.max_length,
        role=QUERY_ROLE,
    )
    scores = query_embeddings @ doc_embeddings.T

    results: dict[str, dict[str, float]] = {}
    max_rank = min(100, len(doc_ids))
    for row_index, query_id in enumerate(query_ids):
        ranking = torch.argsort(scores[row_index], descending=True).tolist()[:max_rank]
        results[query_id] = {
            doc_ids[index]: float(scores[row_index, index])
            for index in ranking
        }

    return compute_all_metrics(qrels, results)


def train_contrastive_model(
    model,
    tokenizer,
    dataloader: DataLoader,
    corpus: dict[str, dict[str, str]],
    device,
    settings: RetrieverFinetuningConfig,
    epochs: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    fp16: bool,
    log_file: Path,
    validation_file: Path,
    validation_log_file: Path,
    eval_batch_size: int,
    wandb_run=None,
) -> None:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=settings.weight_decay,
    )

    update_steps = math.ceil(len(dataloader) / gradient_accumulation_steps) * epochs
    warmup_steps = int(update_steps * settings.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, update_steps)

    use_amp = fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    optimizer_step = 0
    log_file.parent.mkdir(parents=True, exist_ok=True)
    validation_log_file.parent.mkdir(parents=True, exist_ok=True)
    model.train()

    with log_file.open("w", encoding="utf-8") as loss_log, validation_log_file.open(
        "w", encoding="utf-8"
    ) as validation_log:
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            progress = tqdm(dataloader, desc=f"epoch {epoch}/{epochs}")

            for step, batch in enumerate(progress, start=1):
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        loss = compute_contrastive_loss(batch, tokenizer, model, device, settings)
                else:
                    loss = compute_contrastive_loss(batch, tokenizer, model, device, settings)

                batch_loss = float(loss.detach().cpu())
                scaler.scale(loss / gradient_accumulation_steps).backward()
                running_loss += batch_loss

                if step % gradient_accumulation_steps == 0 or step == len(dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                    train_record = {
                        "epoch": epoch,
                        "batch_step": step,
                        "optimizer_step": optimizer_step,
                        "loss": batch_loss,
                        "running_loss": running_loss / step,
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                    loss_log.write(json.dumps(train_record) + "\n")
                    loss_log.flush()
                    if wandb_run:
                        wandb_run.log(
                            {
                                "train/loss": batch_loss,
                                "train/running_loss": running_loss / step,
                                "train/learning_rate": scheduler.get_last_lr()[0],
                                "train/epoch": epoch,
                                "train/batch_step": step,
                                **gpu_memory_stats(device),
                            },
                            step=optimizer_step,
                        )

                progress.set_postfix(loss=f"{running_loss / step:.4f}")

            metrics = evaluate_training_split(
                model=model,
                tokenizer=tokenizer,
                corpus=corpus,
                eval_file=validation_file,
                device=device,
                settings=settings,
                batch_size=eval_batch_size,
            )
            validation_record = {"epoch": epoch, **metrics}
            validation_log.write(json.dumps(validation_record) + "\n")
            validation_log.flush()
            if wandb_run:
                wandb_run.log(
                    {
                        "validation/epoch": epoch,
                        **{
                            f"validation/{metric_name}": metric_value
                            for metric_name, metric_value in metrics.items()
                        },
                        **gpu_memory_stats(device),
                    },
                    step=optimizer_step,
                )
            print(
                "validation "
                f"epoch={epoch} "
                f"Recall@5={metrics['Recall@5']:.4f} "
                f"MRR={metrics['MRR']:.4f} "
                f"nDCG@10={metrics['nDCG@10']:.4f}"
            )
            model.train()
