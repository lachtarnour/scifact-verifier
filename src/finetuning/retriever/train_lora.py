"""Train a LoRA adapter for a SPECTER2 dense retriever."""

import argparse
import os
from pathlib import Path

from src.data.load_scifact import load_corpus

from .config import LORA_CONFIG, RETRIEVER_CONFIG
from .lora import load_lora_model
from .tracking import init_wandb
from .training import build_dataloader, train_contrastive_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=RETRIEVER_CONFIG.train_file)
    parser.add_argument("--validation-file", type=Path, default=RETRIEVER_CONFIG.validation_file)
    parser.add_argument("--output-dir", type=Path, default=LORA_CONFIG.output_dir)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--validation-log-file", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=RETRIEVER_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=RETRIEVER_CONFIG.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=RETRIEVER_CONFIG.eval_batch_size)
    parser.add_argument("--learning-rate", type=float, default=RETRIEVER_CONFIG.learning_rate)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=RETRIEVER_CONFIG.gradient_accumulation_steps,
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "scifact-retriever"))
    parser.add_argument("--wandb-run-name", default=os.environ.get("WANDB_RUN_NAME"))
    parser.add_argument(
        "--wandb-mode",
        default=os.environ.get("WANDB_MODE", "online"),
        choices=["online", "offline", "disabled"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_file.exists():
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    if not args.validation_file.exists():
        raise FileNotFoundError(f"Validation file not found: {args.validation_file}")

    tokenizer, model, device = load_lora_model()
    model.print_trainable_parameters()

    corpus = load_corpus()
    sep_token = tokenizer.sep_token or "[SEP]"
    dataloader = build_dataloader(args.train_file, corpus, sep_token, args.batch_size)
    loss_log_file = args.log_file or args.output_dir / "training_loss.jsonl"
    validation_log_file = args.validation_log_file or args.output_dir / "validation_metrics.jsonl"

    wandb_run = init_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={
            "base_model": RETRIEVER_CONFIG.base_model,
            "query_adapter": RETRIEVER_CONFIG.query_adapter,
            "document_adapter": RETRIEVER_CONFIG.document_adapter,
            "lora_r": LORA_CONFIG.r,
            "lora_alpha": LORA_CONFIG.alpha,
            "lora_dropout": LORA_CONFIG.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "fp16": args.fp16,
            "train_file": str(args.train_file),
            "validation_file": str(args.validation_file),
        },
    )
    try:
        train_contrastive_model(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            corpus=corpus,
            device=device,
            settings=RETRIEVER_CONFIG,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            log_file=loss_log_file,
            validation_file=args.validation_file,
            validation_log_file=validation_log_file,
            eval_batch_size=args.eval_batch_size,
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run:
            wandb_run.finish()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")
    print(f"Saved loss log to {loss_log_file}")
    print(f"Saved validation metrics to {validation_log_file}")


if __name__ == "__main__":
    main()
