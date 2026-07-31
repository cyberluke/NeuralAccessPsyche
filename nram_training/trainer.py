"""Shared production LoRA trainer with lazy ML imports."""
import os
from pathlib import Path
from .config import AdapterRole, config_for
from .manifests import atomic_json, make_manifest

def validate_parent(model_name: str) -> None:
    if "0.6b" in model_name.lower():
        raise ValueError("Qwen3-0.6B cannot be attached to the Qwen3-14B pipeline")
    if model_name != "Qwen/Qwen3-14B":
        raise ValueError("production parent must be Qwen/Qwen3-14B")

def train(adapter: AdapterRole, output_dir: str | Path, *, resume: str = "auto", smoke: bool = False) -> dict:
    cfg = config_for(adapter); validate_parent(cfg.base_model)
    out = Path(output_dir) / adapter; out.mkdir(parents=True, exist_ok=True)
    if smoke:
        manifest = make_manifest(cfg.as_dict()) | {"adapter": adapter, "resume": resume, "smoke": True,
            "status": "TEST_ONLY;NOT_FOR_SCIENTIFIC_USE"}
        atomic_json(out / "training_manifest.json", manifest)
        return manifest
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("training requires the pinned training image dependencies") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requires CUDA; refusing CPU fallback for Qwen3-14B")
    torch.manual_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, revision=cfg.tokenizer_revision)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, revision="main", torch_dtype=torch.bfloat16,
                                                 device_map="auto", attn_implementation="sdpa")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout, target_modules=list(cfg.target_modules), bias="none",
        task_type=TaskType.CAUSAL_LM))
    dataset = load_dataset(cfg.dataset, revision=cfg.dataset_revision, split="train")
    threshold = cfg.low_toxicity_threshold if adapter == "nontoxic" else cfg.high_toxicity_threshold
    dataset = dataset.filter(lambda row: row.get("toxicity") is not None and
        (row["toxicity"] <= threshold if adapter == "nontoxic" else row["toxicity"] >= threshold))
    dataset = dataset.map(lambda row: _format_row(row, tokenizer, cfg.max_length), remove_columns=dataset.column_names)
    args = TrainingArguments(output_dir=str(out), num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size, learning_rate=cfg.learning_rate, bf16=True,
        gradient_checkpointing=True, save_strategy="steps", save_steps=100, save_total_limit=2,
        report_to="none", remove_unused_columns=False, seed=cfg.seed)
    trainer = Trainer(model=model, args=args, train_dataset=dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True))
    checkpoint = None if resume != "auto" else _latest_checkpoint(out)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    manifest = make_manifest(cfg.as_dict()) | {"adapter": adapter, "resume": resume,
        "checkpoint": checkpoint, "status": "TRAINED", "train_loss": result.training_loss}
    atomic_json(out / "training_manifest.json", manifest)
    return manifest

def _format_row(row, tokenizer, max_length):
    messages = [{"role": "user", "content": "Respond to this Civil Comments example."},
        {"role": "assistant", "content": row["text"]}]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prompt = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(full, truncation=True, max_length=max_length, padding="max_length")
    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    labels = list(encoded["input_ids"])
    labels[:min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
    labels = [label if mask != tokenizer.pad_token_id else -100 for label, mask in zip(labels, encoded["input_ids"])]
    encoded["labels"] = labels
    return encoded

def _latest_checkpoint(out: Path):
    checkpoints = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None
