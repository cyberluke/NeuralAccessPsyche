#!/usr/bin/env python3
"""
Train LoRA adapters for DExperts toxicity steering.

Non-toxic expert: toxicity <= 0.05
Toxic anti-expert: toxicity >= 0.80

Dataset: google/civil_comments
Base model: Qwen3-0.6B-Base
Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
"""

import json
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import numpy as np
from tqdm import tqdm


def compute_toxicity_score(text: str, toxicity_classifier) -> float:
    """Compute toxicity score using unitary/toxic-bert."""
    inputs = toxicity_classifier.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(toxicity_classifier.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = toxicity_classifier(**inputs)
    
    # Toxic-bert returns logits for 6 classes: toxic, severe_toxic, obscene, 
    # threat, insult, identity_hate. We use the max probability across toxic classes.
    probs = torch.sigmoid(outputs.logits).cpu().numpy()
    return float(np.max(probs))


def load_civil_comments_dataset(
    toxicity_threshold_low: float = 0.05,
    toxicity_threshold_high: float = 0.80,
    max_samples_per_class: int = 20000,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Load and filter civil_comments dataset.
    
    Returns:
        Tuple of (nontoxic_texts, toxic_texts)
    """
    print("Loading google/civil_comments dataset...")
    dataset = load_dataset("google/civil_comments", split="train")
    
    print("Filtering by toxicity score...")
    nontoxic_texts = []
    toxic_texts = []
    
    # Sample and filter
    indices = list(range(len(dataset)))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    for idx in tqdm(indices, desc="Filtering"):
        example = dataset[idx]
        text = example["text"]
        toxicity = example["toxicity"]
        
        # Skip if toxicity is NaN
        if toxicity != toxicity:  # NaN check
            continue
        
        if toxicity <= toxicity_threshold_low and len(nontoxic_texts) < max_samples_per_class:
            nontoxic_texts.append(text)
        elif toxicity >= toxicity_threshold_high and len(toxic_texts) < max_samples_per_class:
            toxic_texts.append(text)
        
        if len(nontoxic_texts) >= max_samples_per_class and len(toxic_texts) >= max_samples_per_class:
            break
    
    print(f"Collected {len(nontoxic_texts)} non-toxic texts")
    print(f"Collected {len(toxic_texts)} toxic texts")
    
    return nontoxic_texts, toxic_texts


def deduplicate_by_hash(texts: List[str]) -> List[str]:
    """Deduplicate texts by normalized hash."""
    seen = set()
    unique = []
    for text in texts:
        normalized = text.strip().lower()
        text_hash = hashlib.md5(normalized.encode()).hexdigest()
        if text_hash not in seen:
            seen.add(text_hash)
            unique.append(text)
    return unique


class TextDataset(Dataset):
    """Simple text dataset for causal LM."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }


def train_lora_adapter(
    base_model_path: str,
    texts: List[str],
    output_dir: str,
    adapter_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a LoRA adapter on the given texts."""
    
    print(f"\n{'='*80}")
    print(f"Training {adapter_name} adapter")
    print(f"{'='*80}")
    print(f"Base model: {base_model_path}")
    print(f"Samples: {len(texts)}")
    print(f"Output: {output_dir}")
    print(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Training: epochs={num_epochs}, batch={batch_size}, lr={learning_rate}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load base model and tokenizer
    print("\nLoading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Configure LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    print(f"Target modules: {target_modules}")
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    print("\nPreparing dataset...")
    dataset = TextDataset(texts, tokenizer, max_length=512)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    train_result = trainer.train()
    
    # Save adapter
    print(f"\nSaving adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Compute metrics
    metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        "num_samples": len(texts),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "seed": seed,
    }
    
    # Save training metrics
    metrics_path = Path(output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nTraining complete. Loss: {metrics['train_loss']:.4f}")
    print(f"Metrics saved to {metrics_path}")
    
    # Free memory
    del model, tokenizer, trainer
    torch.cuda.empty_cache()
    
    return metrics


def main():
    """Main entry point."""
    # Configuration
    base_model_path = r"E:\_MODELS\huggingface\hub\models--Qwen--Qwen3-0.6B-Base\snapshots\da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    output_base = Path("artifacts/dexperts/adapters")
    
    toxicity_threshold_low = 0.05
    toxicity_threshold_high = 0.80
    max_samples_per_class = 20000
    seed = 42
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Cannot train LoRA adapters.")
        sys.exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load toxicity classifier
    print("\nLoading toxicity classifier (unitary/toxic-bert)...")
    from transformers import AutoModelForSequenceClassification
    toxicity_classifier = AutoModelForSequenceClassification.from_pretrained(
        "unitary/toxic-bert",
        torch_dtype=torch.float16,
    ).to("cuda")
    toxicity_classifier.eval()
    
    # Load and filter dataset
    nontoxic_texts, toxic_texts = load_civil_comments_dataset(
        toxicity_threshold_low=toxicity_threshold_low,
        toxicity_threshold_high=toxicity_threshold_high,
        max_samples_per_class=max_samples_per_class,
        seed=seed,
    )
    
    # Deduplicate
    print("\nDeduplicating...")
    nontoxic_texts = deduplicate_by_hash(nontoxic_texts)
    toxic_texts = deduplicate_by_hash(toxic_texts)
    
    print(f"After deduplication: {len(nontoxic_texts)} non-toxic, {len(toxic_texts)} toxic")
    
    # Match lengths (sample from larger set)
    min_len = min(len(nontoxic_texts), len(toxic_texts))
    if len(nontoxic_texts) > min_len:
        np.random.seed(seed)
        nontoxic_texts = list(np.random.choice(nontoxic_texts, min_len, replace=False))
    if len(toxic_texts) > min_len:
        np.random.seed(seed)
        toxic_texts = list(np.random.choice(toxic_texts, min_len, replace=False))
    
    print(f"Matched sample size: {min_len}")
    
    # Train non-toxic expert
    nontoxic_dir = output_base / "nontoxic"
    nontoxic_dir.mkdir(parents=True, exist_ok=True)
    
    nontoxic_metrics = train_lora_adapter(
        base_model_path=base_model_path,
        texts=nontoxic_texts,
        output_dir=str(nontoxic_dir),
        adapter_name="nontoxic",
        seed=seed,
    )
    
    # Train toxic anti-expert
    toxic_dir = output_base / "toxic"
    toxic_dir.mkdir(parents=True, exist_ok=True)
    
    toxic_metrics = train_lora_adapter(
        base_model_path=base_model_path,
        texts=toxic_texts,
        output_dir=str(toxic_dir),
        adapter_name="toxic",
        seed=seed,
    )
    
    # Create training manifest
    manifest = {
        "base_model": {
            "path": base_model_path,
            "name": "Qwen3-0.6B-Base",
            "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        },
        "dataset": {
            "name": "google/civil_comments",
            "toxicity_threshold_low": toxicity_threshold_low,
            "toxicity_threshold_high": toxicity_threshold_high,
            "max_samples_per_class": max_samples_per_class,
            "seed": seed,
            "nontoxic_samples": len(nontoxic_texts),
            "toxic_samples": len(toxic_texts),
        },
        "lora_config": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "task_type": "CAUSAL_LM",
        },
        "training": {
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-4,
            "seed": seed,
        },
        "adapters": {
            "nontoxic": {
                "path": str(nontoxic_dir),
                "metrics": nontoxic_metrics,
            },
            "toxic": {
                "path": str(toxic_dir),
                "metrics": toxic_metrics,
            },
        },
        "device": {
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        },
    }
    
    manifest_path = Path("artifacts/dexperts/training_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Non-toxic adapter: {nontoxic_dir}")
    print(f"Toxic adapter: {toxic_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
