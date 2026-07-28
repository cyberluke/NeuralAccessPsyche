"""
Trained hidden-state probes for NRAM v5.

Implements real probe training with explicit train/validation/test partitions.
Uses regularized linear probes for interpretability.

Initial probes:
- paraphrase_risk
- novelty
- genericity
- product_concreteness
- evidence_alignment
- cliche_risk

Stores probe artifacts with:
- model/tokenizer hashes
- layer
- feature normalization
- classifier weights
- calibration parameters
- dataset hash
- training seed
- validation metrics
- artifact hash
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

logger = logging.getLogger(__name__)


PROBE_TYPES = [
    "paraphrase_risk",
    "novelty",
    "genericity",
    "product_concreteness",
    "evidence_alignment",
    "cliche_risk",
]


@dataclass
class ProbeArtifact:
    """Trained probe artifact with metadata."""
    probe_id: str
    probe_type: str
    layer_name: str
    layer_index: int
    hidden_dim: int
    weights: torch.Tensor  # (hidden_dim,)
    bias: float
    feature_mean: torch.Tensor  # (hidden_dim,)
    feature_std: torch.Tensor  # (hidden_dim,)
    calibration_slope: float
    calibration_intercept: float
    val_accuracy: float
    val_f1: float
    val_auc: float
    model_hash: str
    tokenizer_hash: str
    dataset_hash: str
    training_seed: int
    artifact_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "probe_type": self.probe_type,
            "layer_name": self.layer_name,
            "layer_index": self.layer_index,
            "hidden_dim": self.hidden_dim,
            "calibration_slope": self.calibration_slope,
            "calibration_intercept": self.calibration_intercept,
            "val_accuracy": self.val_accuracy,
            "val_f1": self.val_f1,
            "val_auc": self.val_auc,
            "model_hash": self.model_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "dataset_hash": self.dataset_hash,
            "training_seed": self.training_seed,
            "artifact_hash": self.artifact_hash,
        }


@dataclass
class ProbeTrainingConfig:
    """Configuration for probe training."""
    probe_type: str
    layer_name: str
    layer_index: int
    hidden_dim: int
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    early_stopping_patience: int = 10
    seed: int = 42


class HiddenStateProbeTrainer:
    """
    Trains linear probes on hidden-state activations.
    
    Uses explicit train/validation/test partitions.
    Regularized linear probe for interpretability.
    """
    
    def __init__(self, config: ProbeTrainingConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        torch.manual_seed(config.seed)
    
    def train(
        self,
        activations: torch.Tensor,  # (num_samples, hidden_dim)
        labels: torch.Tensor,  # (num_samples,) binary
        model_hash: str,
        tokenizer_hash: str,
        dataset_hash: str,
    ) -> Tuple[nn.Linear, ProbeArtifact]:
        """
        Train a linear probe.
        
        Returns (trained_model, artifact).
        """
        activations = activations.float().to(self.device)
        labels = labels.float().to(self.device)
        
        # Feature normalization
        mean = activations.mean(dim=0)
        std = activations.std(dim=0) + 1e-8
        normalized = (activations - mean) / std
        
        # Split data
        dataset = TensorDataset(normalized, labels)
        total = len(dataset)
        train_size = int(self.config.train_split * total)
        val_size = int(self.config.val_split * total)
        test_size = total - train_size - val_size
        
        generator = torch.Generator().manual_seed(self.config.seed)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Initialize model
        model = nn.Linear(self.config.hidden_dim, 1).to(self.device)
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            # Train
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_x).squeeze(-1)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    logits = model(batch_x).squeeze(-1)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        model.eval()
        test_probs = []
        test_labels = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                logits = model(batch_x).squeeze(-1)
                probs = torch.sigmoid(logits)
                test_probs.append(probs)
                test_labels.append(batch_y)
        
        test_probs = torch.cat(test_probs).cpu().numpy()
        test_labels = torch.cat(test_labels).cpu().numpy()
        
        # Compute metrics
        test_preds = (test_probs > 0.5).astype(float)
        accuracy = (test_preds == test_labels).mean()
        
        # F1 score
        tp = ((test_preds == 1) & (test_labels == 1)).sum()
        fp = ((test_preds == 1) & (test_labels == 0)).sum()
        fn = ((test_preds == 0) & (test_labels == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # AUC (simple approximation)
        sorted_indices = test_probs.argsort()[::-1]
        sorted_labels = test_labels[sorted_indices]
        tpr = sorted_labels.cumsum() / (sorted_labels.sum() + 1e-8)
        auc = tpr.mean()
        
        # Calibration (simple linear fit)
        from scipy.stats import linregress
        slope, intercept, _, _, _ = linregress(test_probs, test_labels)
        
        # Compute artifact hash
        weights_bytes = model.weight.data.cpu().numpy().tobytes()
        artifact_hash = hashlib.sha256(weights_bytes).hexdigest()
        
        artifact = ProbeArtifact(
            probe_id=f"{self.config.probe_type}_layer{self.config.layer_index}",
            probe_type=self.config.probe_type,
            layer_name=self.config.layer_name,
            layer_index=self.config.layer_index,
            hidden_dim=self.config.hidden_dim,
            weights=model.weight.data.squeeze(0).cpu(),
            bias=model.bias.item(),
            feature_mean=mean.cpu(),
            feature_std=std.cpu(),
            calibration_slope=float(slope),
            calibration_intercept=float(intercept),
            val_accuracy=float(accuracy),
            val_f1=float(f1),
            val_auc=float(auc),
            model_hash=model_hash,
            tokenizer_hash=tokenizer_hash,
            dataset_hash=dataset_hash,
            training_seed=self.config.seed,
            artifact_hash=artifact_hash,
        )
        
        logger.info(
            f"Trained probe {artifact.probe_id}: "
            f"acc={accuracy:.3f}, f1={f1:.3f}, auc={auc:.3f}"
        )
        
        return model, artifact


class LiveProbeController:
    """
    Live probe controller for runtime inference.
    
    Calculates probe scores from actual hidden states during generation.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.probes: Dict[str, Tuple[nn.Linear, ProbeArtifact]] = {}
        self.enabled = True
        self._invocation_count = 0
    
    def register_probe(
        self,
        model: nn.Linear,
        artifact: ProbeArtifact,
    ) -> None:
        """Register a trained probe."""
        model = model.to(self.device)
        model.eval()
        self.probes[artifact.probe_id] = (model, artifact)
        logger.info(f"Registered probe: {artifact.probe_id}")
    
    @torch.no_grad()
    def score(
        self,
        hidden_states: torch.Tensor,
        probe_id: str,
    ) -> float:
        """
        Compute probe score for hidden states.
        
        Args:
            hidden_states: (batch, seq, hidden_dim) or (batch, hidden_dim)
            probe_id: Probe identifier
        
        Returns:
            Mean probe score in [0, 1]
        """
        if not self.enabled or probe_id not in self.probes:
            return 0.0
        
        self._invocation_count += 1
        model, artifact = self.probes[probe_id]
        
        # Flatten if needed
        if hidden_states.dim() == 3:
            h = hidden_states.reshape(-1, hidden_states.shape[-1])
        else:
            h = hidden_states
        
        # Normalize using stored statistics
        mean = artifact.feature_mean.to(h.device, dtype=h.dtype)
        std = artifact.feature_std.to(h.device, dtype=h.dtype)
        normalized = (h - mean) / std
        
        # Forward pass
        logits = model(normalized).squeeze(-1)
        probs = torch.sigmoid(logits)
        
        return probs.mean().item()
    
    def score_all(
        self,
        hidden_states: torch.Tensor,
    ) -> Dict[str, float]:
        """Score all registered probes."""
        return {
            probe_id: self.score(hidden_states, probe_id)
            for probe_id in self.probes
        }
    
    def reset(self) -> None:
        """Reset invocation count."""
        self._invocation_count = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "enabled": self.enabled,
            "num_probes": len(self.probes),
            "probes": {
                k: {
                    "type": v[1].probe_type,
                    "layer": v[1].layer_name,
                    "val_accuracy": v[1].val_accuracy,
                    "val_f1": v[1].val_f1,
                }
                for k, v in self.probes.items()
            },
            "invocation_count": self._invocation_count,
        }


class ProbeArtifactRegistry:
    """Registry for probe artifacts."""
    
    def __init__(self, artifacts_dir: str = "artifacts/nram_probes"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, ProbeArtifact] = {}
    
    def store(
        self,
        model: nn.Linear,
        artifact: ProbeArtifact,
    ) -> Path:
        """Store a probe artifact."""
        # Save model weights
        model_path = self.artifacts_dir / f"{artifact.probe_id}.pt"
        torch.save({
            "weights": artifact.weights,
            "bias": artifact.bias,
            "feature_mean": artifact.feature_mean,
            "feature_std": artifact.feature_std,
        }, str(model_path))
        
        # Save metadata
        meta_path = self.artifacts_dir / f"{artifact.probe_id}.meta.json"
        with open(meta_path, "w") as f:
            json.dump(artifact.to_dict(), f, indent=2)
        
        self._index[artifact.probe_id] = artifact
        logger.info(f"Stored probe artifact: {artifact.probe_id}")
        return model_path
    
    def load(self, probe_id: str) -> Tuple[nn.Linear, ProbeArtifact]:
        """Load a probe artifact."""
        if probe_id not in self._index:
            meta_path = self.artifacts_dir / f"{probe_id}.meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Probe artifact not found: {probe_id}")
            with open(meta_path) as f:
                data = json.load(f)
            artifact = ProbeArtifact(**data)
            self._index[probe_id] = artifact
        else:
            artifact = self._index[probe_id]
        
        # Load model
        model_path = self.artifacts_dir / f"{probe_id}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Probe model not found: {probe_id}")
        
        data = torch.load(str(model_path), map_location="cpu")
        model = nn.Linear(artifact.hidden_dim, 1)
        model.weight.data = data["weights"].unsqueeze(0)
        model.bias.data = torch.tensor([data["bias"]])
        
        return model, artifact
    
    def list_artifacts(self) -> List[ProbeArtifact]:
        """List all registered artifacts."""
        return list(self._index.values())
