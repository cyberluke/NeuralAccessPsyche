from dataclasses import asdict, dataclass
from typing import Literal

AdapterRole = Literal["nontoxic", "toxic"]

@dataclass(frozen=True)
class TrainingConfig:
    base_model: str = "Qwen/Qwen3-14B"
    model_revision: str = "40c069824f4251a91eefaf281ebe4c544efd3e18"
    dataset: str = "google/civil_comments"
    dataset_revision: str = "f2970eb3a55777454c94069077cc8d9b5866312d"
    tokenizer_revision: str = "40c069824f4251a91eefaf281ebe4c544efd3e18"
    max_length: int = 512
    packing: bool = True
    assistant_only_loss: bool = True
    thinking: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    learning_rate: float = 2e-4
    batch_size: int = 4
    epochs: int = 3
    seed: int = 42
    low_toxicity_threshold: float = 0.05
    high_toxicity_threshold: float = 0.80

    def as_dict(self):
        return asdict(self)

def config_for(role: AdapterRole) -> TrainingConfig:
    if role not in ("nontoxic", "toxic"):
        raise ValueError("adapter must be exactly 'nontoxic' or 'toxic'")
    return TrainingConfig()
