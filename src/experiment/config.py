"""
Configuration management for experiments.
Supports YAML/JSON configs with validation and defaults.
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False


@dataclass
class DataConfig:
    train_path: str = "../data/alpaca-instruction-data.json"
    train_split: float = 0.85
    val_split: float = 0.05
    test_split: float = 0.1
    batch_size: int = 8
    max_length: Optional[int] = 512
    num_workers: int = 0
    prompt_style: str = "enhanced"
    shuffle: bool = True


@dataclass
class OptimizerConfig:
    name: str = "AdamW"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    scheduler: Optional[str] = "cosine"
    warmup_steps: int = 100
    num_training_steps: Optional[int] = None


@dataclass
class TrainingConfig:
    num_epochs: int = 3
    eval_freq: int = 100
    eval_iter: int = 10
    save_freq: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: bool = False
    device: str = "cuda"
    seed: int = 42


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: [
        "loss", "perplexity", "accuracy", "bleu", "rouge"
    ])
    generate_samples: bool = True
    num_samples: int = 5
    max_gen_length: int = 100
    temperature: float = 0.7
    top_k: int = 50


@dataclass
class ExperimentConfig:
    experiment_name: str = "default_experiment"
    description: str = ""
    output_dir: str = "experiments"

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str, format: str = "yaml"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        if format == "yaml":
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif format == "json":
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelConfig(**data['model'])
        if 'data' in data and isinstance(data['data'], dict):
            data['data'] = DataConfig(**data['data'])
        if 'optimizer' in data and isinstance(data['optimizer'], dict):
            if 'betas' in data['optimizer'] and isinstance(data['optimizer']['betas'], list):
                data['optimizer']['betas'] = tuple(data['optimizer']['betas'])
            data['optimizer'] = OptimizerConfig(**data['optimizer'])
        if 'training' in data and isinstance(data['training'], dict):
            data['training'] = TrainingConfig(**data['training'])
        if 'evaluation' in data and isinstance(data['evaluation'], dict):
            data['evaluation'] = EvaluationConfig(**data['evaluation'])
        return cls(**data)

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        path = Path(path)
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        return cls.from_dict(data)


def load_config(path: str) -> ExperimentConfig:
    return ExperimentConfig.load(path)


def save_config(config: ExperimentConfig, path: str, format: str = "yaml"):
    config.save(path, format)


def create_default_configs():
    configs_dir = Path("configs/templates")
    configs_dir.mkdir(parents=True, exist_ok=True)

    fast_config = ExperimentConfig(
        experiment_name="fast_experiment",
        description="Quick test with small model and data",
    )
    fast_config.model.n_layers = 6
    fast_config.model.n_heads = 6
    fast_config.model.emb_dim = 384
    fast_config.data.batch_size = 16
    fast_config.training.num_epochs = 1
    fast_config.save("configs/templates/fast.yaml")

    full_config = ExperimentConfig(
        experiment_name="full_training",
        description="Full model training with all features",
    )
    full_config.data.batch_size = 8
    full_config.training.num_epochs = 3
    full_config.training.mixed_precision = True
    full_config.save("configs/templates/full.yaml")

    lora_config = ExperimentConfig(
        experiment_name="lora_finetune",
        description="Parameter-efficient fine-tuning with LoRA",
    )
    lora_config.optimizer.learning_rate = 1e-4
    lora_config.training.num_epochs = 5
    lora_config.save("configs/templates/lora.yaml")

    print(f"✅ Created default configs in {configs_dir}")


if __name__ == "__main__":
    create_default_configs()






