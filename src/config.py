"""Configuration loading and management."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _resolve_env_vars(value: str) -> str:
    """Replace ${ENV_VAR} patterns with actual environment variable values."""
    pattern = re.compile(r"\$\{(\w+)\}")
    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")
    return pattern.sub(replacer, value)


def _resolve_config(obj):
    """Recursively resolve environment variables in config values."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: _resolve_config(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_config(item) for item in obj]
    return obj


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    bias: str = "none"
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingSchedule:
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 500
    warmup_steps: int = 50
    learning_rate: float = 2e-5
    optimizer: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    fp16: bool = True


@dataclass
class TrainingConfig:
    base_model: str = "meta-llama/Llama-Guard-3-8B"
    output_dir: str = "outputs/models/elderly-guard"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    schedule: TrainingSchedule = field(default_factory=TrainingSchedule)
    grandguard_fraction: float = 0.62
    general_fraction: float = 0.38


@dataclass
class ModelSpec:
    name: str
    provider: str
    display_name: str = ""

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name


@dataclass
class Config:
    # Paths
    dataset_path: str = "data/elderlysafe_final.parquet"
    output_dir: str = "outputs"
    results_dir: str = "outputs/results"
    figures_dir: str = "outputs/figures"
    tables_dir: str = "outputs/tables"
    model_dir: str = "outputs/models"

    # API keys
    api_keys: dict = field(default_factory=dict)

    # Models
    generator_model: str = "grok-4"
    judge_filter_model: str = "gpt-5.1"
    judge_models: list = field(default_factory=lambda: ["gemini-2.5-flash", "gpt-5.1"])
    target_llms: list = field(default_factory=list)
    safeguard_model: str = "gpt-oss-safeguard-20b"
    safeguard_provider: str = "openai"

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Evaluation
    num_sample_prompts: int = 500
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        resolved = _resolve_config(raw)

        # Build paths
        paths = resolved.get("paths", {})

        # Build API keys
        api_keys = resolved.get("api_keys", {})

        # Build model specs
        models_cfg = resolved.get("models", {})
        target_llms = []
        for m in models_cfg.get("target_llms", []):
            target_llms.append(ModelSpec(
                name=m["name"],
                provider=m["provider"],
                display_name=m.get("display_name", m["name"]),
            ))

        # Build training config
        train_cfg = resolved.get("training", {})
        lora_cfg = train_cfg.get("lora", {})
        sched_cfg = train_cfg.get("schedule", {})
        data_mix = train_cfg.get("data_mix", {})

        training = TrainingConfig(
            base_model=train_cfg.get("base_model", "meta-llama/Llama-Guard-3-8B"),
            output_dir=train_cfg.get("output_dir", "outputs/models/elderly-guard"),
            lora=LoRAConfig(
                rank=lora_cfg.get("rank", 16),
                alpha=lora_cfg.get("alpha", 16),
                dropout=lora_cfg.get("dropout", 0.0),
                bias=lora_cfg.get("bias", "none"),
                target_modules=lora_cfg.get("target_modules", [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]),
                task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
            ),
            schedule=TrainingSchedule(
                per_device_train_batch_size=sched_cfg.get("per_device_train_batch_size", 4),
                gradient_accumulation_steps=sched_cfg.get("gradient_accumulation_steps", 4),
                max_steps=sched_cfg.get("max_steps", 500),
                warmup_steps=sched_cfg.get("warmup_steps", 50),
                learning_rate=sched_cfg.get("learning_rate", 2e-5),
                optimizer=sched_cfg.get("optimizer", "adamw_8bit"),
                weight_decay=sched_cfg.get("weight_decay", 0.01),
                lr_scheduler_type=sched_cfg.get("lr_scheduler_type", "cosine"),
                fp16=sched_cfg.get("fp16", True),
            ),
            grandguard_fraction=data_mix.get("grandguard_fraction", 0.62),
            general_fraction=data_mix.get("general_fraction", 0.38),
        )

        safeguard_cfg = resolved.get("safeguard", {})
        eval_cfg = resolved.get("evaluation", {})

        return cls(
            dataset_path=paths.get("dataset", "data/elderlysafe_final.parquet"),
            output_dir=paths.get("output_dir", "outputs"),
            results_dir=paths.get("results_dir", "outputs/results"),
            figures_dir=paths.get("figures_dir", "outputs/figures"),
            tables_dir=paths.get("tables_dir", "outputs/tables"),
            model_dir=paths.get("model_dir", "outputs/models"),
            api_keys=api_keys,
            generator_model=models_cfg.get("generator", "grok-4"),
            judge_filter_model=models_cfg.get("judge_filter", "gpt-5.1"),
            judge_models=models_cfg.get("judges", ["gemini-2.5-flash", "gpt-5.1"]),
            target_llms=target_llms,
            safeguard_model=safeguard_cfg.get("model", "gpt-oss-safeguard-20b"),
            safeguard_provider=safeguard_cfg.get("provider", "openai"),
            training=training,
            num_sample_prompts=eval_cfg.get("num_sample_prompts", 500),
            seed=eval_cfg.get("seed", 42),
        )

    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        for d in [self.output_dir, self.results_dir, self.figures_dir,
                  self.tables_dir, self.model_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
