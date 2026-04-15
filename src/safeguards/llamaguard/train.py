"""LoRA fine-tuning script for Llama-Guard-3.

Fine-tunes Llama-Guard-3-8B with LoRA for elderly-specific safety classification.
Training config from the paper:
- LoRA: r=16, alpha=16, dropout=0, all projection layers
- Schedule: batch=4, grad_accum=4, max_steps=500, lr=2e-5, cosine, fp16
"""

from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

from ...config import TrainingConfig


def setup_model_and_tokenizer(base_model: str, load_in_4bit: bool = True):
    """Load the base Llama-Guard-3 model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.config.use_cache = False
    return model, tokenizer


def setup_lora(model, config: TrainingConfig):
    """Apply LoRA adapter to the model."""
    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        target_modules=config.lora.target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    config: TrainingConfig,
) -> str:
    """Run the training loop.

    Returns path to the saved adapter.
    """
    output_dir = config.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.schedule.per_device_train_batch_size,
        gradient_accumulation_steps=config.schedule.gradient_accumulation_steps,
        max_steps=config.schedule.max_steps,
        warmup_steps=config.schedule.warmup_steps,
        learning_rate=config.schedule.learning_rate,
        weight_decay=config.schedule.weight_decay,
        lr_scheduler_type=config.schedule.lr_scheduler_type,
        fp16=config.schedule.fp16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=2048,
    )

    trainer.train()

    # Save the final adapter
    adapter_path = str(Path(output_dir) / "checkpoint-500")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print(f"Adapter saved to {adapter_path}")
    return adapter_path


def run_training_pipeline(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    config: TrainingConfig,
) -> str:
    """Full training pipeline: load model -> apply LoRA -> train -> save.

    Returns path to the saved adapter.
    """
    print(f"Loading base model: {config.base_model}")
    model, tokenizer = setup_model_and_tokenizer(config.base_model)

    print("Applying LoRA adapter...")
    model = setup_lora(model, config)

    print(f"Training for {config.schedule.max_steps} steps...")
    adapter_path = train(model, tokenizer, train_dataset, eval_dataset, config)

    return adapter_path
