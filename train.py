import os
import json
import argparse
import random
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback

# -----------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_LENGTH = 1024
LEARNING_RATE = 3e-5
NUM_TRAIN_EPOCHS = 2
LOGGING_STEPS = 50
SAVE_STEPS = 100
EVAL_STEPS = 100
SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------

class SaveBestModelCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.best_loss = float('inf')
        self.best_dir = os.path.join(output_dir, "best_model")
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and eval_loss < self.best_loss:
            self.best_loss = eval_loss
            model = kwargs.get("model")
            tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
            if model is not None:
                model.save_pretrained(self.best_dir)
                if tokenizer:
                    tokenizer.save_pretrained(self.best_dir)
                print(f"New best model saved! eval_loss: {eval_loss:.4f}")


class MemoryManagementCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:  # Every 10 steps to avoid overhead
            torch.cuda.empty_cache()

# -----------------------------------------------------------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_splits(dataset_name):
    data_path = os.path.join(BASE_DIR, "unified_dataset", f"{dataset_name}_train.json")
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    train_samples = [sample for sample in raw if sample["metadata"]["split"] == "train"]
    val_samples = [sample for sample in raw if sample["metadata"]["split"] == "validation"]
    return Dataset.from_list(train_samples), Dataset.from_list(val_samples)

def build_processor(tokenizer):
    def _format(example):
        messages = [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["target"]},
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
        labels = tokens["input_ids"][:]
        
        assistant_preamble_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["input"]}, {"role": "assistant", "content": ""}],
            tokenize=False,
            add_generation_prompt=False,
        )
        assistant_preamble_ids = tokenizer(assistant_preamble_text, truncation=True, max_length=MAX_LENGTH, padding=False)["input_ids"]
        assistant_start = min(len(assistant_preamble_ids), len(labels))
        attention_mask = tokens["attention_mask"]

        for idx in range(len(labels)):
            if idx < assistant_start or attention_mask[idx] == 0:
                labels[idx] = -100

        return {"input_ids": tokens["input_ids"], "attention_mask": attention_mask, "labels": labels}

    return _format

def prepare_dataset(dataset, tokenizer):
    processor = build_processor(tokenizer)
    return dataset.map(processor, remove_columns=dataset.column_names)

def load_model(model_name, revision=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", 
        trust_remote_code=True,
        revision=revision,
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir, batch_size, grad_accum, deepspeed_config, resume_from_checkpoint=False):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    optim = "adamw_torch_fused" if deepspeed_config is None else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        save_total_limit=1,
        optim=optim,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=SEED,
        data_seed=SEED,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # callbacks=[SaveBestModelCallback(output_dir), MemoryManagementCallback()],
        callbacks=[SaveBestModelCallback(output_dir)]
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deepspeed", type=str, default=None) 
    parser.add_argument("--revision", type=str, default=None)
    args = parser.parse_args()

    set_seed()
    model, tokenizer = load_model(args.model, args.revision)
    train_split, val_split = load_splits(args.dataset)
    train_dataset = prepare_dataset(train_split, tokenizer)
    val_dataset = prepare_dataset(val_split, tokenizer)
    model_id = args.model.split("/")[-1]
    output_dir = os.path.join(BASE_DIR, "new_models", args.dataset, f"{model_id}-ft")
    os.makedirs(output_dir, exist_ok=True)

    train_model(
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        output_dir,
        args.batch_size,
        args.grad_accum,
        args.deepspeed,
        resume_from_checkpoint=args.resume
    )

if __name__ == "__main__":
    main()
