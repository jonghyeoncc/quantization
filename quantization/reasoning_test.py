import os
import json
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig

# -----------------------------------------------------------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"


DEFAULT_GPU_MEMORY_UTILIZATION = 0.96
MAX_MODEL_LEN = 8192
SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_test_data(dataset_name):
    data_path = os.path.join(BASE_DIR, "unified_dataset", f"{dataset_name}_test.json")
    return load_dataset("json", data_files=data_path, split="train")

def build_model(model_name, tensor_parallel_size, revision=None, gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION, enforce_eager=False, seed=SEED, max_model_len=MAX_MODEL_LEN):
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        revision=revision,
        enforce_eager=enforce_eager,
        seed=seed,
        reasoning_config=ReasoningConfig(
            reasoning_start_str="<think>",
            reasoning_end_str="</think>",
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        revision=revision,
    )
    return llm, tokenizer

def create_inputs(text):
    return text

def run_inference(llm, tokenizer, dataset, temperature, max_new_tokens, thinking_token_budget, seed=SEED):
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
        thinking_token_budget=thinking_token_budget,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    total = len(dataset)
    print("-" * 100)
    print(f"Total samples: {total}")
    print("-" * 100)

    all_prompts = []
    for sample in tqdm(dataset, desc="Preparing prompts"):
        prompt = create_inputs(sample["input"])
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        all_prompts.append(input_text)

    print(f"Running batched inference on {total} prompts...")
    outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

    results = []
    for idx, (sample, output) in enumerate(zip(dataset, outputs)):
        metadata = dict(sample["metadata"])
        metadata.pop("date", None)
        result = {
            "index": idx,
            "id": sample["id"],
            "input": sample["input"],
            "model_prediction": output.outputs[0].text,
            "metadata": metadata,
        }
        results.append(result)
    return results

def save_results(records, dataset_name, model_name, output_dir=None, output_filename_prefix=""):
    model_id = model_name.split("/")[-1]
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "zero-shot-reasoning-results", dataset_name)
    elif not os.path.isabs(output_dir):
        output_dir = os.path.join(BASE_DIR, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{output_filename_prefix}{model_id}-{dataset_name}-test.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")
    print(f"Total samples processed: {len(records)}")

# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=4864)
    parser.add_argument("--thinking_token_budget", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--enforce_eager", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_filename_prefix", type=str, default="")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_model_len", type=int, default=MAX_MODEL_LEN)

    args = parser.parse_args()

    set_seed(args.seed)
    test_data = load_test_data(args.dataset)
    llm, tokenizer = build_model(
        args.model,
        args.tensor_parallel_size,
        args.revision,
        args.gpu_memory_utilization,
        args.enforce_eager,
        args.seed,
        args.max_model_len,
    )
    records = run_inference(llm, tokenizer, test_data, args.temperature, args.max_new_tokens, args.thinking_token_budget, args.seed)
    save_results(records, args.dataset, args.model, args.output_dir, args.output_filename_prefix)


if __name__ == "__main__":
    main()
