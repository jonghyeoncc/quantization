import os
import argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# -------------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Run RTN quantization for a single model.")
    parser.add_argument(
        "--model_name",
        required=True,
        help="Hugging Face repo id or local path for the base model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision (commit SHA, tag, or branch).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory for the quantized model.",
    )
    return parser.parse_args()


def _build_output_path(model_name, explicit_dir):
    if explicit_dir:
        return explicit_dir
    safe_name = model_name.replace("/", "-")
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models/fine-tuned/rtn"))
    return os.path.join(default_root, f"{safe_name}-rtn-4bit")


def _build_hf_load_kwargs(model_source, token, revision):
    kwargs = {
        "trust_remote_code": True,
    }
    if token:
        kwargs["token"] = token
    if revision and not os.path.exists(model_source):
        kwargs["revision"] = revision
    return kwargs


def _load_model_and_processor(model_source, token, revision):
    hf_kwargs = _build_hf_load_kwargs(model_source, token, revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype="auto",
        **hf_kwargs,
    )
    try:
        processor = AutoProcessor.from_pretrained(
            model_source,
            use_fast=True,
            **hf_kwargs,
        )
    except ValueError:
        processor = AutoProcessor.from_pretrained(
            model_source,
            use_fast=False,
            **hf_kwargs,
        )
    return model, processor


args = _parse_args()

load_dotenv()
token = os.getenv("HUGGING_FACE_HUB_TOKEN")

model_path = args.model_name
if args.revision and not os.path.exists(model_path):
    model_path = snapshot_download(
        repo_id=model_path,
        revision=args.revision,
        token=token,
    )
output_path = _build_output_path(model_path, args.output_dir)

os.makedirs(output_path, exist_ok=True)

model, processor = _load_model_and_processor(model_path, token, args.revision)

recipe = [
    QuantizationModifier(
        targets=["Linear"],
        ignore=["lm_head"],
        scheme="W4A16",
    )
]

# -------------------------------------------------------------------------------

oneshot(
    model=model,
    processor=processor,
    recipe=recipe,
    output_dir=output_path,
    save_compressed=True,
    trust_remote_code_model=True,
)

print(f"RTN quantized model saved to {output_path}.")
