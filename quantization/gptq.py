import os
import argparse
import re
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# -------------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        required=True,
    )
    parser.add_argument(
        "--revision",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
    )
    parser.add_argument(
        "--calib_file",
        default=None,
    )
    return parser.parse_args()


def _infer_calib_dirname(calib_file):
    if not calib_file:
        return "calib_1"

    calib_name = os.path.basename(calib_file)
    match = re.search(r"(?:^|_)(?:val|calib)_(\d+)(?:_|\.|$)", calib_name)
    if match:
        return f"calib_{match.group(1)}"

    return "calib_1"


def _build_output_path(model_name, explicit_dir, calib_file):
    if explicit_dir:
        return explicit_dir
    safe_name = model_name.replace("/", "-")
    default_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "models/random_calib/dataset",
            _infer_calib_dirname(calib_file),
        )
    )
    return os.path.join(default_root, f"{safe_name}-gptq-4bit")


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


def _load_calibration_dataset(calib_file):
    if calib_file:
        dataset = load_dataset("json", data_files=calib_file, split="train")
    else:
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation[:1024]")
    return dataset

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
output_path = _build_output_path(args.model_name, args.output_dir, args.calib_file)

os.makedirs(output_path, exist_ok=True)

model, processor = _load_model_and_processor(model_path, token, args.revision)

calibration_dataset = _load_calibration_dataset(args.calib_file)
num_calibration_samples = len(calibration_dataset)

recipe = [
    GPTQModifier(
        scheme="W4A16",
        targets=["Linear"],
        ignore=["lm_head"],
    )
]

# -------------------------------------------------------------------------------

oneshot(
    model=model,
    processor=processor,
    dataset=calibration_dataset,
    recipe=recipe,
    output_dir=output_path,
    num_calibration_samples=num_calibration_samples,
    save_compressed=True,
    trust_remote_code_model=True,
)

print(f"GPTQ quantized model saved to {output_path}.")
