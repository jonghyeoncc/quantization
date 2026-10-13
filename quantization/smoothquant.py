import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import json
import re

import torch
from dotenv import load_dotenv
from datasets import load_dataset
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.utils import align_module_device
from huggingface_hub import snapshot_download
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SmoothQuantModifier
from llmcompressor.modifiers.transform.smoothquant.base import (
    MINIMUM_SMOOTHING_SCALE,
)


SMOOTHQUANT_MAPPINGS = {
    "qwen": [
        [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
        [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
    ],
}


# -------------------------------------------------------------------------------
# Basic configs

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run SmoothQuant W8A8 quantization for a single model."
    )
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
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--smoothing_strength",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--mapping_preset",
        choices=["auto", "qwen", "none"],
        default="auto",
        help=(
            "SmoothQuant layer mapping. auto uses local mappings for Qwen, "
            "otherwise llmcompressor defaults."
        ),
    )
    parser.add_argument(
        "--save_compressed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _infer_calib_dirname(calib_file):
    if not calib_file:
        return "calib_1"

    calib_name = os.path.basename(calib_file)
    match = re.search(r"(?:^|_)(?:val|calib|bootstrap)_(\d+)(?:_|\.|$)", calib_name)
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
    return os.path.join(default_root, f"{safe_name}-smoothquant-w8a8")


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
    processor = AutoProcessor.from_pretrained(
        model_source,
        use_fast=True,
        **hf_kwargs,
    )
    return model, processor


def _load_calibration_dataset(calib_file):
    if calib_file:
        dataset = load_dataset("json", data_files=calib_file, split="train")
    else:
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation[:1024]")
    return _normalize_calibration_dataset(dataset)


def _extract_text(record):
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return json.dumps(record, ensure_ascii=False)

    for key in ("text", "input", "prompt", "question", "content", "sentence", "instruction"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value

    messages = record.get("messages")
    if isinstance(messages, list):
        parts = [
            message["content"]
            for message in messages
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        ]
        if parts:
            return "\n".join(parts)

    return json.dumps(record, ensure_ascii=False)


def _normalize_calibration_dataset(dataset):
    if "text" in dataset.column_names:
        return dataset
    return dataset.map(lambda row: {"text": _extract_text(row)})


def _load_model_config(model_source, token, revision):
    hf_kwargs = _build_hf_load_kwargs(model_source, token, revision)
    return AutoConfig.from_pretrained(model_source, **hf_kwargs)


def _store_parameter_update(module, parameter_name, updated_value):
    current_value = getattr(module, parameter_name)
    updated_value = updated_value.to(device=current_value.device, dtype=current_value.dtype)

    if isinstance(module._parameters, OffloadCache):
        module._parameters[parameter_name] = updated_value
    else:
        current_value.copy_(updated_value)


def _install_smoothquant_offload_fix():
    if getattr(SmoothQuantModifier, "_offload_fix_installed", False):
        return

    @torch.no_grad()
    def _apply_smoothing_with_persistent_offload_updates(self, model):
        for mapping in self.resolved_mappings_:
            if mapping.smooth_name not in self.scales_:
                continue
            logger.info(f"Smoothing with {mapping.smooth_name}")

            activation_scales = (
                self.scales_[mapping.smooth_name].max_channel_vals
                - self.scales_[mapping.smooth_name].min_channel_vals
            )
            smooth_layer = mapping.smooth_layer
            balance_layers = mapping.balance_layers

            scales = self._calculate_smoothing_scales(balance_layers, activation_scales)
            scales = torch.maximum(
                scales,
                torch.tensor([MINIMUM_SMOOTHING_SCALE], device=scales.device),
            )

            def smooth(module):
                with align_module_device(module):
                    module_scales = scales.to(device=module.weight.device)
                    if module in balance_layers:
                        updated_weight = module.weight * module_scales.view(1, -1)
                        _store_parameter_update(module, "weight", updated_weight)
                    elif module == smooth_layer:
                        if module.weight.ndim == 1:
                            updated_weight = module.weight / module_scales
                        else:
                            updated_weight = module.weight / module_scales.view(-1, 1)
                        _store_parameter_update(module, "weight", updated_weight)

                        bias = getattr(module, "bias", None)
                        if bias is not None:
                            updated_bias = bias / module_scales
                            _store_parameter_update(module, "bias", updated_bias)

            for layer in balance_layers:
                smooth(layer)
            smooth(smooth_layer)

            del self.scales_[mapping.smooth_name]

    SmoothQuantModifier._apply_smoothing = _apply_smoothing_with_persistent_offload_updates
    SmoothQuantModifier._offload_fix_installed = True


def _choose_mapping_preset(model_name, model_source, config, preset):
    if preset == "none":
        return None
    if preset != "auto":
        print(f"Using {preset.upper()}-specific SmoothQuant mappings.")
        return SMOOTHQUANT_MAPPINGS[preset]

    model_type = str(getattr(config, "model_type", "") or "").lower()
    architectures = [str(item) for item in getattr(config, "architectures", []) or []]
    arch_text = " ".join(architectures).lower()
    source_text = f"{model_name} {model_source}".lower()

    if model_type.startswith("qwen") or "qwen" in arch_text or "qwen" in source_text:
        print("Using Qwen-specific SmoothQuant mappings.")
        return SMOOTHQUANT_MAPPINGS["qwen"]

    print("Using default SmoothQuant mappings inferred by llmcompressor.")
    return None


def _build_smoothquant_modifier_kwargs(
    model_name,
    model_source,
    config,
    mapping_preset,
    smoothing_strength,
    num_calibration_samples,
):
    kwargs = {
        "smoothing_strength": smoothing_strength,
        "num_calibration_steps": num_calibration_samples,
    }

    mappings = _choose_mapping_preset(model_name, model_source, config, mapping_preset)
    if mappings is not None:
        kwargs["mappings"] = mappings

    return kwargs


args = _parse_args()

load_dotenv()
token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

model_path = args.model_name
if args.revision and not os.path.exists(model_path):
    model_path = snapshot_download(
        repo_id=model_path,
        revision=args.revision,
        token=token,
    )
output_path = _build_output_path(args.model_name, args.output_dir, args.calib_file)

os.makedirs(output_path, exist_ok=True)

model_config = _load_model_config(model_path, token, args.revision)
model, processor = _load_model_and_processor(model_path, token, args.revision)

calibration_dataset = _load_calibration_dataset(args.calib_file)
num_calibration_samples = args.num_calibration_samples or len(calibration_dataset)

_install_smoothquant_offload_fix()

recipe = []
if args.mapping_preset == "none":
    print("Skipping SmoothQuant smoothing; running plain W8A8.")
else:
    recipe.append(
        SmoothQuantModifier(
            **_build_smoothquant_modifier_kwargs(
                args.model_name,
                model_path,
                model_config,
                args.mapping_preset,
                args.smoothing_strength,
                num_calibration_samples,
            )
        )
    )

recipe.append(
    QuantizationModifier(
        targets=["Linear"],
        ignore=["lm_head"],
        scheme="W8A8",
    ),
)

# -------------------------------------------------------------------------------

oneshot(
    model=model,
    processor=processor,
    dataset=calibration_dataset,
    recipe=recipe,
    output_dir=output_path,
    num_calibration_samples=num_calibration_samples,
    max_seq_length=args.max_seq_length,
    save_compressed=args.save_compressed,
    trust_remote_code_model=True,
    precision="bfloat16",
)

print(f"SmoothQuant W8A8 quantized model saved to {output_path}.")
