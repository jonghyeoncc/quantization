import json, re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
POOLS = ROOT / "0_pools"
SAE_DIR = ROOT / "1_sae"
OUT = ROOT / "out"

SEED = 42
MODELS = {"Qwen3-1.7B": "Qwen/Qwen3-1.7B", "Qwen3-14B": "Qwen/Qwen3-14B"}
LAYERS = {"Qwen3-1.7B": [7, 14, 21, 23, 27], "Qwen3-14B": [10, 20, 30, 33, 39]}   # 1/4, 1/2, 3/4, 5/6, last
VARIANTS = ["bf16", "awq", "gptq"]
SAE_CFG = "EF8_K64"

HF_CACHE = ROOT / "hf_cache"
QUANT_DIR = ROOT / "quant_models"
QUANT = {"awq": ("AWQ/calib_1", "awq-4bit"), "gptq": ("GPTQ/calib_1", "gptq-4bit")}
MAX_TOK = 16384
SEGS = ["instr", "span", "post", "think", "think_end", "answer"]
FIELD = re.compile(r'"[^"]*"\s*:\s*("?)')


def checkpoint(model, variant="bf16"):
    """Checkpoint path of `model` for `variant` (bf16 = the single HF-cache snapshot)."""
    repo = MODELS[model]
    if variant == "bf16":
        return next((HF_CACHE / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*/"))
    sub, suffix = QUANT[variant]
    return QUANT_DIR / sub / f"{repo.replace('/', '-')}-{suffix}"


def load_lm(model, variant, device="cuda"):
    """Causal LM in bf16; quantized checkpoints are dequantized (needs an environment with compressed-tensors)."""
    from transformers import AutoModelForCausalLM
    from transformers.utils.quantization_config import CompressedTensorsConfig
    kw = {} if variant == "bf16" else \
        {"quantization_config": CompressedTensorsConfig(run_compressed=False)}
    lm = AutoModelForCausalLM.from_pretrained(
        checkpoint(model, variant), dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa", **kw).eval()
    if variant != "bf16":
        plain(lm)
    return lm


def plain(lm):
    """Make decompressed compressed-tensors Linears plain: drop the instance-level forward
    wrapper (it re-fake-quantizes the already dequantized weight on every call, a no-op on
    grid values but slow) and the packed copies it reads."""
    for m in lm.modules():
        if "forward" in m.__dict__ and hasattr(m, "quantization_scheme"):
            del m.forward
            for k in ("weight_packed", "weight_scale", "weight_zero_point", "weight_g_idx", "weight_shape"):
                if k in m._parameters:
                    delattr(m, k)
    torch.cuda.empty_cache()


def batches(lengths, max_tok):
    """Order-preserving greedy batches capped by padded token count."""
    out, cur, mx = [], [], 0
    for i, n in enumerate(lengths):
        if cur and (len(cur) + 1) * max(mx, n) > max_tok:
            out.append(cur)
            cur, mx = [], 0
        cur.append(i)
        mx = max(mx, n)
    return out + [cur]


def massive(x):
    """Sun et al. 2024 §2 massive-activation rule per row: max|x| > 100 and >= 1000 x median|x|."""
    a = x.abs()
    return (a.amax(1) > 100) & (a.amax(1) >= 1000 * a.median(1).values)


def tokenizer(model, variant="bf16"):
    tok = AutoTokenizer.from_pretrained(checkpoint(model, variant))
    tok.padding_side = "right"
    return tok


def test_docs(model, variant):
    """Test documents: the variant's own trace (0_pools/think) joined to its test-pool row."""
    pool = [json.loads(l) for l in open(POOLS / "test_pool.jsonl")]
    rows = [json.loads(l) for l in open(POOLS / "think" / f"{model}-{variant}.jsonl")]
    return [{**r, "input": pool[r["pool_row"]]["input"], "text": pool[r["pool_row"]]["text"]}
            for r in rows]


def forward(lm, tok, ids, layers, fn):
    """Teacher-force every sequence; call fn(layer, doc index, h) with h = that
    document's resid_post rows (seq_len, d_model), bf16, on the model device."""
    cur = []

    def hook(mod, inp, o, l):
        h = o[0] if isinstance(o, tuple) else o
        for j, i in enumerate(cur):
            fn(l, i, h[j, :len(ids[i])])

    hs = [lm.model.layers[l].register_forward_hook(lambda m, i, o, l=l: hook(m, i, o, l))
          for l in layers]
    with torch.inference_mode():
        for b in batches([len(t) for t in ids], MAX_TOK):
            cur[:] = b
            enc = tok.pad({"input_ids": [ids[i].tolist() for i in cur]},
                          return_tensors="pt").to(lm.device)
            lm.model(**enc, use_cache=False)
    for h in hs:
        h.remove()


def think_seq(tk, row, prediction, think_end, full=False):
    """(input_ids, per-token tags) of one document; `row` needs `input` and `text`.
    full=True also returns the char offsets and the full string."""
    rendered = tk.apply_chat_template([{"role": "user", "content": row["input"]}],
                                      tokenize=False, add_generation_prompt=True,
                                      enable_thinking=True)
    s0 = rendered.rfind(row["text"])
    assert s0 >= 0, (row["dataset"], row["orig_id"])
    s1, t0 = s0 + len(row["text"]), len(rendered)
    t1 = t0 + think_end
    enc = tk(rendered + prediction, add_special_tokens=False, return_offsets_mapping=True)
    seg = ["span" if a >= s0 and b <= s1 and b > a else
           "prompt" if a < t0 else "think" if a < t1 else "answer"
           for a, b in enc["offset_mapping"]]
    return (enc["input_ids"], seg, enc["offset_mapping"], rendered + prediction) if full \
        else (enc["input_ids"], seg)


def tag_doc(tk, row, d, close_id):
    """(ids, int8 segment codes, int8 label_pred marks) of one document."""
    ids, seg, offs, txt = think_seq(tk, row, d["prediction"], d["think_end"], full=True)
    n, c, lab = len(ids), np.full(len(ids), -1, np.int8), np.zeros(len(ids), np.int8)
    sp = [i for i, s in enumerate(seg) if s == "span"]
    th = [i for i, s in enumerate(seg) if s == "think"]
    s0, s1, t0, t1 = sp[0], sp[-1] + 1, th[0], th[-1] + 1
    c[:s0], c[s0:s1], c[s1:t0], c[t0:t1] = 0, 1, 2, 3
    if d["closed"]:
        assert ids[t1 - 1] == close_id, d["orig_id"]
        a = t1
        while a < n and tk.decode([ids[a]]).strip() == "":
            a += 1
        c[t1 - 1:a], c[a:] = 4, 5
        if a < n:
            for f, m in enumerate(list(FIELD.finditer(txt, offs[a][0]))[:2], 1):
                v = m.end()
                k = next((i for i in range(a, n) if offs[i][1] > v), None)
                if k is not None:
                    lab[k - 1] = f
    assert (c >= 0).all()
    return np.asarray(ids, np.int32), c, lab
