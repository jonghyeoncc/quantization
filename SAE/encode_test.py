import argparse, json, time

import numpy as np
import torch

from common import DATASETS, LAYERS, MAX_TOK, OUT, SAE_CFG, SAE_DIR, SEED, SEGS, VARIANTS, forward, load_lm, tag_doc, test_docs, tokenizer
from encode import TokenCodes
from train_sae import load_sae


def prefixes(docs, ids, tags):
    """Common instr token prefix length per dataset."""
    pre = {}
    for d, x, t in zip(docs, ids, tags):
        p = x[:int((t == 0).sum())]
        q = pre.get(d["dataset"], p)
        m = min(len(p), len(q))
        pre[d["dataset"]] = q[:int(np.argmax(np.r_[p[:m] != q[:m], True]))]
    return {k: len(v) for k, v in pre.items()}


def main(model, variant, layers):
    torch.manual_seed(SEED)
    run = variant
    tok = tokenizer(model, variant)
    close_id = tok.convert_tokens_to_ids("</think>")
    alld = test_docs(model, variant)
    ids, tags, labs = zip(*[tag_doc(tok, r, r, close_id) for r in alld])
    pre = prefixes(alld, ids, tags)
    order = list(range(len(alld)))
    owner, keep = {}, []
    for i in order:
        ds, t = alld[i]["dataset"], tags[i]
        k = t != 3
        if ds in owner:
            k[:pre[ds]] = False
        else:
            owner[ds] = alld[i]["pool_row"]
        keep.append(np.flatnonzero(k))
    seq = [ids[i] for i in order]

    sae = {l: load_sae(SAE_DIR / model / f"L{l}" / SAE_CFG / "sae.pt") for l in layers}
    out = {l: OUT / "codes" / model / f"L{l}" / run for l in layers}
    tc = {l: TokenCodes(sae[l], out[l]) for l in layers}
    lm = load_lm(model, variant)

    def fn(l, j, h):
        i, k = order[j], keep[j]
        x = h[torch.from_numpy(k).to(h.device)]
        tc[l].add(x, token_ids=ids[i][k], doc=np.full(len(k), alld[i]["pool_row"]), pos=k,
                  seg=tags[i][k], label=labs[i][k])

    t0 = time.time()
    forward(lm, tok, seq, layers, fn)
    el = time.time() - t0
    for l in layers:
        tc[l].save(doc_order=np.array([alld[i]["pool_row"] for i in order]))
        r = np.load(out[l] / "rows.npz")
        dsr = np.repeat([DATASETS.index(alld[i]["dataset"]) for i in order], np.diff(r["doc_offset"]))
        man = {"model": model, "variant": variant, "run": run, "layer": l, "sae": str(SAE_DIR / model / f"L{l}" / SAE_CFG),
               "segments": SEGS, "seg_codes_kept": {"input": [0, 1, 2], "end": [4, 5]},
               "label": "label_pred mark: 1 = first JSON field (Text categories), 2 = second (Target group)",
               "n_docs": len(order), "n_unclosed": sum(not alld[i]["closed"] for i in order),
               "n_rows": int(len(r["seg"])), "n_ctx_tokens": int(sum(len(s) for s in seq)),
               "rows": {ds: {s: int(((r["seg"] == c) & (dsr == d)).sum()) for c, s in enumerate(SEGS) if c != 3}
                        for d, ds in enumerate(DATASETS)},
               "label_pred": {f"field{f}": int((r["label"] == f).sum()) for f in (1, 2)},
               "massive_rows": {s: int((r["massive"] & (r["seg"] == c)).sum()) for c, s in enumerate(SEGS) if c != 3},
               "instr_prefix_tokens": pre, "instr_prefix_owner_doc": owner,
               "doc_order": "rows.npz doc_order (test-pool rows, processing order), test-pool order",
               "max_batch_tokens": MAX_TOK, "random_seed": SEED, "elapsed_sec": round(el, 1),
               "bytes": sum(f.stat().st_size for f in out[l].iterdir())}
        (out[l] / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"{model} {run}: {len(order)} docs, {man['n_rows']} rows/layer, {el:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LAYERS))
    ap.add_argument("--variant", default="bf16", choices=VARIANTS)
    a = ap.parse_args()
    main(a.model, a.variant, LAYERS[a.model])
