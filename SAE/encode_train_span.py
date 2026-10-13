import argparse, json, time

import numpy as np
import torch

from common import LAYERS, OUT, SAE_CFG, SAE_DIR, SEED
from encode import TokenCodes
from train_sae import load_sae

ACTS = OUT / "activations"
BLOCK = 1 << 17


def span_rows(p, sd):
    """(row index in the part file, token id, doc, pos) of the span rows of documents `sd`, in file order."""
    R, T, D, P, off = [], [], [], [], 0
    for i in sd:
        kp = p["keep"][p["keep_off"][i]:p["keep_off"][i + 1]].astype(np.int64)
        c = p["ctx_off"][i] + kp
        m = p["tags"][c] == 1
        R.append(off + np.flatnonzero(m))
        T.append(p["ids"][c[m]])
        D.append(np.full(int(m.sum()), i))
        P.append(kp[m])
        off += len(kp)
    return [np.concatenate(x) for x in (R, T, D, P)], off


def main(model, layers):
    torch.manual_seed(SEED)
    src = ACTS / model
    p = dict(np.load(src / "plan.npz"))
    parts = [(s, g) for s in ("fit", "val") for g in range(len(list(src.glob(f"{s}.part*.docs.npy"))))]
    meta = {}
    for s, g in parts:
        sd = np.load(src / f"{s}.part{g}.docs.npy")
        meta[s, g], n = span_rows(p, sd)
        assert n == np.load(src / f"L{layers[0]}" / f"{s}.part{g}.npy", mmap_mode="r").shape[0]
    for l in layers:
        t0 = time.time()
        out = OUT / "codes_train" / model / f"L{l}" / "bf16"
        tc = TokenCodes(load_sae(SAE_DIR / model / f"L{l}" / SAE_CFG / "sae.pt"), out)
        for s, g in parts:
            a = np.load(src / f"L{l}" / f"{s}.part{g}.npy", mmap_mode="r")
            R, T, D, P = meta[s, g]
            for i in range(0, len(a), BLOCK):
                j0, j1 = np.searchsorted(R, [i, i + BLOCK])
                x = torch.from_numpy(np.asarray(a[i:i + BLOCK])[R[j0:j1] - i]).view(torch.bfloat16).cuda()
                tc.add(x, token_ids=T[j0:j1], doc=D[j0:j1], pos=P[j0:j1],
                       seg=np.ones(j1 - j0, np.int8), label=np.zeros(j1 - j0, np.int8))
        tc.save()
        r = np.load(out / "rows.npz")
        man = {"model": model, "layer": l, "source": str(src), "sae": str(SAE_DIR / model / f"L{l}" / SAE_CFG),
               "parts": [f"{s}.part{g}" for s, g in parts], "n_rows": int(len(r["doc"])),
               "n_docs": int(len(np.unique(r["doc"]))), "n_massive": int(r["massive"].sum()),
               "elapsed_sec": round(time.time() - t0, 1)}
        (out / "manifest.json").write_text(json.dumps(man, indent=1))
        print(json.dumps(man), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LAYERS))
    a = ap.parse_args()
    main(a.model, LAYERS[a.model])
