import argparse, json, time

import numpy as np
import torch
import torch.nn.functional as F

from common import LAYERS, OUT, SAE_DIR, SEED

SEGS = ["instr", "span", "post", "think", "think_end", "answer", "label_pred"]


def load_split(acts_dir, split, layer, device):
    """(rows, segment codes) of one split: the stored parts in order, without
    the rows flagged massive at this layer (Sun et al. 2024 §2)."""
    man = json.loads((acts_dir / "manifest.json").read_text())
    n_parts = len(list(acts_dir.glob(f"{split}.part*.docs.npy")))
    p = np.load(acts_dir / "plan.npz")
    tags, keep, ctx_off, keep_off = p["tags"].copy(), p["keep"], p["ctx_off"], p["keep_off"]
    tags[p["label"] > 0] = 6
    ok, seg = [], []
    for g in range(n_parts):
        ok.append(~np.load(acts_dir / f"L{layer}" / f"{split}.part{g}.massive.npy"))
        seg.append(np.concatenate([tags[ctx_off[i] + keep[keep_off[i]:keep_off[i + 1]]]
                                   for i in np.load(acts_dir / f"{split}.part{g}.docs.npy")]))
    x = torch.empty(int(sum(o.sum() for o in ok)), man["d_model"], dtype=torch.bfloat16)
    r = 0
    for g in range(n_parts):
        a = np.load(acts_dir / f"L{layer}" / f"{split}.part{g}.npy", mmap_mode="r")
        for i in range(0, len(a), 1 << 18):
            c = torch.from_numpy(np.asarray(a[i:i + (1 << 18)])[ok[g][i:i + (1 << 18)]]).view(torch.bfloat16)
            x[r:r + len(c)] = c
            r += len(c)
    seg = np.concatenate([s[o] for s, o in zip(seg, ok)])
    return x.to(device), torch.from_numpy(seg).to(device)


class BatchTopKSAE(torch.nn.Module):
    def __init__(self, d, d_sae, mu, device):
        super().__init__()
        self.d_sae = d_sae
        g = torch.Generator(device=device).manual_seed(SEED)
        w = torch.randn(d_sae, d, generator=g, device=device)
        self.W_dec = torch.nn.Parameter(w / w.norm(dim=-1, keepdim=True))
        self.W_enc = torch.nn.Parameter(self.W_dec.data.t().clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae, device=device))
        self.b_pre = torch.nn.Parameter(mu.clone())
        self.register_buffer("theta", torch.tensor(0.0, device=device))
        self.register_buffer("last_fired", torch.zeros(d_sae, dtype=torch.long, device=device))

    def acts(self, x):
        return F.relu((x - self.b_pre) @ self.W_enc + self.b_enc)

    @torch.no_grad()
    def infer(self, x):
        a = self.acts(x)
        z = a * (a > self.theta)
        return z, z @ self.W_dec + self.b_pre


def load_sae(path, device="cuda"):
    """A trained SAE from sae.pt, fp32, inference mode."""
    ck = torch.load(path, map_location=device, weights_only=True)
    c = ck["config"]
    sae = BatchTopKSAE(c["d_model"], c["d_sae"], torch.zeros(c["d_model"], device=device), device)
    sae.load_state_dict(ck["state_dict"])
    return sae.eval()


@torch.no_grad()
def val_metrics(sae, x_val, mu, seg=None, bs=16384):
    """EV (mean-centred with the fit mean), inference L0, fired fraction; per segment if seg is given."""
    groups = {"all": None} if seg is None else {"all": None, **{s: j for j, s in enumerate(SEGS)}}
    acc = {g: [0.0, 0.0, 0, 0] for g in groups}
    fired = torch.zeros(sae.d_sae, dtype=torch.bool, device=x_val.device)
    for i in range(0, len(x_val), bs):
        x = x_val[i : i + bs]
        with torch.autocast("cuda", torch.bfloat16):
            z, recon = sae.infer(x)
        e = (recon.float() - x.float()).pow(2).sum(1).double()
        v = (x.float() - mu).pow(2).sum(1).double()
        l0 = (z > 0).sum(1)
        fired |= (z > 0).any(0)
        for g, j in groups.items():
            m = torch.ones_like(l0, dtype=torch.bool) if j is None else seg[i : i + bs] == j
            a = acc[g]
            a[0] += e[m].sum(); a[1] += v[m].sum(); a[2] += l0[m].sum(); a[3] += int(m.sum())
    out = {g: {"ev": float(1 - a[0] / a[1]), "l0": float(a[2] / a[3]), "n": a[3]}
           for g, a in acc.items() if a[3]}
    res = {"ev": out["all"]["ev"], "l0": out["all"]["l0"], "val_fired_frac": float(fired.float().mean())}
    return res if seg is None else {**res, "per_segment": out}


def train(cfg, x_fit, x_val, mu, out_dir, seg_val=None):
    B, steps, k = cfg["batch_size"], cfg["steps"], cfg["k"]
    sae = BatchTopKSAE(x_fit.shape[1], cfg["d_sae"], mu, mu.device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], eps=6.25e-10, fused=True)
    g = torch.Generator().manual_seed(SEED)
    perm, off = torch.randperm(len(x_fit), generator=g), 0

    out_dir.mkdir(parents=True, exist_ok=True)
    log, t0 = open(out_dir / "train_log.jsonl", "w"), time.time()
    for step in range(steps):
        warm, decay = cfg["warmup_steps"], cfg["decay_steps"]
        scale = min(1.0, (step + 1) / warm, max(0.0, (steps - step) / decay))
        for pg in opt.param_groups:
            pg["lr"] = cfg["lr"] * scale

        if off + B > len(perm):
            perm, off = torch.randperm(len(x_fit), generator=g), 0
        x = x_fit[perm[off : off + B]].to(sae.b_pre.device, non_blocking=True)
        off += B

        with torch.autocast("cuda", torch.bfloat16):
            a = sae.acts(x)
            flat = a.flatten()
            # exact BatchTopK; prefilter at 0.5*theta only shrinks the topk input
            # (any global top-B*K value exceeds theta/2 whenever >= B*K survive)
            with torch.no_grad():
                cand = (flat > 0.5 * sae.theta).nonzero(as_tuple=True)[0]
                if cand.numel() >= B * k:
                    _, ii = torch.topk(flat[cand], B * k, sorted=False)
                    idx = cand[ii]
                else:
                    _, idx = torch.topk(flat, B * k, sorted=False)
            vals = flat.gather(0, idx)
            z = torch.zeros_like(flat).scatter(0, idx, vals).view(B, -1)
            recon = z @ sae.W_dec + sae.b_pre
        loss = (recon.float() - x.float()).pow(2).sum(-1).mean()

        with torch.no_grad():
            pos = vals > 0
            sae.last_fired[(idx[pos] % sae.d_sae).unique()] = step
            m = vals[pos].min().float()
            sae.theta.copy_(m if step == 0 else 0.99 * sae.theta + 0.01 * m)
            dead = (step - sae.last_fired) > cfg["dead_window"]
        aux = torch.tensor(0.0, device=x.device)
        dead_idx = dead.nonzero(as_tuple=True)[0]
        if dead_idx.numel():
            with torch.autocast("cuda", torch.bfloat16):
                ad = a[:, dead_idx]
                if dead_idx.numel() > cfg["k_aux"]:
                    av, ai = torch.topk(ad, cfg["k_aux"], dim=-1)
                    ad = torch.zeros_like(ad).scatter(-1, ai, av)
                ehat = ad @ sae.W_dec[dead_idx]
            e = (x.float() - recon.float()).detach()
            aux = (ehat.float() - e).pow(2).sum(-1).mean()
        (loss + aux / 32).backward()

        with torch.no_grad():
            w, gr = sae.W_dec.data, sae.W_dec.grad
            gr -= (gr * w).sum(-1, keepdim=True) * w
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            sae.W_dec.data /= sae.W_dec.data.norm(dim=-1, keepdim=True)

        if (step + 1) % 500 == 0 or step + 1 == steps:
            rec = {"step": step + 1, "loss": round(loss.item(), 2),
                   "aux": round(aux.item(), 2), "theta": round(sae.theta.item(), 4),
                   "dead_frac": round(dead.float().mean().item(), 5),
                   "elapsed": round(time.time() - t0, 1), **val_metrics(sae, x_val, mu)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{out_dir.name}] {rec}", flush=True)

    metrics = {**cfg, "train_sec": round(time.time() - t0, 1),
               "dead_frac_final": round(dead.float().mean().item(), 5),
               **val_metrics(sae, x_val, mu, seg_val)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    torch.save({"state_dict": sae.state_dict(), "config": cfg}, out_dir / "sae.pt")
    log.close()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LAYERS))
    args = ap.parse_args()
    ef, k, steps, batch_size = 8, 64, 6000, 4096

    torch.manual_seed(SEED)
    acts_dir, out = OUT / "activations" / args.model, SAE_DIR / args.model
    man = json.loads((acts_dir / "manifest.json").read_text())
    d = man["d_model"]
    for layer in man["layers"]:
        x_fit, _ = load_split(acts_dir, "fit", layer, "cpu")
        x_val, seg_val = load_split(acts_dir, "val", layer, "cuda")
        mu = torch.zeros(d, dtype=torch.float64, device="cuda")
        for i in range(0, len(x_fit), 1 << 18):
            mu += x_fit[i:i + (1 << 18)].to("cuda").double().sum(0)
        mu = (mu / len(x_fit)).float()
        (out / f"L{layer}").mkdir(parents=True, exist_ok=True)
        np.save(out / f"L{layer}" / "fit_mean.npy", mu.cpu().numpy())
        cfg = {"layer": layer, "ef": ef, "k": k, "d_model": d, "d_sae": ef * d,
               "steps": steps, "batch_size": batch_size, "lr": 3e-4,
               "warmup_steps": 200, "decay_steps": 1200, "k_aux": 512, "dead_window": 250,
               "aux_coef": 1 / 32, "random_seed": SEED, "acts_dir": str(acts_dir),
               "n_fit": len(x_fit), "n_val": len(x_val)}
        m = train(cfg, x_fit, x_val, mu, out / f"L{layer}" / f"EF{ef}_K{k}", seg_val)
        print("DONE", json.dumps(m), flush=True)
        del x_fit, x_val, mu
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
