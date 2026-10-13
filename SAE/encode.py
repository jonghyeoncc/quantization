from pathlib import Path

import numpy as np
import torch

from common import massive

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class Bin:
    """Append-only raw array file; `done()` reopens it as a memmap for np.savez."""

    def __init__(self, path, dtype):
        self.path, self.dtype, self.n = Path(path), np.dtype(dtype), 0
        self.f = open(self.path, "wb")

    def add(self, a):
        a = np.ascontiguousarray(a, self.dtype)
        a.tofile(self.f)
        self.n += a.size

    def done(self):
        self.f.close()
        return np.memmap(self.path, self.dtype, "r", shape=(self.n,)) if self.n \
            else np.empty(0, self.dtype)


class TokenCodes:
    """Per-token codes of one (layer, run), written to `out` by `save`.
    META = per-row fields passed by the caller to `add`."""
    META = [("token_ids", np.int32), ("doc", np.int32), ("pos", np.int32), ("seg", np.int8),
            ("label", np.int8)]
    FIELDS = [("data", np.float32), ("indices", np.uint16), ("nnz", np.int32),
              ("norm", np.float32), ("err_norm", np.float32), ("massive", np.bool_)] + META

    def __init__(self, sae, out, chunk=2048):
        assert sae.d_sae <= 65536
        self.sae, self.out, self.chunk = sae, Path(out), chunk
        self.out.mkdir(parents=True, exist_ok=True)
        self.b = {k: Bin(self.out / f"_{k}.bin", t) for k, t in self.FIELDS}
        self.n_rows = [0]

    @torch.no_grad()
    def add(self, x, **meta):
        """Encode rows x (n, d_model) of one document; meta = the META arrays of these rows."""
        x = x.float()
        for s in range(0, len(x), self.chunk):
            xb = x[s:s + self.chunk]
            z, rec = self.sae.infer(xb)
            m = z > 0
            self.b["indices"].add(m.nonzero(as_tuple=True)[1].cpu().numpy())
            self.b["data"].add(z[m].cpu().numpy())
            self.b["nnz"].add(m.sum(1, dtype=torch.int32).cpu().numpy())
            self.b["norm"].add(xb.norm(dim=1).cpu().numpy())
            self.b["err_norm"].add((xb - rec).norm(dim=1).cpu().numpy())
            self.b["massive"].add(massive(xb).cpu().numpy())
        for k, _ in self.META:
            self.b[k].add(meta[k])
        self.n_rows.append(self.n_rows[-1] + len(x))

    def save(self, **extra):
        """z.csr.npz (scipy.sparse.load_npz) and rows.npz (META, norms, massive, doc_offset, extra)."""
        v = {k: b.done() for k, b in self.b.items()}
        ptr = np.concatenate([[0], np.cumsum(v["nnz"], dtype=np.int64)])
        np.savez(self.out / "z.csr.npz", format=np.array(b"csr"),
                 shape=np.array((self.n_rows[-1], self.sae.d_sae), np.int64),
                 data=v["data"], indices=v["indices"], indptr=ptr)
        np.savez(self.out / "rows.npz", doc_offset=np.array(self.n_rows, np.int64), **extra,
                 **{k: v[k] for k in ("norm", "err_norm", "massive", *[k for k, _ in self.META])})
        for b in self.b.values():
            b.path.unlink()
