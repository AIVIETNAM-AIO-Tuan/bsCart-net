"""Danh gia downstream CLS M3T + biomarker (S7/S8) - MOT dinh nghia dung chung cho hai notebook.

Ke hoach: `docs/ke_hoach_m3t_cls.md` muc 5-6. Giao thuc da dang ky: `configs/m3t_s9.json["evaluation"]`.

- Feature set moi (ten co dinh), luat chon dac trung (`__nosel` khong LASSO), mat na nhanh anh (radiomics VA CLS la
  nhanh anh, khong thi CLS lang le roi vao nhanh bio cua I_mlp_fusion), ho dac trung.
- Giao ca theo npz: CLS v2 chi co cho 1.221/1.229 ca phat trien (luat `npz_intersection` da dang ky) -> MOI feature
  set chay tren cung giao do; split/fold GHIM tren 1.229 ca roi LOC, khong chia lai (`subset_folds`).
- Head M3T dong bang (argmax logit) lam nhanh so sanh `m3t_head`.
- Phep so dang ky truoc: `registered_pairs` + `compare_pair` (bootstrap theo subject x seed + bon muc).
- `parallel_map`: chay fold song song ma ket qua KHONG phu thuoc so worker.
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import m3t as M3T
from . import ordinal as ORD

NOSEL = "__nosel"
HEAD = "m3t_head"
RAD_PREFIX = "rad_"


# ============================================================ CLS

def load_cls(csv_path, meta_path, epoch=None, known_leaky=()):
    """Doc bang CLS cua S9 (m3t_cls_v2*.csv) va kiem provenance -> DataFrame case_id + m3t_* + m3tlogit_*.

    `epoch=None` = checkpoint chinh (`main_epoch` cua meta). Kiem: case_id duy nhat, du CLS_DIM chieu, khong NaN,
    khong chieu hang, MOT weights_hash dung bang meta cua epoch do va KHONG thuoc trong so ro ri.
    """
    import pandas as pd
    meta = json.load(open(meta_path, encoding="utf-8"))
    ep = int(meta["main_epoch"] if epoch is None else epoch)
    want = meta["weights_hash"][str(ep)]
    df = pd.read_csv(csv_path)
    feat = [c for c in df.columns if M3T.is_cls_col(c)]
    logit = [c for c in df.columns if M3T.is_logit_col(c)]
    if len(feat) != M3T.CLS_DIM:
        raise ValueError(f"{csv_path}: {len(feat)} cot CLS, can {M3T.CLS_DIM}")
    if not logit:
        raise ValueError(f"{csv_path}: thieu cot logit {M3T.LOGIT_PREFIX}*")
    if not df["case_id"].is_unique:
        raise ValueError(f"{csv_path}: case_id trung")
    x = df[feat].to_numpy(np.float64)
    if not np.isfinite(x).all() or not np.isfinite(df[logit].to_numpy(np.float64)).all():
        raise ValueError(f"{csv_path}: CLS/logit co NaN/inf")
    if len(df) > 1 and (x.std(axis=0) == 0).any():
        raise ValueError(f"{csv_path}: co chieu CLS hang so")
    hashes = set(df[f"{M3T.PROV_PREFIX}weights_hash"].astype(str))
    if hashes != {want}:
        raise ValueError(f"{csv_path}: weights_hash {sorted(hashes)} != meta epoch {ep} ({want})")
    leaky = set(M3T.LEAKY_HASHES) | set(meta.get("known_leaky", [])) | set(known_leaky)
    if want in leaky:
        raise ValueError(f"{csv_path}: trong so RO RI ({want[:12]}) - khong duoc dung lam dac trung")
    out = df[["case_id"] + feat + logit].copy()
    out["case_id"] = out["case_id"].astype(str)
    out.attrs.update(weights_hash=want, epoch=ep)
    return out


def attach_cls(df, cls, expected_n=None):
    """Ghep CLS vao bang ca (`validate="one_to_one"`, giu thu tu dong) -> (bang da ghep, mask co CLS, ca thieu).

    Khong impute: ca thieu CLS bi loai o buoc giao (mask), khong bao gio lap median. `expected_n` = so ca giao bat
    buoc (vd 1221 theo S9 muc 13) - khac thi dung.
    """
    feat = [c for c in cls.columns if M3T.is_cls_col(c)]
    clash = [c for c in cls.columns if c != "case_id" and c in df.columns]
    if clash:
        raise ValueError(f"bang ca da co cot {clash[:3]} trung ten voi bang CLS")
    left = df.assign(case_id=df["case_id"].astype(str))
    out = left.merge(cls, on="case_id", how="left", validate="one_to_one")
    if len(out) != len(df):
        raise AssertionError("merge doi so dong")
    has = out[feat].notna().all(axis=1).to_numpy()
    part = out[feat].notna().any(axis=1).to_numpy() & ~has
    if part.any():
        raise ValueError(f"{int(part.sum())} ca co CLS thieu mot phan")
    if expected_n is not None and int(has.sum()) != int(expected_n):
        raise AssertionError(f"giao co CLS = {int(has.sum())} ca, can {expected_n}")
    return out, has, out.loc[~has, "case_id"].tolist()


def head_pred(cls, classes):
    """Du doan cua head M3T dong bang (argmax logit) -> chi so lop theo `classes` (phai la 0..n_logit-1)."""
    logit = [c for c in cls.columns if M3T.is_logit_col(c)]
    if list(classes) != list(range(len(logit))):
        raise ValueError(f"lop cohort {list(classes)} khong khop {len(logit)} logit cua head M3T")
    return cls[logit].to_numpy(np.float64).argmax(axis=1).astype(np.int64)


# ============================================================ feature set

def is_image_col(c: str) -> bool:
    """Nhanh ANH: radiomics (`rad_`) va CLS M3T (`m3t_000`...). Logit/provenance khong phai dac trung."""
    return c.startswith(RAD_PREFIX) or M3T.is_cls_col(c)


def bio_mask(cols) -> np.ndarray:
    """True = nhanh bio (hinh hoc legacy + be mat S6), False = nhanh anh. Dung cho I_mlp_fusion."""
    return np.array([not is_image_col(c) for c in cols], bool)


def family(c: str, surface_prefixes) -> str:
    if c.startswith(RAD_PREFIX):
        return "radiomics"
    if M3T.is_cls_col(c):
        return "m3t"
    return "S6_surface" if c.startswith(tuple(surface_prefixes)) else "legacy"


def m3t_feature_sets(legacy, surface, rad, cls) -> dict:
    """Feature set moi cua ke hoach muc 5 (ten co dinh). `rad` rong -> bo cac set co radiomics.

    Ba ban `__nosel` (khong loc tuong quan / LASSO) cho cac set CLS khong co radiomics; cap chinh la
    `m3t_only__nosel` (128 cot) vs `s6_all_plus_m3t__nosel` (198 = 15 + 55 + 128).
    """
    legacy, surface, rad, cls = (list(v) for v in (legacy, surface, rad, cls))
    if not cls or not all(M3T.is_cls_col(c) for c in cls):
        raise ValueError("cls phai la danh sach cot m3t_XXX")
    fs = {}
    if rad:
        fs["rad_only"] = rad
    fs["m3t_only"] = cls
    fs["legacy_plus_m3t"] = legacy + cls
    fs["s6_all_plus_m3t"] = legacy + surface + cls
    if rad:
        fs["s6_all_plus_radiomics_plus_m3t"] = legacy + surface + rad + cls
    for name in ("m3t_only", "legacy_plus_m3t", "s6_all_plus_m3t"):
        fs[name + NOSEL] = list(fs[name])
    for name, cols in fs.items():
        if len(set(cols)) != len(cols):
            raise ValueError(f"{name}: cot trung")
    return fs


def needs_selection(fs_name: str, n_cols: int, select_above: int = 120) -> bool:
    """Luat chon dac trung trong fold: nhieu hon `select_above` cot VA khong phai ban `__nosel`."""
    return n_cols > select_above and not fs_name.endswith(NOSEL)


# ============================================================ fold

def subset_folds(folds, keep):
    """Loc fold GHIM tren toan cohort ve cac ca `keep` - KHONG chia lai, khong doi thanh vien.

    `folds`: [(seed, fold, tr, te)] chi so tren bang day du. Tra cung cau truc, chi so tren bang da loc (thu tu
    giu nguyen). Moi seed: moi ca giu lai nam trong dung MOT fold test.
    """
    keep = np.asarray(keep, bool)
    pos = np.full(len(keep), -1, np.int64)
    pos[keep] = np.arange(int(keep.sum()))
    out = []
    for s, f, tr, te in folds:
        tr, te = np.asarray(tr), np.asarray(te)
        out.append((s, f, pos[tr[keep[tr]]], pos[te[keep[te]]]))
    for s in sorted({s for s, *_ in out}):
        cnt = np.zeros(int(keep.sum()), np.int64)
        for s2, _, _, te in out:
            if s2 == s:
                np.add.at(cnt, te, 1)
        if not (cnt == 1).all():
            raise AssertionError(f"seed {s}: ca khong nam trong dung mot fold test sau khi loc")
    return out


# ============================================================ phep so dang ky truoc

def registered_pairs(eval_cfg) -> list:
    """[{a, b, question, primary}] theo thu tu dang ky: cap chinh truoc, roi `secondary`."""
    p = eval_cfg["primary"]
    pairs = [dict(a=p["a"], b=p["b"], question="chinh: biomarker co cai thien nhanh anh M3T", primary=True)]
    pairs += [dict(a=s["a"], b=s["b"], question=s["question"], primary=False) for s in eval_cfg["secondary"]]
    return pairs


def branch_metrics(y, P, n_classes: int) -> dict:
    """Metric mot nhanh, trung binh qua seed (P [N] hoac [N,S]): qwk, mae, off_by_2, recall/precision lop cuoi."""
    y = np.asarray(y, np.int64)
    P = np.asarray(P, np.int64).reshape(len(y), -1)
    rows = []
    for s in range(P.shape[1]):
        prf = ORD.per_class_prf(y, P[:, s], n_classes)
        rows.append(dict(qwk=ORD.qwk(y, P[:, s], n_classes), mae=ORD.mae(y, P[:, s]),
                         off_by_2=ORD.off_by_rate(y, P[:, s], 2), acc=float((P[:, s] == y).mean()),
                         recall_last=float(prf["recall"][-1]), precision_last=float(prf["precision"][-1])))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def compare_pair(y, pa, pb, groups, n_classes: int, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05,
                 min_delta: float = 0.02) -> dict:
    """Mot phep so b - a theo giao thuc da dang ky: Delta QWK (trung binh qua seed), CI bootstrap theo subject,
    bon muc (`ordinal.classify_delta`), hieu tung seed, metric phu cua tung nhanh."""
    res = ORD.bootstrap_delta(y, pa, pb, "qwk", n_classes, n_boot=n_boot, seed=seed, alpha=alpha, groups=groups)
    lvl = ORD.classify_delta(res, min_delta=min_delta)
    ma, mb = branch_metrics(y, pa, n_classes), branch_metrics(y, pb, n_classes)
    return dict(delta=res["delta"], ci_low=res["ci_low"], ci_high=res["ci_high"], n_boot=res["n_boot"],
                level=lvl["level"], label=lvl["label"], note=lvl["note"], min_delta=float(min_delta),
                delta_per_seed=ORD.seed_deltas(y, pa, pb, "qwk", n_classes).round(6).tolist(),
                metrics_a=ma, metrics_b=mb, d_metrics={k: mb[k] - ma[k] for k in ma})


def stack_seeds(preds: dict, seeds) -> np.ndarray:
    """{seed: [N]} -> [N, S] theo thu tu `seeds`; du doan thieu (-1) -> loi."""
    P = np.stack([np.asarray(preds[s], np.int64) for s in seeds], axis=1)
    if (P < 0).any():
        raise ValueError(f"{int((P < 0).sum())} du doan OOF thieu")
    return P


# ============================================================ chay song song

def _single_thread_call(fn, args):
    import torch
    torch.set_num_threads(1)
    return fn(*args)


def parallel_map(fn, tasks, n_jobs: int = 1) -> list:
    """[fn(*t) for t in tasks], tuy chon song song bang joblib/loky - ket qua KHONG phu thuoc `n_jobs`.

    torch chay 1 luong o CA HAI che do (MLP full-batch cho cung so hoc); XGB/LASSO trong notebook co so luong co
    dinh (n_jobs=2 / 1). Worker loky la tien trinh moi: PYTHONPATH duoc them thu muc chua `bsc` de import duoc.
    `fn` dinh nghia trong notebook duoc cloudpickle theo gia tri; mang numpy lon nen truyen qua `tasks`
    (joblib memmap) thay vi de trong closure.
    """
    tasks = list(tasks)
    if n_jobs is None or n_jobs <= 1 or len(tasks) <= 1:
        return [_single_thread_call(fn, t) for t in tasks]
    from joblib import Parallel, delayed
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if root not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([root] + paths)
    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(_single_thread_call)(fn, t) for t in tasks)
