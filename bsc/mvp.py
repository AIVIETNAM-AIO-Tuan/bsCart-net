"""Dieu phoi MVP Stage 1 - toan bo logic ma notebook can, o dang TEST DUOC.

VI SAO FILE NAY TON TAI
-----------------------
Truoc do `build_for` / `train_eval` / `domain_for` duoc viet THANG trong notebook. Chung
khong bao gio duoc chay truoc khi nguoi dung chay tren Colab, nen moi lan co bug la mot
vong "push -> tai lai notebook -> chay -> vo -> va". Ba bug da di qua duong do:
core.SPACING hardcode, atlas grid index tran, tham chieu bien chua ton tai.

Nguyen tac tu day: notebook chi duoc chua CAU HINH va LOI GOI. Moi logic o day, co test
chay bang phantom (`tests/test_mvp.py`) khong can nibabel/Colab.

Truc cau hinh S/D/I/H xem `bsc/experiment.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import atlas as atlas_mod
from . import core, headroom, metrics, model
from .core import SPACING, RayConfig
from .experiment import RunConfig


# ------------------------------------------------------------ nguon du lieu

class CaseSource:
    """Giao dien doc du lieu mot ca. Tach ra de test bang phantom.

    Ban that (`NiftiCaseSource`) can nibabel; ban phantom trong test thi khong. Nho vay
    toan bo pipeline duoi day chay duoc tren may local.
    """

    def spacing(self, cid) -> tuple: raise NotImplementedError
    def mri(self, cid): raise NotImplementedError
    def bone_gt(self, cid): raise NotImplementedError
    def cart_gt(self, cid): raise NotImplementedError
    def bone_pred(self, cid): raise NotImplementedError      # cho S2
    def prob(self, cid): raise NotImplementedError           # cho I3
    def baseline_cart(self, cid): raise NotImplementedError  # baseline de so sanh
    def has_pred(self, cid) -> bool: return True
    def has_prob(self, cid) -> bool: return True


class NiftiCaseSource(CaseSource):
    """Doc tu cay thu muc nnUNet. Chi dung tren Colab (can nibabel)."""

    def __init__(self, raw: str, cls: str, cart_label: int, bone_label: int,
                 baseline_path_fn, prob_dir: "str | None" = None):
        self.raw, self.cls = raw, cls
        self.cart_label, self.bone_label = cart_label, bone_label
        self._baseline_path = baseline_path_fn
        self.prob_dir = prob_dir
        self._cache = {}

    def _lab(self, cid):
        if cid not in self._cache:
            from . import io_utils
            self._cache[cid] = io_utils.load_nii(f"{self.raw}/labelsTr/{cid}.nii.gz")
            if len(self._cache) > 4:                 # gioi han bo nho
                self._cache.pop(next(iter(self._cache)))
        return self._cache[cid]

    def spacing(self, cid):  return self._lab(cid)[1]
    def bone_gt(self, cid):  return self._lab(cid)[0] == self.bone_label
    def cart_gt(self, cid):  return self._lab(cid)[0] == self.cart_label

    def mri(self, cid):
        from . import io_utils
        return io_utils.load_nii(f"{self.raw}/imagesTr/{cid}_0000.nii.gz")[0]

    def bone_pred(self, cid):
        from . import io_utils
        return io_utils.load_nii(self._baseline_path(cid))[0] == self.bone_label

    def baseline_cart(self, cid):
        from . import io_utils
        return io_utils.load_nii(self._baseline_path(cid))[0] == self.cart_label

    def prob(self, cid):
        from . import io_utils
        return io_utils.load_nii(f"{self.prob_dir}/{self.cls}/{cid}.nii.gz")[0]

    def has_pred(self, cid):
        return self._baseline_path(cid) is not None

    def has_prob(self, cid):
        import os
        return bool(self.prob_dir) and os.path.exists(
            f"{self.prob_dir}/{self.cls}/{cid}.nii.gz")


# ------------------------------------------------------- dung dataline mot run

def _bone_for(run: RunConfig, src: CaseSource, cid):
    """S0/S1 dung xuong GT; S2 dung xuong DU DOAN."""
    return src.bone_pred(cid) if run.surface == "S2" else src.bone_gt(cid)


def domain_for(run: RunConfig, verts, occ, cfg: RayConfig,
               atlas: "atlas_mod.ArticularAtlas | None"):
    """D0 = oracle (tu sun GT) | D1 = atlas theo fold."""
    if run.domain == "D0":
        return core.oracle_domain(verts, occ, cfg)
    if atlas is None:
        raise ValueError(f"{run.experiment_id}: domain D1 can `atlas`, nhung nhan None.")
    return atlas_mod.atlas_domain(verts, atlas, thr=0.2)


def build_case(run: RunConfig, src: CaseSource, cid, cfg: RayConfig = RayConfig(),
               atlas=None, direction: str = "normal",
               jitter_s_mm: float = 0.0, jitter_theta_deg: float = 0.0, seed: int = 0):
    """Mot ca -> (X, occ, presence, verts, dirs). Spacing lay THEO CA, khong mac dinh.

    S1 (jitter) duoc lay tu chinh `run.surface`, khong phai tham so rieng - de cau hinh
    va hanh vi khong the lech nhau.
    """
    sp = src.spacing(cid)
    bone, cart = _bone_for(run, src, cid), src.cart_gt(cid)
    if not bone.any() or not cart.any():
        return None

    if run.surface == "S1" and jitter_s_mm == 0.0 and jitter_theta_deg == 0.0:
        jitter_s_mm, jitter_theta_deg = 0.5, 10.0     # muc mac dinh cua S1

    # Domain phai tinh tren hinh hoc CHUA jitter/doi huong: no la vung giai phau,
    # khong phu thuoc cach ta phong tia.
    _, verts0, dirs0 = core.bone_geometry(bone, sp, cfg)
    occ0 = core.occupancy_target(cart, verts0, dirs0, cfg, sp)
    dom = domain_for(run, verts0, occ0, cfg, atlas)

    extra = {"prob": src.prob(cid)} if "prob" in run.channels else None
    return model.build_case_rays(
        src.mri(cid), bone, cart, sp, cfg, domain=dom, channels=run.channels,
        extra_vols=extra, direction=direction, jitter_s_mm=jitter_s_mm,
        jitter_theta_deg=jitter_theta_deg, seed=seed)


def build_dataset(run: RunConfig, src: CaseSource, case_ids, cfg: RayConfig = RayConfig(),
                  atlas=None, rays_per_case: int = 20000, seed: int = 0,
                  direction: str = "normal", jitter_s_mm: float = 0.0,
                  jitter_theta_deg: float = 0.0, verbose: bool = False):
    """Nhieu ca -> (X, occ, presence) da ghep, co lay mau con khi TRAIN."""
    Xs, os_, ps = [], [], []
    for i, cid in enumerate(case_ids):
        out = build_case(run, src, cid, cfg, atlas, direction, jitter_s_mm,
                         jitter_theta_deg, seed + i)
        if out is None:
            continue
        X, occ, pres, _, _ = out
        if len(X) == 0:
            continue
        if rays_per_case and len(X) > rays_per_case:
            k = np.random.default_rng(seed + i).choice(len(X), rays_per_case, replace=False)
            X, occ, pres = X[k], occ[k], pres[k]
        Xs.append(X); os_.append(occ); ps.append(pres)
        if verbose:
            print(f"  {cid}: {len(X)} tia, absent {100*(pres==0).mean():.1f}%")
    if not Xs:
        raise ValueError(f"{run.experiment_id}: khong dung duoc tia nao.")
    return np.concatenate(Xs), np.concatenate(os_), np.concatenate(ps)


# ------------------------------------------------------- QC hinh hoc (§6 b3/5/6)

#: Khoa cua ban ghi QC. LA MOT HOP DONG BEN VUNG - checkpoint tren Drive dung ten nay.
#: DOI TEN = pha resume cua nguoi dung (da xay ra mot lan: KeyError 'm3'). Neu buoc phai
#: doi, phai viet ham di tru, khong duoc doi lang le.
QC_KEYS = ("m3_normals_ok", "m4_single_interval", "m2_dice", "m2_assd_mm",
           "frame_degenerate")


def geometry_qc_case(src: CaseSource, cid, cls_name: str, bone_mask, cart_mask,
                     cfg: RayConfig = RayConfig()) -> dict:
    """M3 + M4 + M2 cho MOT (ca, lop). Spacing lay theo ca."""
    sp = src.spacing(cid)
    sdf, verts, normals = core.bone_geometry(bone_mask, sp, cfg)
    occ = core.occupancy_target(cart_mask, verts, normals, cfg, sp)
    rec = core.splat_rays(occ.astype(np.float32), verts, normals, cart_mask.shape, cfg, sp)
    pr = rec > 0.5
    return {
        "case": cid, "cls": cls_name,
        "m3_normals_ok": float(core.check_normals(sdf, verts, normals, sp)[0]),
        "m4_single_interval": float(core.single_interval_ratio(occ)),
        "m2_dice": float(2 * (pr & cart_mask).sum() / (pr.sum() + cart_mask.sum() + 1e-8)),
        "m2_assd_mm": float(metrics.assd(cart_mask, pr, sp)),
        "frame_degenerate": bool(atlas_mod.fit_frame(verts).is_degenerate),
    }


def summarize_geometry_qc(rows, cls_name: str) -> "dict | None":
    """Trung binh cac chi so QC cho mot lop. Tra None neu khong co ban ghi hop le.

    Ban ghi thieu khoa (checkpoint tu phien ban cu) bi BO QUA co bao, thay vi lam vo
    ca cell bang KeyError.
    """
    r = [x for x in rows if x.get("cls") == cls_name]
    good = [x for x in r if all(k in x for k in QC_KEYS)]
    if len(good) < len(r):
        print(f"  canh bao: bo qua {len(r)-len(good)}/{len(r)} ban ghi {cls_name} "
              f"thieu khoa (checkpoint tu ban cu?)")
    if not good:
        return None
    out = {k: float(np.mean([x[k] for x in good])) for k in QC_KEYS}
    out["n"] = len(good)
    return out


def geometry_qc_gate(summary: dict, m3_min: float = 0.995, m4_min: float = 0.90,
                     m2_assd_max: float = 0.1) -> dict:
    """Cong §3.5 cho QC hinh hoc. Tra dict cac tieu chi + `pass`."""
    g = {
        "m3_pass": summary["m3_normals_ok"] >= m3_min,
        "m4_pass": summary["m4_single_interval"] >= m4_min,
        "m2_pass": summary["m2_assd_mm"] <= m2_assd_max,
        "frame_ok": summary["frame_degenerate"] == 0.0,
    }
    g["pass"] = all(g.values())
    return g


# -------------------------------------------------------------- train + danh gia

@dataclass
class RunResult:
    run: RunConfig
    net: object
    val_occ_dice: float
    presence: "dict | None"
    n_train_rays: int
    history: list = field(default_factory=list)


def train_run(run: RunConfig, src: CaseSource, train_ids, val_ids,
              cfg: RayConfig = RayConfig(), atlas=None, rays_per_case: int = 20000,
              epochs: int = 30, lr: float = 3e-4, batch_size: int = 4096,
              direction: str = "normal", device: str = "cpu",
              verbose: bool = False) -> RunResult:
    """Train + danh gia tren tap val. Truc H quyet dinh co presence head hay khong."""
    Xa, oa, pa = build_dataset(run, src, train_ids, cfg, atlas, rays_per_case,
                               seed=0, direction=direction, verbose=verbose)
    Xb, ob, pb = build_dataset(run, src, val_ids, cfg, atlas, rays_per_case,
                               seed=1000, direction=direction)

    net = model.RayEncoder1D(in_channels=len(run.channels),
                             with_presence=run.with_presence)
    hist = model.fit(net, Xa, oa, pa, epochs=epochs, batch_size=batch_size, lr=lr,
                     seed=run.seed, device=device)
    op, pp = model.predict_rays(net, Xb, device=device)
    dice = float(2 * ((op > 0.5) & ob.astype(bool)).sum()
                 / ((op > 0.5).sum() + ob.sum() + 1e-8))
    pres = metrics.presence_f1(pb.astype(bool), pp > 0.5) if pp is not None else None
    return RunResult(run, net, dice, pres, int(len(Xa)), hist)


def evaluate_case(run: RunConfig, net, src: CaseSource, cid,
                  cfg: RayConfig = RayConfig(), atlas=None, device: str = "cpu") -> dict:
    """Danh gia MOT ca o MAU SO VUNG MONG (§3.7), so voi baseline.

    Dung FULL marching-cubes density khi inference - KHONG lay mau con (core.py: o mat
    do thua h=1.0mm round-trip mat 20% voxel sun).
    """
    sp = src.spacing(cid)
    bone, cart = _bone_for(run, src, cid), src.cart_gt(cid)
    out = build_case(run, src, cid, cfg, atlas, seed=0)
    if out is None:
        return {}
    X, occ, pres, verts, dirs = out
    op, pp = model.predict_rays(net, X, device=device)
    vol = model.reconstruct_volume(op, verts, dirs, cart.shape, cfg, sp,
                                   presence_prob=pp)

    tf = headroom.gt_thickness_per_node(bone, cart, sp, cfg)
    row = {
        "case": cid,
        "thin_ray": headroom.thin_region_boundary_error(cart, vol > 0.5, bone, sp, cfg, tf),
    }
    if src.has_pred(cid):
        row["thin_baseline"] = headroom.thin_region_boundary_error(
            cart, src.baseline_cart(cid), bone, sp, cfg, tf)
    if pp is not None:
        row["presence"] = metrics.presence_f1(pres.astype(bool), pp > 0.5)
    return row


def summarize_gate37(rows) -> dict:
    """Tong hop cong §3.7 tu cac ban ghi per-case.

    MAU SO LA VUNG MONG, khong phai ASSD tong (M0 §6: muc tieu quy ve ASSD tong chi
    0.006mm, nho ngang sai khac giua hai implementation metric).
    """
    b = np.array([r["thin_baseline"]["thin_mean_err_mm"]
                  for r in rows if "thin_baseline" in r], float)
    m = np.array([r["thin_ray"]["thin_mean_err_mm"]
                  for r in rows if "thin_baseline" in r], float)
    ok = np.isfinite(b) & np.isfinite(m)
    b, m = b[ok], m[ok]
    if b.size == 0:
        return {"n": 0}
    rel = (b - m) / b
    boot = metrics.paired_bootstrap(m, b)          # b - m > 0 = ray tot hon
    out = {
        "n": int(b.size),
        "thin_err_baseline_mm": float(b.mean()),
        "thin_err_ray_mm": float(m.mean()),
        "rel_improve": float(rel.mean()),
        "abs_improve_mm": float(boot["delta_mean"]),
        "ci": (float(boot["ci_low"]), float(boot["ci_high"])),
        "n_better": int(boot["n_better"]),
        "pass_10pct": bool(rel.mean() >= 0.10),
        "ci_low_positive": bool(boot["ci_low"] > 0),
    }
    pres = [r["presence"] for r in rows if "presence" in r]
    if pres:
        out["presence_f1"] = float(np.mean([p["f1"] for p in pres]))
        out["absent_recall"] = float(np.nanmean([p["absent_recall"] for p in pres]))
    return out
