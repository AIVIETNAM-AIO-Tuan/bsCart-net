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

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field

import numpy as np

from . import atlas as atlas_mod
from . import core, headroom, metrics, model
from .core import SPACING, RayConfig
from .experiment import RunConfig


# ------------------------------------------- provenance & checkpoint (Step 0)
#
# VI SAO: review phuong phap chi ra bug bookkeeping - khong luu checkpoint (tron 2
# instance P2), eval jsonl chi khoa case_id (dung ket qua model CU), khong khoa git
# commit. Cac ham duoi day sua tan goc: MOI run luu checkpoint + hash + manifest, va
# eval PHAI reload tu dia (khong dung `net` con trong RAM). Xem
# `mapping_split_canonical_p2_decisions_vi.md` §4, §7.
#
# RANG BUOC (§4.3): Step 0 chi THEM bookkeeping, KHONG doi numerics. Test round-trip
# duoi day chung minh reload cho ket qua Y HET train.

def git_commit(cwd: str = ".") -> str:
    """SHA commit hien tai, hoac 'unknown' neu khong trong git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _hash_json(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(run: RunConfig, cfg: RayConfig, *, epochs: int, lr: float,
                rays_per_case: int, batch_size: int, extra: "dict | None" = None) -> str:
    """Hash MOI thu anh huong numerics cua mot run. Doi bat ky cai nao => hash khac.

    Dung de phat hien 'cung experiment_id nhung config that su khac' (vd doi epoch/lr).
    """
    payload = {
        "surface": run.surface, "domain": run.domain, "inputs": run.inputs,
        "heads": run.heads, "cls": run.cls, "seed": run.seed, "version": run.version,
        "channels": list(run.channels), "with_presence": run.with_presence,
        "prob_source": run.prob_source,
        "ray": {"k": cfg.k, "d_min": cfg.d_min, "d_max": cfg.d_max,
                "smooth_mm": cfg.smooth_mm},
        "train": {"epochs": epochs, "lr": lr, "rays_per_case": rays_per_case,
                  "batch_size": batch_size},
    }
    if extra:
        payload["extra"] = extra
    return _hash_json(payload)[:16]


def save_run(result: "RunResult", bsc_root: str, cfg: RayConfig, *, epochs: int,
             lr: float, rays_per_case: int, batch_size: int = 4096,
             atlas_hash: str = "", split_hash: str = "", dataset_revision: str = "",
             git_cwd: str = ".", extra_metrics: "dict | None" = None) -> str:
    """Luu checkpoint + manifest, tra ve `run_dir = runs/<experiment_id>/<ckpt_hash>/`.

    Duong dan CHUA ckpt_hash => checkpoint moi -> thu muc moi -> khong the reuse eval cu
    (sua bug 'stale eval'). Mo hinh luu du state_dict + in_channels + with_presence de
    load lai dung kien truc.
    """
    import torch
    run = result.run
    tmp = f"{bsc_root}/runs/_tmp_{run.experiment_id}.pt"
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    torch.save({"state_dict": result.net.state_dict(),
                "in_channels": len(run.channels),
                "with_presence": run.with_presence,
                "experiment_id": run.experiment_id}, tmp)
    ckpt_hash = _hash_file(tmp)[:16]
    run_dir = f"{bsc_root}/runs/{run.experiment_id}/{ckpt_hash}"
    os.makedirs(run_dir, exist_ok=True)
    os.replace(tmp, f"{run_dir}/model.pt")

    manifest = run.to_registry(dataset_revision=dataset_revision, extra={
        "checkpoint_sha256": ckpt_hash,
        "git_commit": git_commit(git_cwd),
        "config_hash": config_hash(run, cfg, epochs=epochs, lr=lr,
                                   rays_per_case=rays_per_case, batch_size=batch_size),
        "atlas_hash": atlas_hash, "split_manifest_hash": split_hash,
        "epochs": epochs, "lr": lr, "rays_per_case": rays_per_case,
        "batch_size": batch_size,
        "ray_k": cfg.k, "ray_d_min": cfg.d_min, "ray_d_max": cfg.d_max,
        "smooth_mm": cfg.smooth_mm,
        "val_occ_dice": result.val_occ_dice, "presence": result.presence,
        "n_train_rays": result.n_train_rays, **(extra_metrics or {})})
    with open(f"{run_dir}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return run_dir


def load_run_model(run_dir: str, device: str = "cpu"):
    """Reload model tu checkpoint tren dia. MOI eval phai qua day, khong dung `net` RAM."""
    import torch
    ck = torch.load(f"{run_dir}/model.pt", map_location=device)
    net = model.RayEncoder1D(in_channels=ck["in_channels"],
                             with_presence=ck["with_presence"])
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval()


def run_manifest(run_dir: str) -> dict:
    with open(f"{run_dir}/manifest.json", encoding="utf-8") as f:
        return json.load(f)


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


def _stratified_ray_indices(occ, cfg: RayConfig, n: int, rng) -> np.ndarray:
    """Lay mau con CAN BANG theo bin do day (Phase B, review §8).

    Uniform sampling lam tia sun MONG (hiem) it duoc dai dien => recall 37% o vung mong.
    Ham nay cap phat DEU cho moi bin do day co mat (ke ca `absent`), roi lap day phan
    con lai. Nho vay tia mong/absent duoc oversample so voi uniform.
    """
    th = core.ray_stats(occ, cfg)["thickness"]
    b = core.assign_thickness_bin(th)
    groups = [np.flatnonzero(b == k) for k in range(len(core.THICKNESS_NAMES))]
    nonempty = [g for g in groups if len(g)]
    per = max(1, n // len(nonempty))
    picked = [rng.choice(g, min(len(g), per), replace=False) for g in nonempty]
    idx = np.concatenate(picked)
    if len(idx) < n:                                   # lap day tu phan chua chon
        rest = np.setdiff1d(np.arange(len(occ)), idx, assume_unique=False)
        if len(rest):
            idx = np.concatenate([idx, rng.choice(rest, min(len(rest), n - len(idx)),
                                                  replace=False)])
    rng.shuffle(idx)
    return idx[:n]


def build_dataset(run: RunConfig, src: CaseSource, case_ids, cfg: RayConfig = RayConfig(),
                  atlas=None, rays_per_case: int = 20000, seed: int = 0,
                  direction: str = "normal", jitter_s_mm: float = 0.0,
                  jitter_theta_deg: float = 0.0, stratify: bool = False,
                  verbose: bool = False):
    """Nhieu ca -> (X, occ, presence) da ghep, co lay mau con khi TRAIN.

    `stratify=False` (mac dinh) = uniform, GIU nguyen hanh vi canonical. `stratify=True`
    = can bang theo bin do day (Phase B) - KHONG dung cho canonical.
    """
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
            rng = np.random.default_rng(seed + i)
            k = (_stratified_ray_indices(occ, cfg, rays_per_case, rng) if stratify
                 else rng.choice(len(X), rays_per_case, replace=False))
            X, occ, pres = X[k], occ[k], pres[k]
        Xs.append(X); os_.append(occ); ps.append(pres)
        if verbose:
            print(f"  {cid}: {len(X)} tia, absent {100*(pres==0).mean():.1f}%")
    if not Xs:
        raise ValueError(f"{run.experiment_id}: khong dung duoc tia nao.")
    return np.concatenate(Xs), np.concatenate(os_), np.concatenate(ps)


def micro_overfit(run: RunConfig, src: CaseSource, cid, cfg: RayConfig = RayConfig(),
                  atlas=None, *, rays: int = 1200, epochs: int = 150, lr: float = 3e-3,
                  stratify: bool = True, device: str = "cpu", seed: int = 0) -> dict:
    """Phase B step 1 (review §11): overfit 1 ca cau hinh S0-D1-H1, bao cao THIN/ABSENT rieng.

    Muc dich: xac nhan cau hinh dang HONG (S0-D1-H1) CO THE thuoc long tia. Neu khong dat
    ~0.97-0.99 => bug o input/target/kien truc/loss, KHONG phai chuyen scale. Bao cao rieng
    thin-ray recall va absent presence-acc vi do la cho §3.7 that bai.
    """
    X, occ, pres = build_dataset(run, src, [cid], cfg, atlas=atlas, rays_per_case=rays,
                                 stratify=stratify, seed=seed)
    net = model.RayEncoder1D(in_channels=len(run.channels),
                             with_presence=run.with_presence)
    hist = model.fit(net, X, occ, pres, epochs=epochs, batch_size=256, lr=lr,
                     seed=seed, device=device)
    op, pp = model.predict_rays(net, X, device=device)

    tgt = occ.astype(bool)
    thr, dice = 0.5, 0.0
    for t in np.arange(0.3, 0.81, 0.05):
        d = 2 * ((op > t) & tgt).sum() / ((op > t).sum() + tgt.sum() + 1e-8)
        if d > dice:
            dice, thr = float(d), float(t)

    th_ray = core.ray_stats(occ, cfg)["thickness"]
    thin = (th_ray > 0) & (th_ray <= 1.0)
    thin_recall = (float(((op > thr) & tgt)[thin].sum() / max(tgt[thin].sum(), 1))
                   if thin.any() else float("nan"))
    absent = pres == 0
    absent_acc = (float((pp[absent] <= 0.5).mean())
                  if (pp is not None and absent.any()) else float("nan"))
    return {"occ_dice": dice, "best_thr": thr, "thin_ray_recall": thin_recall,
            "absent_pres_acc": absent_acc, "n_rays": int(len(X)),
            "absent_frac": float(absent.mean()),
            "loss0": hist[0]["loss"], "lossN": hist[-1]["loss"], "net": net}


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
              stratify: bool = False, loss: "model.LossWeights | None" = None,
              verbose: bool = False) -> RunResult:
    """Train + danh gia tren tap val. Truc H quyet dinh co presence head hay khong.

    PHASE B: `stratify` va `loss` mac dinh = hanh vi canonical (test khoa lai).
    """
    Xa, oa, pa = build_dataset(run, src, train_ids, cfg, atlas, rays_per_case,
                               seed=0, direction=direction, stratify=stratify,
                               verbose=verbose)
    Xb, ob, pb = build_dataset(run, src, val_ids, cfg, atlas, rays_per_case,
                               seed=1000, direction=direction)   # val KHONG stratify

    net = model.RayEncoder1D(in_channels=len(run.channels),
                             with_presence=run.with_presence)
    hist = model.fit(net, Xa, oa, pa, epochs=epochs, batch_size=batch_size, lr=lr,
                     seed=run.seed, device=device, w=loss or model.LossWeights())
    op, pp = model.predict_rays(net, Xb, device=device)
    dice = float(2 * ((op > 0.5) & ob.astype(bool)).sum()
                 / ((op > 0.5).sum() + ob.sum() + 1e-8))
    pres = metrics.presence_f1(pb.astype(bool), pp > 0.5) if pp is not None else None
    return RunResult(run, net, dice, pres, int(len(Xa)), hist)


def evaluate_case(run: RunConfig, net, src: CaseSource, cid,
                  cfg: RayConfig = RayConfig(), atlas=None, device: str = "cpu",
                  presence_gate: str = "hard", presence_thr: float = 0.5) -> dict:
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
                                   presence_prob=pp, presence_thr=presence_thr,
                                   presence_gate=presence_gate)

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


# ------------------------------------------ Mapping audit (Phase A step 8)
#
# Review §6 phat hien 1123 voxel sun GT bi gan vao bin `absent` -> phep gan
# nearest-bone-node (Euclidean) khong phai anh xa giai phau chinh xac. Decision
# (`mapping_split_canonical_p2_decisions_vi.md` §2): so Mapping A (nearest node) voi
# Mapping B (nearest sampled-ray proxy) tren 10 ca de xem ket luan §3.7 co nhay voi
# cach mapping khong. B KHONG phai full provenance (splat_rays dung bincount, mat source
# ray) - do la Mapping C danh cho final Gate.

def _nearest_ray_thickness(query_mm, verts, dirs, thick_node, cfg: RayConfig,
                           max_dist_mm: float, depth_stride: int = 2):
    """Mapping B: thickness cua RAY co sample gan query nhat. NaN neu xa hon max_dist.

    Dung ca vi tri node, huong ray va depth (gan tinh than 'ray ownership' hon nearest
    node). `depth_stride` giam so diem cay KDTree (K=64 -> 32) cho nhe bo nho.
    """
    from scipy.spatial import cKDTree
    depths = cfg.depths[::depth_stride]
    pts = (verts[:, None, :] + depths[None, :, None] * dirs[:, None, :]).reshape(-1, 3)
    ray_of = np.repeat(np.arange(len(verts)), len(depths))
    d, i = cKDTree(pts.astype(np.float64)).query(np.asarray(query_mm, np.float64))
    th = thick_node[ray_of[i]].astype(np.float32)
    th[d > max_dist_mm] = np.nan
    return th, d.astype(np.float32)


def default_max_dist_mm(cfg: RayConfig, spacing=SPACING) -> float:
    """Nua duong cheo voxel + nua buoc ray (§2.5). Query xa hon => `unassigned`."""
    return 0.5 * float(np.linalg.norm(spacing)) + 0.5 * cfg.step


def mapping_audit_case(gt_cart, pred_cart, bone_mask, spacing, cfg: RayConfig,
                       max_dist_mm: "float | None" = None,
                       thin_bins=("absent", "<=0.5mm", "<=1.0mm")) -> "dict | None":
    """So lỗi vùng mỏng + mis-binning giua Mapping A (node) va B (ray) cho MOT ca."""
    from scipy.ndimage import distance_transform_edt
    if max_dist_mm is None:
        max_dist_mm = default_max_dist_mm(cfg, spacing)

    verts, normals, thick_node = headroom.gt_thickness_per_node(bone_mask, gt_cart, spacing, cfg)
    sg, sp_ = metrics.surface_mask(gt_cart), metrics.surface_mask(pred_cart)
    if not sg.any() or not sp_.any():
        return None
    dist_gt = distance_transform_edt(~sp_, sampling=spacing)[sg]   # GT surf -> PRED
    dist_pr = distance_transform_edt(~sg, sampling=spacing)[sp_]   # PRED surf -> GT
    gt_pts = core.voxel_centers_mm(sg, spacing)
    pr_pts = core.voxel_centers_mm(sp_, spacing)
    thin_idx = [core.THICKNESS_NAMES.index(b) for b in thin_bins]

    def thin_err(th_gt, th_pr):
        d = np.concatenate([dist_gt, dist_pr])
        t = np.concatenate([th_gt, th_pr])
        ok = np.isfinite(d) & np.isfinite(t)
        sel = np.isin(core.assign_thickness_bin(t[ok]), thin_idx)
        dd = d[ok][sel]
        return (float(dd.mean()) if dd.size else np.nan), int(sel.sum())

    # Mapping A - nearest node
    errA, nA = thin_err(core.nearest_surface_value(verts, thick_node, gt_pts),
                        core.nearest_surface_value(verts, thick_node, pr_pts))
    # Mapping B - nearest sampled-ray
    tgB, dgB = _nearest_ray_thickness(gt_pts, verts, normals, thick_node, cfg, max_dist_mm)
    tpB, dpB = _nearest_ray_thickness(pr_pts, verts, normals, thick_node, cfg, max_dist_mm)
    errB, nB = thin_err(tgB, tpB)

    # Voxel sun GT bi gan vao bin `absent` (red flag §6) duoi moi mapping
    gt_vox = core.voxel_centers_mm(gt_cart, spacing)
    absent = core.THICKNESS_NAMES.index("absent")
    binA = core.assign_thickness_bin(core.nearest_surface_value(verts, thick_node, gt_vox))
    tvB, _ = _nearest_ray_thickness(gt_vox, verts, normals, thick_node, cfg, max_dist_mm)
    binB = core.assign_thickness_bin(np.nan_to_num(tvB, nan=1.0))   # unassigned -> khong absent

    dB = np.concatenate([dgB, dpB])
    return {
        "thin_err_A_mm": errA, "thin_err_B_mm": errB,
        "n_thin_surf_A": nA, "n_thin_surf_B": nB,
        "gt_vox_absent_A": int((binA == absent).sum()),
        "gt_vox_absent_B": int((binB == absent).sum()),
        "n_gt_vox": int(gt_cart.sum()),
        "unassigned_frac_B": float(np.mean(~np.isfinite(np.concatenate([tgB, tpB])))),
        "distB_median_mm": float(np.median(dB)),
        "distB_p95_mm": float(np.percentile(dB, 95)),
        "max_dist_mm": float(max_dist_mm),
    }


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
