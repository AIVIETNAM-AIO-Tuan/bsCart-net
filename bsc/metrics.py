"""Metric phan doan - MOT dinh nghia duy nhat cho toan bo Stage 1.

VI SAO FILE NAY TON TAI
-----------------------
`surf()` da bi copy-paste thanh hai ban trong repo (merge_s6_evaluate cell 22 va
merge_s9_5fold_150ep cell 18) roi troi khac nhau. Voi mot hieu ung co 0.1mm, metric
phai co DUNG MOT dinh nghia. Khong copy ham nao trong day sang notebook - import.

PHU THUOC
---------
Chi scipy + skimage + numpy. KHONG dung SimpleITK, de:
  (a) chay duoc tren may local (SimpleITK chua cai);
  (b) doc lap voi ban SimpleITK cu => Gate 0 doi chieu HAI cai dat doc lap.
Notebook bootstrap se so ban nay voi ban SimpleITK cu tren du lieu that. Neu lech,
DUNG - khong the do hieu ung 0.1mm bang metric chua kiem chung (Gate 0).

HD95: KHAC BIET PHAI GHI RO TRONG BAI BAO
------------------------------------------
`surf()` cu gop khoang cach hai chieu vao MOT mang roi lay p95 => "HD95 pooled".
Chuan hon la max(p95(gt->pred), p95(pred->gt)). Ban pooled doc LAC QUAN hon vi mot
chieu tot co the che chieu kia. Ta cung cap CA HAI:
  hd95(..., mode="pooled")     - khop code cu, de tai lap bang da cong bo
  hd95(..., mode="max")        - dinh nghia chuan, dung cho bai bao
Mac dinh "max". Bang doi chieu Gate 0 phai dung "pooled".
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries

SPACING = (0.70, 0.3646, 0.3646)


# ------------------------------------------------------------ ha tang be mat

def surface_mask(mask: np.ndarray) -> np.ndarray:
    """Voxel bien trong cua mask (tuong duong sitk.LabelContour)."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return np.zeros_like(m)
    return find_boundaries(m, mode="inner")


def distance_to_surface(mask: np.ndarray, spacing=SPACING) -> np.ndarray:
    """Khoang cach (mm) tu MOI voxel toi be mat cua mask. +inf neu mask rong."""
    s = surface_mask(mask)
    if not s.any():
        return np.full(mask.shape, np.inf, np.float32)
    return distance_transform_edt(~s, sampling=spacing).astype(np.float32)


def surface_distances(gt: np.ndarray, pred: np.ndarray, spacing=SPACING):
    """Tra (d_gt_to_pred, d_pred_to_gt): khoang cach mm tu moi voxel be mat ben nay
    toi be mat ben kia. Day la nguyen lieu cho MOI metric be mat ben duoi.

    Tra (None, None) neu mot trong hai rong - de goi ham quyet dinh xu ly, thay vi
    lang le tra 0 (se lam bang ket qua trong dep gia tao o cac ca that bai hoan toan).
    """
    gt, pred = np.asarray(gt).astype(bool), np.asarray(pred).astype(bool)
    if not gt.any() or not pred.any():
        return None, None
    sg, sp_ = surface_mask(gt), surface_mask(pred)
    d_to_pred = distance_transform_edt(~sp_, sampling=spacing)
    d_to_gt = distance_transform_edt(~sg, sampling=spacing)
    return d_to_pred[sg].astype(np.float32), d_to_gt[sp_].astype(np.float32)


# ------------------------------------------------------------ metric the tich

def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Dice. Quy uoc: ca hai rong => 1.0 (dong y hoan hao ve "khong co gi").

    Quy uoc nay QUAN TRONG cho bin `absent`: mot ca OA nang mat sun hoan toan ma
    model cung du doan khong co sun la DUNG, khong phai NaN.
    """
    gt, pred = np.asarray(gt).astype(bool), np.asarray(pred).astype(bool)
    s = gt.sum() + pred.sum()
    if s == 0:
        return 1.0
    return float(2.0 * (gt & pred).sum() / s)


def precision_recall(gt: np.ndarray, pred: np.ndarray):
    gt, pred = np.asarray(gt).astype(bool), np.asarray(pred).astype(bool)
    tp = float((gt & pred).sum())
    p = tp / pred.sum() if pred.sum() else (1.0 if not gt.any() else 0.0)
    r = tp / gt.sum() if gt.sum() else (1.0 if not pred.any() else 0.0)
    return p, r


def volume_error(gt: np.ndarray, pred: np.ndarray, spacing=SPACING) -> dict:
    """Sai so the tich (mm^3). `signed` giu dau: duong = du doan THUA."""
    vox = float(np.prod(spacing))
    v_gt = float(np.asarray(gt).astype(bool).sum()) * vox
    v_pr = float(np.asarray(pred).astype(bool).sum()) * vox
    return {
        "vol_gt_mm3": v_gt, "vol_pred_mm3": v_pr,
        "vol_err_signed_mm3": v_pr - v_gt,
        "vol_err_abs_mm3": abs(v_pr - v_gt),
        "vol_err_rel": (v_pr - v_gt) / v_gt if v_gt > 0 else np.nan,
        "fp_vol_mm3": float((np.asarray(pred).astype(bool) & ~np.asarray(gt).astype(bool)).sum()) * vox,
        "fn_vol_mm3": float((~np.asarray(pred).astype(bool) & np.asarray(gt).astype(bool)).sum()) * vox,
    }


# ------------------------------------------------------------- metric be mat

def assd(gt: np.ndarray, pred: np.ndarray, spacing=SPACING) -> float:
    """Average Symmetric Surface Distance (mm). NaN neu mot ben rong."""
    a, b = surface_distances(gt, pred, spacing)
    if a is None:
        return float("nan")
    return float(np.concatenate([a, b]).mean())


def hd95(gt: np.ndarray, pred: np.ndarray, spacing=SPACING, mode: str = "max") -> float:
    """95th-percentile Hausdorff (mm).

    mode="max"    : max(p95(gt->pred), p95(pred->gt))  - dinh nghia chuan, dung cho bai bao
    mode="pooled" : p95 tren khoang cach hai chieu gop chung - khop `surf()` cu.
                    Doc LAC QUAN hon (mot chieu tot che chieu kia). Dung mode nay khi
                    doi chieu bang da cong bo o Gate 0.
    """
    a, b = surface_distances(gt, pred, spacing)
    if a is None:
        return float("nan")
    if mode == "pooled":
        return float(np.percentile(np.concatenate([a, b]), 95))
    if mode == "max":
        return float(max(np.percentile(a, 95), np.percentile(b, 95)))
    raise ValueError(f"mode khong hop le: {mode!r} (chon 'max' hoac 'pooled')")


def surface_dice(gt: np.ndarray, pred: np.ndarray, tol_mm: float, spacing=SPACING) -> float:
    """Surface Dice o dung sai `tol_mm` (§2.3) - MOI, chua tung co trong repo.

    Ty le voxel be mat (ca hai phia) nam trong tol_mm cua be mat kia. Day la metric
    do dung thu ma gia thuyet hua: bien dung cho, o dung sai NHO.

    tol 0.5mm va 1.0mm la hai muc plan doc yeu cau. Luu y 0.5mm ~ 1.4 voxel in-plane
    nhung chi 0.71 voxel theo z - nen o dung sai nay, sai so theo z chiem uu the.
    Do cung la ly do M0c (chan doan bac thang GT theo z) dang gia.
    """
    a, b = surface_distances(gt, pred, spacing)
    if a is None:
        return float("nan")
    both = np.concatenate([a, b])
    return float((both <= tol_mm).mean())


def boundary_mae(gt: np.ndarray, pred: np.ndarray, bone_sdf: np.ndarray,
                 spacing=SPACING) -> dict:
    """MAE bien TRONG va bien NGOAI, tach theo khoang cach toi be mat XUONG (§2.3).

    Vi sao tach: gia thuyet noi rieng ve bien NGOAI (mat khop, giap khe khop) - do la
    noi sun mon di. Bien TRONG (giap xuong) gan nhu co dinh vi no bam vao xuong. Gop
    chung hai bien lai se pha loang chinh hieu ung can do.

    Cach tach: voi moi voxel be mat, xet SDF xuong tai do. Voxel co SDF nho => gan
    xuong => bien trong. Ta chia tai TRUNG VI SDF cua be mat GT - mot nguong theo du
    lieu, khong phai hang so tuy tien.

    Tra dict: inner_mae_mm, outer_mae_mm, n_inner, n_outer.
    """
    gt, pred = np.asarray(gt).astype(bool), np.asarray(pred).astype(bool)
    if not gt.any() or not pred.any():
        return {"inner_mae_mm": np.nan, "outer_mae_mm": np.nan, "n_inner": 0, "n_outer": 0}

    sg, sp_ = surface_mask(gt), surface_mask(pred)
    d_to_pred = distance_transform_edt(~sp_, sampling=spacing)
    d_to_gt = distance_transform_edt(~sg, sampling=spacing)

    thr = float(np.median(bone_sdf[sg]))       # nguong theo du lieu, khong tuy tien
    out = {}
    for name, sel_g, sel_p in [
        ("inner", sg & (bone_sdf <= thr), sp_ & (bone_sdf <= thr)),
        ("outer", sg & (bone_sdf > thr), sp_ & (bone_sdf > thr)),
    ]:
        d = np.concatenate([d_to_pred[sel_g], d_to_gt[sel_p]])
        out[f"{name}_mae_mm"] = float(d.mean()) if d.size else np.nan
        out[f"n_{name}"] = int(sel_g.sum() + sel_p.sum())
    return out


# ------------------------------------------------------------------ tong hop

def all_metrics(gt: np.ndarray, pred: np.ndarray, spacing=SPACING,
                bone_sdf: "np.ndarray | None" = None) -> dict:
    """Bo metric day du cho MOT lop, MOT ca. Dung cho bang per-case (§7).

    Tinh khoang cach be mat MOT lan roi tai su dung - EDT la phan dat nhat
    (~30-60s/ca tren volume 23.6M voxel).
    """
    gt, pred = np.asarray(gt).astype(bool), np.asarray(pred).astype(bool)
    p, r = precision_recall(gt, pred)
    m = {"dice": dice(gt, pred), "precision": p, "recall": r}
    m.update(volume_error(gt, pred, spacing))

    a, b = surface_distances(gt, pred, spacing)
    if a is None:
        m.update({"assd_mm": np.nan, "hd95_mm": np.nan, "hd95_pooled_mm": np.nan,
                  "sdice_0.5mm": np.nan, "sdice_1.0mm": np.nan})
    else:
        both = np.concatenate([a, b])
        m["assd_mm"] = float(both.mean())
        m["hd95_mm"] = float(max(np.percentile(a, 95), np.percentile(b, 95)))
        m["hd95_pooled_mm"] = float(np.percentile(both, 95))
        m["sdice_0.5mm"] = float((both <= 0.5).mean())
        m["sdice_1.0mm"] = float((both <= 1.0).mean())

    if bone_sdf is not None:
        m.update(boundary_mae(gt, pred, bone_sdf, spacing))
    return m


# ------------------------------------------- metric hinh thai tren tia (§2.3)

def presence_f1(pres_true, pres_pred) -> dict:
    """Presence/absence F1 tren be mat khop (§2.3 morphological).

    Quy uoc: lop DUONG = "CO sun". Nhung con so dang quan tam nhat la recall cua lop
    AM (phat hien dung sun MAT) - do la thu M0 chi ra ResEnc yeu nhat (bin `absent`
    sai gap ~4 lan cac bin khac). Nen tra ca `absent_recall`.
    """
    t = np.asarray(pres_true).astype(bool)
    p = np.asarray(pres_pred).astype(bool)
    tp = float((t & p).sum()); fp = float((~t & p).sum()); fn = float((t & ~p).sum())
    tn = float((~t & ~p).sum())
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if not t.any() else 0.0)
    rec = tp / (tp + fn) if (tp + fn) else (1.0 if not p.any() else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "absent_recall": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "n_present": int(t.sum()), "n_absent": int((~t).sum()),
    }


def thickness_mae(thick_true, thick_pred, only_present: bool = False) -> dict:
    """MAE do day sun (mm) tren tung tia (§2.3 morphological).

    only_present=False (mac dinh): tinh tren MOI tia, ke ca tia absent (do day thuc = 0).
    Doan thua sun o cho khong co sun LA loi do day - bo qua se giau dung cai lop
    `absent` ma gia thuyet nham toi.
    """
    a = np.asarray(thick_true, float)
    b = np.asarray(thick_pred, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if only_present:
        ok &= a > 0
    if not ok.any():
        return {"mae_mm": np.nan, "bias_mm": np.nan, "n": 0}
    d = b[ok] - a[ok]
    return {"mae_mm": float(np.abs(d).mean()), "bias_mm": float(d.mean()),
            "n": int(ok.sum())}


# ------------------------------------------------- thong ke (§2.4)

def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                     seed: int = 0, alpha: float = 0.05) -> dict:
    """CI bootstrap cho hieu GHEP CAP theo tung ca: mean(b - a).  (§2.4)

    Ghep cap la bat buoc: cac ca khac nhau ve do kho hon nhieu so voi khac biet giua
    hai model, nen so sanh khong ghep cap se bi chim trong phuong sai giua cac ca.

    Tra dict: delta_mean, ci_low, ci_high, n_better/worse/equal, p_wilcoxon (neu co scipy).
    Bo cap co NaN (lop vang trong GT) truoc khi tinh.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return {"delta_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}

    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)

    out = {
        "n": int(d.size),
        "delta_mean": float(d.mean()),
        "ci_low": float(np.percentile(boots, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        "n_better": int((d > 0).sum()),
        "n_worse": int((d < 0).sum()),
        "n_equal": int((d == 0).sum()),
    }
    try:
        from scipy.stats import wilcoxon
        if np.any(d != 0):
            out["p_wilcoxon"] = float(wilcoxon(a, b).pvalue)
    except Exception:
        pass
    return out
