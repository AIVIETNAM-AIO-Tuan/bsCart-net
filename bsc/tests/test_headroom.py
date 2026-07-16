"""Kiem chung M0 tren phantom co HEADROOM DA BIET.

M0 la cong chan quyet dinh so phan Stage 1, nen no phai duoc kiem tren cac case ma
ta CO Y dat loi vao dung cho da biet, roi xac nhan M0 chi dung cho do.

Hai kich ban doi cuc:
  A) Loi tap trung o sun MONG  -> M0 phai bao thin+absent chiem phan lon error mass,
     va phan thuong lon  -> PROCEED
  B) Loi tap trung o sun DAY   -> M0 phai bao phan thuong nho  -> STOP
"""

from __future__ import annotations

import numpy as np

from bsc import core, headroom
from bsc.core import RayConfig

SP = core.SPACING


def _graded_shell(shape=(50, 130, 130), spacing=SP, r_mm=16.0):
    """Vo sun co do day BIEN THIEN theo goc: mong o mot phia, day o phia kia.

    Tra (bone, cart, thickness_field_dung_de_kiem). Nho vay ta biet vung nao mong.
    """
    zz, yy, xx = np.meshgrid(*[np.arange(s) * sp for s, sp in zip(shape, spacing)],
                             indexing="ij")
    c = [s * sp / 2 for s, sp in zip(shape, spacing)]
    r = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    # Do day 0.3mm -> 2.5mm theo goc quanh truc x
    ang = np.arctan2(yy - c[1], zz - c[0])
    t = 0.3 + 2.2 * (np.cos(ang) * 0.5 + 0.5)
    bone = r <= r_mm
    cart = (r > r_mm) & (r <= r_mm + t)
    return bone, cart


def test_m0_detects_prize_when_error_is_in_thin():
    """Kich ban A: loi dat vao vung MONG -> M0 phai thay phan thuong + PROCEED."""
    bone, gt = _graded_shell()
    # "Pred" = GT nhung bao mon them o vung mong (bao mon lop ngoai noi sun mong)
    cfg = RayConfig()
    _, verts, normals = core.bone_geometry(bone, SP, cfg)
    occ = core.occupancy_target(gt, verts, normals, cfg, SP)
    st = core.ray_stats(occ, cfg)

    # Xoa sun o cac node mong nhat (mo phong ResEnc bo sot sun mong)
    pred = gt.copy()
    thin_nodes = verts[st["thickness"] < 0.8]
    from scipy.spatial import cKDTree
    if len(thin_nodes):
        gt_pts = core.voxel_centers_mm(gt, SP)
        d, _ = cKDTree(thin_nodes).query(gt_pts, distance_upper_bound=2.0)
        near_thin = np.isfinite(d)
        zyx = np.argwhere(gt)[near_thin]
        pred[tuple(zyx.T)] = False

    mass = headroom.error_mass_by_thickness(gt, pred, bone, SP, cfg)
    thin = sum(mass["per_bin"][n]["frac_error_mass"] for n in ("absent", "<=0.5mm", "<=1.0mm"))
    prize = headroom.prize_counterfactual(gt, pred, bone, SP, cfg)

    assert thin > 0.5, f"loi dat o vung mong nhung M0 bao thin chi {thin:.2f} error mass"
    assert prize["d_assd_prize_mm"] > 0, "phan thuong phai duong khi loi o vung mong"


def test_m0_no_prize_when_error_is_in_thick():
    """Kich ban B: loi dat vao vung DAY -> M0 phai thay phan thuong nho -> khong PROCEED."""
    bone, gt = _graded_shell()
    cfg = RayConfig()
    _, verts, normals = core.bone_geometry(bone, SP, cfg)
    occ = core.occupancy_target(gt, verts, normals, cfg, SP)
    st = core.ray_stats(occ, cfg)

    # Xoa sun o cac node DAY nhat
    pred = gt.copy()
    thick_nodes = verts[st["thickness"] > 2.0]
    from scipy.spatial import cKDTree
    if len(thick_nodes):
        gt_pts = core.voxel_centers_mm(gt, SP)
        d, _ = cKDTree(thick_nodes).query(gt_pts, distance_upper_bound=2.0)
        zyx = np.argwhere(gt)[np.isfinite(d)]
        pred[tuple(zyx.T)] = False

    mass = headroom.error_mass_by_thickness(gt, pred, bone, SP, cfg)
    thin = sum(mass["per_bin"][n]["frac_error_mass"] for n in ("absent", "<=0.5mm", "<=1.0mm"))
    assert thin < 0.4, f"loi dat o vung day nhung M0 bao thin toi {thin:.2f} error mass"


def test_gate1_decision_logic():
    """Logic quyet dinh Gate 1 dung nguong."""
    def fake(d_assd, thin_frac, n=20):
        prize = [{"d_assd_prize_mm": d_assd} for _ in range(n)]
        mass = [{"per_bin": {nm: {"frac_error_mass": (thin_frac / 3 if nm in
                 ("absent", "<=0.5mm", "<=1.0mm") else (1 - thin_frac) / 2)}
                 for nm in core.THICKNESS_NAMES}} for _ in range(n)]
        return headroom.evaluate_gate1(prize, mass, n_boot=200)

    assert fake(0.10, 0.50)["decision"] == "PROCEED"
    assert fake(0.06, 0.50)["decision"] == "RESCOPE"
    assert fake(0.10, 0.20)["decision"] == "RESCOPE"   # phan thuong to nhung khong o vung mong
    assert fake(0.02, 0.50)["decision"] == "STOP"


def test_error_mass_bins_sum_to_total():
    """Tong error mass cac bin = tong (khong mat voxel nao trong phan ra)."""
    bone, gt = _graded_shell()
    pred = gt.copy()
    pred[25:, :, :] = False              # bo mot nua
    mass = headroom.error_mass_by_thickness(gt, pred, bone, SP)
    s = sum(b["error_mass_mm"] for b in mass["per_bin"].values())
    assert abs(s - mass["total_error_mass_mm"]) < 1e-3
    fr = sum(b["frac_error_mass"] for b in mass["per_bin"].values())
    assert abs(fr - 1.0) < 1e-6
