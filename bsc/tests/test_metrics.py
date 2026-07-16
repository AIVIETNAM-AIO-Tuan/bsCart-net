"""Kiem chung metric bang hinh hoc CO DAP AN GIAI TICH.

Vi sao: bsc/metrics.py la thu se phan xu moi cong chan. Neu no sai, ca du an do
theo. Nen no phai duoc kiem bang cac truong hop ma ta BIET TRUOC dap an dung
(dich mask di dung d mm => ASSD phai ra dung d), khong phai bang cach so voi
chinh no.

Gate 0 con mot lop kiem nua tren DU LIEU THAT: doi chieu voi bang da cong bo
(femoral_cart Dice 0.891, ASSD 0.21) trong +-0.005. Xem notebook bootstrap.
"""

from __future__ import annotations

import numpy as np
import pytest

from bsc import metrics as M

SP = M.SPACING
ISO = (1.0, 1.0, 1.0)


def _cube(shape=(40, 40, 40), lo=10, hi=30, shift=(0, 0, 0)):
    m = np.zeros(shape, bool)
    sl = tuple(slice(lo + s, hi + s) for s in shift)
    m[sl] = True
    return m


# ------------------------------------------------------------------ the tich

def test_dice_identical_and_disjoint():
    a = _cube()
    assert M.dice(a, a) == pytest.approx(1.0)
    assert M.dice(a, ~a) == pytest.approx(0.0)


def test_dice_both_empty_is_one_not_nan():
    """Ca hai rong => 1.0, KHONG phai NaN hay 0.

    Quan trong cho bin `absent`: ca OA nang mat sun hoan toan ma model cung du doan
    khong co sun la DUNG HOAN TOAN. Neu tra NaN, cac ca nay bi loai khoi thong ke va
    ta mat dung nhom benh nhan ma gia thuyet quan tam nhat.
    """
    e = np.zeros((10, 10, 10), bool)
    assert M.dice(e, e) == 1.0


def test_dice_known_overlap():
    """Chong lan da biet: hai khoi 20^3 lech 10 voxel => giao = 10x20x20."""
    a = _cube(lo=10, hi=30)
    b = _cube(lo=10, hi=30, shift=(10, 0, 0))
    inter, expect = 10 * 20 * 20, None
    expect = 2 * inter / (8000 + 8000)
    assert M.dice(a, b) == pytest.approx(expect, abs=1e-6)


def test_volume_error_sign_and_magnitude():
    """Dau PHAI la: duong = du doan THUA."""
    gt = _cube(lo=10, hi=30)            # 20^3 = 8000 voxel
    pred = _cube(lo=10, hi=32)          # 22x20x20... thuc ra 22^3? -> kiem bang so that
    r = M.volume_error(gt, pred, ISO)
    assert r["vol_pred_mm3"] > r["vol_gt_mm3"]
    assert r["vol_err_signed_mm3"] > 0, "du doan thua phai cho sai so DUONG"
    assert r["fn_vol_mm3"] == 0.0, "pred bao gt => khong co false negative"
    assert r["fp_vol_mm3"] == pytest.approx(r["vol_err_abs_mm3"])


# -------------------------------------------------------- be mat: dap an biet truoc

@pytest.mark.parametrize("shift", [1, 2, 3])
def test_assd_equals_known_shift_isotropic(shift):
    """Dich mat phang di `shift` voxel tren luoi vuong => ASSD ~ shift mm.

    Day la kiem chung MANH NHAT co the: dap an biet truoc chinh xac.
    Dung khoi bet (slab) de bien la mat phang, tranh hieu ung goc/canh.
    """
    shape = (40, 40, 40)
    gt = np.zeros(shape, bool); gt[10:30, :, :] = True
    pr = np.zeros(shape, bool); pr[10 + shift:30 + shift, :, :] = True
    # Mat phang song song, dich `shift` mm => phan lon voxel bien cach nhau shift mm
    assert M.assd(gt, pr, ISO) == pytest.approx(shift, abs=0.35)


def test_assd_uses_spacing_anisotropically():
    """Dich 1 voxel theo z (0.70mm) va theo x (0.3646mm) PHAI cho ASSD khac nhau.

    Bat loi thieu `sampling=spacing`: neu quen, ca hai se ra ~1.0 (don vi voxel).

    Dung slab VUONG GOC voi truc dich, dai het khung theo hai truc kia: nhu vay dich
    1 voxel lam TOAN BO be mat cach deu, khong bi cac mat song song (khoang cach 0)
    pha loang. (Khoi hop 6 mat cho ket qua tron dung nhung khong khop dap an giai
    tich vi 4 mat ben truot doc chinh no.)
    """
    shape = (40, 60, 60)
    gz = np.zeros(shape, bool); gz[10:30, :, :] = True   # mat vuong goc z
    gx = np.zeros(shape, bool); gx[:, :, 10:30] = True   # mat vuong goc x
    az = M.assd(gz, np.roll(gz, 1, axis=0), SP)          # -> 0.70mm
    ax = M.assd(gx, np.roll(gx, 1, axis=2), SP)          # -> 0.3646mm
    assert az > ax * 1.5, f"ASSD khong nhay theo spacing: z={az:.3f} vs x={ax:.3f}"
    assert az == pytest.approx(0.70, abs=0.02)
    assert ax == pytest.approx(0.3646, abs=0.02)


def test_assd_identical_is_zero():
    """Du doan khop HOAN HAO mask GT => ASSD = 0.

    Day la test khoa lai luan diem "KHONG co san roi rac hoa". Roi rac hoa KHONG
    tao san cho ASSD; san that la nhieu chu thich. (Toi da tung nham dieu nay.)
    """
    a = _cube()
    assert M.assd(a, a, SP) == 0.0


def test_empty_gives_nan_not_zero():
    """Mot ben rong => NaN, KHONG phai 0.

    Neu tra 0, mot that bai hoan toan (du doan rong) se doc nhu "hoan hao" va lam
    dep bang ket qua mot cach gia tao. SKM-TEA da co dung tinh huong nay: Dice 0.000
    kem ASSD nan.
    """
    a = _cube()
    e = np.zeros_like(a)
    assert np.isnan(M.assd(a, e, SP))
    assert np.isnan(M.hd95(a, e, SP))
    assert np.isnan(M.surface_dice(a, e, 0.5, SP))


# ------------------------------------------------------------------ HD95

def test_hd95_pooled_is_optimistic_vs_max():
    """Ban `pooled` (khop code cu) PHAI <= ban `max` (chuan).

    Day la ly do phai ghi ro trong bai bao: `surf()` cu gop hai chieu roi lay p95,
    nen mot chieu tot co the che chieu kia => doc lac quan hon.
    Dung hinh hoc bat doi xung de lam lo khac biet.
    """
    # Khoi lon + mot gai nho tach roi o goc: cac voxel gai o XA be mat GT (chieu
    # pred->gt lon), nhung be mat GT thi luon co pred gan (chieu gt->pred nho). Gop
    # chung, chi ~gai/tong voxel bien vuot nguong p95 => pooled BO SOT hoan toan.
    # max giu duoc vi no khong cho hai chieu che nhau. Gai 12^3 la kich thuoc da do
    # cho pooled=0 con max=6.3.
    shape = (60, 60, 60)
    gt = np.zeros(shape, bool); gt[10:50, 10:50, 10:50] = True
    pr = gt.copy(); pr[2:14, 2:14, 2:14] = True         # gai 12^3 tach roi o goc
    p = M.hd95(gt, pr, ISO, mode="pooled")
    x = M.hd95(gt, pr, ISO, mode="max")
    assert p <= x, f"pooled ({p:.3f}) phai <= max ({x:.3f})"
    assert x > p + 1.0, f"pooled ({p:.3f}) phai LAC QUAN ro so voi max ({x:.3f})"


def test_hd95_invalid_mode():
    a = _cube()
    with pytest.raises(ValueError):
        M.hd95(a, a, SP, mode="linh tinh")


# --------------------------------------------------------------- Surface Dice

def test_surface_dice_perfect_is_one():
    a = _cube()
    assert M.surface_dice(a, a, 0.5, SP) == pytest.approx(1.0)


def test_surface_dice_monotone_in_tolerance():
    """Dung sai rong hon => surface Dice khong bao gio giam."""
    shape = (40, 40, 40)
    gt = np.zeros(shape, bool); gt[10:30, :, :] = True
    pr = np.roll(gt, 2, axis=0)
    vals = [M.surface_dice(gt, pr, t, SP) for t in (0.25, 0.5, 1.0, 2.0, 4.0)]
    assert all(x <= y + 1e-9 for x, y in zip(vals, vals[1:])), vals
    assert vals[-1] == pytest.approx(1.0)


def test_surface_dice_catches_subvoxel_shift():
    """Surface Dice @0.5mm phai phan biet duoc lech 1 voxel z (0.70mm).

    Day la ly do metric nay ton tai: Dice the tich gan nhu khong nhuc nhich voi lech
    1 voxel tren cau truc day, nhung surface Dice o dung sai nho thi sup.
    """
    # Slab vuong goc z: dich 1 voxel (0.70mm > tol 0.5mm) lam TOAN BO be mat vuot
    # nguong => surface Dice sup ve ~0, trong khi Dice the tich van 0.95. Do chinh
    # la khoang cach ma metric nay sinh ra de chi ra.
    shape = (40, 60, 60)
    gt = np.zeros(shape, bool); gt[10:30, :, :] = True
    pr = np.roll(gt, 1, axis=0)
    assert M.surface_dice(gt, pr, 0.5, SP) < 0.05
    assert M.dice(gt, pr) > 0.9, "Dice the tich VAN cao - dung la van de can chi ra"


# ------------------------------------------------------------- boundary MAE

def test_boundary_mae_splits_inner_outer():
    """Bien trong/ngoai phai duoc tach va dem duoc."""
    shape = (40, 40, 40)
    bone = np.zeros(shape, bool); bone[:, :, :20] = True
    sdf = M.distance_transform_edt(~bone, sampling=ISO) - \
          M.distance_transform_edt(bone, sampling=ISO)
    gt = np.zeros(shape, bool); gt[10:30, 10:30, 20:24] = True
    pr = np.zeros(shape, bool); pr[10:30, 10:30, 20:25] = True
    r = M.boundary_mae(gt, pr, sdf.astype(np.float32), ISO)
    assert r["n_inner"] > 0 and r["n_outer"] > 0
    assert np.isfinite(r["inner_mae_mm"]) and np.isfinite(r["outer_mae_mm"])
    # pred day them ve phia NGOAI => bien ngoai phai sai nhieu hon bien trong
    assert r["outer_mae_mm"] > r["inner_mae_mm"]


# ------------------------------------------------------------------ thong ke

def test_paired_bootstrap_detects_real_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0.85, 0.03, 100)
    b = a + 0.02                                  # cai thien nhat quan
    r = M.paired_bootstrap(a, b, n_boot=2000, seed=1)
    assert r["delta_mean"] == pytest.approx(0.02, abs=2e-3)
    assert r["ci_low"] > 0, "CI phai loai tru 0 khi hieu ung la that"
    assert r["n_better"] == 100


def test_paired_bootstrap_null_includes_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(0.85, 0.03, 100)
    b = rng.normal(0.85, 0.03, 100)
    r = M.paired_bootstrap(a, b, n_boot=2000, seed=1)
    assert r["ci_low"] < 0 < r["ci_high"], "CI phai chua 0 khi khong co hieu ung"


def test_paired_bootstrap_drops_nan_pairs():
    """Cap co NaN (lop vang trong GT) phai bi bo, khong lam hong ca thong ke."""
    a = np.array([0.8, 0.9, np.nan, 0.7])
    b = np.array([0.85, 0.95, 0.5, np.nan])
    r = M.paired_bootstrap(a, b, n_boot=500, seed=0)
    assert r["n"] == 2
    assert r["delta_mean"] == pytest.approx(0.05, abs=1e-6)


# ------------------------------------------------------------------ tong hop

def test_all_metrics_keys_and_perfect_case():
    a = _cube()
    m = M.all_metrics(a, a, SP)
    for k in ("dice", "assd_mm", "hd95_mm", "hd95_pooled_mm", "sdice_0.5mm",
              "sdice_1.0mm", "precision", "recall", "vol_err_signed_mm3"):
        assert k in m, f"thieu key {k}"
    assert m["dice"] == pytest.approx(1.0)
    assert m["assd_mm"] == 0.0
    assert m["sdice_0.5mm"] == pytest.approx(1.0)
