"""Phantom asserts cho lop hinh hoc (test M1/M2/M3/M4 thu nho).

Chay LOCAL, khong can data, khong can GPU:
    python -m pytest bsc/tests -v

Vi sao ton tai: plan doc §3.5 de xuat M1 "synthetic shell phantom" nhu mot thi nghiem
rieng. Nhung no test cai PHANTOM GENERATOR, khong test gia thuyet - nen da cat xuong
thanh unit test. Gia tri that cua no la bat REGRESSION o cac hang so da do:

  * dao dau phap tuyen (extract_surface): sai dau -> occupancy toan 0, bug IM LANG
  * doi mm->voxel cua sigma: sai -> M3 tu 99.89% roi xuong 99.11%, truot cong
  * sampling= cua EDT: thieu -> SDF sai ~2x giua z va in-plane
  * pts/spacing truoc map_coordinates: thieu -> lay mau sai cho hoan toan

Cac nguong duoi day la SO DA DO tren phantom o dung spacing [0.70, 0.3646, 0.3646],
khong phai uoc luong.
"""

from __future__ import annotations

import numpy as np
import pytest

from bsc import core
from bsc.core import RayConfig

SP = core.SPACING


def _assd(a, b, spacing=SP) -> float:
    """ASSD doc lap (scipy EDT) - CO Y dung lai bsc.metrics de test khong tu xac nhan."""
    from scipy.ndimage import distance_transform_edt as edt
    from skimage.segmentation import find_boundaries
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not a.any() or not b.any():
        return float("nan")
    sa, sb = find_boundaries(a, mode="inner"), find_boundaries(b, mode="inner")
    da, db = edt(~sa, sampling=spacing), edt(~sb, sampling=spacing)
    return float(np.concatenate([db[sa], da[sb]]).mean())


# ------------------------------------------------------------------ phantom

def make_shell(shape=(60, 140, 140), spacing=SP, r_mm=14.0, t_mm=1.6,
               defect=False, bumpy=False):
    """Ellipsoid "xuong" + vo "sun" day t_mm bao ngoai.

    Tra (bone, cart). Toa do tinh theo MM nen phantom dung ANISOTROPIC that,
    khong phai luoi vuong gia lap.
    """
    zz, yy, xx = np.meshgrid(*[np.arange(s) * sp for s, sp in zip(shape, spacing)],
                             indexing="ij")
    c = [s * sp / 2 for s, sp in zip(shape, spacing)]
    r = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)

    rad = np.full(r.shape, r_mm, np.float32)
    if bumpy:
        # Nhap nho be mat -> ep vung cong cao, la noi M4 va phap tuyen de hong nhat
        rad = rad + 1.5 * np.sin(3 * yy / r_mm) * np.cos(3 * xx / r_mm)

    bone = r <= rad
    cart = (r > rad) & (r <= rad + t_mm)

    if defect:
        # Mat sun toan phan tren mot chom: mo phong full-thickness loss cua OA nang
        cart &= ~((xx > c[2]) & (yy > c[1]))
    return bone, cart


# ------------------------------------------------------------------ M3

@pytest.mark.parametrize("bumpy", [False, True])
def test_m3_normals_point_outward(bumpy):
    """M3: >99.5% phap tuyen huong RA ngoai (plan doc §3.5 M3).

    Day la test bat loi DAO DAU. Neu ai do bo dau tru trong extract_surface,
    frac_ok se sup ve ~0 chu khong phai giam nhe.
    """
    bone, _ = make_shell(bumpy=bumpy)
    sdf, verts, normals = core.bone_geometry(bone, SP, RayConfig())
    frac_ok, _ = core.check_normals(sdf, verts, normals, SP)
    assert frac_ok > 0.995, f"M3 truot: chi {frac_ok:.4f} phap tuyen huong ra"


def test_m3_catches_flipped_normals():
    """Dao dau phap tuyen PHAI lam M3 sup - chung minh test co suc bat loi.

    Khong co test nay thi test_m3 o tren co the dang dau vi ly do sai.
    """
    bone, _ = make_shell()
    sdf, verts, normals = core.bone_geometry(bone, SP, RayConfig())
    frac_ok, _ = core.check_normals(sdf, verts, -normals, SP)   # co tinh lam sai
    assert frac_ok < 0.01, f"M3 KHONG bat duoc phap tuyen dao dau (frac_ok={frac_ok:.4f})"


def test_smoothing_sigma_is_mm_not_voxels():
    """sigma phai duoc doi mm -> voxel theo tung truc (cam bay 2).

    Neu truyen thang sigma_mm vao gaussian_filter, truc z (0.70mm/voxel) se bi lam
    mo ~2x qua tay so voi in-plane, va SDF meo di theo huong z. Test bat dieu do
    bang cach so voi ket qua da doi don vi dung.
    """
    bone, _ = make_shell()
    sdf_raw = core.signed_distance(bone, SP, smooth_mm=0.0)

    from scipy.ndimage import gaussian_filter
    correct = gaussian_filter(sdf_raw, sigma=[0.5 / s for s in SP])
    wrong = gaussian_filter(sdf_raw, sigma=[0.5, 0.5, 0.5])       # bug: mm nhu voxel

    got = core.signed_distance(bone, SP, smooth_mm=0.5)
    assert np.allclose(got, correct, atol=1e-4), "signed_distance khong doi mm->voxel"
    assert not np.allclose(correct, wrong, atol=1e-3), "phantom qua tho de phan biet"


# ------------------------------------------------------------------ M2

@pytest.mark.parametrize("bumpy,defect", [(False, False), (True, False), (True, True)])
def test_m2_roundtrip(bumpy, defect):
    """M2: GT -> tia -> voxel phai gan nhu vo ton (plan doc §3.5 M2).

    Do duoc tren phantom: Dice 0.987, ASSD 0.011mm - thap hon phan thuong ~20x.
    Do la ly do M2 la THU TUC chu khong phai cong chan: khong co "san roi rac hoa".
    Mot du doan khop hoan hao mask GT cho ASSD = 0; san that la NHIEU CHU THICH.
    """
    bone, cart = make_shell(bumpy=bumpy, defect=defect)
    rec = core.roundtrip(cart, bone, SP, RayConfig())
    pred = rec > 0.5

    inter = (pred & cart).sum()
    dice = 2.0 * inter / (pred.sum() + cart.sum())
    assert dice > 0.97, f"M2 round-trip Dice {dice:.4f} < 0.97"


def test_m2_thin_ceiling_dice_collapses_but_assd_holds():
    """PHAT HIEN QUAN TRONG: tran round-trip SUP o vung sun mong.

    Do duoc tren phantom nay:
        do day    M2 Dice    M2 ASSD
        1.6mm     0.9995     0.001mm
        0.5mm     0.925      0.030mm
        0.4mm     0.848      0.065mm

    Voi sun 0.4mm, phep doi toa do TU NO mat 15% Dice du du doan hoan hao. Nghia la
    cong M2 cua plan doc (Dice >= 0.97) se TRUOT o dung cai vung ma gia thuyet nham
    toi - khong phai vi gia thuyet sai, ma vi Dice tren cau truc day ~1 voxel nhay
    den muc tan nhan.

    Nhung ASSD van 0.065mm - duoi xa muc tieu 0.1mm. => DAT CONG M2 TREN ASSD.

    Test nay khoa ca hai chieu: Dice PHAI sup (neu khong, phantom qua tho de noi len
    dieu gi), va ASSD PHAI giu (neu khong, bieu dien tia that su khong du phan giai
    va gia thuyet gap rac roi that su).
    """
    cfg = RayConfig()
    dice, assd = {}, {}
    for t in (1.6, 0.4):
        bone, cart = make_shell(t_mm=t)
        rec = core.roundtrip(cart, bone, SP, cfg) > 0.5
        dice[t] = 2 * (rec & cart).sum() / (rec.sum() + cart.sum())
        assd[t] = _assd(rec, cart, SP)

    assert dice[1.6] > 0.99, f"sun day phai gan nhu hoan hao, duoc {dice[1.6]:.4f}"
    assert dice[0.4] < 0.95, (
        f"Dice sun cuc mong KHONG sup ({dice[0.4]:.4f}) - phantom co the qua tho, "
        f"hoac gia dinh 've tran mong' can xem lai"
    )
    # Mat con lai: bien VAN chinh xac du Dice te. Day la ly do bai bao phai lay
    # metric bien lam chinh chu khong phai Dice.
    assert assd[0.4] < 0.10, (
        f"ASSD sun cuc mong {assd[0.4]:.3f}mm >= muc tieu 0.1mm - neu the thi bieu "
        f"dien tia khong du phan giai va gia thuyet gap rac roi that"
    )


def test_m2_full_density_covers_all_cartilage():
    """O FULL marching-cubes density, splat khong duoc bo sot voxel sun nao.

    Day la mat con lai cua quyet dinh "KHONG remesh": xem test_subsample_breaks_it.
    """
    bone, cart = make_shell()
    rec = core.roundtrip(cart, bone, SP, RayConfig())
    missed = (cart & (rec <= 0.5)).sum() / cart.sum()
    assert missed < 0.02, f"Bo sot {missed:.1%} voxel sun o full density"


def test_subsample_breaks_reconstruction():
    """Chung minh Step 2 cua plan doc ("remesh cho dong deu") LA BUG.

    Doc bao: "The surface should then be remeshed or subsampled so that point spacing
    is reasonably uniform." Do duoc: h=1.0mm -> Dice 0.811, mat 20% voxel sun.

    Test nay khoa lai phat hien do. Neu ai do them remeshing vao duong inference,
    day la thu se do do.
    """
    bone, cart = make_shell()
    cfg = RayConfig()
    _, verts, normals = core.bone_geometry(bone, SP, cfg)

    occ = core.occupancy_target(cart, verts, normals, cfg, SP)
    full = core.splat_rays(occ.astype(np.float32), verts, normals, cart.shape, cfg, SP) > 0.5
    d_full = 2 * (full & cart).sum() / (full.sum() + cart.sum())

    sub = slice(None, None, 8)   # lam thua manh -> mo phong remesh tho
    occ_s = core.occupancy_target(cart, verts[sub], normals[sub], cfg, SP)
    sparse = core.splat_rays(occ_s.astype(np.float32), verts[sub], normals[sub],
                             cart.shape, cfg, SP) > 0.5
    d_sparse = 2 * (sparse & cart).sum() / (sparse.sum() + cart.sum())

    assert d_full > 0.97
    assert d_sparse < d_full - 0.05, (
        f"Subsample dang KHONG lam hong reconstruct (full {d_full:.3f} vs "
        f"sparse {d_sparse:.3f}) - kiem tra lai gia dinh 'khong remesh'"
    )


# ------------------------------------------------------------------ M4 + tia

def test_m4_single_interval_ratio():
    """M4: vo cau don gian => gan nhu moi tia chi cat 1 khoang (plan doc §3.5 M4).

    Do tren phantom: 0.998. Thuc te se thap hon o ria sun / vung cong cao.
    """
    bone, cart = make_shell(bumpy=True)
    cfg = RayConfig()
    _, verts, normals = core.bone_geometry(bone, SP, cfg)
    occ = core.occupancy_target(cart, verts, normals, cfg, SP)
    assert core.single_interval_ratio(occ) > 0.95


def test_ray_thickness_matches_shell():
    """Do day suy ra tu occupancy phai khop do day vo that (~t_mm)."""
    t_mm = 1.6
    bone, cart = make_shell(t_mm=t_mm)
    cfg = RayConfig()
    _, verts, normals = core.bone_geometry(bone, SP, cfg)
    occ = core.occupancy_target(cart, verts, normals, cfg, SP)
    st = core.ray_stats(occ, cfg)

    med = np.median(st["thickness"][st["presence"]])
    assert abs(med - t_mm) < 0.25, f"do day suy ra {med:.2f}mm vs that {t_mm}mm"


def test_defect_gives_absent_rays():
    """Vung mat sun toan phan phai sinh ra tia presence=False.

    Neu khong, presence head khong co negative va luan diem 've sun mat' khong
    kiem chung duoc - xem oracle_domain docstring.
    """
    bone, cart = make_shell(defect=True)
    cfg = RayConfig()
    _, verts, normals = core.bone_geometry(bone, SP, cfg)
    occ = core.occupancy_target(cart, verts, normals, cfg, SP)
    st = core.ray_stats(occ, cfg)
    frac_absent = 1.0 - st["presence"].mean()
    assert 0.05 < frac_absent < 0.95, f"ty le tia absent bat thuong: {frac_absent:.2f}"


def test_oracle_domain_includes_absent_rays():
    """oracle_domain PHAI chua tia absent nho vanh no 5mm.

    Day la chi tiet plan doc bo sot. Thieu vanh, domain == present, presence head
    khong co negative, va khong the test duoc gia thuyet chinh.
    """
    bone, cart = make_shell(defect=True)
    cfg = RayConfig()
    _, verts, normals = core.bone_geometry(bone, SP, cfg)
    occ = core.occupancy_target(cart, verts, normals, cfg, SP)
    st = core.ray_stats(occ, cfg)

    dom_rim = core.oracle_domain(verts, occ, cfg, rim_mm=5.0)
    dom_norim = core.oracle_domain(verts, occ, cfg, rim_mm=0.0)

    n_abs_rim = (dom_rim & ~st["presence"]).sum()
    n_abs_norim = (dom_norim & ~st["presence"]).sum()
    assert n_abs_rim > n_abs_norim, "vanh no khong them duoc tia absent nao"
    assert n_abs_rim > 100, f"chi {n_abs_rim} tia absent trong domain - qua it de train"


# ------------------------------------------------------------------ splat

def test_splat_is_adjoint_of_sampling():
    """Splat phai la nghich dao cua lay mau: truong hang so -> tra ve hang so do.

    Day la tinh chat lam round-trip gan nhu vo ton. Neu chuan hoa trong so sai,
    test nay se lech ngay.
    """
    bone, _ = make_shell()
    cfg = RayConfig()
    _, verts, normals = core.bone_geometry(bone, SP, cfg)
    vals = np.full((len(verts), cfg.k), 0.7, np.float32)
    rec = core.splat_rays(vals, verts, normals, bone.shape, cfg, SP)
    hit = rec > 0
    assert np.allclose(rec[hit], 0.7, atol=1e-4), "splat khong bao toan truong hang so"


def test_thickness_bins():
    t = np.array([0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0], np.float32)
    b = core.assign_thickness_bin(t)
    assert b.tolist() == [0, 1, 1, 2, 2, 3, 3, 4]
