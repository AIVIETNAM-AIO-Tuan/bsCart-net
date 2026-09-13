"""Phantom asserts cho bsc/biomarkers.py - chay LOCAL, khong can data.

    python -m pytest bsc/tests/test_biomarkers.py -v

Cac phantom deu dung spacing THAT (0.70, 0.3646, 0.3646) => sai so anisotropy duoc test.

Dieu test nay CHUNG MINH (khong phai gia dinh):
  * thickness proxy cu (the tich / tiep xuc) MU voi mat sun toan be day; thc_tab thi khong
  * denuded_ratio cu doi rat it khi mat sun (mau so la ca xuong); fcl_pct doi dung bang lo
  * closing chi lap lo nho hon ~2*ban kinh -> ban kinh la tham so nhay, phai bao cao
"""

from __future__ import annotations

import numpy as np
import pytest

from bsc import biomarkers as BM

SP = (0.70, 0.3646, 0.3646)


# ------------------------------------------------------------------ phantoms

def _grid(shape, spacing):
    zz, yy, xx = np.meshgrid(*[np.arange(s) * sp for s, sp in zip(shape, spacing)], indexing="ij")
    return zz, yy, xx


def sphere_knee(shape=(56, 120, 120), spacing=SP, r_mm=12.0, t_mm=1.6,
                cap_deg=None, cap_dir=(0.0, 0.0, 1.0)):
    """Mot xuong cau (nhan 1) boc sun (nhan 2) day t_mm; tuy chon KHOET mot chom (FCL)."""
    zz, yy, xx = _grid(shape, spacing)
    c = np.array([s * sp / 2 for s, sp in zip(shape, spacing)])
    d = np.stack([zz - c[0], yy - c[1], xx - c[2]], -1)
    r = np.linalg.norm(d, axis=-1)
    bone = r <= r_mm
    cart = (r > r_mm) & (r <= r_mm + t_mm)
    if cap_deg is not None:
        u = np.asarray(cap_dir, np.float64)
        u /= np.linalg.norm(u)
        cosang = (d @ u) / np.maximum(r, 1e-6)
        cart &= ~(cosang >= np.cos(np.deg2rad(cap_deg)))
    arr = np.zeros(shape, np.uint8)
    arr[bone] = 1
    arr[cart] = 2
    return arr


def two_bone_knee(shape=(64, 136, 128), spacing=SP, r_mm=9.0, t_mm=1.6, defect=False,
                  defect_deg=20.0, defect_tilt_deg=35.0):
    """Xuong dui (cau tren) + xuong chay (cau duoi), truc tren-duoi = truc 1 (y), trong-ngoai = truc 2 (x).

    Sun dui = vo nua duoi cau tren (nhan 2). Sun chay = vo nua tren cau duoi, chia trong (4)
    / ngoai (5) theo x. `defect`: khoet chom tren mam chay TRONG, nghieng ve phia trong de
    nam tron trong khoang trong.
    """
    zz, yy, xx = _grid(shape, spacing)
    ext = np.array([s * sp for s, sp in zip(shape, spacing)])
    cz, cx = ext[0] / 2, ext[2] / 2
    cf = np.array([cz, 36.0, cx])
    ct = np.array([cz, 12.0, cx])

    def sphere(c):
        d = np.stack([zz - c[0], yy - c[1], xx - c[2]], -1)
        return d, np.linalg.norm(d, axis=-1)

    arr = np.zeros(shape, np.uint8)
    df, rf = sphere(cf)
    arr[rf <= r_mm] = 1
    fem_cart = (rf > r_mm) & (rf <= r_mm + t_mm) & (yy < cf[1])
    arr[fem_cart] = 2

    dt, rt = sphere(ct)
    arr[rt <= r_mm] = 3
    tib_cart = (rt > r_mm) & (rt <= r_mm + t_mm) & (yy > ct[1])
    if defect:
        tilt = np.deg2rad(defect_tilt_deg)
        u = np.array([0.0, np.cos(tilt), -np.sin(tilt)])       # huong len, nghieng ve -x (trong)
        cosang = (dt @ u) / np.maximum(rt, 1e-6)
        tib_cart &= ~(cosang >= np.cos(np.deg2rad(defect_deg)))
    arr[tib_cart & (xx < cx)] = 4
    arr[tib_cart & (xx >= cx)] = 5
    return arr


def cap_area(r_mm, deg):
    return 2 * np.pi * r_mm ** 2 * (1 - np.cos(np.deg2rad(deg)))


# ------------------------------------------------------------------ luoi + phap tuyen

def test_bone_mesh_is_at_bone_boundary_and_normals_point_outward():
    arr = sphere_knee()
    verts, faces, normals, info = BM.bone_mesh(arr == 1, SP)
    c = np.array([s * sp / 2 for s, sp in zip(arr.shape, SP)])
    rad = np.linalg.norm(verts - c, axis=1)
    assert abs(np.median(rad) - 12.0) < 0.3, f"be mat lech tam: r={np.median(rad):.2f}"
    outward = ((verts - c) * normals).sum(1) > 0
    assert outward.mean() > 0.99, f"phap tuyen huong ra chi {outward.mean():.3f}"
    # quy uoc skimage nhat quan: hoac lat gan het, hoac khong lat gi
    assert info["flip_frac"] < 0.02 or info["flip_frac"] > 0.98, info
    area = BM.vertex_areas(verts, faces).sum()
    assert abs(area - 4 * np.pi * 144) / (4 * np.pi * 144) < 0.06, f"dien tich cau lech: {area:.0f}"


def test_bone_mesh_empty_mask_is_safe():
    verts, faces, normals, info = BM.bone_mesh(np.zeros((10, 10, 10), bool), SP)
    assert len(verts) == 0 and info["n_verts"] == 0


# ------------------------------------------------------------------ do day doc phap tuyen

@pytest.mark.parametrize("t_mm,tol,min_presence", [(1.6, 0.25, 0.97), (0.5, 0.3, 0.85)])
def test_thickness_along_normals_recovers_shell(t_mm, tol, min_presence):
    # Vo 0.5mm MONG HON voxel z (0.70mm): chinh mask sun bi dut o 2 cuc => presence
    # khong the 100%. Do la gioi han cua rasterization (xem core.roundtrip), khong phai loi do.
    arr = sphere_knee(t_mm=t_mm)
    verts, faces, normals, _ = BM.bone_mesh(arr == 1, SP)
    thick, first = BM.thickness_along_normals(arr == 2, verts, normals, SP)
    assert (thick > 0).mean() > min_presence, "vo sun kin ma nhieu dinh khong thay sun"
    med = np.median(thick[thick > 0])
    assert abs(med - t_mm) < tol, f"do day {med:.2f} vs {t_mm}"
    assert np.nanmedian(first) <= 0.6, "diem cham dau tien phai sat be mat xuong"


def test_thickness_zero_when_cartilage_far_from_bone():
    # Sun cach xuong 2mm > gap_mm=1.0 => khong tinh la sun cua be mat nay
    zz, yy, xx = _grid((56, 120, 120), SP)
    c = [s * sp / 2 for s, sp in zip((56, 120, 120), SP)]
    r = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    arr = np.zeros(r.shape, np.uint8)
    arr[r <= 12.0] = 1
    arr[(r > 14.0) & (r <= 15.6)] = 2
    verts, _, normals, _ = BM.bone_mesh(arr == 1, SP)
    thick, _ = BM.thickness_along_normals(arr == 2, verts, normals, SP, gap_mm=1.0)
    assert (thick > 0).mean() < 0.01


# ------------------------------------------------------------------ FCL tren cau

def test_full_shell_has_no_fcl():
    arr = sphere_knee()
    out = BM.surface_biomarkers(arr, SP)
    assert out["fcl_fem_pct"] < 0.5, out["fcl_fem_pct"]
    assert abs(out["tab_fem_mm2"] - 4 * np.pi * 144) / (4 * np.pi * 144) < 0.08
    assert abs(out["thc_cab_fem_mm"] - 1.6) < 0.25
    assert out["fcl_fem_ndef"] == 0


def test_fcl_recovers_cap_area_when_closing_covers_hole():
    deg = 20.0
    arr = sphere_knee(cap_deg=deg)
    expect = cap_area(12.0, deg)                     # ~54.6 mm^2, ban kinh trac dia ~4.2mm
    out = BM.surface_biomarkers(arr, SP, close_mm=8.0)
    got = out["fcl_fem_mm2"]
    assert abs(got - expect) / expect < 0.25, f"FCL {got:.1f} vs ky vong {expect:.1f}"
    assert abs(out["fcl_fem_pct"] - 100 * expect / (4 * np.pi * 144)) < 1.0
    assert out["fcl_fem_ndef"] == 1
    assert abs(out["fcl_fem_maxdef_mm2"] - got) < 1e-6
    # ThC.tAB giam dung bang phan mat, ThC.cAB khong doi
    assert abs(out["thc_cab_fem_mm"] - 1.6) < 0.25
    assert out["thc_tab_fem_mm"] < out["thc_cab_fem_mm"] * (1 - 0.02)


def test_fcl_closing_radius_sensitivity():
    """Lo ban kinh trac dia ~4.2mm: closing 8mm lap duoc, closing 2mm KHONG => FCL ~0.

    Day la HAN CHE co chu y cua footprint-bang-closing: ban kinh phai duoc bao cao.
    """
    arr = sphere_knee(cap_deg=20.0)
    big = BM.surface_biomarkers(arr, SP, close_mm=8.0)["fcl_fem_mm2"]
    small = BM.surface_biomarkers(arr, SP, close_mm=2.0)["fcl_fem_mm2"]
    # Do duoc: 49.8 vs 8.7 mm^2 (phan con lai o R=2 la vanh rang cua cua mep lo)
    assert big > 40.0 and small < 0.3 * big, (big, small)


def test_legacy_thickness_is_blind_to_fcl_but_thc_tab_is_not():
    """Chung minh ly do can ho surface: proxy cu mu voi mat sun toan be day."""
    full = BM.all_biomarkers(sphere_knee(), SP)
    hole = BM.all_biomarkers(sphere_knee(cap_deg=35.0), SP, close_mm=10.0)   # chom ~9% dien tich
    frac = hole["fcl_fem_pct"] / 100.0
    assert 0.06 < frac < 0.12, frac
    rel_legacy = (full["thickness_femoral_mm"] - hole["thickness_femoral_mm"]) / full["thickness_femoral_mm"]
    rel_thc = (full["thc_tab_fem_mm"] - hole["thc_tab_fem_mm"]) / full["thc_tab_fem_mm"]
    # thc_tab giam DUNG bang phan dien tich mat (theo cau truc). Proxy cu giam it hon nhieu,
    # va phan giam do chi den tu artefact huong (mat do voxel bien khac nhau theo huong),
    # khong phai tu ton thuong. Do duoc: legacy 2.8% vs thc_tab 8.8% khi mat 8.5%.
    assert abs(rel_thc - frac) < 0.03, (rel_thc, frac)
    assert rel_legacy < 0.7 * frac, (rel_legacy, frac)
    # proxy cu con sai tuyet doi ~2x (3.1mm cho vo 1.6mm) vi dem voxel bien
    assert full["thickness_femoral_mm"] > 2.5 and abs(full["thc_cab_fem_mm"] - 1.6) < 0.25
    # denuded_ratio cu: mau so la CA xuong nen nhich it hon fcl_pct
    assert (hole["denuded_ratio_femoral"] - full["denuded_ratio_femoral"]) < frac


# ------------------------------------------------------------------ 2 xuong, tach khoang

def test_two_bone_end_to_end_compartments_and_fcl():
    arr0 = two_bone_knee(defect=False)
    arr1 = two_bone_knee(defect=True)
    out0, surf0 = BM.all_biomarkers(arr0, SP, return_surfaces=True)
    out1 = BM.all_biomarkers(arr1, SP)

    # legacy: du cot, the tich sun dui ~ nua vo cau
    for k in ("vol_femoral_cart_mm3", "thickness_med_tib_mm", "denuded_ratio_tibial",
              "extrusion_med_meniscus"):
        assert k in out0
    half_shell = 0.5 * 4 / 3 * np.pi * (10.6 ** 3 - 9.0 ** 3)
    assert abs(out0["vol_femoral_cart_mm3"] - half_shell) / half_shell < 0.10

    # khoang: 2 mam chay xap xi bang nhau, sun dui tach duoc 2 nua
    assert out0["qc_axis_ok"] == 1.0
    assert out0["tab_mt_mm2"] > 150 and out0["tab_lt_mm2"] > 150
    assert abs(out0["tab_mt_mm2"] - out0["tab_lt_mm2"]) / out0["tab_mt_mm2"] < 0.2
    assert out0["tab_fem_med_mm2"] > 150 and out0["tab_fem_lat_mm2"] > 150
    assert abs(out0["tab_fem_mm2"] - out0["tab_fem_med_mm2"] - out0["tab_fem_lat_mm2"]) < 1e-3
    for c in ("fem", "mt", "lt"):
        assert abs(out0[f"thc_cab_{c}_mm"] - 1.6) < 0.3, (c, out0[f"thc_cab_{c}_mm"])
        assert out0[f"fcl_{c}_pct"] < 1.0, (c, out0[f"fcl_{c}_pct"])

    # khoet mam chay TRONG: FCL chi xuat hien o mt
    expect = cap_area(9.0, 20.0)                     # ~30.7 mm^2
    assert abs(out1["fcl_mt_mm2"] - expect) / expect < 0.35, (out1["fcl_mt_mm2"], expect)
    assert out1["fcl_lt_mm2"] < 3.0 and out1["fcl_fem_mm2"] < 5.0
    assert out1["fcl_mt_ndef"] == 1
    # thc_tab giam dung bang phan mat; proxy cu giam it hon (do duoc 5.4% vs 10.4%)
    frac = out1["fcl_mt_pct"] / 100.0
    rel_legacy = (out0["thickness_med_tib_mm"] - out1["thickness_med_tib_mm"]) / out0["thickness_med_tib_mm"]
    rel_thc = (out0["thc_tab_mt_mm"] - out1["thc_tab_mt_mm"]) / out0["thc_tab_mt_mm"]
    assert abs(rel_thc - frac) < 0.03, (rel_thc, frac)
    assert rel_legacy < 0.7 * frac, (rel_legacy, frac)

    # be mat tra ve dung de QC
    assert set(surf0) == {"fem", "tib"}
    paint = BM.paint_vertices(surf0["tib"]["verts"], surf0["tib"]["footprint"], arr0.shape, SP)
    assert paint.sum() > 1000


def test_no_cartilage_gives_nan_not_crash():
    arr = sphere_knee()
    arr[arr == 2] = 0
    out = BM.all_biomarkers(arr, SP)
    assert np.isnan(out["fcl_fem_pct"]) and np.isnan(out["tab_fem_mm2"])
    assert np.isnan(out["thickness_femoral_mm"])


def test_all_keys_are_features_or_qc():
    out = BM.all_biomarkers(two_bone_knee(), SP)
    bad = [k for k in out if not (k.startswith(BM.FEATURE_PREFIXES) or k.startswith("qc_"))]
    assert not bad, f"cot khong co tien to hop le (S7 se bo sot): {bad}"
    for c in BM.COMPARTMENTS:
        assert f"fcl_{c}_pct" in out and f"thc_tab_{c}_mm" in out


def test_spacing_is_required_and_validated():
    arr = sphere_knee()
    with pytest.raises(TypeError):
        BM.all_biomarkers(arr)                       # thieu spacing
    with pytest.raises(ValueError):
        BM.all_biomarkers(arr, (0.7, 0.36))          # sai so truc


# ------------------------------------------------------------------ closing trac dia

def test_geodesic_closing_contains_mask_and_fills_small_hole_only():
    arr = sphere_knee()
    verts, faces, _, _ = BM.bone_mesh(arr == 1, SP)
    g = BM.mesh_graph(verts, faces)
    c = np.array([s * sp / 2 for s, sp in zip(arr.shape, SP)])
    u = (verts - c) / np.linalg.norm(verts - c, axis=1, keepdims=True)
    hole = u[:, 2] > np.cos(np.deg2rad(15))          # chom 15 deg, ban kinh trac dia ~3.1mm
    mask = ~hole
    closed = BM.geodesic_closing(g, mask, 6.0)
    assert (closed | ~mask).all(), "closing phai chua tap goc"
    assert closed[hole].mean() > 0.9, "closing 6mm phai lap chom 3.1mm"
    closed_small = BM.geodesic_closing(g, mask, 1.0)
    assert closed_small[hole].mean() < 0.3, "closing 1mm khong duoc lap chom 3.1mm"
