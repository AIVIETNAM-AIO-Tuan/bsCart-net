"""Atlas quan the theo fold (D1) - kiem dung theo plan §3.4 Step 4 + §2.2."""

from __future__ import annotations

import numpy as np
import pytest

from bsc import atlas, core
from bsc.core import RayConfig

SP = core.SPACING


def _knee_like(shape=(40, 110, 110), spacing=SP, t_mm=2.0,
               cap_frac=0.30, shift_mm=0.0, seed=0):
    """Xuong giong DAU XUONG THAT: ellipsoid ba ban truc khac nhau, BI CAT boi FOV.

    Vi sao KHONG dung hinh cau: cau co ba tri rieng PCA bang nhau (truc tuy y) va doi
    xung tam (skew = 0, dau truc tuy y) => khung toa do khong xac dinh => atlas gop
    thanh chao. Xuong that trong OAI-ZIB bi mat cat FOV cat cut nen bat doi xung manh.
    Phantom phai phan anh dieu do, neu khong test se do nham mot che do khong ton tai.
    """
    zz, yy, xx = np.meshgrid(*[np.arange(s) * sp for s, sp in zip(shape, spacing)],
                             indexing="ij")
    c = [s * sp / 2 for s, sp in zip(shape, spacing)]
    c[2] += shift_mm
    # Ba ban truc khac nhau => tri rieng PCA tach bach
    rz, ry, rx = 9.0, 15.0, 11.0
    q = (((zz - c[0]) / rz) ** 2 + ((yy - c[1]) / ry) ** 2 + ((xx - c[2]) / rx) ** 2)
    ell = q <= 1.0
    shell = (q > 1.0) & (q <= (1.0 + t_mm / rx) ** 2)

    # CAT bo mot dau (mo phong FOV) => bat doi xung => skew xac dinh dau truc
    cut = yy > (c[1] - 0.45 * ry)
    bone = ell & cut
    cart = shell & cut & (xx > c[2] + (1 - cap_frac) * rx)
    mri = np.zeros(shape, np.float32)
    return mri, bone, cart


def _loader_factory(cases):
    def loader(cid):
        return cases[cid]
    return loader


def test_frame_is_deterministic_and_not_degenerate():
    """Khung phai deterministic va khong suy bien tren xuong co be rong hop ly."""
    _, bone, _ = _knee_like()
    _, verts, _ = core.bone_geometry(bone, SP)

    f1 = atlas.fit_frame(verts)
    f2 = atlas.fit_frame(verts.copy())
    assert np.allclose(f1.centroid, f2.centroid) and np.allclose(f1.scale, f2.scale)
    assert not f1.is_degenerate, f"khung suy bien: extent {f1.extent}"

    # Toa do chuan hoa phai nam quanh [-1, 1]
    u = atlas.to_normalized(verts, f1)
    assert np.abs(u).max() < 1.6, f"chuan hoa lech: max|u| = {np.abs(u).max():.2f}"


def test_frame_strict_catches_flat_surface():
    """Be mat det tren mot truc => strict=True phai bao loi, khong tra khung vo nghia."""
    flat = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
    flat[:, 0] *= 0.01           # gan nhu phang theo truc z
    flat[:, 1:] *= 20.0
    with pytest.raises(ValueError, match="SUY BIEN"):
        atlas.fit_frame(flat, strict=True)


def test_normalized_coords_are_translation_invariant():
    """Cung hinh dang, tinh tien khac nhau => toa do chuan hoa phai TRUNG nhau.

    Day la dieu kien de atlas gop duoc thong tin giua cac ca.
    """
    _, b0, _ = _knee_like(shift_mm=0.0)
    _, b1, _ = _knee_like(shift_mm=3.0)
    _, v0, _ = core.bone_geometry(b0, SP)
    _, v1, _ = core.bone_geometry(b1, SP)

    u0 = atlas.to_normalized(v0, atlas.fit_frame(v0))
    u1 = atlas.to_normalized(v1, atlas.fit_frame(v1))
    # So phan bo (khong so tung dinh vi so luong dinh co the khac)
    assert np.allclose(u0.mean(0), u1.mean(0), atol=0.15)
    assert np.allclose(u0.std(0), u1.std(0), atol=0.15)


def test_grid_index_never_overflows():
    """Chi so o luoi PHAI trong [0, n_bins-1] o moi truong hop bien.

    Bug da gap: kep PHAN SO bang `clip(t, 0, 1 - 1e-9)` vo o float32 (eps ~1.19e-7 nen
    1-1e-9 lam tron thanh 1.0) => chi so = n_bins => ravel_multi_index no
    'invalid entry in coordinates array'. Phai kep CHI SO.
    """
    n = 24
    for lo, hi in [(-1.2, 1.2), (-3.0, 3.0)]:
        for dtype in (np.float32, np.float64):
            u = np.array([[lo, hi, 0.0],
                          [hi, hi, hi],                 # dung bien tren
                          [lo - 99, hi + 99, 0.0],      # ngoai bien xa
                          [np.nan, np.inf, -np.inf]],   # khong huu han
                         dtype=dtype)
            i = atlas.grid_index(u, n, lo, hi)
            assert i.min() >= 0 and i.max() <= n - 1, f"tran: {i}"
            assert i.dtype == np.int64


def test_build_atlas_survives_vertex_at_upper_bound():
    """Dinh nam DUNG bien tren cua luoi khong duoc lam vo build_articular_atlas."""
    cases = {"c0": _knee_like()}
    a = atlas.build_articular_atlas(list(cases), _loader_factory(cases), SP,
                                    RayConfig(), n_bins=24, lo=-1.0, hi=1.0,
                                    min_count=1)
    assert np.isfinite(a.prob).any()


def test_atlas_records_case_ids_for_audit():
    """§2.2 doi audit duoc: atlas phai ghi lai da dung ca nao."""
    cases = {f"tr{i}": _knee_like(shift_mm=0.5 * i, seed=i) for i in range(4)}
    a = atlas.build_articular_atlas(list(cases), _loader_factory(cases), SP,
                                    RayConfig(), n_bins=12, fold=0)
    assert set(a.case_ids) == set(cases)
    assert a.fold == 0


def test_assert_no_leak_catches_heldout_case():
    """§2.2: mask sun cua ca held-out KHONG duoc dung xay atlas -> phai bao loi."""
    cases = {f"c{i}": _knee_like(shift_mm=0.5 * i, seed=i) for i in range(4)}
    a = atlas.build_articular_atlas(list(cases), _loader_factory(cases), SP,
                                    RayConfig(), n_bins=12)

    atlas.assert_no_leak(a, ["val1", "val2"])              # khong giao => OK
    with pytest.raises(ValueError, match="RO RI ATLAS"):
        atlas.assert_no_leak(a, ["val1", "c2"])            # c2 da dung => phai no


def test_atlas_domain_selects_articular_region_on_heldout_case():
    """Atlas xay tu ca TRAIN phai khoanh dung vung khop cua mot ca CHUA TUNG THAY.

    Day la phep thu that su cua D1: no co tong quat hoa duoc khong, hay chi thuoc long
    cac ca da xay.
    """
    cfg = RayConfig()
    train = {f"tr{i}": _knee_like(shift_mm=0.4 * i, seed=i) for i in range(5)}
    a = atlas.build_articular_atlas(list(train), _loader_factory(train), SP, cfg,
                                    n_bins=16, min_count=1)

    # Ca held-out: cung kieu hinh, tinh tien khac
    _, bone_h, cart_h = _knee_like(shift_mm=2.0, seed=99)
    atlas.assert_no_leak(a, ["heldout"])
    _, verts, normals = core.bone_geometry(bone_h, SP, cfg)
    dom = atlas.atlas_domain(verts, a, thr=0.2)

    occ = core.occupancy_target(cart_h, verts, normals, cfg, SP)
    truth = core.oracle_domain(verts, occ, cfg)

    assert dom.any(), "atlas khong chon duoc node nao"
    assert not dom.all(), "atlas chon TAT CA node - khong loc duoc gi"
    # Atlas la bo sinh ung vien THIEN VE RECALL (core.py) => uu tien phu kin vung that
    recall = (dom & truth).sum() / max(truth.sum(), 1)
    assert recall > 0.5, f"atlas bo sot qua nhieu vung khop that: recall {recall:.2f}"


def test_min_count_leaves_sparse_cells_undefined():
    """O luoi it quan sat phai la NaN - mot ca ca biet khong duoc tu tao ra vung atlas."""
    cases = {"only": _knee_like()}
    a = atlas.build_articular_atlas(list(cases), _loader_factory(cases), SP,
                                    RayConfig(), n_bins=32, min_count=5)
    assert np.isnan(a.prob).any(), "min_count khong chan duoc o thua du lieu"
    # query van phai an toan (NaN -> 0)
    _, bone, _ = _knee_like()
    _, verts, _ = core.bone_geometry(bone, SP)
    p = a.query(verts, atlas.fit_frame(verts))
    assert np.isfinite(p).all() and (p >= 0).all()
