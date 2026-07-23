"""Test TICH HOP: chay dung luong ma notebook chay, tren phantom, o may local.

MUC DICH: bat het lop bug "chi lo ra khi chay tren Colab". Truoc day build_for/
train_eval/domain_for nam trong notebook nen khong ai chay chung truoc; ba bug da di
qua duong do (spacing hardcode, atlas index tran, bien chua ton tai). Test nay dam bao
moi to hop S/D/I/H deu chay tron ven TRUOC khi ban day len Colab.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bsc import atlas as atlas_mod
from bsc import core, experiment as X, model, mvp
from bsc.core import RayConfig

# Spacing CO Y dat truc 0.70mm o CUOI - giong du lieu that (Dataset001), KHAC
# core.SPACING. Neu code nao lo dung mac dinh thay vi spacing theo ca, test se lech.
SP_REAL = (0.3646, 0.3646, 0.70)


def _phantom(shape=(90, 90, 34), spacing=SP_REAL, t_mm=2.0, cap=0.30,
             shift=0.0, seed=0, erode_pred=True):
    """Xuong ellipsoid bi cat + vo sun mot phia + vung MAT sun. Kem 'du doan' xau hon GT."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(*[np.arange(s) * sp for s, sp in zip(shape, spacing)],
                             indexing="ij")
    c = [s * sp / 2 for s, sp in zip(shape, spacing)]
    c[0] += shift
    rz, ry, rx = 13.0, 11.0, 8.0
    q = ((zz - c[0]) / rz) ** 2 + ((yy - c[1]) / ry) ** 2 + ((xx - c[2]) / rx) ** 2
    bone = (q <= 1.0) & (yy > c[1] - 0.45 * ry)
    shell = (q > 1.0) & (q <= (1.0 + t_mm / rx) ** 2) & (yy > c[1] - 0.45 * ry)
    cart = shell & (zz > c[0] + (1 - cap) * rz)

    mri = np.full(shape, 0.3, np.float32)
    mri[bone] = 0.1
    mri[cart] = 0.9
    mri += rng.normal(0, 0.05, shape).astype(np.float32)

    # "Baseline": sun bi an mon + xuong hoi khac => co loi de do cai thien
    cart_pred = cart & (zz > c[0] + (1 - cap * 0.8) * rz)
    bone_pred = (q <= 1.02) & (yy > c[1] - 0.45 * ry)
    prob = cart_pred.astype(np.float32) * 0.85 + 0.02
    return dict(mri=mri, bone=bone, cart=cart, bone_pred=bone_pred,
                cart_pred=cart_pred, prob=prob)


class PhantomSource(mvp.CaseSource):
    """Nguon du lieu gia - cung giao dien voi NiftiCaseSource, khong can nibabel."""

    def __init__(self, n=4, spacing=SP_REAL, seed=0):
        self.sp = spacing
        self.cases = {f"c{i}": _phantom(spacing=spacing, shift=0.4 * i, seed=seed + i)
                      for i in range(n)}

    def ids(self):            return sorted(self.cases)
    def spacing(self, cid):   return self.sp
    def mri(self, cid):       return self.cases[cid]["mri"]
    def bone_gt(self, cid):   return self.cases[cid]["bone"]
    def cart_gt(self, cid):   return self.cases[cid]["cart"]
    def bone_pred(self, cid): return self.cases[cid]["bone_pred"]
    def prob(self, cid):      return self.cases[cid]["prob"]
    def baseline_cart(self, cid): return self.cases[cid]["cart_pred"]


@pytest.fixture(scope="module")
def src():
    return PhantomSource(n=4)


@pytest.fixture(scope="module")
def atlas_fold(src):
    def loader(cid):
        return src.mri(cid), src.bone_gt(cid), src.cart_gt(cid)
    return atlas_mod.build_articular_atlas(src.ids()[:3], loader, SP_REAL, RayConfig(),
                                           n_bins=12, min_count=1, fold=0)


# ------------------------------------------------------------------ luong chinh

@pytest.mark.parametrize("alias", ["P0", "P1", "P2"])
def test_plan_matrix_runs_end_to_end(alias, src, atlas_fold):
    """P0/P1 (D0,H0) va P2 (D1,H1) deu phai chay tron ven: dataset -> train -> eval."""
    run = X.from_plan(alias, "femoral_cart", seed=1)
    res = mvp.train_run(run, src, src.ids()[:3], src.ids()[3:], RayConfig(),
                        atlas=atlas_fold, rays_per_case=400, epochs=3)
    assert res.n_train_rays > 0
    assert 0.0 <= res.val_occ_dice <= 1.0
    # Truc H phai the hien dung: H0 khong co presence, H1 co
    if run.with_presence:
        assert res.presence is not None and 0.0 <= res.presence["f1"] <= 1.0
    else:
        assert res.presence is None


def test_p3_needs_prob_and_runs(src, atlas_fold):
    """P3 (=M8-D) dung kenh prob; phai chay va mang dung vai tro ablation."""
    run = X.from_plan("P3", "femoral_cart", seed=1,
                      prob_source=X.DEFAULT_PROB_SOURCE)
    assert run.m8_role == "M8-D"
    assert "prob" in run.channels
    res = mvp.train_run(run, src, src.ids()[:3], src.ids()[3:], RayConfig(),
                        atlas=atlas_fold, rays_per_case=400, epochs=3)
    assert res.n_train_rays > 0


def test_d1_without_atlas_fails_loudly(src):
    """Quen truyen atlas cho run D1 phai BAO LOI, khong duoc am tham roi ve oracle."""
    run = X.from_plan("P2", "femoral_cart")
    with pytest.raises(ValueError, match="can `atlas`"):
        mvp.build_dataset(run, src, src.ids()[:1], RayConfig(), atlas=None,
                          rays_per_case=100)


def test_s2_uses_predicted_bone_not_gt(src, atlas_fold):
    """S2 phai dung xuong DU DOAN. Kiem bang so tia khac han so voi S0."""
    cfg = RayConfig()
    r0 = X.from_plan("P2", "femoral_cart")
    r5 = X.from_plan("P5", "femoral_cart", inputs="I2")
    assert r5.surface == "S2"
    n0 = len(mvp.build_case(r0, src, "c0", cfg, atlas_fold)[0])
    n5 = len(mvp.build_case(r5, src, "c0", cfg, atlas_fold)[0])
    assert n0 > 0 and n5 > 0
    assert n0 != n5, "S2 cho ket qua y het S0 => co the dang dung nham xuong GT"


def test_s1_applies_jitter_by_default(src, atlas_fold):
    """S1 phai TU DONG ap jitter - cau hinh va hanh vi khong duoc lech nhau."""
    cfg = RayConfig()
    r4 = X.from_plan("P4", "femoral_cart", inputs="I2")
    assert r4.surface == "S1"
    X4 = mvp.build_case(r4, src, "c0", cfg, atlas_fold, seed=0)[0]
    X2 = mvp.build_case(X.from_plan("P2", "femoral_cart"), src, "c0", cfg,
                        atlas_fold, seed=0)[0]
    assert X4.shape[1:] == X2.shape[1:]
    assert not np.allclose(X4[:min(len(X4), len(X2))], X2[:min(len(X4), len(X2))]), \
        "S1 khong khac S0 => jitter chua duoc ap"


def test_m6_direction_variants_all_run(src, atlas_fold):
    """M6: moi huong doi chung phai chay duoc (co the te hon, nhung khong duoc VO)."""
    run = X.from_plan("P2", "femoral_cart")
    for mode in ("normal", "axial", "tangent", "random"):
        Xa, oa, pa = mvp.build_dataset(run, src, src.ids()[:2], RayConfig(),
                                       atlas=atlas_fold, rays_per_case=200,
                                       direction=mode)
        assert len(Xa) > 0, f"huong {mode} khong sinh duoc tia nao"
        assert np.isfinite(Xa).all(), f"huong {mode} sinh ra NaN/Inf"


def test_eval_and_gate37_summary(src, atlas_fold):
    """Danh gia per-case + tong hop cong §3.7 phai chay va tra du truong."""
    run = X.from_plan("P2", "femoral_cart", seed=1)
    res = mvp.train_run(run, src, src.ids()[:3], src.ids()[3:], RayConfig(),
                        atlas=atlas_fold, rays_per_case=400, epochs=3)
    rows = [mvp.evaluate_case(run, res.net, src, cid, RayConfig(), atlas_fold)
            for cid in src.ids()]
    rows = [r for r in rows if r]
    assert rows, "khong danh gia duoc ca nao"
    assert all("thin_ray" in r and "thin_baseline" in r for r in rows)

    g = mvp.summarize_gate37(rows)
    for k in ("n", "thin_err_baseline_mm", "thin_err_ray_mm", "rel_improve",
              "abs_improve_mm", "ci", "pass_10pct", "ci_low_positive"):
        assert k in g, f"thieu truong {k}"
    assert g["n"] == len(rows)
    assert np.isfinite(g["thin_err_baseline_mm"])


def test_qc_keys_are_a_frozen_contract():
    """Khoa QC la HOP DONG BEN VUNG - checkpoint tren Drive dung dung ten nay.

    Doi ten = pha resume cua nguoi dung (da xay ra: KeyError 'm3'). Test nay ghim lai
    de lan sau doi ten se lam do test o local, KHONG phai do tren Colab.
    """
    assert mvp.QC_KEYS == ("m3_normals_ok", "m4_single_interval", "m2_dice",
                           "m2_assd_mm", "frame_degenerate")


def test_qc_summary_survives_legacy_checkpoint_rows(src, capsys):
    """Ban ghi thieu khoa (checkpoint ban cu) phai bi BO QUA co bao, khong KeyError."""
    good = mvp.geometry_qc_case(src, "c0", "femoral_cart",
                                src.bone_gt("c0"), src.cart_gt("c0"), RayConfig())
    legacy = {"case": "cX", "cls": "femoral_cart", "m3": 0.99, "m4": 0.99}  # ten CU

    s = mvp.summarize_geometry_qc([good, legacy], "femoral_cart")
    assert s is not None and s["n"] == 1, "phai dung duoc ban ghi hop le"
    assert "bo qua" in capsys.readouterr().out

    assert mvp.summarize_geometry_qc([legacy], "femoral_cart") is None
    assert mvp.summarize_geometry_qc([], "femoral_cart") is None


def test_qc_gate_logic(src):
    """Cong QC phai bat dung tung tieu chi."""
    base = {"m3_normals_ok": 0.998, "m4_single_interval": 0.99, "m2_dice": 0.99,
            "m2_assd_mm": 0.01, "frame_degenerate": 0.0}
    assert mvp.geometry_qc_gate(base)["pass"]
    assert not mvp.geometry_qc_gate({**base, "m3_normals_ok": 0.99})["pass"]
    assert not mvp.geometry_qc_gate({**base, "m4_single_interval": 0.80})["pass"]
    assert not mvp.geometry_qc_gate({**base, "m2_assd_mm": 0.5})["pass"]
    assert not mvp.geometry_qc_gate({**base, "frame_degenerate": 0.2})["pass"]


def test_geometry_qc_case_returns_full_contract(src):
    """geometry_qc_case phai tra DU moi khoa trong hop dong."""
    r = mvp.geometry_qc_case(src, "c0", "femoral_cart", src.bone_gt("c0"),
                             src.cart_gt("c0"), RayConfig())
    for k in mvp.QC_KEYS:
        assert k in r, f"thieu khoa {k}"
    assert r["case"] == "c0" and r["cls"] == "femoral_cart"
    assert 0.0 <= r["m3_normals_ok"] <= 1.0
    assert r["m2_assd_mm"] >= 0.0


def test_spacing_is_per_case_not_module_default(src, atlas_fold):
    """Spacing PHAI lay theo ca. Doi spacing cua nguon => hinh hoc phai doi theo.

    Neu o dau do con hardcode core.SPACING, hai lan goi se ra ket qua giong nhau va
    test nay bat duoc.
    """
    cfg = RayConfig()
    run = X.from_plan("P0", "femoral_cart")
    a = mvp.build_case(run, src, "c0", cfg, None, seed=0)[0]

    src2 = PhantomSource(n=1, spacing=(0.5, 0.5, 0.5))     # isotropic - khac han
    b = mvp.build_case(run, src2, "c0", cfg, None, seed=0)[0]
    assert a.shape[0] != b.shape[0], \
        "doi spacing khong lam doi hinh hoc => co cho dang dung spacing mac dinh"


def test_checkpoint_roundtrip_is_numerically_identical(src, atlas_fold, tmp_path):
    """§4.3: Step 0 KHONG doi numerics. Reload checkpoint phai cho du doan Y HET.

    Day la bang chung he checkpoint chi la bookkeeping, khong dung vao ket qua.
    """
    run = X.from_plan("P2", "femoral_cart", seed=1)
    res = mvp.train_run(run, src, src.ids()[:3], src.ids()[3:], RayConfig(),
                        atlas=atlas_fold, rays_per_case=400, epochs=3)

    run_dir = mvp.save_run(res, str(tmp_path), RayConfig(), epochs=3, lr=3e-4,
                           rays_per_case=400, atlas_hash="a1", split_hash="s1")
    net2 = mvp.load_run_model(run_dir)

    # Du doan tren cung input phai TRUNG tuyet doi
    Xb, _, _ = mvp.build_dataset(run, src, src.ids()[3:], RayConfig(),
                                 atlas=atlas_fold, rays_per_case=400, seed=7)
    o1, p1 = model.predict_rays(res.net, Xb)
    o2, p2 = model.predict_rays(net2, Xb)
    assert np.array_equal(o1, o2), "reload lam doi occupancy prediction"
    assert np.array_equal(p1, p2), "reload lam doi presence prediction"


def test_save_run_paths_and_manifest(src, atlas_fold, tmp_path):
    """Duong dan chua checkpoint_hash; manifest ghi du truong audit (§7.1)."""
    run = X.from_plan("P2", "femoral_cart", seed=1)
    res = mvp.train_run(run, src, src.ids()[:3], src.ids()[3:], RayConfig(),
                        atlas=atlas_fold, rays_per_case=400, epochs=2)
    run_dir = mvp.save_run(res, str(tmp_path), RayConfig(), epochs=2, lr=3e-4,
                           rays_per_case=400, atlas_hash="ah", split_hash="sh",
                           dataset_revision="d001")

    assert os.path.exists(f"{run_dir}/model.pt")
    assert run.experiment_id in run_dir
    m = mvp.run_manifest(run_dir)
    assert m["checkpoint_sha256"] in run_dir          # hash nam trong duong dan
    for k in ("experiment_id", "git_commit", "config_hash", "atlas_hash",
              "split_manifest_hash", "checkpoint_sha256", "epochs", "val_occ_dice"):
        assert k in m, f"manifest thieu {k}"
    assert m["atlas_hash"] == "ah" and m["split_manifest_hash"] == "sh"


def test_different_weights_give_different_checkpoint_hash(src, atlas_fold, tmp_path):
    """Model khac weight => ckpt_hash khac => khong the vo tinh reuse eval cu."""
    ids_tr, ids_va = src.ids()[:3], src.ids()[3:]
    run = X.from_plan("P2", "femoral_cart", seed=1)
    d1 = mvp.save_run(mvp.train_run(run, src, ids_tr, ids_va, RayConfig(), atlas=atlas_fold,
                                    rays_per_case=300, epochs=2), str(tmp_path / "a"),
                      RayConfig(), epochs=2, lr=3e-4, rays_per_case=300)
    run2 = X.from_plan("P2", "femoral_cart", seed=2)   # seed khac => weight khac
    d2 = mvp.save_run(mvp.train_run(run2, src, ids_tr, ids_va, RayConfig(), atlas=atlas_fold,
                                    rays_per_case=300, epochs=2), str(tmp_path / "b"),
                      RayConfig(), epochs=2, lr=3e-4, rays_per_case=300)
    assert os.path.basename(d1) != os.path.basename(d2), "hai model khac cho cung hash"


def test_config_hash_sensitive_and_deterministic():
    """config_hash: cung input => cung hash; doi bat ky yeu to numerics => hash khac."""
    run = X.from_plan("P2", "femoral_cart", seed=1)
    cfg = RayConfig()
    base = dict(epochs=30, lr=3e-4, rays_per_case=20000, batch_size=4096)
    h0 = mvp.config_hash(run, cfg, **base)
    assert h0 == mvp.config_hash(run, cfg, **base)                       # deterministic
    assert h0 != mvp.config_hash(run, cfg, **{**base, "epochs": 60})     # doi epoch
    assert h0 != mvp.config_hash(run, cfg, **{**base, "lr": 1e-4})       # doi lr
    assert h0 != mvp.config_hash(run, RayConfig(k=48), **base)           # doi RayConfig
    assert h0 != mvp.config_hash(X.from_plan("P1", "femoral_cart", seed=1), cfg, **base)


def test_mapping_audit_returns_both_mappings(src):
    """Mapping audit: tra du so cho ca A (node) va B (ray) + QC khoang cach."""
    mri, bone, cart = src.cases["c0"]["mri"], src.bone_gt("c0"), src.cart_gt("c0")
    pred = cart.copy()
    pred[:, :, cart.shape[2] // 2:] = False        # tao loi de co surface points

    r = mvp.mapping_audit_case(cart, pred, bone, SP_REAL, RayConfig())
    assert r is not None
    for k in ("thin_err_A_mm", "thin_err_B_mm", "n_thin_surf_A", "n_thin_surf_B",
              "gt_vox_absent_A", "gt_vox_absent_B", "unassigned_frac_B",
              "distB_median_mm", "distB_p95_mm", "max_dist_mm"):
        assert k in r, f"thieu {k}"
    assert np.isfinite(r["thin_err_A_mm"]) and np.isfinite(r["thin_err_B_mm"])
    assert 0.0 <= r["unassigned_frac_B"] <= 1.0
    assert r["distB_median_mm"] <= r["distB_p95_mm"]


def test_nearest_ray_respects_max_dist():
    """Query xa hon max_dist phai thanh NaN (unassigned), khong ep vao bin."""
    verts = np.array([[0, 0, 0], [0, 0, 5]], np.float32)
    dirs = np.array([[1, 0, 0], [1, 0, 0]], np.float32)
    thick = np.array([2.0, 3.0], np.float32)
    cfg = RayConfig()
    far = np.array([[0, 0, 100]], np.float32)       # cach xa moi ray sample
    th, d = mvp._nearest_ray_thickness(far, verts, dirs, thick, cfg, max_dist_mm=1.0)
    assert np.isnan(th[0]), "diem xa phai unassigned"
    near = np.array([[cfg.d_min, 0, 0]], np.float32)  # ngay tren ray 0
    th2, d2 = mvp._nearest_ray_thickness(near, verts, dirs, thick, cfg, max_dist_mm=1.0)
    assert th2[0] == 2.0 and d2[0] < 1.0


def test_default_max_dist_positive():
    md = mvp.default_max_dist_mm(RayConfig(), SP_REAL)
    assert md > 0 and md < 5.0        # nua duong cheo voxel + nua buoc ray, hop ly


def test_domain_computed_before_jitter(src, atlas_fold):
    """Mien khop phai tinh tren hinh hoc CHUA jitter - no la vung giai phau.

    Neu tinh sau jitter, moi muc jitter lai cho mot domain khac va so sanh M7 tro nen
    lan lon giua 'mo hinh kem hon' voi 'do tren tap tia khac'.
    """
    cfg = RayConfig()
    run = X.from_plan("P2", "femoral_cart")
    n_no = len(mvp.build_case(run, src, "c0", cfg, atlas_fold,
                              jitter_s_mm=0.0, seed=0)[0])
    n_jit = len(mvp.build_case(run, src, "c0", cfg, atlas_fold,
                               jitter_s_mm=1.0, jitter_theta_deg=15.0, seed=0)[0])
    assert n_no == n_jit, "so tia doi theo jitter => domain dang tinh SAU khi jitter"
