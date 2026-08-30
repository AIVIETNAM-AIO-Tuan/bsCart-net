"""Kiem cac phep do headroom ngu canh be mat, tren phantom - chay local, khong can du lieu."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bsc import atlas as atlas_mod
from bsc import core, experiment as X, model, surface_context as SC
from bsc.core import RayConfig
from test_mvp import SP_REAL, PhantomSource


@pytest.fixture(scope="module")
def src():
    return PhantomSource(n=4)


@pytest.fixture(scope="module")
def atlas_fold(src):
    def loader(cid):
        return src.mri(cid), src.bone_gt(cid), src.cart_gt(cid)
    return atlas_mod.build_articular_atlas(src.ids()[:3], loader, SP_REAL, RayConfig(),
                                           n_bins=12, min_count=1, fold=0)


def test_surface_neighbors_shape_and_excludes_self():
    v = np.random.default_rng(0).random((50, 3)).astype(np.float32) * 10
    nb = SC.surface_neighbors(v, k=5)
    assert nb.shape == (50, 5)
    assert not any(i in nb[i] for i in range(50)), "lang gieng khong duoc chua chinh no"


def test_absence_coherence_detects_clustered_vs_random():
    """Vang sun THANH MANG phai cho lift cao; ngau nhien phai cho lift ~1."""
    rng = np.random.default_rng(0)
    v = np.stack(np.meshgrid(*[np.arange(12)] * 2, indexing="ij"), -1).reshape(-1, 2)
    v = np.column_stack([v, np.zeros(len(v))]).astype(np.float32)
    nb = SC.surface_neighbors(v, k=4)

    clustered = np.ones(len(v), bool)
    clustered[v[:, 0] < 5] = False                       # mot mang lien tuc bi absent
    scattered = np.ones(len(v), bool)
    scattered[rng.choice(len(v), (~clustered).sum(), replace=False)] = False

    lc = SC.absence_coherence(clustered, nb)["lift"]
    ls = SC.absence_coherence(scattered, nb)["lift"]
    assert lc > ls, f"mang ({lc:.2f}) phai co lift cao hon ngau nhien ({ls:.2f})"
    assert lc > 1.5, "vang sun thanh mang phai cho lift ro rang"


def test_neighbor_oracle_high_when_spatially_smooth():
    """Presence lien tuc theo khong gian => bo phieu lang gieng GT doan tot."""
    v = np.stack(np.meshgrid(*[np.arange(12)] * 2, indexing="ij"), -1).reshape(-1, 2)
    v = np.column_stack([v, np.zeros(len(v))]).astype(np.float32)
    nb = SC.surface_neighbors(v, k=4)
    smooth = v[:, 0] >= 6                                # nua co sun, nua khong
    r = SC.neighbor_oracle_presence(smooth, nb)
    assert r["acc"] > 0.85, f"truong lien tuc ma oracle chi {r['acc']:.2f}"


def test_fp_isolation_distinguishes_scattered_from_patch():
    """FP roi rac => nb_correct cao; FP thanh mang => thap."""
    v = np.stack(np.meshgrid(*[np.arange(14)] * 2, indexing="ij"), -1).reshape(-1, 2)
    v = np.column_stack([v, np.zeros(len(v))]).astype(np.float32)
    nb = SC.surface_neighbors(v, k=6)
    truth = np.zeros(len(v), bool)                       # tat ca deu absent

    rng = np.random.default_rng(1)
    scattered = np.zeros(len(v), bool)
    scattered[rng.choice(len(v), 12, replace=False)] = True
    patch = (v[:, 0] < 4) & (v[:, 1] < 4)                # mang FP lien tuc

    iso_s = SC.fp_isolation(truth, scattered, nb)["nb_correct_around_fp"]
    iso_p = SC.fp_isolation(truth, patch, nb)["nb_correct_around_fp"]
    assert iso_s > iso_p, f"FP roi rac ({iso_s:.2f}) phai cao hon FP mang ({iso_p:.2f})"
    assert SC.fp_isolation(truth, truth, nb)["n_fp"] == 0     # khong co FP


def test_smooth_presence_alpha_endpoints():
    p = np.array([1.0, 0.0, 1.0, 0.0], np.float32)
    nb = np.array([[1, 2], [0, 3], [0, 3], [1, 2]])
    assert np.allclose(SC.smooth_presence(p, nb, 0.0), p)          # alpha=0 giu nguyen
    s1 = SC.smooth_presence(p, nb, 1.0)
    assert np.allclose(s1, p[nb].mean(axis=1))                     # alpha=1 = TB lang gieng
    s5 = SC.smooth_presence(p, nb, 0.5)
    assert ((s5 >= np.minimum(p, s1)) & (s5 <= np.maximum(p, s1))).all()


def test_surface_context_case_end_to_end(src, atlas_fold):
    """Chay tron ven tren phantom, tra du 4 phep do."""
    from bsc import mvp
    run = X.from_plan("P2", "femoral_cart", seed=1)
    res = mvp.train_run(run, src, src.ids()[:3], src.ids()[3:], RayConfig(),
                        atlas=atlas_fold, rays_per_case=400, epochs=3)
    r = SC.surface_context_case(run, res.net, src, src.ids()[3], RayConfig(),
                                atlas=atlas_fold, k=6, alphas=(0.0, 0.5, 1.0))
    assert r is not None
    for key in ("coherence", "nb_oracle", "fp_iso", "sweep"):
        assert key in r
    assert set(r["sweep"]) == {"0.00", "0.50", "1.00"}
    # BAT BIEN: thin_err = NaN KHI VA CHI KHI du doan rong. NaN la ket qua CO NGHIA
    # (lam muot qua tay xoa sach du doan), khong phai loi - ghi lai qua n_pred_vox.
    for a, v in r["sweep"].items():
        assert np.isfinite(v["thin_err_mm"]) or v["n_pred_vox"] == 0, \
            f"alpha {a}: NaN nhung van co {v['n_pred_vox']} voxel du doan"
        assert 0.0 <= v["presence_f1"] <= 1.0

    s = SC.summarize_surface_context([r], baseline_thin_mm=0.55)
    for key in ("coherence_lift", "nb_oracle_f1", "fp_nb_correct", "best_alpha",
                "smoothing_gain_mm", "signal_exists", "fp_isolated", "smoothing_helps",
                "verdict", "best_vs_baseline_mm", "alphas_wiped_out"):
        assert key in s, f"thieu {key}"
    # best_alpha khong duoc roi vao alpha bi xoa sach
    assert s["best_alpha"] not in s["alphas_wiped_out"]


def test_h0_run_raises_clear_error(src, atlas_fold):
    """Run khong co presence head phai bao loi RO RANG, khong im lang."""
    from bsc import mvp
    run = X.from_plan("P0", "femoral_cart", seed=1)       # H0 - khong presence
    res = mvp.train_run(run, src, src.ids()[:2], src.ids()[3:], RayConfig(),
                        atlas=None, rays_per_case=200, epochs=2)
    with pytest.raises(ValueError, match="presence head"):
        SC.surface_context_case(run, res.net, src, src.ids()[3], RayConfig(), atlas=None)


def test_with_neighbors_concatenates_correctly():
    """Ghep dac trung: [N, K, C] + n lang gieng -> [N, (1+n)*K*C], dung thu tu."""
    X = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    nb = np.array([[1, 2], [0, 2], [0, 1], [0, 1]])
    out = SC._with_neighbors(X, nb, n_use=2)
    assert out.shape == (4, 3 * 2 * 3)
    flat = X.reshape(4, -1)
    assert np.array_equal(out[:, :6], flat)                 # phan dau = chinh node
    assert np.array_equal(out[0, 6:12], flat[1])            # lang gieng 1 cua node 0
    assert np.array_equal(out[0, 12:], flat[2])             # lang gieng 2


def test_knn_presence_recovers_separable_classes():
    """k-NN phai tach duoc hai cum ro rang - kiem ham hoat dong dung."""
    rng = np.random.default_rng(0)
    Xtr = np.concatenate([rng.normal(0, 0.3, (60, 4)), rng.normal(5, 0.3, (60, 4))])
    ytr = np.array([0] * 60 + [1] * 60, np.int8)
    Xte = np.concatenate([rng.normal(0, 0.3, (20, 4)), rng.normal(5, 0.3, (20, 4))])
    yte = np.array([0] * 20 + [1] * 20, bool)
    pred = SC._knn_presence(Xtr, ytr, Xte, k=5)
    assert (pred == yte).mean() > 0.95


def test_raw_context_knn_end_to_end(src, atlas_fold):
    """raw_context_knn chay tron ven va tra du truong quyet dinh."""
    from bsc import mvp
    run = X.from_plan("P2", "femoral_cart", seed=1)
    r = SC.raw_context_knn(run, src, src.ids()[:3], src.ids()[3:], RayConfig(),
                           atlas=atlas_fold, k_nb=6, n_use=2, k_knn=5,
                           rays_per_case=200, verbose=False)
    for key in ("single_ray", "with_neighbors", "gain_acc", "gain_f1",
                "majority_baseline", "context_adds_info", "features_informative",
                "n_train_rays", "n_val_rays"):
        assert key in r, f"thieu {key}"
    for side in ("single_ray", "with_neighbors"):
        assert 0.0 <= r[side]["acc"] <= 1.0
    assert r["n_train_rays"] > 0 and r["n_val_rays"] > 0


def test_channel_information_knn_compares_subsets(src, atlas_fold):
    """So nhieu bo kenh trong MOT luot build; tra du truong quyet dinh."""
    r = SC.channel_information_knn(
        src, src.ids()[:3], src.ids()[3:], "femoral_cart", RayConfig(), atlas=atlas_fold,
        subsets={"mri": ("mri",), "mri+grad": ("mri", "grad")},
        k_nb=4, n_use=2, k_knn=5, rays_per_case=150, verbose=False)
    assert set(r["by_channels"]) == {"mri", "mri+grad"}
    for name, v in r["by_channels"].items():
        for tag in ("single", "with_nb"):
            assert 0.0 <= v[tag]["acc"] <= 1.0, f"{name}/{tag} acc bat thuong"
        assert "lift_vs_majority" in v and "gain_from_neighbors" in v
    for key in ("majority_baseline", "best_subset", "best_lift", "best_gain_nb",
                "any_features_informative", "neighbors_add_info"):
        assert key in r, f"thieu {key}"
    assert r["best_subset"] in r["by_channels"]


def test_i6_channels_are_union_of_i1_i2():
    """I6 (chan doan) phai la hop cua I1 va I2, thu tu mri,grad,sdf."""
    assert X.INPUT["I6"] == ("mri", "grad", "sdf")
    assert set(X.INPUT["I6"]) == set(X.INPUT["I1"]) | set(X.INPUT["I2"])
    # I6 KHONG duoc nam trong ma tran ke hoach (chi la chan doan)
    assert all(v[2] != "I6" for v in X.PLAN_MATRIX.values() if v[2])


def test_summarize_empty():
    assert SC.summarize_surface_context([])["n"] == 0
    assert SC.summarize_surface_context([None])["n"] == 0
