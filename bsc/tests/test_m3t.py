"""Test bsc/m3t.py - CPU, cau hinh THU NHO, khong can du lieu that.

    python -m pytest bsc/tests/test_m3t.py -v

Chi phi 2D co dinh theo N x target_size^2 (cau hinh that mat ~3 s CPU ke ca dau vao 8^3)
=> test thu nho CAU HINH (C3d=4, N=4, emb=16, C2d=8, target=12), khong chi dau vao.
Kiem port tren trong so that (tai lap softmax ghi trong knee_testing_v3.ipynb) chay ngoai
bo test vi can file trong so + zip 29.5 GB - xem Event.md 2026-09-25.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bsc import m3t

TINY = dict(C3d=4, N=4, emb_dim=16, C2d=8, target_size=12)
# nguong THAT nam o configs/m3t_s9.json; o day la dau vao cua test (so khop duoc kiem rieng)
IMG_TH = dict(r_median=0.99, r_p01=0.97, margin=0.1, slope_lo=0.95, slope_hi=1.05)
CLS_TH = dict(nn_self=0.98, dist_ratio=0.25, head_agree=0.95, d_ekl=0.1)


def tiny(seed=0, **kw):
    torch.manual_seed(seed)
    return m3t.build_m3t(**{**TINY, **kw}).eval()


class _OriginalForward(m3t.M3TModelFull):
    """forward NGUYEN VAN cell 18 (truoc khi tach forward_features) - chuan de so."""

    def forward(self, x):
        B = x.shape[0]
        X = self.d3d(x)
        S = self.extractor(X)
        B, T, C, L_s, _ = S.shape
        S_flat = S.reshape(B*T, C, L_s, L_s)
        K = self.d2d(S_flat)
        T_proj = self.projection(K)
        T_proj = T_proj.reshape(B, 3*self.N, -1)
        Tcor = T_proj[:, :self.N, :]
        Tsag = T_proj[:, self.N:2*self.N, :]
        Tax  = T_proj[:, 2*self.N:, :]
        Z0 = self.pos_plane(Tcor, Tsag, Tax)
        Z = self.transformer(Z0)
        cls_token = Z[:, 0, :]
        logits = self.fc(cls_token)
        return logits


# ------------------------------------------------------------ 1-3. mo hinh

def test_real_config_param_count_and_embedding_shape_without_forward():
    m = m3t.build_m3t()
    assert m3t.count_params(m) == m3t.N_PARAMS == 1_160_677
    assert tuple(m.pos_plane.pos_embedding.shape) == (1, 3 * 20 + 4, 128)
    assert m.fc.out_features == 5 and m.extractor.target_size == 160
    with pytest.raises(ValueError, match="khong ton tai"):
        m3t.build_m3t(depth=3)


def test_forward_equals_fc_of_forward_features_and_the_original_forward():
    m = tiny()
    ref = _OriginalForward(**{**m3t.M3T_CFG, **TINY}).eval()
    ref.load_state_dict(m.state_dict(), strict=True)
    x = torch.rand(2, 1, 7, 9, 11)
    with torch.no_grad():
        f = m.forward_features(x)
        assert f.shape == (2, TINY["emb_dim"])
        assert torch.equal(m(x), m.fc(f))
        assert torch.equal(m(x), ref(x))


def test_tiny_config_runs_on_odd_shapes_and_axis_shorter_than_n():
    m = tiny()
    with torch.no_grad():
        assert m(torch.rand(1, 1, 5, 13, 6)).shape == (1, 5)
        assert m(torch.rand(1, 1, 3, 10, 9)).shape == (1, 5)          # truc 3 < N=4: chi so lap


def test_target_size_none_breaks_on_non_cubic_input():
    m = tiny(target_size=None)
    with torch.no_grad():
        assert m(torch.rand(1, 1, 8, 8, 8)).shape == (1, 5)             # lap phuong thi chay
        with pytest.raises(RuntimeError):
            m(torch.rand(1, 1, 6, 8, 10))


# ------------------------------------------------------------ 4. nap + hash + chan ro ri

def test_load_raw_and_wrapped_state_dict_strict_with_same_hash(tmp_path):
    m = tiny(seed=1)
    sd = m.state_dict()
    torch.save(sd, tmp_path / "raw.pth")
    torch.save({"epoch": 3, "model_state_dict": sd, "best_val_acc": 0.5}, tmp_path / "wrap.pth")
    a, ha = m3t.load_m3t(tmp_path / "raw.pth", **TINY)
    b, hb = m3t.load_m3t(tmp_path / "wrap.pth", **TINY)
    assert ha == hb == m3t.weights_hash(sd) and len(ha) == 64
    assert not a.training and a.weights_hash == ha
    x = torch.rand(1, 1, 6, 6, 6)
    with torch.no_grad():
        assert torch.equal(a(x), b(x))
    other = tiny(seed=2)
    assert m3t.weights_hash(other.state_dict()) != ha
    bad = dict(sd)
    bad.pop("fc.bias")
    torch.save(bad, tmp_path / "bad.pth")
    with pytest.raises(RuntimeError, match="Missing key"):
        m3t.load_m3t(tmp_path / "bad.pth", **TINY)


def test_leaky_hash_is_blocked_unless_explicitly_allowed(tmp_path, monkeypatch):
    m = tiny(seed=3)
    torch.save(m.state_dict(), tmp_path / "w.pth")
    h = m3t.weights_hash(m.state_dict())
    monkeypatch.setattr(m3t, "LEAKY_HASHES", frozenset({h}))
    with pytest.raises(PermissionError, match="ro ri"):
        m3t.load_m3t(tmp_path / "w.pth", **TINY)
    model, _ = m3t.load_m3t(tmp_path / "w.pth", allow_leaky=True, **TINY)
    vols = [np.random.default_rng(0).random((5, 6, 7))]
    with pytest.raises(PermissionError):
        m3t.extract(model, vols, expect_shape=None)                      # chan ca o buoc trich
    cls, _ = m3t.extract(model, vols, expect_shape=None, allow_leaky=True)
    assert cls.shape == (1, TINY["emb_dim"])


def test_leaky_hashes_are_the_three_known_files():
    assert len(m3t.LEAKY_HASHES) == 3 and all(len(h) == 64 for h in m3t.LEAKY_HASHES)


# ------------------------------------------------------------ 5. minmax01

def test_minmax01_edge_cases():
    v = np.array([[[3, 5], [7, 11]]], dtype=np.uint16)
    out = m3t.minmax01(v)
    assert out.dtype == np.float32 and out.min() == 0.0 and out.max() == 1.0
    assert np.allclose(out.ravel(), [0, 0.25, 0.5, 1.0])
    assert v.tolist() == [[[3, 5], [7, 11]]]                             # khong sua dau vao
    assert (m3t.minmax01(np.full((2, 2, 2), 7)) == 0).all()             # khoi hang -> 0
    neg = m3t.minmax01(np.array([-2.0, 0.0, 2.0]))
    assert neg.tolist() == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="NaN"):
        m3t.minmax01(np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="rong"):
        m3t.minmax01(np.zeros((0, 3)))


def test_minmax01_matches_torchio_arithmetic():
    # torchio RescaleIntensity: mang float32, tru/chia TAI CHO voi can float64 tu np.percentile
    rng = np.random.default_rng(0)
    v = rng.integers(0, 600, size=(6, 7, 8)).astype(np.uint16)
    arr = v.astype(np.float32)
    lo, hi = np.percentile(arr, (0, 100))
    arr -= lo
    arr /= (hi - lo)
    assert np.array_equal(m3t.minmax01(v), arr)


def test_extract_batches_are_independent_in_eval_and_train_mode_is_refused():
    m = tiny(seed=4)
    rng = np.random.default_rng(1)
    vols = [rng.integers(0, 500, (6, 8, 7)).astype(np.uint16) for _ in range(5)]
    c1, l1 = m3t.extract(m, vols, bs=1, expect_shape=None)
    c3, l3 = m3t.extract(m, iter(vols), bs=3, expect_shape=None)
    assert c1.shape == (5, 16) and l1.shape == (5, 5)
    assert np.allclose(c1, c3, atol=1e-5) and np.allclose(l1, l3, atol=1e-5)
    with pytest.raises(ValueError, match="shape"):
        m3t.extract(m, vols[:1], expect_shape=(1, 2, 3))
    m.train()
    with pytest.raises(RuntimeError, match="eval"):
        m3t.extract(m, vols[:1], expect_shape=None)


# ------------------------------------------------------------ 9-11. chuyen doi

def test_orientation_table_and_slice_axis():
    assert len(m3t.ORIENTATIONS) == 48
    assert m3t.slice_axis((0.3646, 0.3646, 0.70)) == 2
    with pytest.raises(ValueError, match="duy nhat"):
        m3t.slice_axis((0.7, 0.7, 0.36))
    o = m3t.orientations_for(2)
    assert len(o) == 16 and all(p[0] == 2 for p, _ in o.values())


@pytest.mark.parametrize("method", m3t.RESIZE_METHODS)
def test_resize3d_output_shape_and_dtype(method):
    out = m3t.resize3d(np.random.default_rng(0).random((9, 13, 11)), (4, 6, 5), method)
    assert out.shape == (4, 6, 5) and out.dtype == np.float32
    with pytest.raises(ValueError):
        m3t.resize3d(np.zeros((3, 3)), (2, 2, 2), method)


def _phantom(shape=(20, 24, 12), seed=0):
    """Khoi co cau truc BAT DOI XUNG (moi huong cho anh khac nhau) + nhieu nho."""
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing="ij")
    v = 300 * z + 150 * y ** 2 + 80 * np.sin(6 * x) + 200 * ((z - 0.3) ** 2 + (y - 0.7) ** 2 < 0.05)
    return (v + rng.normal(scale=3, size=shape)).astype(np.float32)


def test_search_recovers_known_orientation_and_resize():
    src = _phantom()                                    # [Z,Y,X], truc X (cuoi) la truc 0.70
    spacing = (0.3646, 0.3646, 0.70)
    target = (6, 10, 8)
    truth_orient = "p210_f101"
    assert m3t.ORIENTATIONS[truth_orient][0][0] == 2
    npz = m3t.resize3d(m3t.reorient(src, truth_orient), target, "area")
    pairs = [("c0", src, spacing, npz)]
    df = m3t.search_conversion(pairs, shape=target)
    assert len(df) == 16 * len(m3t.RESIZE_METHODS)
    s = m3t.summarize_conversion(df)
    assert s.iloc[0].orient == truth_orient and s.iloc[0].method == "area"
    assert s.iloc[0].r_median > 0.999 and s.iloc[0].margin > 0.1
    exact = m3t.nifti_to_m3t(src, spacing, m3t.spec_name(truth_orient, "area"), shape=target)
    assert np.allclose(exact, npz)
    with pytest.raises(ValueError, match="truc lat cat"):
        m3t.nifti_to_m3t(src, spacing, "p012_f000|area", shape=target)


def test_dst_axis_generalizes_the_slice_axis_assumption(tmp_path):
    src = _phantom()                                      # truc lat cat = 2
    spacing = (0.3646, 0.3646, 0.70)
    target = (6, 10, 8)
    orient = "p021_f010"                                  # truc 2 cua src roi vao truc 1 cua M3T
    assert m3t.dst_axis_of(orient, 2) == 1
    npz = m3t.resize3d(m3t.reorient(src, orient), target, "trilinear")
    df = m3t.search_conversion([("c", src, spacing, npz)], shape=target, dst_axis=1)
    assert m3t.summarize_conversion(df).iloc[0].orient == orient
    spec = m3t.spec_name(orient, "trilinear")
    assert np.allclose(m3t.nifti_to_m3t(src, spacing, spec, shape=target, dst_axis=1), npz)
    with pytest.raises(ValueError, match="truc lat cat"):
        m3t.nifti_to_m3t(src, spacing, spec, shape=target)            # mac dinh dst_axis=0
    wide = m3t.search_conversion([("c", src, spacing, npz)], orients=list(m3t.ORIENTATIONS),
                                 methods=("trilinear",), shape=target)
    assert len(wide) == 48 and m3t.summarize_conversion(wide).iloc[0].orient == orient
    nib = pytest.importorskip("nibabel")
    arr_xyz = np.transpose(src, (2, 1, 0))                # load_nii dao (X,Y,Z) -> (Z,Y,X)
    img = nib.Nifti1Image(arr_xyz, np.diag([0.70, 0.3646, 0.3646, 1.0]))
    nib.save(img, str(tmp_path / "c.nii.gz"))
    out = m3t.convert_case(tmp_path / "c.nii.gz", spec, dst_axis=1, shape=target)
    assert np.allclose(out, npz, atol=1e-4)


def test_pick_best_tag_keeps_the_matching_side_only():
    from scipy.ndimage import gaussian_filter
    src = _phantom()
    spacing = (0.3646, 0.3646, 0.70)
    target = (6, 10, 8)
    right = m3t.resize3d(m3t.reorient(src, "p210_f000"), target, "trilinear")
    other = gaussian_filter(np.random.default_rng(3).normal(size=src.shape), 2).astype(np.float32)
    left = m3t.resize3d(m3t.reorient(other, "p210_f000"), target, "trilinear")
    df = m3t.search_conversion([("c0", src, spacing, right, "R"), ("c0", src, spacing, left, "L")], shape=target)
    assert set(df.tag) == {"R", "L"} and len(df) == 2 * 16 * len(m3t.RESIZE_METHODS)
    keep, verdict = m3t.pick_best_tag(df)
    assert verdict.iloc[0].best_tag == "R" and set(keep.tag) == {"R"}
    assert verdict.iloc[0].r_best > 0.99 > verdict.iloc[0].r_other
    with pytest.raises(ValueError, match="npz shape"):
        m3t.search_conversion([("c0", src, spacing, right[:-1])], shape=target)


def test_summarize_margin_is_against_a_different_orientation():
    import pandas as pd
    df = pd.DataFrame(dict(case_id=["a"] * 3, orient=["p210_f000", "p210_f000", "p201_f000"],
                           method=["area", "trilinear", "area"], r=[0.995, 0.994, 0.80],
                           slope=[1.0, 1.0, 0.9]))
    s = m3t.summarize_conversion(df)
    assert s.iloc[0].spec == "p210_f000|area"
    assert s.iloc[0].margin == pytest.approx(0.195)          # vs huong khac, KHONG vs 'trilinear'
    g = m3t.image_gate(s, IMG_TH)
    assert g["passed"] and g["checks"]["margin"]
    with pytest.raises(KeyError, match="m3t_s9.json"):
        m3t.image_gate(s, dict(r_median=0.99))


def test_volume_corr_invariant_to_linear_intensity_change():
    a = _phantom()
    r, slope = m3t.volume_corr(3.0 * a + 50.0, a)
    assert r == pytest.approx(1.0, abs=1e-6) and slope == pytest.approx(1.0, abs=1e-6)
    r2, _ = m3t.volume_corr(a, a[::-1].copy())
    assert r2 < 0.9
    with pytest.raises(ValueError):
        m3t.volume_corr(a, a[:-1])


# ------------------------------------------------------------ 12. cong CLS

def test_conversion_gate_identical_offset_and_noisy():
    rng = np.random.default_rng(0)
    cls = rng.normal(size=(40, 16))
    lg = rng.normal(size=(40, 5)) * 3
    same = m3t.conversion_gate(cls, cls.copy(), lg, lg.copy(), CLS_TH)
    assert same["passed"] and same["nn_self"] == 1.0 and same["dist_ratio"] == 0.0
    shifted = m3t.conversion_gate(cls + 5.0, cls, lg, lg, CLS_TH)    # lech hang lon hon khoang cach giua ca
    assert not shifted["checks"]["dist_ratio"]
    noisy = m3t.conversion_gate(cls + rng.normal(scale=2.0, size=cls.shape), cls,
                                lg + rng.normal(scale=3.0, size=lg.shape), lg, CLS_TH)
    assert not noisy["passed"] and noisy["nn_self"] < 0.98 and noisy["head_agree"] < 0.95
    with pytest.raises(ValueError):
        m3t.conversion_gate(cls, cls[:-1], lg, lg, CLS_TH)
    with pytest.raises(KeyError):
        m3t.conversion_gate(cls, cls, lg, lg, dict(nn_self=0.98))


# ------------------------------------------------------------ 13. dau van tay

def test_fingerprint_catches_planted_duplicate_but_not_other_knees():
    rng = np.random.default_rng(0)
    bank_vols = [_phantom(shape=(24, 32, 32), seed=s) * rng.uniform(0.5, 2) + rng.normal(scale=40, size=(24, 32, 32))
                 for s in range(12)]
    bank_vols = [np.roll(v, shift=3 * i, axis=1) for i, v in enumerate(bank_vols)]
    bank = np.stack([m3t.fingerprint(v) for v in bank_vols])
    q = bank_vols[7] * 1.7 + 20 + rng.normal(scale=2, size=bank_vols[7].shape)   # cung anh, doi cuong do
    r, i = m3t.max_corr(m3t.fingerprint(q), bank)
    assert i == 7 and r > 0.99
    r_other, _ = m3t.max_corr(m3t.fingerprint(q), np.delete(bank, 7, axis=0))
    assert r_other < 0.97
    with pytest.raises(ValueError, match="hang"):
        m3t.fingerprint(np.ones((4, 4, 4)))


# ------------------------------------------------------------ 17. schema CSV

def test_cls_frame_schema_and_column_regex():
    rng = np.random.default_rng(0)
    df = m3t.cls_frame(["b", "a", "c"], rng.normal(size=(3, 128)), rng.normal(size=(3, 5)),
                       dict(weights_hash="abc", input="npz"))
    feat = [c for c in df.columns if m3t.is_cls_col(c)]
    assert feat == [f"m3t_{i:03d}" for i in range(128)]
    assert [c for c in df.columns if m3t.is_logit_col(c)] == [f"m3tlogit_{k}" for k in range(5)]
    assert list(df.columns[:2]) == ["case_id", "m3t_000"]
    assert list(df.columns[-2:]) == ["prov_weights_hash", "prov_input"]
    assert df["case_id"].tolist() == ["b", "a", "c"]
    assert not any(m3t.is_cls_col(c) for c in ["m3tlogit_0", "prov_weights_hash", "m3t_1000", "rad_x"])
    with pytest.raises(ValueError, match="trung"):
        m3t.cls_frame(["a", "a"], rng.normal(size=(2, 4)), rng.normal(size=(2, 5)), {})
    const = rng.normal(size=(3, 4))
    const[:, 2] = 1.0
    with pytest.raises(ValueError, match="hang so"):
        m3t.cls_frame(["a", "b", "c"], const, rng.normal(size=(3, 5)), {})
    nan = rng.normal(size=(2, 4))
    nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        m3t.cls_frame(["a", "b"], nan, rng.normal(size=(2, 5)), {})
