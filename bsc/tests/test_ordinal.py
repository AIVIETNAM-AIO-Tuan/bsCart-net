"""Test bsc/ordinal.py tren du lieu tong hop - chay LOCAL (torch CPU), khong can data.

    python -m pytest bsc/tests/test_ordinal.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from bsc import ordinal as ORD

K = 5


def make_synthetic(n=800, n_feat=6, seed=0, noise=0.8):
    """Bien an z = Xw + nhieu; KL = digitize(z) voi phan bo lech (nhieu KL0/KL2, it KL4)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    w = rng.normal(size=n_feat)
    z = X @ w + rng.normal(scale=noise, size=n)
    cuts = np.quantile(z, [0.30, 0.45, 0.75, 0.93])
    y = np.digitize(z, cuts)
    return X.astype(np.float32), y.astype(np.int64)


@pytest.fixture(scope="module")
def data():
    X, y = make_synthetic()
    tr, te = np.arange(600), np.arange(600, 800)
    return X[tr], y[tr], X[te], y[te]


# ------------------------------------------------------------ nhan nguong / giai ma

def test_threshold_roundtrip():
    y = np.array([0, 1, 2, 3, 4, 3, 2])
    t = ORD.to_thresholds(y, K)
    assert t.shape == (7, 4)
    assert t[3].tolist() == [1, 1, 1, 0]                      # KL3 -> [1,1,1,0] nhu slide
    assert (ORD.decode_count(t) == y).all()
    assert (ORD.decode_cumdiff(t) == y).all()


def test_decode_rules_document_the_slide_question():
    """Cau hoi: p1, p2 'cao qua' co lam KL3 sai khong? KHONG - chi p3 hoac p2 quyet dinh."""
    assert ORD.decode_count(np.array([[0.98, 0.99, 0.99, 0.15]]))[0] == 3     # KL3 dung
    assert ORD.decode_count(np.array([[0.98, 0.99, 0.99, 0.55]]))[0] == 4     # sai do p3
    assert ORD.decode_count(np.array([[0.98, 0.95, 0.45, 0.10]]))[0] == 2     # sai do p2
    assert ORD.decode_count(np.array([[0.98, 0.95, 0.70, 0.10]]))[0] == 3     # KL2 that -> sai LEN
    p = np.array([[0.9, 0.4, 0.6, 0.1]])                                      # khong don dieu
    assert ORD.decode_count(p)[0] == 2
    assert ORD.monotonic_violation_rate(p) == 1.0
    assert ORD.monotonic_violation_rate(ORD.monotone_cummin(p)) == 0.0
    assert ORD.monotone_cummin(p)[0].tolist() == pytest.approx([0.9, 0.4, 0.4, 0.1])


def test_qwk_penalises_far_errors_more():
    y = np.array([0, 1, 2, 3, 4] * 20)
    near = (y + np.tile([1, 0, 0, 0, 0], 20)).clip(0, 4)
    far = np.where(np.tile([1, 0, 0, 0, 0], 20) == 1, 4, y)
    assert ORD.qwk(y, near, K) > ORD.qwk(y, far, K)
    assert ORD.qwk(y, y, K) == pytest.approx(1.0)


# ------------------------------------------------------------ chon dac trung

def test_select_features_drops_duplicates_and_noise():
    """Bo cot trung, giu cot co tin hieu, va tra chi so hop le."""
    rng = np.random.default_rng(0)
    n = 400
    info = rng.normal(size=(n, 3))
    z = info @ np.array([1.0, -0.7, 0.4]) + rng.normal(scale=0.3, size=n)
    y = np.digitize(z, np.quantile(z, [0.2, 0.4, 0.6, 0.8]))
    dup = info[:, 0] + 1e-8 * rng.normal(size=n)              # cot 3 = ban sao cua cot 0
    X = np.column_stack([info, dup, rng.normal(size=(n, 40))])

    sel = ORD.select_features(X, y, seed=0)
    assert sel.ndim == 1 and set(sel) <= set(range(X.shape[1]))
    assert not (0 in sel and 3 in sel), "giu ca hai ban sao gan trung nhau"
    assert 3 not in sel, "loc tuong quan phai giu cot chi so NHO hon"
    assert sum(j in sel for j in (0, 1, 2)) >= 2, "bo mat cot co tin hieu"
    assert len(sel) < X.shape[1], "khong loai duoc cot nhieu nao"


def test_select_features_is_deterministic_and_handles_degenerate():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 30))
    X[:, 5] = 0.0                                              # cot hang so
    y = rng.integers(0, 5, size=120)
    a = ORD.select_features(X, y, seed=0)
    b = ORD.select_features(X, y, seed=0)
    assert np.array_equal(a, b)
    assert 5 not in a, "cot hang so phai bi bo"
    # it cot hon min_keep thi tra ve tat ca, khong sap
    assert len(ORD.select_features(X[:, :4], y)) == 4


# ------------------------------------------------------------ Frank & Hall

def test_frank_hall_learns_ordinal_structure(data):
    Xtr, ytr, Xte, yte = data
    fh = ORD.FrankHall(lambda: LogisticRegression(max_iter=2000)).fit(Xtr, ytr, K)
    p = fh.predict_proba_thresholds(Xte)
    assert p.shape == (len(Xte), K - 1)
    assert ORD.monotonic_violation_rate(p) == 0.0                # cummin ep don dieu
    q = ORD.qwk(yte, fh.predict(Xte), K)
    assert q > 0.7, q
    tab = ORD.mean_p_by_class(p, yte, K)
    assert (tab[0] < 0.5).all()                                  # lop thap nhat: moi nguong < 0.5
    assert (np.diff(tab, axis=0) > -0.02).all()                  # p_k tang theo lop that (bac thang)
    # Lop hiem nhat (KL4 ~7%): P(KL>3) trung binh chi ~0.46 vi mat can bang - dung nhu
    # canh bao "nguong KL>3 hieu chinh kem nhat" -> KHONG doi > 0.5 o day.
    auc = ORD.threshold_auc(p, ORD.to_thresholds(yte, K))
    assert np.nanmin(auc) > 0.85, auc


def test_frank_hall_handles_degenerate_threshold():
    X = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 3, size=50)         # chi co lop 0..2 trong K=5
    fh = ORD.FrankHall(lambda: LogisticRegression(max_iter=500)).fit(X, y, K)
    p = fh.predict_proba_thresholds(X)
    assert (p[:, 3] == 0).all() and (p[:, 2] == 0).all()
    assert fh.predict(X).max() <= 2


# ------------------------------------------------------------ diem cat

def test_fit_cutpoints_beats_naive_rounding_on_biased_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, K, size=600)
    scores = y + 0.7 + rng.normal(scale=0.3, size=600)          # lech he thong +0.7
    naive = ORD.qwk(y, np.clip(np.rint(scores), 0, K - 1).astype(int), K)
    cuts = ORD.fit_cutpoints(scores, y, K)
    fitted = ORD.qwk(y, ORD.apply_cutpoints(scores, cuts), K)
    assert fitted > naive + 0.05, (fitted, naive)
    assert (np.diff(cuts) > 0).all()


# ------------------------------------------------------------ MLP voi loss cua slide

def test_mlp_ordinal_only_learns_and_is_monotone(data):
    Xtr, ytr, Xte, yte = data
    model, info = ORD.train_ordinal_mlp(Xtr, ytr, K, lambdas=ORD.ORDINAL_ONLY_LAMBDAS,
                                        epochs=300, seed=0)
    h = info["history"]
    assert h[-1]["total"] < 0.5 * h[0]["total"], "loss khong giam"
    assert h[-1]["cls"] > 0 and info["lambdas"]["cls"] == 0     # CE duoc ghi nhung khong toi uu
    out = ORD.predict_ordinal_mlp(model, info, Xte)
    assert out["p_thr"].shape == (len(Xte), K - 1)
    assert ORD.qwk(yte, out["y_count"], K) > 0.7
    assert out["mono_violation"] < 0.05                           # L_mono la phat mem: gan 0, khong dam bao 0
    # hai cach giai ma chi lech o hang p gan 0.5 / khong don dieu (do duoc ~0.9)
    assert (out["y_count"] == out["y_cumdiff"]).mean() > 0.85
    tab = ORD.mean_p_by_class(out["p_thr"], yte, K)
    assert (tab[0] < 0.5).all()                                  # lop thap nhat: moi nguong < 0.5
    assert (np.diff(tab, axis=0) > -0.02).all()                  # bac thang: p_k tang theo lop that
    # lop hiem nhat (~7%) co the co P(KL>3) trung binh < 0.5 vi mat can bang - khong doi > 0.5


def test_mlp_slide_multitask_all_heads_trained(data):
    Xtr, ytr, Xte, yte = data
    oa = (ytr >= 2).astype(np.float32)
    model, info = ORD.train_ordinal_mlp(Xtr, ytr, K, oa_target=oa, lambdas=ORD.SLIDE_LAMBDAS,
                                        epochs=300, seed=0)
    h = info["history"]
    for k in ("ord", "cls", "oa"):
        assert h[-1][k] < h[0][k], f"loss {k} khong giam"
    out = ORD.predict_ordinal_mlp(model, info, Xte)
    assert ORD.qwk(yte, out["y_count"], K) > 0.7
    assert ORD.qwk(yte, out["y_softmax"], K) > 0.7
    # hai head co the bat dong - do la ly do notebook phai ghi ro dau ra nao la chinh thuc
    agree = (out["y_softmax"] == out["y_count"]).mean()
    assert agree > 0.7, agree
    # p_oa ~ p_1 (cung su kien KL>=2 == KL>1): tuong quan cao
    assert np.corrcoef(out["p_oa"], out["p_thr"][:, 1])[0, 1] > 0.9


def test_mlp_is_deterministic_given_seed(data):
    Xtr, ytr, Xte, _ = data
    a = ORD.predict_ordinal_mlp(*ORD.train_ordinal_mlp(Xtr, ytr, K, epochs=50, seed=3), Xte)
    b = ORD.predict_ordinal_mlp(*ORD.train_ordinal_mlp(Xtr, ytr, K, epochs=50, seed=3), Xte)
    assert np.allclose(a["p_thr"], b["p_thr"])
