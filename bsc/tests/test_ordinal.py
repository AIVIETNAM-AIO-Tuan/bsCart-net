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


# ------------------------------------------------------------ chia train/test

def test_stratified_group_split_keeps_ratio_and_subjects():
    """Phan tang dung ty le lop, va khong subject nao bi tach ra hai phia."""
    rng = np.random.default_rng(0)
    n_case, n_subj = 1229, 1215
    kl = [284, 233, 295, 311, 106]
    subj = np.array(list(range(n_subj)) + list(rng.choice(n_subj, n_case - n_subj, replace=False)))
    y = np.repeat(np.arange(5), kl)
    rng.shuffle(y)

    tr, te = ORD.stratified_group_split(y, subj, test_size=0.2, seed=42)
    assert len(tr) + len(te) == n_case
    assert not (set(subj[tr]) & set(subj[te])), "subject bi tach doi"
    assert 0.17 < len(te) / n_case < 0.23

    for k in range(5):
        p_all = (y == k).mean()
        p_te = (y[te] == k).mean()
        assert abs(p_te - p_all) < 0.02, f"KL{k} lech ty trong: {p_te:.3f} vs {p_all:.3f}"

    # Lop hiem phai ON DINH qua nhieu seed - day la dieu StratifiedGroupKFold KHONG lam duoc
    # (do duoc: sd 3.8 so voi 0.23 cua ham nay).
    c = [int((y[ORD.stratified_group_split(y, subj, 0.2, seed=s)[1]] == 4).sum())
         for s in range(30)]
    assert np.std(c) < 1.0, f"so ca KL4 dao dong sd={np.std(c):.2f}"


def test_stratified_group_split_handles_multi_case_subject():
    """Subject nhieu ca khac lop: khong bi tach, nhan dai dien la lop CAO NHAT."""
    n = 12
    subj = np.array([f"s{i//2:02d}" for i in range(2 * n)])       # moi subject dung 2 ca
    y = np.concatenate([np.array([0, k]) for k in np.repeat([0, 1, 2, 3, 4], n // 5 + 1)[:n]])
    tr, te = ORD.stratified_group_split(y, subj, test_size=0.34, seed=0)
    assert not (set(subj[tr]) & set(subj[te]))
    for s in np.unique(subj):                       # moi subject phai nam tron mot phia
        i = np.flatnonzero(subj == s)
        assert set(i) <= set(tr) or set(i) <= set(te), f"subject {s} bi tach doi"


def test_stratified_group_split_falls_back_when_class_too_rare(capsys):
    """Lop chi co 1 subject thi khong phan tang duoc: phai LUI VE chia nhom, khong duoc sap."""
    subj = np.array([f"s{i}" for i in range(20)])
    y = np.array([0] * 10 + [1] * 9 + [4])          # lop 4 chi 1 subject
    tr, te = ORD.stratified_group_split(y, subj, test_size=0.3, seed=0)
    assert len(tr) + len(te) == len(y)
    assert not (set(subj[tr]) & set(subj[te]))
    assert "CANH BAO" in capsys.readouterr().out


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


def test_frank_hall_survives_pickle(data):
    """Pickle duoc du `make_clf` la lambda, va nap lai van du doan y het."""
    import pickle
    Xtr, ytr, Xte, _ = data
    fh = ORD.FrankHall(lambda: LogisticRegression(max_iter=2000)).fit(Xtr, ytr, K)
    before = fh.predict(Xte)

    fh2 = pickle.loads(pickle.dumps(fh))           # truoc day nem AttributeError o day
    assert np.array_equal(fh2.predict(Xte), before)
    assert np.allclose(fh2.predict_proba_thresholds(Xte), fh.predict_proba_thresholds(Xte))

    # Nap lai thi KHONG fit lai duoc, va phai bao ro thay vi nem loi kho hieu
    assert fh2.make_clf is None
    with pytest.raises(RuntimeError, match="khong con"):
        fh2.fit(Xtr, ytr, K)


def test_frank_hall_handles_degenerate_threshold():
    X = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 3, size=50)         # chi co lop 0..2 trong K=5
    fh = ORD.FrankHall(lambda: LogisticRegression(max_iter=500)).fit(X, y, K)
    p = fh.predict_proba_thresholds(X)
    assert (p[:, 3] == 0).all() and (p[:, 2] == 0).all()
    assert fh.predict(X).max() <= 2


# ------------------------------------------------------------ metric theo lop

def test_per_class_prf_matches_sklearn():
    """Ban numpy phai trung KHIT sklearn, ke ca khi mot lop khong bao gio duoc doan."""
    from sklearn.metrics import precision_recall_fscore_support

    rng = np.random.default_rng(7)
    y = rng.integers(0, K, size=300)
    yp = rng.integers(0, K - 1, size=300)                  # lop 4 khong bao gio duoc doan
    pr, rc, f1, sup = precision_recall_fscore_support(y, yp, labels=list(range(K)), zero_division=0)
    m = ORD.per_class_prf(y, yp, K)
    assert np.allclose(m["precision"], pr) and np.allclose(m["recall"], rc)
    assert np.allclose(m["f1"], f1) and np.array_equal(m["support"], sup)
    assert ORD.macro_recall(y, yp, K) == pytest.approx(rc.mean())
    assert ORD.macro_f1(y, yp, K) == pytest.approx(f1.mean())
    assert np.array_equal(ORD.per_class_recall(y, yp, K), m["recall"])

    cm = ORD.confusion(y, yp, K)
    assert cm.shape == (K, K) and cm.sum() == len(y)
    assert (cm[:, 4] == 0).all()                           # cot lop khong duoc doan
    assert ORD.per_class_recall(y, y, K).tolist() == [1.0] * K


def test_bootstrap_delta_ci_brackets_zero_and_real_gap():
    y = np.random.default_rng(3).integers(0, K, size=250)
    same = ORD.bootstrap_delta(y, y, y, "qwk", K, n_boot=300)
    assert same["delta"] == 0.0 and same["ci_low"] <= 0 <= same["ci_high"]
    gap = ORD.bootstrap_delta(y, np.zeros_like(y), y, "qwk", K, n_boot=300)
    assert gap["delta"] > 0 and gap["ci_low"] > 0, gap


# ------------------------------------------------------------ bootstrap theo subject + seed

def _bootstrap_delta_v1(y_true, yp_a, yp_b, metric, n_classes, n_boot=2000, seed=0, alpha=0.05):
    """Ban CU nguyen van (truoc 25/09/2026) - chuan de khoa duong 1D khong groups."""
    f = ORD.OBJECTIVES[metric] if isinstance(metric, str) else metric
    y = np.asarray(y_true, np.int64).reshape(-1)
    a = np.asarray(yp_a, np.int64).reshape(-1)
    b = np.asarray(yp_b, np.int64).reshape(-1)
    n = len(y)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = f(y[idx], b[idx], n_classes) - f(y[idx], a[idx], n_classes)
    delta = f(y, b, n_classes) - f(y, a, n_classes)
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return dict(delta=float(delta), ci_low=float(lo), ci_high=float(hi), n_boot=int(n_boot))


def _pair(n=300, seed=11):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    a = np.clip(y + rng.integers(-1, 2, n), 0, K - 1)
    b = np.clip(y + rng.integers(-1, 1, n), 0, K - 1)
    return y, a, b


def _seeded(n_subj=120, n_seeds=3, seed=0):
    """y theo subject (vai subject co 2-3 dong), du doan KHAC NHAU giua cac seed."""
    rng = np.random.default_rng(seed)
    reps = rng.choice([1, 1, 1, 2, 3], size=n_subj)
    groups = np.repeat([f"s{i:04d}" for i in range(n_subj)], reps)
    y = rng.integers(0, K, len(groups))
    A = np.column_stack([np.clip(y + rng.integers(-2, 2, len(y)), 0, K - 1) for _ in range(n_seeds)])
    B = np.column_stack([np.clip(y + rng.integers(-1, 2, len(y)), 0, K - 1) for _ in range(n_seeds)])
    return y, A, B, groups


@pytest.mark.parametrize("metric", ["qwk", "macro_recall"])
def test_bootstrap_delta_1d_path_is_byte_identical_to_v1(metric):
    y, a, b = _pair()
    new = ORD.bootstrap_delta(y, a, b, metric, K, n_boot=400, seed=3)
    assert new == _bootstrap_delta_v1(y, a, b, metric, K, n_boot=400, seed=3)
    assert list(new) == ["delta", "ci_low", "ci_high", "n_boot"]          # schema giu nguyen


def test_bootstrap_delta_1d_golden_values_pinned_before_extension():
    # ghim tu ban cu ngay 25/09/2026 (numpy 2.2.6, sklearn 1.7.2), TRUOC khi mo rong
    y, a, b = _pair()
    r = ORD.bootstrap_delta(y, a, b, "qwk", K, n_boot=400, seed=3)
    assert r["delta"] == pytest.approx(0.026828992268487784, abs=1e-12)
    assert r["ci_low"] == pytest.approx(0.008937967830654429, abs=1e-12)
    assert r["ci_high"] == pytest.approx(0.046151037785287674, abs=1e-12)


def test_bootstrap_delta_one_seed_column_and_singleton_groups_reduce_to_1d():
    y, a, b = _pair()
    ref = ORD.bootstrap_delta(y, a, b, "qwk", K, n_boot=300, seed=5)
    assert ORD.bootstrap_delta(y, a[:, None], b[:, None], "qwk", K, n_boot=300, seed=5) == ref
    # moi subject 1 dong, ID da sap xep => cung chuoi boc voi duong cu
    assert ORD.bootstrap_delta(y, a, b, "qwk", K, n_boot=300, seed=5, groups=np.arange(len(y))) == ref


def test_cluster_rows_take_whole_cluster_and_repeat_it_k_times():
    g = np.array(["s3", "s1", "s1", "s2", "s1", "s3"])
    order, starts, counts = ORD._cluster_index(g)
    assert counts.tolist() == [3, 1, 2]                                   # s1, s2, s3 theo np.unique
    rows = ORD._cluster_rows(order, starts, counts, np.array([0, 0, 2]))
    assert rows.tolist() == [1, 2, 4, 1, 2, 4, 0, 5]                      # s1 hai lan, s3 mot lan, s2 khong


def test_grouped_bootstrap_widens_ci_when_rows_are_clustered():
    # moi subject 2 dong GIONG HET: boc theo dong dem mot subject nhu 2 quan sat doc lap
    y, a, b = _pair(n=150, seed=4)
    y2, a2, b2 = np.repeat(y, 2), np.repeat(a, 2), np.repeat(b, 2)
    g = np.repeat(np.arange(150), 2)
    rows = ORD.bootstrap_delta(y2, a2, b2, "qwk", K, n_boot=1500, seed=0)
    subj = ORD.bootstrap_delta(y2, a2, b2, "qwk", K, n_boot=1500, seed=0, groups=g)
    assert subj["delta"] == rows["delta"]
    w_rows, w_subj = rows["ci_high"] - rows["ci_low"], subj["ci_high"] - subj["ci_low"]
    assert w_subj > 1.2 * w_rows, (w_subj, w_rows)                         # ly thuyet ~ sqrt(2)


def test_bootstrap_delta_averages_per_seed_metric_not_pooled_rows():
    y, A, B, g = _seeded()
    A[:, 0] = 2          # seed 0 cua nhanh a doan hang => QWK 0; gop dong se lech xa trung binh
    per_seed = np.array([ORD.qwk(y, B[:, s], K) - ORD.qwk(y, A[:, s], K) for s in range(3)])
    assert np.allclose(ORD.seed_deltas(y, A, B, "qwk", K), per_seed, atol=0, rtol=0)
    r = ORD.bootstrap_delta(y, A, B, "qwk", K, n_boot=200, seed=0, groups=g)
    assert r["delta"] == pytest.approx(per_seed.mean(), abs=1e-15)
    pooled = ORD.qwk(np.tile(y, 3), B.T.reshape(-1), K) - ORD.qwk(np.tile(y, 3), A.T.reshape(-1), K)
    assert abs(pooled - per_seed.mean()) > 0.05                            # test co phan biet duoc


def test_duplicating_a_seed_column_does_not_narrow_the_ci():
    y, A, B, g = _seeded()
    one = ORD.bootstrap_delta(y, A[:, :1], B[:, :1], "qwk", K, n_boot=300, seed=0, groups=g)
    dup = ORD.bootstrap_delta(y, np.repeat(A[:, :1], 3, 1), np.repeat(B[:, :1], 3, 1), "qwk", K,
                              n_boot=300, seed=0, groups=g)
    assert dup == one


def test_same_draw_for_both_branches_and_every_seed():
    y, A, B, g = _seeded()
    same = ORD.bootstrap_delta(y, A, A.copy(), "qwk", K, n_boot=200, seed=0, groups=g)
    assert same["delta"] == 0.0 and same["ci_low"] == 0.0 and same["ci_high"] == 0.0
    r = ORD.bootstrap_delta(y, A, B, "qwk", K, n_boot=400, seed=0, groups=g)
    s = ORD.bootstrap_delta(y, B, A, "qwk", K, n_boot=400, seed=0, groups=g)
    assert s["delta"] == pytest.approx(-r["delta"], abs=1e-12)
    assert s["ci_low"] == pytest.approx(-r["ci_high"], abs=1e-12)
    assert s["ci_high"] == pytest.approx(-r["ci_low"], abs=1e-12)


def test_bootstrap_delta_rejects_bad_shapes_missing_predictions_and_ids():
    y, A, B, g = _seeded()
    with pytest.raises(ValueError, match="cung shape"):
        ORD.bootstrap_delta(y, A, B[:, :2], "qwk", K, n_boot=5, groups=g)
    with pytest.raises(ValueError, match="dong"):
        ORD.bootstrap_delta(y[:-1], A, B, "qwk", K, n_boot=5)
    with pytest.raises(ValueError, match="ndim"):
        ORD.bootstrap_delta(y, A[..., None], B[..., None], "qwk", K, n_boot=5)
    with pytest.raises(ValueError, match="ndim"):
        ORD.bootstrap_delta(y[:, None], A, B, "qwk", K, n_boot=5)
    miss = A.astype(float)
    miss[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ORD.bootstrap_delta(y, miss, B, "qwk", K, n_boot=5, groups=g)
    neg = A.copy()
    neg[0, 0] = -1                                                          # S7: -1 = chua co du doan
    with pytest.raises(ValueError, match="ngoai"):
        ORD.bootstrap_delta(y, neg, B, "qwk", K, n_boot=5, groups=g)
    with pytest.raises(ValueError, match="groups"):
        ORD.bootstrap_delta(y, A, B, "qwk", K, n_boot=5, groups=g[:-1])
    for bad in (None, "nan", ""):
        gg = g.astype(object)
        gg[7] = bad
        with pytest.raises(ValueError, match="ID thieu"):
            ORD.bootstrap_delta(y, A, B, "qwk", K, n_boot=5, groups=gg)
    gf = np.arange(len(y), dtype=float)
    gf[2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ORD.bootstrap_delta(y, A, B, "qwk", K, n_boot=5, groups=gf)


@pytest.mark.parametrize("lo,hi,level,note", [
    (0.03, 0.08, "vuot_nguong", "L > 0.02"),
    (0.005, 0.015, "duoi_nguong", "nho hon nguong"),
    (-0.05, -0.01, "te_hon", "hieu am"),
    (-0.01, 0.01, "chua_du", "DAU; loai duoc"),                 # cat 0, loai duoc loi >= delta
    (0.01, 0.05, "chua_du", "DA co bang chung cai thien"),     # cat delta: 0 < L <= delta <= U
    (-0.01, 0.05, "chua_du", "DAU lan DO LON"),                # cat ca hai
    (0.0, 0.01, "chua_du", "cham bien 0"),                     # L == 0
    (-0.03, 0.0, "chua_du", "cham bien 0"),                    # U == 0
    (0.02, 0.05, "chua_du", "cham bien delta"),                # L == delta
    (0.01, 0.02, "chua_du", "cham bien delta"),                # U == delta
])
def test_classify_delta_four_levels_with_strict_boundaries(lo, hi, level, note):
    r = ORD.classify_delta(dict(delta=(lo + hi) / 2, ci_low=lo, ci_high=hi), min_delta=0.02)
    assert r["level"] == level, r
    assert note in r["note"], r
    assert r["label"] == ORD.DELTA_LEVELS[level]


def test_classify_delta_rejects_invalid_results():
    with pytest.raises(ValueError, match="NaN"):
        ORD.classify_delta(dict(delta=np.nan, ci_low=0.0, ci_high=0.1))
    with pytest.raises(ValueError, match="ci_low"):
        ORD.classify_delta(dict(delta=0.0, ci_low=0.1, ci_high=-0.1))
    with pytest.raises(ValueError, match="thieu khoa"):
        ORD.classify_delta(dict(delta=0.0, ci_low=0.0))
    with pytest.raises(ValueError, match="min_delta"):
        ORD.classify_delta(dict(delta=0.0, ci_low=0.0, ci_high=0.1), min_delta=0.0)


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


def test_fit_cutpoints_default_is_byte_identical():
    """Refactor sang _coordinate_descent KHONG duoc doi mot chu so nao cua duong mac dinh.

    Con so vang duoi day lay tu ban TRUOC refactor, tren dung du lieu cua test phia tren.
    Moi con so S7/S8 da bao cao deu sinh ra tu duong nay, nen no la hop dong.
    """
    rng = np.random.default_rng(0)
    y = rng.integers(0, K, size=600)
    scores = y + 0.7 + rng.normal(scale=0.3, size=600)
    gold = np.array([1.13266468, 2.22593294, 3.30818994, 4.15042206])
    assert np.allclose(ORD.fit_cutpoints(scores, y, K), gold, atol=1e-8)


def make_shrunk(seed, n=600):
    """Diem bi CO ve trung binh - dung tinh huong cua bo hoi quy XGB o model C.

    Var(s) < Var(y) => diem cat toi uu QWK se noi hai bin ngoai cung de bom Var(yp) len.
    """
    r = np.random.default_rng(seed)
    y = r.choice(K, size=n, p=[.23, .19, .24, .25, .09])
    s = 0.6 * (y - 2) + 2 + r.normal(scale=0.45, size=n)
    return y, s


def test_quantile_cutpoints_reproduce_train_marginal():
    y, s = make_shrunk(0)
    cuts = ORD.quantile_cutpoints(s, y, K)
    assert (np.diff(cuts) >= 0).all()
    assert np.array_equal(np.bincount(ORD.apply_cutpoints(s, cuts), minlength=K),
                          np.bincount(y, minlength=K)), "marginal du doan khac marginal that"

    # Lop vang mat: hai cut trung nhau => lop do khong bao gio duoc doan, khong duoc nem loi
    y2 = np.where(y == 3, 2, y)
    c2 = ORD.quantile_cutpoints(s, y2, K)
    assert 3 not in set(ORD.apply_cutpoints(s, c2).tolist())


def test_quantile_beats_variance_inflation_on_shrunk_scores():
    """Diem cat phan vi (0 tham so) danh bai diem cat toi uu QWK tren HELD-OUT.

    Day la bang chung cho luan diem trung tam: trade-off "QWK cao <-> recall lop giua thap"
    den tu MUC TIEU dat diem cat, khong phai tu bo hoi quy. Tim QWK tren ~400 hang con
    overfit vi tri cat, nen bo cai tim kiem di lai duoc CA HAI mat.
    """
    y, s = make_shrunk(0)
    tr, te = slice(0, 400), slice(400, 600)
    c_qwk = ORD.fit_cutpoints(s[tr], y[tr], K)
    c_qnt = ORD.quantile_cutpoints(s[tr], y[tr], K)
    yq, yn = ORD.apply_cutpoints(s[te], c_qwk), ORD.apply_cutpoints(s[te], c_qnt)
    m_qwk, m_qnt = ORD.per_class_recall(y[te], yq, K).min(), ORD.per_class_recall(y[te], yn, K).min()
    assert m_qnt >= m_qwk + 0.2, (m_qwk, m_qnt)                      # do duoc: 0.135 -> 0.486
    assert ORD.qwk(y[te], yn, K) >= ORD.qwk(y[te], yq, K) - 0.03     # do duoc: +0.021

    # Tren nhieu seed: recall thap nhat KHONG BAO GIO te hon, va QWK trung binh khong te hon
    d_q = []
    for seed in range(6):
        y, s = make_shrunk(seed)
        a = ORD.apply_cutpoints(s[te], ORD.fit_cutpoints(s[tr], y[tr], K))
        b = ORD.apply_cutpoints(s[te], ORD.quantile_cutpoints(s[tr], y[tr], K))
        assert ORD.per_class_recall(y[te], b, K).min() >= ORD.per_class_recall(y[te], a, K).min() - 1e-9
        d_q.append(ORD.qwk(y[te], b, K) - ORD.qwk(y[te], a, K))
    assert np.mean(d_q) > -0.01, d_q


def test_fit_cutpoints_objective_and_slack_guarantees():
    """Chi kiem tinh chat BAO DAM in-sample. Khong claim held-out cho macro_recall: no bat on."""
    y, s = make_shrunk(0)
    dec = lambda c: ORD.apply_cutpoints(s, c)
    c_qwk = ORD.fit_cutpoints(s, y, K)
    c_mr = ORD.fit_cutpoints(s, y, K, objective="macro_recall")
    assert ORD.macro_recall(y, dec(c_mr), K) >= ORD.macro_recall(y, dec(c_qwk), K) - 1e-9

    q_best = ORD.qwk(y, dec(c_qwk), K)
    c_sl = ORD.fit_cutpoints(s, y, K, objective="macro_recall", qwk_slack=0.02)
    assert ORD.qwk(y, dec(c_sl), K) >= q_best - 0.02 - 1e-9, "vi pham tran QWK"
    # Pass 2 khoi dau TU diem toi uu QWK nen khong the te hon no ve macro_recall
    assert ORD.macro_recall(y, dec(c_sl), K) >= ORD.macro_recall(y, dec(c_qwk), K) - 1e-9

    # objective callable
    c_cb = ORD.fit_cutpoints(s, y, K, objective=lambda a, b: -ORD.mae(a, b))
    assert (np.diff(c_cb) > 0).all()
    with pytest.raises(ValueError, match="objective"):
        ORD.fit_cutpoints(s, y, K, objective="khong_ton_tai")


def test_fit_cutpoints_min_recall_feasible_and_fallback(capsys):
    y, s = make_shrunk(0)
    c0 = ORD.fit_cutpoints(s, y, K)
    assert ORD.per_class_recall(y, ORD.apply_cutpoints(s, c0), K).min() < 0.50   # san 0.5 co RANG BUOC

    c = ORD.fit_cutpoints(s, y, K, min_recall=0.50)
    assert ORD.per_class_recall(y, ORD.apply_cutpoints(s, c), K).min() >= 0.50
    assert "CANH BAO" not in capsys.readouterr().out

    c_bad = ORD.fit_cutpoints(s, y, K, min_recall=0.60)                          # bat kha thi
    assert "CANH BAO" in capsys.readouterr().out
    assert np.array_equal(c_bad, c0), "fallback phai tra ve DUNG ket qua khong rang buoc"


def test_apply_cutpoints_rejects_nonmonotone():
    """np.digitize nhan bins giam dan ma khong keu - phai chan o day."""
    with pytest.raises(AssertionError, match="khong giam"):
        ORD.apply_cutpoints(np.array([1.0, 2.0]), np.array([3.0, 1.0]))


# ------------------------------------------------------------ nguong quyet dinh tung k

def make_underconfident(seed, n=600):
    """p hop le nhung nguong CUOI bi thieu tu tin - dung trieu chung do duoc o S7."""
    r = np.random.default_rng(seed)
    y = r.choice(K, size=n, p=[.23, .19, .24, .25, .09])
    z = 2.0 * (y[:, None] - np.arange(K - 1)[None, :] - 0.5) + r.normal(scale=1.2, size=(n, K - 1))
    p = ORD.monotone_cummin(1.0 / (1.0 + np.exp(-z)))
    p[:, 3] *= 0.6
    return y, p


def test_fit_threshold_cuts_recover_underconfident_last_threshold():
    y, p = make_underconfident(0)
    tr, te = slice(0, 400), slice(400, 600)
    thr = ORD.fit_threshold_cuts(p[tr], y[tr], K, objective="macro_recall")
    assert thr.shape == (K - 1,) and (thr >= 0.05).all() and (thr <= 0.95).all()
    assert thr[-1] < 0.5, "khong ha duoc nguong cuoi du no bi thieu tu tin"

    y_fix = ORD.decode_count(p[te])
    y_tun = ORD.decode_count(p[te], thr=thr)
    r_fix = ORD.per_class_recall(y[te], y_fix, K)
    r_tun = ORD.per_class_recall(y[te], y_tun, K)
    assert r_tun[-1] >= r_fix[-1] + 0.3, (r_fix[-1], r_tun[-1])     # do duoc: 0.20 -> 1.00
    assert ORD.qwk(y[te], y_tun, K) >= ORD.qwk(y[te], y_fix, K) - 0.02

    # Luoi khong chua 0.5 nen in-sample khong bao gio te hon vach co dinh
    assert ORD.qwk(y[tr], ORD.decode_count(p[tr], thr=thr), K) >= ORD.qwk(y[tr], ORD.decode_count(p[tr]), K) - 1e-9
    # thr KHONG bi ep don dieu (co y): dem co the nhan mau nhu [1,0,1,0] - do va BAO CAO
    assert 0.0 <= ORD.monotonic_violation_rate((p[te] > thr).astype(float)) <= 1.0


def test_decode_count_vector_thr_and_shape_guard():
    p = np.array([[0.9, 0.8, 0.4, 0.3], [0.9, 0.2, 0.1, 0.05]])
    assert ORD.decode_count(p, thr=np.array([0.5, 0.5, 0.5, 0.5])).tolist() == ORD.decode_count(p).tolist()
    assert ORD.decode_count(p, thr=np.array([0.5, 0.5, 0.35, 0.25])).tolist() == [4, 1]
    for bad in (np.array([0.5, 0.5]), np.full(K, 0.5)):             # (N,) voi N==K-1, va (K,)
        with pytest.raises(AssertionError, match="vector"):
            ORD.decode_count(p, thr=bad)


# ------------------------------------------------------------ diem honest & trong so lop

def test_inner_oof_no_leakage_and_matches_sklearn():
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, cross_val_predict

    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 5))
    y = rng.normal(size=200)
    g = rng.integers(0, 40, size=200)

    # Fold PHAI trung cross_val_predict(GroupKFold(4)) - neu khong, dong C cua S7/S8 se doi so
    a = ORD.inner_oof(lambda Xa, ya, Xb: Ridge().fit(Xa, ya).predict(Xb), X, y, g)
    b = cross_val_predict(Ridge(), X, y, groups=g, cv=GroupKFold(n_splits=4))
    assert np.allclose(a, b, atol=1e-12)

    # Ro ri: scorer nho moi hang da thay va tra +100 cho no
    def memoriser(Xa, ya, Xb):
        seen = {r.tobytes() for r in Xa}
        return np.array([100.0 if r.tobytes() in seen else 0.0 for r in Xb])
    assert ORD.inner_oof(memoriser, X, y, g).max() == 0.0, "co hang duoc cham diem boi model da thay no"

    # Dau ra 2-D (p_thr cua FrankHall / MLP)
    p = ORD.inner_oof(lambda Xa, ya, Xb: np.zeros((len(Xb), K - 1)), X, y, g)
    assert p.shape == (200, K - 1) and np.isfinite(p).all()

    # row_kwargs phai duoc cat theo train cua tung fold
    w = np.arange(200, dtype=float)
    def check_w(Xa, ya, Xb, w=None):
        assert w is not None and len(w) == len(Xa)
        return np.zeros(len(Xb))
    ORD.inner_oof(check_w, X, y, g, row_kwargs={"w": w})


def test_class_balanced_weights_equalise_class_mass():
    y, _ = make_shrunk(0)
    w = ORD.class_balanced_weights(y, K)
    assert w.mean() == pytest.approx(1.0)
    mass = [w[y == k].sum() for k in range(K)]
    assert np.allclose(mass, mass[0]), mass                 # tong trong so moi lop bang nhau
    assert w[y == 4].max() > w[y == 3].max()                # lop hiem duoc nang len

    # Lop vang mat khong duoc lam hong chuan hoa
    y2 = np.where(y == 4, 3, y)
    w2 = ORD.class_balanced_weights(y2, K)
    assert w2.mean() == pytest.approx(1.0) and np.isfinite(w2).all()


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


def test_softmax_only_isolates_loss_from_architecture(data):
    """Doi chung: cung than MLP, chi doi loss sang softmax CE.

    Day la thu duy nhat cho phep noi "loss ordinal dong gop bao nhieu" ma khong tron voi
    "mang khac cay". Neu thieu no, chenh lech D tru A chi noi duoc rang hai HE THONG khac
    nhau, khong noi duoc vi sao.
    """
    Xtr, ytr, Xte, yte = data
    ctrl, info = ORD.train_ordinal_mlp(Xtr, ytr, K, lambdas=ORD.SOFTMAX_ONLY_LAMBDAS,
                                       epochs=300, seed=0)
    h = info["history"]
    assert h[-1]["cls"] < 0.6 * h[0]["cls"], "head softmax khong hoc duoc"
    assert info["lambdas"]["ord"] == 0.0

    out = ORD.predict_ordinal_mlp(ctrl, info, Xte)
    assert ORD.qwk(yte, out["y_softmax"], K) > 0.65

    # MOI MODEL PHAI DUOC GIAI MA BANG HEAD CUA CHINH NO.
    # Head nguong o day khong nhan gradient nao tu l_ord, nen p_thr KHONG sup ve hon loan
    # ma DET quanh 0.5: mot lop tuyen tinh ngau nhien dat tren mot than DA hoc duoc thu tu
    # van cho ra thu tu mo nhat, nhung bien do gan nhu bang khong. Do duoc: 0.48-0.51.
    tab = ORD.mean_p_by_class(out["p_thr"], yte, K)
    spread = float(tab.max() - tab.min())
    assert spread < 0.15, f"head nguong khong duoc huan luyen ma bien do toi {spread:.2f}"
    assert ORD.qwk(yte, out["y_count"], K) < ORD.qwk(yte, out["y_softmax"], K) - 0.05

    # Chieu nguoc lai cung phai dung: ordinal_only thi head SOFTMAX moi la cai khong hoc.
    m2, i2 = ORD.train_ordinal_mlp(Xtr, ytr, K, lambdas=ORD.ORDINAL_ONLY_LAMBDAS,
                                   epochs=300, seed=0)
    o2 = ORD.predict_ordinal_mlp(m2, i2, Xte)
    assert ORD.qwk(yte, o2["y_softmax"], K) < ORD.qwk(yte, o2["y_count"], K) - 0.2


def test_head_hidden_makes_heads_mlp_and_keeps_linear_default(data):
    """head_hidden rong => dau la MOT Linear (giu hanh vi cu). Co gia tri => dau thanh MLP."""
    Xtr, ytr, Xte, yte = data
    lin, _ = ORD.train_ordinal_mlp(Xtr, ytr, K, epochs=20, seed=0)
    mlp, info = ORD.train_ordinal_mlp(Xtr, ytr, K, epochs=20, seed=0, head_hidden=(32,))

    n_lin = sum(1 for m in lin.head_ord.modules() if isinstance(m, ORD.nn.Linear))
    n_mlp = sum(1 for m in mlp.head_ord.modules() if isinstance(m, ORD.nn.Linear))
    assert n_lin == 1 and n_mlp == 2, (n_lin, n_mlp)
    assert info["head_hidden"] == (32,) and info["latent_dim"] == 32

    # Dau MLP phai co NHIEU tham so hon, va van hoc duoc
    assert sum(p.numel() for p in mlp.parameters()) > sum(p.numel() for p in lin.parameters())
    m2, i2 = ORD.train_ordinal_mlp(Xtr, ytr, K, lambdas=ORD.ORDINAL_ONLY_LAMBDAS,
                                   epochs=300, seed=0, head_hidden=(32,))
    assert ORD.qwk(yte, ORD.predict_ordinal_mlp(m2, i2, Xte)["y_count"], K) > 0.65


def test_trunk_output_is_returned_and_informative(data):
    """Dau ra cua than duoc tra ve, dung shape, khong NaN, va mang thong tin ve nhan.

    CHUA DUNG VAO VIEC GI. Y tuong trich "cls / latent vector" den tu mo hinh phan loai ANH,
    ma nhanh anh thi chua co, nen S8 KHONG hien thuc no. Test nay chi khoa lai rang module
    tra dung thu no noi la tra, va rang than that su hoc duoc mot bieu dien co nghia.
    """
    Xtr, ytr, Xte, yte = data
    model, info = ORD.train_ordinal_mlp(Xtr, ytr, K, lambdas=ORD.ORDINAL_ONLY_LAMBDAS,
                                        epochs=300, seed=0)
    out = ORD.predict_ordinal_mlp(model, info, Xte)
    z = out["latent"]
    assert z.shape == (len(Xte), info["latent_dim"]) == (len(Xte), 32)
    assert np.isfinite(z).all()

    # Kiem latent co nghia: mot hoi quy tuyen tinh TU LATENT phai doan duoc lop tot hon nhieu
    # so voi doan mu. Neu latent vo nghia thi moi y tuong dung lai no deu vo ich.
    from sklearn.linear_model import LinearRegression
    r = LinearRegression().fit(z, yte)
    rss = float(((r.predict(z) - yte) ** 2).sum())
    tss = float(((yte - yte.mean()) ** 2).sum())
    assert 1 - rss / tss > 0.5, f"R2 tu latent chi {1 - rss / tss:.2f}"

    # Chieu latent phai doi theo `hidden`
    _, i3 = ORD.train_ordinal_mlp(Xtr, ytr, K, hidden=(64, 16), epochs=20, seed=0)
    assert i3["latent_dim"] == 16


# ------------------------------------------------------------ ASL & nguong hoc duoc

def test_asymmetric_loss_downweights_easy_negatives():
    """gamma_neg > 0 phai ha trong so cua AM DE, va khong dung toi mau DUONG."""
    import torch
    z = torch.tensor([[-4.0, 0.0, 4.0]])                  # am de, mo ho, duong tu tin
    t_neg = torch.zeros_like(z)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    l_bce = bce(z, t_neg, reduction="none")
    l_asl = -((1 - t_neg) * ((torch.sigmoid(z) - 0.05).clamp(min=0) ** 4.0)
              * torch.log((1 - (torch.sigmoid(z) - 0.05).clamp(min=0)).clamp(min=1e-8)))
    assert l_asl[0, 0] < l_bce[0, 0] * 0.01, "am DE khong duoc ha trong so"
    assert l_asl[0, 2] > l_asl[0, 0] * 100, "am KHO phai van nang"

    # gamma_pos = 0 va clip = 0 => nhanh DUONG trung khop BCE
    t_pos = torch.ones_like(z)
    a = ORD.asymmetric_loss(z, t_pos, gamma_neg=4.0, gamma_pos=0.0, clip=0.0)
    b = bce(z, t_pos)
    assert abs(float(a) - float(b)) < 1e-5, (float(a), float(b))


def test_asl_model_trains_and_lifts_rare_threshold(data):
    """Model dung ASL hoc duoc, va nang xac suat o nguong LECH NHAT (lop cuoi)."""
    Xtr, ytr, Xte, yte = data
    common = dict(oa_target=(ytr >= 2).astype(np.float32), epochs=300, seed=0)
    m_bce, i_bce = ORD.train_ordinal_mlp(Xtr, ytr, K, lambdas=ORD.SLIDE_LAMBDAS, **common)
    m_asl, i_asl = ORD.train_ordinal_mlp(Xtr, ytr, K, lambdas=ORD.ASL_LAMBDAS,
                                         asl=ORD.ASL_KW, **common)
    assert i_asl["history"][-1]["total"] < i_asl["history"][0]["total"]
    assert i_asl["asl"] == ORD.ASL_KW

    o_bce = ORD.predict_ordinal_mlp(m_bce, i_bce, Xte)
    o_asl = ORD.predict_ordinal_mlp(m_asl, i_asl, Xte)
    assert ORD.qwk(yte, o_asl["y_count"], K) > 0.6

    # Tren lop CAO NHAT, ASL phai day P(y>K-2) len cao hon BCE - do la muc dich cua no
    hi = yte == K - 1
    assert hi.sum() > 5
    assert o_asl["p_thr"][hi, -1].mean() > o_bce["p_thr"][hi, -1].mean()


def test_learned_tau_needs_exp_term_and_changes_decode(data):
    """`tau` CHI hoc duoc qua thanh phan `exp`; khong bat `exp` thi no dung yen o 0."""
    Xtr, ytr, Xte, yte = data

    # Bat learn_thresholds nhung KHONG bat `exp` => tau khong co gradient
    _, i0 = ORD.train_ordinal_mlp(Xtr, ytr, K, lambdas=ORD.SLIDE_LAMBDAS,
                                  learn_thresholds=True, epochs=100, seed=0,
                                  oa_target=(ytr >= 2).astype(np.float32))
    assert np.allclose(i0["tau"], 0.0), "tau doi ma khong co thanh phan exp"

    m, info = ORD.train_ordinal_mlp(Xtr, ytr, K, lambdas=ORD.LEARNED_THR_LAMBDAS,
                                    learn_thresholds=True, epochs=400, seed=0,
                                    oa_target=(ytr >= 2).astype(np.float32))
    assert info["tau"].shape == (K - 1,)
    assert np.abs(info["tau"]).max() > 1e-3, "tau van dung yen du da bat exp"
    assert info["history"][-1]["exp"] < info["history"][0]["exp"], "thanh phan exp khong giam"

    out = ORD.predict_ordinal_mlp(m, info, Xte)
    assert out["p_thr_tau"].shape == out["p_thr"].shape
    assert not np.allclose(out["p_thr_tau"], out["p_thr"]), "tau khong anh huong xac suat"
    assert ORD.qwk(yte, out["y_count_tau"], K) > 0.6

    # Giai ma co the TRUNG y het: tren du lieu tong hop de nay tau hoc ra rat nho va it khi
    # day duoc ca nao qua lan cat 0.5. Dieu PHAI dung la xac suat dich DUNG CHIEU cua tau,
    # vi p_adj = sigmoid(z - tau).
    k = int(np.argmax(np.abs(info["tau"])))
    lo, hi = out["p_thr_tau"][:, k].mean(), out["p_thr"][:, k].mean()
    assert (lo < hi) == (info["tau"][k] > 0), (info["tau"][k], lo, hi)

    # Model THUONG: tau = 0 nen hai cach giai ma phai trung khop tuyet doi
    m2, i2 = ORD.train_ordinal_mlp(Xtr, ytr, K, lambdas=ORD.ORDINAL_ONLY_LAMBDAS,
                                   epochs=100, seed=0)
    o2 = ORD.predict_ordinal_mlp(m2, i2, Xte)
    assert np.array_equal(o2["y_count_tau"], o2["y_count"])
    assert np.allclose(o2["tau"], 0.0)


def test_two_branch_fusion_learns_alpha_and_respects_constraint(data):
    """z = alpha*bio + beta*rad voi alpha + beta = 1 BAO DAM theo cau truc, va alpha hoc duoc."""
    Xtr, ytr, Xte, yte = data
    n_feat = Xtr.shape[1]
    bio_mask = np.zeros(n_feat, bool)
    bio_mask[: n_feat // 2] = True                       # nua dau = bio, nua sau = "radio"

    model, info = ORD.train_ordinal_mlp(Xtr, ytr, K, lambdas=ORD.ORDINAL_ONLY_LAMBDAS,
                                        epochs=300, seed=0, bio_mask=bio_mask)
    assert isinstance(model.trunk, ORD.TwoBranchTrunk)
    a = info["alpha"]
    assert a is not None and 0.0 < a < 1.0, a
    assert abs(a + (1 - a) - 1.0) < 1e-9                 # rang buoc la dong nhat thuc
    assert abs(a - 0.5) > 1e-4, "alpha dung yen o gia tri khoi tao"
    assert ORD.qwk(yte, ORD.predict_ordinal_mlp(model, info, Xte)["y_count"], K) > 0.6

    # Moi nhanh chi duoc nhin phan cot cua no
    assert len(model.trunk.bio_idx) == int(bio_mask.sum())
    assert len(model.trunk.rad_idx) == int((~bio_mask).sum())
    assert set(model.trunk.bio_idx.tolist()) & set(model.trunk.rad_idx.tolist()) == set()


def test_fusion_alpha_leans_to_the_informative_branch(data):
    """Nhanh nhieu thong tin hon phai duoc trong so lon hon."""
    Xtr, ytr, Xte, _ = data
    # Seed PHAI khac seed cua make_synthetic: default_rng(0).normal cho dung cung luong so,
    # nen "nhieu" sinh bang seed 0 se TRUNG KHOP Xtr va hai nhanh nhan cung mot ma tran.
    rng = np.random.default_rng(12345)
    noise_tr = rng.normal(size=Xtr.shape).astype(np.float32)
    assert not np.allclose(noise_tr, Xtr), "nhieu trung du lieu that - doi seed"
    bio_mask = np.r_[np.ones(Xtr.shape[1], bool), np.zeros(Xtr.shape[1], bool)]

    # bio = cot that, radio = nhieu thuan => alpha phai > 0.5
    _, i1 = ORD.train_ordinal_mlp(np.hstack([Xtr, noise_tr]), ytr, K, epochs=400, seed=0,
                                  lambdas=ORD.ORDINAL_ONLY_LAMBDAS, bio_mask=bio_mask)
    # dao lai hai nhom => alpha phai < 0.5
    _, i2 = ORD.train_ordinal_mlp(np.hstack([noise_tr, Xtr]), ytr, K, epochs=400, seed=0,
                                  lambdas=ORD.ORDINAL_ONLY_LAMBDAS, bio_mask=bio_mask)
    assert i1["alpha"] > i2["alpha"], (i1["alpha"], i2["alpha"])


def test_fusion_falls_back_when_one_group_empty(data):
    """Mot nhom rong thi khong co gi de gop: lui ve than thuong, khong tao nhanh 0 cot."""
    Xtr, ytr, Xte, _ = data
    for mask in (np.ones(Xtr.shape[1], bool), np.zeros(Xtr.shape[1], bool)):
        model, info = ORD.train_ordinal_mlp(Xtr, ytr, K, epochs=20, seed=0, bio_mask=mask)
        assert not isinstance(model.trunk, ORD.TwoBranchTrunk)
        assert info["alpha"] is None
        assert ORD.predict_ordinal_mlp(model, info, Xte)["p_thr"].shape == (len(Xte), K - 1)


def test_mlp_is_deterministic_given_seed(data):
    Xtr, ytr, Xte, _ = data
    a = ORD.predict_ordinal_mlp(*ORD.train_ordinal_mlp(Xtr, ytr, K, epochs=50, seed=3), Xte)
    b = ORD.predict_ordinal_mlp(*ORD.train_ordinal_mlp(Xtr, ytr, K, epochs=50, seed=3), Xte)
    assert np.allclose(a["p_thr"], b["p_thr"])
