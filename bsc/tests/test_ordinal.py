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
