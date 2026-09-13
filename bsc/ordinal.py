"""Phan loai KL theo THU TU (ordinal) tren bang biomarker - MOT dinh nghia, notebook chi goi.

Giao dien du lieu chung: X [N,F] float, y_idx [N] int trong 0..K-1 (K = so muc KL CO MAT
trong cohort, map bang sorted(unique(KL)) - cohort co the thieu KL0).

BA CACH XU LY THU TU
--------------------
1. FrankHall      K-1 bo phan loai nhi phan P(y>k) (Frank & Hall 2001; Niu et al. 2016),
                  ep don dieu bang cummin, giai ma bang DEM nguong.
2. fit_cutpoints  hoi quy diem lien tuc -> toi uu K-1 diem cat theo QWK (coordinate descent).
3. OrdinalMLP     MLP voi du 4 thanh phan LOSS cua slide "Biomarker-guided ordinal multi-task":
                  L = l_ord*BCE(nguong) + l_mono*sum relu(p_{k+1}-p_k) + l_cls*CE(softmax)
                      + l_oa*BCE(OA status).
                  SLIDE_LAMBDAS bat ca 4; ORDINAL_ONLY_LAMBDAS chi ord+mono.

DINH CHINH QUAN TRONG: day la LOSS cua slide, KHONG phai KIEN TRUC cua slide
---------------------------------------------------------------------------
Slide mo ta mot mang hoc DAU CUOI tu anh: encoder dung chung tren X_MRI, head phan doan
ra M_hat, head biomarker ra b_hat, roi module lam sang tren
z = Fusion(GAP(F), Pool(F, M_hat), phi(b_hat)).

Module nay chi lam phan phi(b_hat), tren mot BANG biomarker DA TINH SAN. Khong co encoder
nen khong co GAP(F) lan Pool(F, M_hat); khong co head nao nen khong co L_seg lan L_bio.
Chu "MLP" KHONG co trong slide - hai lop an 64/32 la lua chon cho phi, slide khong chi dinh.

Ly do rut gon: phep thu RE truoc khi bo hang tuan GPU vao kien truc day du. Neu bo may
ordinal khong thang noi softmax ngay tren bang so, noi moi thu khac da co dinh, thi kho tin
no la thu lam nen khac biet trong mang dau cuoi. CHIEU NGUOC LAI KHONG SUY RA DUOC.

GIAI MA LA QUYET DINH RIENG VOI LOSS - PHAI GHI RO KHI BAO CAO
--------------------------------------------------------------
  decode_count  : y = #{k : p_k > 0.5}                         (Niu/CORAL, mac dinh)
  decode_cumdiff: y = argmax_k (p_{k-1} - p_k), p_{-1}=1, p_{K-1}=0 (can p don dieu)

TANG QUYET DINH - fit tren diem INNER-OOF cua tap train, KHONG train lai scorer
  fit_cutpoints(objective=, min_recall=, qwk_slack=)   diem cat cho diem hoi quy (C)
  quantile_cutpoints                                    diem cat theo phan vi, 0 tham so
  fit_threshold_cuts                                    thr[K-1] thay 0.5 cho p_k (B/D/E)
  Trade-off "QWK cao <-> recall lop giua thap" la tinh chat cua MUC TIEU dat diem cat, khong
  phai cua scorer: xem docstring quantile_cutpoints. Moi quy tac deu dung chung
  _coordinate_descent; objective="qwk" khong rang buoc thi byte-identical voi ban cu.
Voi ca KL3 that, p_1, p_2 "cao qua" KHONG gay sai (t_1 = t_2 = 1); sai chi den tu
p_3 > 0.5 (thanh KL4) hoac p_2 < 0.5 (thanh KL2). Xem test_decode_rules_document_the_slide_question.

L_OA (KL>=2) TRUNG voi nguong p_1 = P(KL>1): no chi la tang trong so cho mot BCE da co.
Giu de tai hien slide, nhung `oa_target` la tham so - notebook tinh tu gia tri KL THAT
(KL>=2), khong tu chi so lop (cohort thieu KL0 thi chi so lech).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

SLIDE_LAMBDAS = dict(ord=1.0, mono=1.0, cls=1.0, oa=1.0)
ORDINAL_ONLY_LAMBDAS = dict(ord=1.0, mono=1.0, cls=0.0, oa=0.0)

#: DOI CHUNG BAT BUOC cho ORDINAL_ONLY_LAMBDAS - cung than MLP, cung so epoch, cung seed,
#: chi doi loss sang softmax CE. Thieu no thi chenh lech giua "MLP + loss nguong" va
#: "XGBoost + softmax" TRON hai thu: doi loss VA doi ho mo hinh (mang vs cay). Voi doi chung
#: nay, chenh lech D tru A2 la dong gop RIENG cua loss ordinal.
#: Giai ma bang argmax head softmax; head nguong khong duoc huan luyen nen p_thr vo nghia.
SOFTMAX_ONLY_LAMBDAS = dict(ord=0.0, mono=0.0, cls=1.0, oa=0.0)

#: ASL thay BCE o thanh phan nguong. Cung trong so voi SLIDE_LAMBDAS de so duoc 1-1.
#: Dung kem `asl=ASL_KW` khi goi train_ordinal_mlp.
ASL_LAMBDAS = dict(ord=1.0, mono=1.0, cls=1.0, oa=1.0)
ASL_KW = dict(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)

#: Them thanh phan `exp` (expected-count) - thanh phan DUY NHAT huan luyen duoc `tau`.
#: Dung kem `learn_thresholds=True`.
LEARNED_THR_LAMBDAS = dict(ord=1.0, mono=1.0, cls=1.0, oa=1.0, exp=1.0)


def asymmetric_loss(logits, targets, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                    clip: float = 0.05, eps: float = 1e-8):
    """Asymmetric Loss cho nhan nhi phan (Ridnik/Ben-Baruch et al., ICCV 2021).

        L+ = (1 - p)^gamma_pos * log(p)
        L- = p_m^gamma_neg * log(1 - p_m),   p_m = max(p - clip, 0)
        L  = -mean[ y*L+ + (1-y)*L- ]

    VI SAO HOP O DAY, khong phai y tuong ngau nhien. Phan ra nguong bien KL thanh mot bai
    toan DA NHAN voi K-1 nhan nhi phan, va cac nhan do lech RAT khac nhau tren cohort 1229 ca:

        t_0 = 1(KL>0)   945 duong / 1229   (77% duong - AM la thieu so)
        t_1 = 1(KL>1)   712 / 1229         (58%)
        t_2 = 1(KL>2)   417 / 1229         (34%)
        t_3 = 1(KL>3)   106 / 1229         ( 8.6% duong - DUONG la thieu so)

    Nguong cuoi lech 1:11. Do dung la trieu chung da do duoc: ca KL4 that co trung binh
    P(KL>3) chi 0.26-0.38, nen gan nhu khong ai duoc doan la KL4. `gamma_neg > gamma_pos`
    ha trong so cac AM DE, tuc dung cai dang lan at nguong cuoi.

    `clip` day xac suat cua mau am xuong truoc khi tinh, bo qua han cac am qua de.
    """
    p = torch.sigmoid(logits)
    t = targets
    pm = (p - clip).clamp(min=0.0) if clip > 0 else p
    l_pos = t * ((1.0 - p) ** gamma_pos) * torch.log(p.clamp(min=eps))
    l_neg = (1.0 - t) * (pm ** gamma_neg) * torch.log((1.0 - pm).clamp(min=eps))
    return -(l_pos + l_neg).mean()


# ------------------------------------------------------------ nhan nguong & giai ma

def to_thresholds(y_idx, n_classes: int) -> np.ndarray:
    """y -> t [N, K-1], t_k = 1(y > k). KL3 (K=5) -> [1,1,1,0]."""
    y = np.asarray(y_idx, np.int64).reshape(-1)
    return (y[:, None] > np.arange(n_classes - 1)[None, :]).astype(np.float32)


def monotone_cummin(p) -> np.ndarray:
    """Ep p_0 >= p_1 >= ... bang cummin (cach sua kinh dien cua Frank & Hall)."""
    return np.minimum.accumulate(np.asarray(p, np.float64), axis=1)


def decode_count(p, thr=0.5) -> np.ndarray:
    """y = so nguong vuot thr. Do lon p_k KHONG quan trong, chi phia nao cua thr.

    `thr` co the la VECTOR [K-1]: muc quyet dinh rieng tung nguong (xem fit_threshold_cuts).
    Chan hinh dang tuong minh: thr [N] voi N == K-1 se broadcast SAI truc ma numpy khong keu.
    """
    p = np.asarray(p)
    thr = np.asarray(thr, np.float64)
    assert thr.ndim == 0 or thr.shape == (p.shape[1],), \
        f"thr phai la vo huong hoac vector [K-1]={p.shape[1]}, nhan shape {thr.shape}"
    return (p > thr).sum(axis=1).astype(np.int64)


def decode_cumdiff(p) -> np.ndarray:
    """y = argmax_k (p_{k-1} - p_k). p khong don dieu => 'xac suat' am, van argmax duoc."""
    p = np.asarray(p, np.float64)
    n = len(p)
    ext = np.concatenate([np.ones((n, 1)), p, np.zeros((n, 1))], axis=1)
    q = ext[:, :-1] - ext[:, 1:]
    return q.argmax(axis=1).astype(np.int64)


def monotonic_violation_rate(p, eps: float = 1e-6) -> float:
    """Ty le hang co p_{k+1} > p_k (L_mono la phat MEM, khong dam bao = 0)."""
    p = np.asarray(p, np.float64)
    if p.shape[1] < 2:
        return 0.0
    return float((np.diff(p, axis=1) > eps).any(axis=1).mean())


# ------------------------------------------------------------ metric

def stratified_group_split(y_idx, groups, test_size: float = 0.2, seed: int = 42):
    """Chia train/test VUA phan tang theo lop VUA khong tach mot subject ra hai phia.

    Tra (train_idx, test_idx).

    VI SAO KHONG DUNG StratifiedGroupKFold CUA SKLEARN
    ---------------------------------------------------
    Do tren dung cohort nay (1229 ca / 1215 subject, KL4 chi 106 ca), so ca KL4 roi vao fold
    test:

        GroupShuffleSplit      21.6 +- 4.0    (khong phan tang gi ca)
        StratifiedGroupKFold   20.7 +- 4.0    <-- gan nhu KHONG cai thien
        StratifiedKFold        21.2 +- 0.39   (phan tang tot, nhung LAM RO RI subject)

    Trong MOT lan chia 5 fold, so ca KL4 cua StratifiedGroupKFold chay tu 16 den 25.
    Ly do: tieu chi tham lam cua no lay TRUNG BINH do lech tren MOI lop, nen lop hiem gan
    nhu khong anh huong den quyet dinh va bi hy sinh de can bang cac lop lon.

    CACH O DAY: gop ca theo subject, gan cho moi subject MOT nhan dai dien, roi phan tang
    tren SUBJECT bang StratifiedShuffleSplit. Khong subject nao bi tach doi, va lop hiem
    duoc phan bo dung ty le.

    Nhan dai dien = lop CAO NHAT trong cac ca cua subject do. Chon cao nhat vi lop nang moi
    la thu hiem va can duoc rai deu; lay trung binh hay lay ca dau tien deu lam loang no.
    Trong cohort nay chi 14/1215 subject co hon mot ca nen lua chon do it anh huong.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    y = np.asarray(y_idx, np.int64)
    g = np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)

    lab = np.zeros(len(uniq), np.int64)
    np.maximum.at(lab, inv, y)                    # nhan dai dien moi subject

    # Phan tang can moi lop co it nhat 2 subject. Tren cohort that thi thoa (106 subject
    # KL4), nhung tren tap con nho - vd sensitivity chi tren ca tin cay cao - co the khong.
    # Luc do lui ve chia theo nhom khong phan tang, va NOI RO, thay vi de no nem ValueError.
    counts = np.bincount(lab, minlength=int(lab.max()) + 1)
    if (counts[counts > 0] < 2).any():
        from sklearn.model_selection import GroupShuffleSplit
        hiem = [int(k) for k, c in enumerate(counts) if 0 < c < 2]
        print(f"CANH BAO stratified_group_split: lop {hiem} chi co 1 subject nen KHONG phan "
              f"tang duoc -> lui ve GroupShuffleSplit. Ty trong lop trong tap test se ngau nhien.")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(np.zeros((len(y), 1)), y, g))
        return tr, te

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, te_subj = next(sss.split(np.zeros((len(uniq), 1)), lab))
    te_mask = np.isin(inv, te_subj)
    tr, te = np.flatnonzero(~te_mask), np.flatnonzero(te_mask)
    assert not (set(g[tr]) & set(g[te])), "subject bi tach ra hai phia"
    return tr, te


def select_features(X, y, max_corr: float = 0.9, min_keep: int = 10,
                    max_keep: "int | None" = None, seed: int = 0) -> np.ndarray:
    """Chon dac trung khi so cot >> so ca. Tra CHI SO cac cot duoc giu.

    CHI DUOC GOI TREN TRAIN CUA TUNG FOLD. Goi mot lan tren toan bo du lieu roi dung cho
    moi fold la RO RI: buoc chon da nhin thay nhan cua tap test, va moi con so sau do deu
    lac quan. Day dung la cho ma bang dac trung san co cua S5 khong dung lai duoc.

    Buoc 1: bo cot hang so, roi bo bot cac cot gan trung nhau (|r| > max_corr), giu cot co
            chi so nho hon.
    Buoc 2: LassoCV tren y coi nhu LIEN TUC. Hop voi nhan co thu tu hon logistic da lop,
            va nhanh hon nhieu - day la khac biet co y so voi LASSO logistic cua S5.
    """
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, np.float64)
    n_feat = X.shape[1]
    if n_feat <= min_keep:
        return np.arange(n_feat)

    alive = np.flatnonzero(X.std(axis=0) > 1e-12)
    if alive.size <= min_keep:
        return alive if alive.size else np.arange(min(n_feat, min_keep))

    c = np.abs(np.nan_to_num(np.corrcoef(X[:, alive], rowvar=False)))
    keep_mask = np.ones(alive.size, bool)
    for j in range(alive.size):
        if not keep_mask[j]:
            continue
        dup = np.flatnonzero(c[j] > max_corr)
        keep_mask[dup[dup > j]] = False
    keep = alive[keep_mask]
    if keep.size <= min_keep:
        return keep

    Xs = StandardScaler().fit_transform(X[:, keep])
    # Luoi alpha truyen TUONG MINH: `n_alphas` da deprecated o sklearn moi con `alphas=int`
    # chua co o ban cu, nen truyen mang la cach duy nhat chay duoc o ca hai.
    # max_iter 10000 chu khong 3000: tren radiomics that, nhieu cot tuong quan quanh 0.85
    # lot qua duoc bo loc 0.9, va ma tran nhu vay lam coordinate descent bo rat cham =>
    # ConvergenceWarning o cac alpha nho. Do duoc tren ca that: duality gap dung o 1.1-3.0
    # lan tolerance, tuc gan hoi tu. Noi them vong lap KHONG doi hanh vi, chi cho no toi
    # dung nguong cu. (Thu hep luoi alpha cung het canh bao nhung do la doi MO HINH.)
    las = LassoCV(cv=3, alphas=np.logspace(-3, 0, 20), max_iter=10000,
                  random_state=seed, n_jobs=1)
    las.fit(Xs, np.asarray(y, np.float64))
    coef = np.abs(las.coef_)
    k = max(int((coef > 1e-8).sum()), min_keep)
    if max_keep:
        k = min(k, max_keep)
    return keep[np.argsort(-coef)[:k]]


def qwk(y_true, y_pred, n_classes: int) -> float:
    """Quadratic-weighted kappa - metric chuan cho KL grading (phat sai xa nang hon)."""
    return float(cohen_kappa_score(np.asarray(y_true), np.asarray(y_pred), weights="quadratic",
                                   labels=list(range(n_classes))))


def mae(y_true, y_pred) -> float:
    return float(np.abs(np.asarray(y_true, np.float64) - np.asarray(y_pred, np.float64)).mean())


def off_by_rate(y_true, y_pred, n: int = 2) -> float:
    """Ty le ca lech TU n bac tro len. Bao cao 13/9 dung off-by>=2 lam "loi lam sang nang".

    Kappa va accuracy la mot con so, chung khong phan biet sai gan voi sai xa. Day la con so
    tra loi truc tiep cau hoi "mo hinh co bot doan lech xa khong".
    """
    d = np.abs(np.asarray(y_true, np.int64) - np.asarray(y_pred, np.int64))
    return float((d >= n).mean())


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Ma tran nham lan [K,K]: hang = lop THAT, cot = lop DOAN. Thuan numpy (bincount)."""
    y = np.asarray(y_true, np.int64).reshape(-1)
    yp = np.asarray(y_pred, np.int64).reshape(-1)
    return np.bincount(y * n_classes + yp, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def per_class_prf(y_true, y_pred, n_classes: int) -> dict:
    """precision / recall / f1 / support tung lop, moi mang [K].

    Cung ket qua voi sklearn precision_recall_fscore_support(labels=range(K), zero_division=0)
    (test_per_class_prf_matches_sklearn) nhung thuan numpy: nhanh ~200 lan, dung duoc TRONG
    vong tim diem cat. Truoc day S7/S8 moi notebook goi sklearn inline mot ban - gop ve MOT.
    """
    cm = confusion(y_true, y_pred, n_classes).astype(np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros(n_classes), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(n_classes), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros(n_classes), where=denom > 0)
    return dict(precision=precision, recall=recall, f1=f1, support=support.astype(np.int64))


def per_class_recall(y_true, y_pred, n_classes: int) -> np.ndarray:
    return per_class_prf(y_true, y_pred, n_classes)["recall"]


def macro_recall(y_true, y_pred, n_classes: int) -> float:
    return float(per_class_recall(y_true, y_pred, n_classes).mean())


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    return float(per_class_prf(y_true, y_pred, n_classes)["f1"].mean())


#: Muc tieu cho tang quyet dinh (fit_cutpoints / fit_threshold_cuts / bootstrap_delta).
#: Cung chu ky (y_true, y_pred, n_classes) -> float, cao hon = tot hon.
OBJECTIVES = {"qwk": qwk, "macro_recall": macro_recall, "macro_f1": macro_f1}


def bootstrap_delta(y_true, yp_a, yp_b, metric, n_classes: int, n_boot: int = 2000,
                    seed: int = 0, alpha: float = 0.05) -> dict:
    """CI bootstrap cho hieu metric(b) - metric(a) tren CUNG tap test, resample CHI SO ca.

    Khac metrics.paired_bootstrap: ham do resample hieu TUNG CA, chi hop metric phan ra theo
    ca (Dice/ASSD). QWK hay macro-recall la metric cap TAP, phai tinh LAI tren moi lan
    resample. `metric`: ten trong OBJECTIVES hoac callable (y_true, y_pred, n_classes).
    Tra dict delta, ci_low, ci_high, n_boot. CI khong chua 0 => khac biet vuot nhieu resample.
    Cho n_test ~250: QWK ~0.65 ms/lan => 2000 lan x 2 ~ 3 s.
    """
    f = OBJECTIVES[metric] if isinstance(metric, str) else metric
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


def threshold_auc(p, t) -> np.ndarray:
    """AUC tung nguong P(y>k) vs t_k; NaN neu nguong chi co 1 lop."""
    p, t = np.asarray(p), np.asarray(t)
    out = np.full(p.shape[1], np.nan)
    for k in range(p.shape[1]):
        if len(np.unique(t[:, k])) == 2:
            out[k] = roc_auc_score(t[:, k], p[:, k])
    return out


def mean_p_by_class(p, y_idx, n_classes: int) -> np.ndarray:
    """Bang [K, K-1]: hang = lop that, cot = trung binh P(y>k). Ly tuong la bac thang 1..1,0..0."""
    p = np.asarray(p, np.float64)
    y = np.asarray(y_idx)
    out = np.full((n_classes, p.shape[1]), np.nan)
    for c in range(n_classes):
        m = y == c
        if m.any():
            out[c] = p[m].mean(axis=0)
    return out


# ------------------------------------------------------------ 1. Frank & Hall

class FrankHall:
    """K-1 bo phan loai nhi phan P(y>k), cung factory `make_clf` (sklearn-style fit/predict_proba)."""

    def __init__(self, make_clf, monotone: bool = True):
        self.make_clf = make_clf
        self.monotone = monotone
        self.models_ = []
        self.n_classes_ = None

    def __getstate__(self):
        """Bo `make_clf` khi pickle.

        `make_clf` thuong la mot lambda dinh nghia ngay tai cho goi, ma lambda thi khong
        pickle duoc: `AttributeError: Can't get local object '<locals>.<lambda>'`. No chi
        can luc fit; sau khi fit xong, `models_` giu cac bo phan loai DA fit va chung pickle
        binh thuong. Nen doi tuong nap lai van du doan duoc, chi la khong fit lai duoc.
        """
        st = self.__dict__.copy()
        st["make_clf"] = None
        return st

    def fit(self, X, y_idx, n_classes: int):
        if self.make_clf is None:
            raise RuntimeError(
                "FrankHall nay duoc nap tu file nen khong con `make_clf` (lambda khong pickle "
                "duoc). Doi tuong nap lai chi du doan duoc, khong fit lai duoc. Muon fit thi "
                "tao moi: FrankHall(lambda: XGBClassifier(...))."
            )
        t = to_thresholds(y_idx, n_classes)
        self.n_classes_ = n_classes
        self.models_ = []
        for k in range(n_classes - 1):
            tk = t[:, k].astype(int)
            if tk.min() == tk.max():                    # nguong suy bien trong fold nay
                self.models_.append(float(tk[0]))
                continue
            clf = self.make_clf()
            clf.fit(X, tk)
            self.models_.append(clf)
        return self

    def predict_proba_thresholds(self, X) -> np.ndarray:
        n = len(X)
        p = np.zeros((n, self.n_classes_ - 1))
        for k, m in enumerate(self.models_):
            p[:, k] = m if isinstance(m, float) else m.predict_proba(X)[:, 1]
        return monotone_cummin(p) if self.monotone else p

    def predict(self, X) -> np.ndarray:
        return decode_count(self.predict_proba_thresholds(X))


# ------------------------------------------------------------ 2. diem cat tren hoi quy

def apply_cutpoints(scores, cuts) -> np.ndarray:
    cuts = np.asarray(cuts, np.float64)
    # np.digitize nhan bins GIAM DAN ma khong keu -> mot bug sinh cut khong don dieu se cho rac.
    assert (np.diff(cuts) >= 0).all(), f"diem cat phai khong giam, nhan {cuts}"
    return np.digitize(np.asarray(scores, np.float64), cuts).astype(np.int64)


def _coordinate_descent(objective, init, bounds, n_pass: int, n_grid: int):
    """Leo doi tung toa do tren luoi. Tra (x, best).

    `objective(x) -> float`: cao hon = tot hon; -inf = bat kha thi.
    `bounds(k, x) -> (left, right)`: khoang luoi cho toa do k tai trang thai x hien tai.
    GIU NGUYEN tung chi tiet cua vong lap fit_cutpoints cu (linspace, le 1e-3 do bounds cap,
    chap nhan khi > best + 1e-9, dung khi mot pass khong cai thien) - test golden bao ve.
    """
    x = np.array(init, np.float64)
    best = objective(x)
    for _ in range(n_pass):
        improved = False
        for k in range(len(x)):
            left, right = bounds(k, x)
            if right <= left:
                continue
            for c in np.linspace(left, right, n_grid):
                trial = x.copy()
                trial[k] = c
                v = objective(trial)
                if v > best + 1e-9:
                    best, x, improved = v, trial, True
        if not improved:
            break
    return x, best


def _resolve_objective(objective, n_classes: int):
    """Ten trong OBJECTIVES -> callable (y_true, y_pred); callable thi giu nguyen."""
    if isinstance(objective, str):
        if objective not in OBJECTIVES:
            raise ValueError(f"objective phai la mot trong {list(OBJECTIVES)} hoac callable, nhan {objective!r}")
        f = OBJECTIVES[objective]
        return lambda yt, yp: f(yt, yp, n_classes)
    return objective


def _fit_decision(y, decode, init, bounds, n_classes, objective, min_recall, qwk_slack,
                  n_pass, n_grid, who: str) -> np.ndarray:
    """Loi chung cua fit_cutpoints va fit_threshold_cuts.

    Khong rang buoc: mot lan leo doi theo `objective`.
    qwk_slack   : toi da `objective` s.t. qwk >= qwk_best - slack. HAI PASS: pass 1 tim diem
                  toi uu QWK (qwk_best), pass 2 KHOI DAU TU diem do - kha thi theo cau truc -
                  roi leo doi theo `objective` chi qua cac diem con thoa rang buoc.
    min_recall  : loai moi vector co lop nao recall < san. Khoi dau co the bat kha thi (-inf);
                  khi do bat ky diem kha thi nao tren luoi deu duoc nhan. Khong tim thay diem
                  kha thi => CANH BAO va tra ket qua KHONG rang buoc (bang tham chieu).
    """
    f = _resolve_objective(objective, n_classes)
    obj = lambda x: f(y, decode(x))
    x_unc, best_unc = _coordinate_descent(obj, init, bounds, n_pass, n_grid)
    if min_recall is None and qwk_slack is None:
        return x_unc

    start, q_floor = x_unc, None
    if qwk_slack is not None:
        if objective == "qwk":
            x_q, q_best = x_unc, best_unc
        else:
            x_q, q_best = _coordinate_descent(lambda x: qwk(y, decode(x), n_classes),
                                              init, bounds, n_pass, n_grid)
        q_floor, start = q_best - qwk_slack, x_q

    def constrained(x):
        yp = decode(x)
        if min_recall is not None and per_class_recall(y, yp, n_classes).min() < min_recall:
            return -np.inf
        if q_floor is not None and qwk(y, yp, n_classes) < q_floor:
            return -np.inf
        return f(y, yp)

    x_c, best_c = _coordinate_descent(constrained, start, bounds, n_pass, n_grid)
    if not np.isfinite(best_c):
        print(f"CANH BAO {who}: khong co diem nao thoa rang buoc (min_recall={min_recall}, "
              f"qwk_slack={qwk_slack}) -> tra ve ket qua KHONG rang buoc")
        return x_unc
    return x_c


def fit_cutpoints(scores, y_idx, n_classes: int, objective="qwk", min_recall=None, qwk_slack=None,
                  n_pass: int = 3, n_grid: int = 41) -> np.ndarray:
    """Toi uu K-1 diem cat bang coordinate descent (khoi tao k+0.5). Mac dinh = QWK, khong rang buoc.

    PHAI fit tren diem HONEST (inner out-of-fold trong tap train), khong phai diem train cua
    model da fit - neu khong diem cat bi lech theo overfit. Dung inner_oof de lay diem do.

    objective  : "qwk" | "macro_recall" | "macro_f1" | callable (y_true, y_pred) -> float.
    min_recall : san recall cho MOI lop (None = khong rang buoc).
    qwk_slack  : cho phep QWK thap hon toi uu toi da `slack` de doi lay `objective`.
    Mac dinh (objective="qwk", khong rang buoc) BYTE-IDENTICAL voi ban truoc: xem
    test_fit_cutpoints_default_is_byte_identical. CANH BAO: objective="macro_recall" khong
    rang buoc overfit vi tri cat tren ~80 ca lop hiem (do tren tong hop: 2/4 seed te hon ca hai
    mat tren held-out). Uu tien quantile_cutpoints hoac qwk_slack.
    """
    s = np.asarray(scores, np.float64).reshape(-1)
    y = np.asarray(y_idx, np.int64).reshape(-1)
    init = np.arange(n_classes - 1) + 0.5
    lo, hi = s.min() - 1.0, s.max() + 1.0

    def bounds(k, x):
        left = x[k - 1] + 1e-3 if k > 0 else lo
        right = x[k + 1] - 1e-3 if k + 1 < len(x) else hi
        return left, right

    return _fit_decision(y, lambda x: apply_cutpoints(s, x), init, bounds, n_classes, objective,
                         min_recall, qwk_slack, n_pass, n_grid, who="fit_cutpoints")


def quantile_cutpoints(scores, y_idx, n_classes: int) -> np.ndarray:
    """Diem cat theo PHAN VI: marginal du doan tren train == marginal that. KHONG co tham so de fit.

    VI SAO. QWK quadratic = 2 Cov(y, yp) / (Var y + Var yp + (mean y - mean yp)^2), tuc he so
    tuong hop Lin. Diem hoi quy bi CO ve trung binh (Var score < Var y), nen diem cat toi uu
    QWK NOI hai bin ngoai cung de bom Var(yp) len: co y doan THUA lop dau/cuoi va BOP lop giua
    (S8: C recall KL3 48% thap nhat, KL4 81% cao nhat, precision KL4 72%). Dat cut sao cho ty
    le du doan = ty le that thi Var(yp) ~ Var(y) ma khong phai fit gi. Tren diem co tong hop
    (test_quantile_beats_variance_inflation_on_shrunk_scores): thang cut-QWK ca ve QWK
    held-out lan recall thap nhat, 4/4 seed.

    Chinh xac khi diem khong trung nhau (tie tai diem cat roi len bin tren). Lop vang trong y
    => hai cut bang nhau => lop do khong bao gio duoc doan (dung y).
    """
    s = np.asarray(scores, np.float64).reshape(-1)
    y = np.asarray(y_idx, np.int64).reshape(-1)
    n = len(s)
    ss = np.sort(s)
    cum = np.cumsum(np.bincount(y, minlength=n_classes))[:-1]
    cuts = np.empty(n_classes - 1)
    for k, c in enumerate(cum):
        if c <= 0:
            cuts[k] = ss[0] - 1.0
        elif c >= n:
            cuts[k] = ss[-1] + 1.0
        else:
            cuts[k] = 0.5 * (ss[c - 1] + ss[c])
    return cuts


def fit_threshold_cuts(p, y_idx, n_classes: int, objective="qwk", min_recall=None, qwk_slack=None,
                       n_pass: int = 3, n_grid: int = 19, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    """Muc quyet dinh RIENG tung nguong thr[K-1] thay cho 0.5 co dinh; giai ma decode_count(p, thr).

    Nham dung nguong cuoi: ca KL4 that co P(KL>3) trung binh 0.46-0.57 (S7) nen gan mot nua
    thua vach 0.5, trong khi AUC cua chinh nguong do la 0.952 - cao nhat trong bon. Tuc phan
    biet duoc, chi hieu chinh sai. Fit tren p INNER-OOF cua tap train (nhu fit_cutpoints).
    Luoi 19 diem tren [0.05, 0.95] chua dung 0.5 => in-sample khong bao gio te hon 0.5 co dinh.

    KHONG ep thr don dieu: toi uu do duoc tren tong hop la [0.65, 0.35, 0.55, 0.20]; ep se chan
    dung cai can sua. He qua la dem co the nhan mau nhu [1,0,1,0] - do bang
    monotonic_violation_rate((p > thr).astype(float)) va BAO CAO, khong chan.
    Tuong duong `tau` cua OrdinalMLP(learn_thresholds=True): thr_k = sigmoid(tau_k).
    """
    pp = np.asarray(p, np.float64)
    y = np.asarray(y_idx, np.int64).reshape(-1)
    init = np.full(pp.shape[1], 0.5)
    return _fit_decision(y, lambda x: decode_count(pp, thr=x), init, lambda k, x: (lo, hi),
                         n_classes, objective, min_recall, qwk_slack, n_pass, n_grid,
                         who="fit_threshold_cuts")


# ------------------------------------------------------------ 2b. diem honest & trong so lop

def inner_oof(fit_predict, X, y_idx, groups, n_splits: int = 4, row_kwargs: "dict | None" = None) -> np.ndarray:
    """Diem HONEST (out-of-fold) tren chinh tap train, cho scorer BAT KY (FrankHall, MLP torch...).

    `fit_predict(Xtr, ytr, Xte, **kw) -> diem cho Xte` (1-D hoac [n, d]). Moi phan tu cua
    `row_kwargs` la mang theo HANG (vd sample_weight, oa_target), duoc cat theo tr roi truyen.
    GroupKFold KHONG shuffle => trung fold voi cross_val_predict(cv=GroupKFold(n)) ma S7/S8
    dang dung cho C, nen C tai lap bit-doi-bit (test_inner_oof_no_leakage_and_matches_sklearn).
    Khong hang nao duoc cham diem boi model da thay no.
    """
    from sklearn.model_selection import GroupKFold

    X = np.asarray(X)
    y = np.asarray(y_idx)
    g = np.asarray(groups)
    row_kwargs = {k: np.asarray(v) for k, v in (row_kwargs or {}).items()}
    out = None
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, g):
        kw = {k: v[tr] for k, v in row_kwargs.items()}
        pred = np.asarray(fit_predict(X[tr], y[tr], X[te], **kw), np.float64)
        if out is None:
            out = np.full((len(X),) + pred.shape[1:], np.nan)
        out[te] = pred
    assert out is not None and not np.isnan(out).any(), "inner_oof: con hang chua duoc cham diem"
    return out


def class_balanced_weights(y_idx, n_classes: int) -> np.ndarray:
    """w_i = N / (K' * n_{y_i}) voi K' = so lop CO MAT: tong trong so moi lop bang nhau, mean w = 1.

    Truc KHAC voi tang quyet dinh: doi DIEM (bo hoi quy bot co ve trung binh o lop hiem), khong
    doi diem cat. Danh gia rieng, dung tron voi cac quy tac cat khi doc bang.
    """
    y = np.asarray(y_idx, np.int64).reshape(-1)
    cnt = np.bincount(y, minlength=n_classes).astype(np.float64)
    present = cnt > 0
    w_cls = np.zeros(n_classes)
    w_cls[present] = len(y) / (present.sum() * cnt[present])
    return w_cls[y]


# ------------------------------------------------------------ 3. MLP voi loss cua slide

def _mlp(d_in: int, d_out: int, hidden=(), dropout: float = 0.1) -> nn.Module:
    """Chuoi Linear-ReLU-Dropout roi mot Linear cuoi. `hidden` rong => dung mot Linear."""
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def _trunk(d_in: int, hidden, dropout: float):
    """Chuoi Linear-ReLU-Dropout, KHONG co Linear cuoi. Tra (Sequential, chieu ra)."""
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers), d


class TwoBranchTrunk(nn.Module):
    """Hai nhanh rieng cho hai NHOM dac trung, gop bang trong so hoc duoc.

        z = alpha * f_bio(x_bio) + beta * f_rad(x_rad),   alpha + beta = 1

    Rang buoc duoc BAO DAM theo cau truc, khong phai bang phat: alpha = sigmoid(gate) va
    beta = 1 - alpha, nen tong luon bang 1 va ca hai luon trong [0, 1]. Khong can chuan hoa
    lai, khong the troi ra ngoai.

    VI SAO DANG LAM. Bao cao 13/9 ket luan radiomics dong gop phan lon muc tang kappa, nhung
    do la suy ra TU CHENH LECH giua hai mo hinh. O day `alpha` la mot tham so DOC RA DUOC:
    sau khi huan luyen, no noi thang mo hinh dua vao biomarker hinh hoc bao nhieu phan.
    Hai nhanh cho ra cung chieu latent nen phep gop co nghia.

    Chu y khi doc alpha: no la trong so tren BIEU DIEN da hoc, khong phai ty le thong tin.
    Mot nhanh co the cho vector bien do lon hon va bu lai bang alpha nho. So sanh alpha giua
    cac lan chay chi co nghia khi cung feature set, cung seed va cung so epoch.
    """

    def __init__(self, bio_mask, hidden=(64, 32), dropout: float = 0.1):
        super().__init__()
        m = np.asarray(bio_mask, bool)
        self.register_buffer("bio_idx", torch.as_tensor(np.flatnonzero(m), dtype=torch.long))
        self.register_buffer("rad_idx", torch.as_tensor(np.flatnonzero(~m), dtype=torch.long))
        self.branch_bio, d1 = _trunk(int(m.sum()), hidden, dropout)
        self.branch_rad, d2 = _trunk(int((~m).sum()), hidden, dropout)
        assert d1 == d2, "hai nhanh phai cho cung chieu latent moi gop duoc"
        self.out_dim = d1
        self.gate = nn.Parameter(torch.zeros(1))      # alpha = sigmoid(0) = 0.5 luc khoi tao

    @property
    def alpha(self) -> float:
        return float(torch.sigmoid(self.gate).detach())

    def forward(self, x):
        a = torch.sigmoid(self.gate)
        return a * self.branch_bio(x[:, self.bio_idx]) + (1 - a) * self.branch_rad(x[:, self.rad_idx])


class OrdinalMLP(nn.Module):
    """Than chung -> LATENT -> 3 dau: ord (K-1 logit nguong), cls (K logit), oa (1 logit).

    `hidden`      kien truc THAN. Dau ra cua than la LATENT, chieu = hidden[-1].
    `head_hidden` kien truc tung DAU. Rong () => dau la mot lop Linear (mac dinh, giu nguyen
                  hanh vi cu de so sanh duoc voi cac lan chay truoc). Vd (32,) => dau la MLP.

    LATENT dung de lam gi: no la vector dai dien cua mot ca sau khi mo hinh da nen bang
    biomarker lai. Dung duoc cho t-SNE/UMAP, phan cum, do tuong dong giua ca, hoac lam dau
    vao cho mot mo hinh khac. Tuong duong voi vector CLS trong cac kien truc transformer:
    cung la mot vector duy nhat ma moi dau du doan deu doc tu do.

    CANH BAO khi dung latent: no duoc HUAN LUYEN tren tap train. Trich latent cho ca trong
    tap train roi phan tich chung voi latent cua tap test la tron hai che do khac nhau -
    latent cua ca train da bi mo hinh nhin thay nhan. Luon giu cot danh dau train/test khi
    xuat latent ra file.
    """

    def __init__(self, n_in: int, n_classes: int, hidden=(64, 32), dropout: float = 0.1,
                 head_hidden=(), learn_thresholds: bool = False, bio_mask=None):
        super().__init__()
        # bio_mask khac None va CA HAI nhom deu co cot => than hai nhanh co trong so alpha.
        # Neu mot nhom rong (vd chua nap radiomics, hoac buoc chon loai het) thi khong co gi
        # de gop, lui ve than thuong thay vi tao mot nhanh 0 cot.
        two = bio_mask is not None and 0 < int(np.sum(np.asarray(bio_mask, bool))) < len(bio_mask)
        if two:
            self.trunk = TwoBranchTrunk(bio_mask, hidden, dropout)
            d = self.trunk.out_dim
        else:
            self.trunk, d = _trunk(n_in, hidden, dropout)
        self.latent_dim = d
        self.head_ord = _mlp(d, n_classes - 1, head_hidden, dropout)
        self.head_cls = _mlp(d, n_classes, head_hidden, dropout)
        self.head_oa = _mlp(d, 1, head_hidden, dropout)
        self.n_classes = n_classes
        # `tau` = do lech quyet dinh cua TUNG nguong, hoc duoc. Giai ma thanh z_k > tau_k
        # thay vi p_k > 0.5. No CHI co gradient qua thanh phan `exp` trong ordinal_losses;
        # neu khong bat `exp` thi tau dung yen o 0 va model y het ban thuong.
        if learn_thresholds:
            self.tau = nn.Parameter(torch.zeros(n_classes - 1))
        else:
            self.register_parameter("tau", None)

    def forward(self, x):
        h = self.trunk(x)
        return dict(ord=self.head_ord(h), cls=self.head_cls(h),
                    oa=self.head_oa(h).squeeze(1), latent=h, tau=self.tau)

    @property
    def alpha(self):
        """Trong so cua nhanh biomarker, None neu khong dung than hai nhanh."""
        return self.trunk.alpha if isinstance(self.trunk, TwoBranchTrunk) else None


def ordinal_losses(out: dict, y_idx: torch.Tensor, t_thr: torch.Tensor,
                   oa_t: "torch.Tensor | None", lambdas: dict,
                   asl: "dict | None" = None, temp: float = 1.0) -> dict:
    """Cac thanh phan loss trong slide (trang 10-11), cong hai mo rong. Tra dict co 'total'.

    `asl`  khac None => thanh phan nguong dung Asymmetric Loss thay BCE.
    `exp`  trong `lambdas` bat thanh phan EXPECTED-COUNT:

               y_soft = sum_k sigmoid((z_k - tau_k) / temp)      va   MSE(y_soft, y)

           Day la ban LIEN TUC, kha vi cua chinh quy tac giai ma "dem so nguong vuot".
           No la thanh phan DUY NHAT co gradient chay vao `tau`: BCE chi quan tam tung
           nguong co dung khong, khong quan tam TONG cua chung co ra dung lop khong.
    """
    z = out["ord"]
    l_ord = asymmetric_loss(z, t_thr, **asl) if asl else         F.binary_cross_entropy_with_logits(z, t_thr)
    p = torch.sigmoid(z)
    l_mono = F.relu(p[:, 1:] - p[:, :-1]).sum(dim=1).mean() if p.shape[1] > 1 else p.new_zeros(())
    l_cls = F.cross_entropy(out["cls"], y_idx)
    if oa_t is not None:
        l_oa = F.binary_cross_entropy_with_logits(out["oa"], oa_t)
    else:
        l_oa = p.new_zeros(())
    if lambdas.get("exp", 0) > 0:
        tau = out.get("tau")
        zz = z if tau is None else z - tau
        y_soft = torch.sigmoid(zz / temp).sum(dim=1)
        l_exp = F.mse_loss(y_soft, y_idx.float())
    else:
        l_exp = p.new_zeros(())

    total = (lambdas.get("ord", 0) * l_ord + lambdas.get("mono", 0) * l_mono
             + lambdas.get("cls", 0) * l_cls + lambdas.get("oa", 0) * l_oa
             + lambdas.get("exp", 0) * l_exp)
    return dict(ord=l_ord, mono=l_mono, cls=l_cls, oa=l_oa, exp=l_exp, total=total)


def train_ordinal_mlp(X, y_idx, n_classes: int, oa_target=None, lambdas: "dict | None" = None,
                      hidden=(64, 32), dropout: float = 0.1, epochs: int = 400, lr: float = 1e-3,
                      weight_decay: float = 1e-4, seed: int = 0, device: str = "cpu",
                      head_hidden=(), learn_thresholds: bool = False,
                      asl: "dict | None" = None, temp: float = 1.0, bio_mask=None):
    """Full-batch Adam tren bang nho. Tra (model, info) - info co mu/sd de chuan hoa luc predict."""
    lambdas = dict(SLIDE_LAMBDAS if lambdas is None else lambdas)
    X = np.asarray(X, np.float32)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    xs = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(y_idx, np.int64), device=device)
    t = torch.tensor(to_thresholds(y_idx, n_classes), device=device)
    oa = None
    if oa_target is not None and lambdas.get("oa", 0) > 0:
        oa = torch.tensor(np.asarray(oa_target, np.float32), device=device)

    torch.manual_seed(seed)
    model = OrdinalMLP(X.shape[1], n_classes, hidden, dropout, head_hidden,
                       learn_thresholds, bio_mask).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        losses = ordinal_losses(model(xs), y, t, oa, lambdas, asl=asl, temp=temp)
        losses["total"].backward()
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            rec = {k: float(v.detach()) for k, v in losses.items()}
            rec["epoch"] = ep
            history.append(rec)
    model.eval()
    info = dict(mu=mu, sd=sd, n_classes=n_classes, lambdas=lambdas, history=history,
                hidden=tuple(hidden), head_hidden=tuple(head_hidden),
                latent_dim=model.latent_dim, temp=temp, asl=asl,
                tau=(model.tau.detach().cpu().numpy().copy() if model.tau is not None else None),
                alpha=model.alpha)
    return model, info


@torch.no_grad()
def predict_ordinal_mlp(model: OrdinalMLP, info: dict, X, device: str = "cpu") -> dict:
    """p_thr [N,K-1], p_cls [N,K], p_oa [N], latent [N,d] + 3 cach giai ma.

    `latent` la dau ra cua than, tuc vector dai dien cua tung ca. Xem docstring OrdinalMLP
    ve canh bao tron latent cua tap train voi tap test.
    """
    model.eval()
    xs = torch.tensor((np.asarray(X, np.float32) - info["mu"]) / info["sd"],
                      dtype=torch.float32, device=device)
    out = model(xs)
    z = out["ord"]
    p_thr = torch.sigmoid(z).cpu().numpy()
    # Giai ma CO tau: z_k > tau_k, tuong duong sigmoid(z_k - tau_k) > 0.5.
    # tau = 0 (model thuong) thi y_count_tau trung y_count.
    tau = model.tau.detach() if model.tau is not None else torch.zeros(z.shape[1], device=z.device)
    p_adj = torch.sigmoid(z - tau).cpu().numpy()
    p_cls = torch.softmax(out["cls"], dim=1).cpu().numpy()
    p_oa = torch.sigmoid(out["oa"]).cpu().numpy()
    return dict(p_thr=p_thr, p_thr_tau=p_adj, p_cls=p_cls, p_oa=p_oa,
                tau=tau.cpu().numpy(),
                latent=out["latent"].cpu().numpy(),
                y_count=decode_count(p_thr), y_count_tau=decode_count(p_adj),
                y_cumdiff=decode_cumdiff(p_thr),
                y_softmax=p_cls.argmax(axis=1).astype(np.int64),
                mono_violation=monotonic_violation_rate(p_thr))
