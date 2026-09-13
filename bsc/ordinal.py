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


# ------------------------------------------------------------ nhan nguong & giai ma

def to_thresholds(y_idx, n_classes: int) -> np.ndarray:
    """y -> t [N, K-1], t_k = 1(y > k). KL3 (K=5) -> [1,1,1,0]."""
    y = np.asarray(y_idx, np.int64).reshape(-1)
    return (y[:, None] > np.arange(n_classes - 1)[None, :]).astype(np.float32)


def monotone_cummin(p) -> np.ndarray:
    """Ep p_0 >= p_1 >= ... bang cummin (cach sua kinh dien cua Frank & Hall)."""
    return np.minimum.accumulate(np.asarray(p, np.float64), axis=1)


def decode_count(p, thr: float = 0.5) -> np.ndarray:
    """y = so nguong vuot thr. Do lon p_k KHONG quan trong, chi phia nao cua thr."""
    return (np.asarray(p) > thr).sum(axis=1).astype(np.int64)


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

    def fit(self, X, y_idx, n_classes: int):
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
    return np.digitize(np.asarray(scores, np.float64), np.asarray(cuts, np.float64)).astype(np.int64)


def fit_cutpoints(scores, y_idx, n_classes: int, n_pass: int = 3, n_grid: int = 41) -> np.ndarray:
    """Toi uu K-1 diem cat theo QWK bang coordinate descent (khoi tao k+0.5).

    PHAI fit tren diem HONEST (out-of-fold trong tap train), khong phai diem train cua model
    da fit - neu khong diem cat bi lech theo overfit. Notebook S7 dung cross_val_predict.
    """
    s = np.asarray(scores, np.float64)
    y = np.asarray(y_idx)
    cuts = np.arange(n_classes - 1) + 0.5
    lo, hi = s.min() - 1.0, s.max() + 1.0
    best = qwk(y, apply_cutpoints(s, cuts), n_classes)
    for _ in range(n_pass):
        improved = False
        for k in range(len(cuts)):
            left = cuts[k - 1] + 1e-3 if k > 0 else lo
            right = cuts[k + 1] - 1e-3 if k + 1 < len(cuts) else hi
            if right <= left:
                continue
            for c in np.linspace(left, right, n_grid):
                trial = cuts.copy()
                trial[k] = c
                q = qwk(y, apply_cutpoints(s, trial), n_classes)
                if q > best + 1e-9:
                    best, cuts, improved = q, trial, True
        if not improved:
            break
    return cuts


# ------------------------------------------------------------ 3. MLP voi loss cua slide

class OrdinalMLP(nn.Module):
    """Than chung -> 3 dau: ord (K-1 logit nguong), cls (K logit softmax), oa (1 logit)."""

    def __init__(self, n_in: int, n_classes: int, hidden=(64, 32), dropout: float = 0.1):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head_ord = nn.Linear(d, n_classes - 1)
        self.head_cls = nn.Linear(d, n_classes)
        self.head_oa = nn.Linear(d, 1)
        self.n_classes = n_classes

    def forward(self, x):
        h = self.trunk(x)
        return dict(ord=self.head_ord(h), cls=self.head_cls(h), oa=self.head_oa(h).squeeze(1))


def ordinal_losses(out: dict, y_idx: torch.Tensor, t_thr: torch.Tensor,
                   oa_t: "torch.Tensor | None", lambdas: dict) -> dict:
    """Cac thanh phan loss trong slide (trang 10-11). Tra dict co 'total'."""
    l_ord = F.binary_cross_entropy_with_logits(out["ord"], t_thr)
    p = torch.sigmoid(out["ord"])
    l_mono = F.relu(p[:, 1:] - p[:, :-1]).sum(dim=1).mean() if p.shape[1] > 1 else p.new_zeros(())
    l_cls = F.cross_entropy(out["cls"], y_idx)
    if oa_t is not None:
        l_oa = F.binary_cross_entropy_with_logits(out["oa"], oa_t)
    else:
        l_oa = p.new_zeros(())
    total = (lambdas.get("ord", 0) * l_ord + lambdas.get("mono", 0) * l_mono
             + lambdas.get("cls", 0) * l_cls + lambdas.get("oa", 0) * l_oa)
    return dict(ord=l_ord, mono=l_mono, cls=l_cls, oa=l_oa, total=total)


def train_ordinal_mlp(X, y_idx, n_classes: int, oa_target=None, lambdas: "dict | None" = None,
                      hidden=(64, 32), dropout: float = 0.1, epochs: int = 400, lr: float = 1e-3,
                      weight_decay: float = 1e-4, seed: int = 0, device: str = "cpu"):
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
    model = OrdinalMLP(X.shape[1], n_classes, hidden, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        losses = ordinal_losses(model(xs), y, t, oa, lambdas)
        losses["total"].backward()
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            rec = {k: float(v.detach()) for k, v in losses.items()}
            rec["epoch"] = ep
            history.append(rec)
    model.eval()
    info = dict(mu=mu, sd=sd, n_classes=n_classes, lambdas=lambdas, history=history)
    return model, info


@torch.no_grad()
def predict_ordinal_mlp(model: OrdinalMLP, info: dict, X, device: str = "cpu") -> dict:
    """p_thr [N,K-1], p_cls [N,K], p_oa [N] + 3 cach giai ma (count / cumdiff / softmax)."""
    model.eval()
    xs = torch.tensor((np.asarray(X, np.float32) - info["mu"]) / info["sd"],
                      dtype=torch.float32, device=device)
    out = model(xs)
    p_thr = torch.sigmoid(out["ord"]).cpu().numpy()
    p_cls = torch.softmax(out["cls"], dim=1).cpu().numpy()
    p_oa = torch.sigmoid(out["oa"]).cpu().numpy()
    return dict(p_thr=p_thr, p_cls=p_cls, p_oa=p_oa,
                y_count=decode_count(p_thr), y_cumdiff=decode_cumdiff(p_thr),
                y_softmax=p_cls.argmax(axis=1).astype(np.int64),
                mono_violation=monotonic_violation_rate(p_thr))
