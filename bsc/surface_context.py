"""Do HEADROOM cua ngu canh be mat (intra-surface context) - TRUOC khi xay graph.

VI SAO FILE NAY TON TAI
-----------------------
Stage 1 that bai voi co che: ao giac o vung `absent` chiem ~82% khoi luong loi vung mong.
Gia thuyet sua: cho moi node thay LANG GIENG tren be mat xuong (plan §4.4 Step 5,
intra-surface message passing) de biet "ca vung nay khong co sun".

Nhung plan cung canh bao (§4.4 Step 5 + test MM3): lam muot tren be mat CO THE lap luon
focal defect that. Voi failure mode la ao giac, no co the di HAI CHIEU NGUOC NHAU:
  * FP ROI RAC  -> hang xom bo phieu DAP duoc  -> graph huu ich
  * FP thanh MANG -> hang xom CUNG CO cai sai -> graph lam TE hon

=> Do TRUOC, xay SAU. Dung tinh than M0 (ROI cascade da mat mot chu ky vi xay truoc khi do).

CAC PHEP DO
-----------
1. `absence_coherence`      - vang sun co lien tuc theo khong gian khong? (co tin hieu?)
2. `neighbor_oracle_presence` - neu BIET presence GT cua lang gieng, doan duoc presence
                              cua chinh node khong? => TRAN cua ngu canh be mat.
3. `fp_isolation`           - FP cua model roi rac hay thanh mang? => graph giup hay hai?
4. `smooth_presence` + sweep - "graph nha ngheo": lam muot presence tren be mat roi do lai
                              loi vung mong. KHONG can train gi, dung checkpoint da co.

Phep 4 la quyet dinh nhat: no MO PHONG truc tiep thu ma mot lop graph lam (tong hop lang
gieng), tren chinh model da train.
"""

from __future__ import annotations

import numpy as np

from . import core, headroom, metrics, model, mvp
from .core import SPACING, RayConfig


def surface_neighbors(verts: np.ndarray, k: int = 8) -> np.ndarray:
    """K lang gieng gan nhat tren be mat (khong ke chinh no) -> [N, k] chi so.

    Dung kNN Euclid tren toa do mm thay vi ke canh mesh: node be mat tu marching-cubes
    day dac (~140k node/ca) nen kNN xap xi tot lan can trac dia, va khong phu thuoc
    chat luong tam giac hoa.
    """
    from scipy.spatial import cKDTree
    v = np.asarray(verts, np.float64)
    return cKDTree(v).query(v, k=k + 1)[1][:, 1:]


def absence_coherence(pres_true, nb: np.ndarray) -> dict:
    """Vang sun co LIEN TUC theo khong gian khong? (co tin hieu cho graph khai thac?)

    So ty le lang gieng cung `absent` cua mot node absent, voi ty le nen. Cao han han
    nen => vang sun thanh MANG, ngu canh be mat co thong tin.
    """
    a = np.asarray(pres_true).astype(bool) == False          # noqa: E712 - absent mask
    if not a.any():
        return {"base_absent_rate": float(a.mean()), "nb_absent_given_absent": np.nan,
                "lift": np.nan, "n_absent": 0}
    nb_absent = a[nb][a].mean()
    base = a.mean()
    return {"base_absent_rate": float(base),
            "nb_absent_given_absent": float(nb_absent),
            "lift": float(nb_absent / base) if base > 0 else np.nan,
            "n_absent": int(a.sum())}


def neighbor_oracle_presence(pres_true, nb: np.ndarray) -> dict:
    """TRAN cua ngu canh be mat: biet presence GT cua lang gieng thi doan duoc gi?

    Bo phieu da so tu lang gieng (KHONG dung chinh node) de du doan presence cua node.
    Day la phien ban "oracle" cua mot lop graph: no chi thay lang gieng, va lang gieng
    la GROUND TRUTH. Neu no van doan te => ngu canh be mat KHONG du de xac dinh presence
    => graph khong the cuu, du xay to den may.
    """
    t = np.asarray(pres_true).astype(bool)
    vote = t[nb].mean(axis=1) >= 0.5
    f1 = metrics.presence_f1(t, vote)
    return {"acc": float((vote == t).mean()), **f1}


def fp_isolation(pres_true, pres_pred, nb: np.ndarray) -> dict:
    """FP cua model ROI RAC hay thanh MANG? => bo phieu lang gieng dap duoc hay khong.

    Voi moi node FP (doan CO sun nhung that ra absent), do ty le lang gieng duoc doan
    DUNG (absent). Cao => FP co lap giua vung doan dung => lam muot dap duoc.
    Thap => FP thanh mang, lam muot CUNG CO cai sai (canh bao MM3 cua plan).
    """
    t = np.asarray(pres_true).astype(bool)
    p = np.asarray(pres_pred).astype(bool)
    fp = p & ~t
    if not fp.any():
        return {"n_fp": 0, "nb_correct_around_fp": np.nan, "fp_rate": 0.0}
    return {"n_fp": int(fp.sum()), "fp_rate": float(fp.mean()),
            "nb_correct_around_fp": float((~p)[nb][fp].mean())}


def smooth_presence(pres_prob, nb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """"Graph nha ngheo": tron xac suat presence voi trung binh lang gieng.

    alpha=0 -> giu nguyen; alpha=1 -> thay han bang trung binh lang gieng. Day la dang
    don gian nhat cua mot lop message-passing, du de biet TONG HOP LANG GIENG co giup
    hay khong ma khong phai train gi.
    """
    p = np.asarray(pres_prob, np.float64)
    return ((1.0 - alpha) * p + alpha * p[nb].mean(axis=1)).astype(np.float32)


def surface_context_case(run, net, src, cid, cfg: RayConfig = RayConfig(), atlas=None,
                         k: int = 8, alphas=(0.0, 0.3, 0.5, 0.7, 1.0),
                         presence_gate: str = "soft", device: str = "cpu") -> "dict | None":
    """Chay ca 4 phep do tren MOT ca. Tra dict, hoac None neu ca khong dung duoc.

    Phep 4 (sweep alpha) dung LAI checkpoint da train - khong train gi moi.
    """
    sp = src.spacing(cid)
    bone, cart = src.bone_gt(cid), src.cart_gt(cid)
    out = mvp.build_case(run, src, cid, cfg, atlas)
    if out is None:
        return None
    X, occ, pres, verts, dirs = out
    if len(X) == 0:
        return None

    op, pp = model.predict_rays(net, X, device=device)
    if pp is None:
        raise ValueError("Run nay khong co presence head (H0) - can H1 cho phep do nay.")

    nb = surface_neighbors(verts, k)
    t = pres.astype(bool)

    res = {
        "case": cid, "k": k, "n_nodes": int(len(verts)),
        "coherence": absence_coherence(t, nb),
        "nb_oracle": neighbor_oracle_presence(t, nb),
        "fp_iso": fp_isolation(t, pp > 0.5, nb),
        "sweep": {},
    }

    # Phep 4: lam muot presence roi do lai LOI VUNG MONG (metric quyet dinh §3.7)
    tf = headroom.gt_thickness_per_node(bone, cart, sp, cfg)
    for a in alphas:
        ps = smooth_presence(pp, nb, a)
        vol = model.reconstruct_volume(op, verts, dirs, cart.shape, cfg, sp,
                                       presence_prob=ps, presence_gate=presence_gate)
        pred = vol > 0.5
        e = headroom.thin_region_boundary_error(cart, pred, bone, sp, cfg, tf)
        f1 = metrics.presence_f1(t, ps > 0.5)
        # n_pred_vox=0 => lam muot XOA SACH du doan; thin_err thanh NaN. Do la KET QUA
        # co nghia (lam muot qua tay), khong phai loi - ghi lai de doc duoc.
        res["sweep"][f"{a:.2f}"] = {"thin_err_mm": e["thin_mean_err_mm"],
                                    "presence_f1": f1["f1"],
                                    "absent_recall": f1["absent_recall"],
                                    "n_pred_vox": int(pred.sum())}
    return res


# ------------------------- Ngu canh THO co them thong tin khong? (k-NN test)
#
# bsc_03 phep 1-4 do viec tong hop DU DOAN cua lang gieng (lam muot) - that bai vi model
# da sai theo mang. Nhung do KHONG tra loi: neu cho model nhin DAC TRUNG THO cua tia lan
# can thi sao? Do la thu ma 3D CNN tren slab neo phap tuyen (plan §5.2 Extension B) lam,
# va no khac han lam muot dau ra.
#
# Phep do duoi day tra loi bang k-NN PHI THAM SO - khong train gi:
#   A = doan presence tu dac trung MOT tia
#   B = doan presence tu dac trung tia + K lang gieng ghep lai
# B >> A  => ngu canh THO co them thong tin  => slab + 3D CNN dang xay
# B ~= A  => lang gieng khong them gi o muc dac trung => xay cung vo ich
#
# k-NN la baseline manh nhat co the (no "thuoc long" toan bo tap train), nen neu no khong
# tach duoc thi kien truc nao cung kho.

def _knn_presence(Xtr, ytr, Xte, k: int = 15, chunk: int = 512):
    """Bo phieu k-NN tren dac trung da chuan hoa, BRUTE-FORCE theo khoi.

    KHONG dung cKDTree: dac trung o day co ~640 chieu (tia + 4 lang gieng), ma KD-tree
    suy bien thanh duyet tuyen tinh khi >~20 chieu => cham hang GIO. Brute-force qua
    matmul (BLAS) vua CHINH XAC vua nhanh hon nhieu lan o so chieu nay:
        ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    Chi can ||b||^2 va tich vo huong vi ||a||^2 khong doi thu tu trong moi hang.
    """
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    A = ((Xtr - mu) / sd).astype(np.float32)
    B = ((Xte - mu) / sd).astype(np.float32)
    a2 = (A * A).sum(1)                                    # [Ntr]
    out = np.empty(len(B), bool)
    for i in range(0, len(B), chunk):
        blk = B[i:i + chunk]
        d2 = a2[None, :] - 2.0 * (blk @ A.T)               # bo ||b||^2 (hang so moi hang)
        idx = np.argpartition(d2, k, axis=1)[:, :k]
        out[i:i + chunk] = ytr[idx].mean(axis=1) >= 0.5
    return out


def _with_neighbors(X, nb, n_use: int = 4, sel=None):
    """Ghep dac trung node voi n_use lang gieng -> [N, (1+n_use)*D].

    `sel`: chi dung dac trung cho cac chi so nay (van lay lang gieng tu TOAN BO X).
    Khong co no, ham dung mang 140k x 640 float32 (~358 MB/ca) roi vut gan het di.
    """
    flat = X.reshape(len(X), -1)
    if sel is None:
        sel = np.arange(len(X))
    return np.concatenate([flat[sel]] + [flat[nb[sel, i]] for i in range(n_use)], axis=1)


def raw_context_knn(run, src, train_ids, val_ids, cfg: RayConfig = RayConfig(),
                    atlas=None, k_nb: int = 8, n_use: int = 4, k_knn: int = 15,
                    rays_per_case: int = 4000, hard_only: bool = True,
                    seed: int = 0, verbose: bool = True) -> dict:
    """Ngu canh THO co them thong tin khong? So k-NN mot-tia vs tia+lang-gieng.

    hard_only=True: chi xet tia KHO (absent hoac do day <=1mm) - do la cho §3.7 that bai.
    Do tren tia val, hoc tu tia train - KHONG dung nhan cua val.
    """
    def collect(ids, sd):
        Xs, ys, hs = [], [], []
        it = ids
        if verbose:
            try:
                from tqdm.auto import tqdm
                it = tqdm(ids, desc="knn build", leave=False)
            except Exception:
                pass
        for i, cid in enumerate(it):
            out = mvp.build_case(run, src, cid, cfg, atlas)
            if out is None:
                continue
            X, occ, pres, verts, _ = out
            if len(X) == 0:
                continue
            nb = surface_neighbors(verts, k_nb)
            th = core.ray_stats(occ, cfg)["thickness"]
            hard = (th <= 1.0)                       # absent + cuc mong + mong
            rng = np.random.default_rng(sd + i)
            sel = np.flatnonzero(hard) if hard_only else np.arange(len(X))
            if len(sel) > rays_per_case:
                sel = rng.choice(sel, rays_per_case, replace=False)
            if len(sel) == 0:
                continue
            # Lay mau con TRUOC roi moi ghep dac trung (tiet kiem ~350MB/ca)
            Xs.append(_with_neighbors(X, nb, n_use, sel))
            ys.append(pres[sel].astype(np.int8))
            hs.append(X.reshape(len(X), -1)[sel])
        if not Xs:
            raise ValueError("khong thu duoc tia nao")
        return np.concatenate(hs), np.concatenate(Xs), np.concatenate(ys)

    Xa1, Xan, ya = collect(train_ids, seed)          # 1 = mot tia, n = tia + lang gieng
    Xb1, Xbn, yb = collect(val_ids, seed + 500)

    pa = _knn_presence(Xa1, ya, Xb1, k_knn)
    pn = _knn_presence(Xan, ya, Xbn, k_knn)
    fa, fn = metrics.presence_f1(yb.astype(bool), pa), metrics.presence_f1(yb.astype(bool), pn)
    acc_a = float((pa == yb.astype(bool)).mean())
    acc_n = float((pn == yb.astype(bool)).mean())
    base = float(max(yb.mean(), 1 - yb.mean()))      # doan lop da so

    return {
        "n_train_rays": int(len(ya)), "n_val_rays": int(len(yb)),
        "present_rate_val": float(yb.mean()), "majority_baseline": base,
        "single_ray": {"acc": acc_a, "f1": fa["f1"], "absent_recall": fa["absent_recall"]},
        "with_neighbors": {"acc": acc_n, "f1": fn["f1"], "absent_recall": fn["absent_recall"]},
        "gain_acc": acc_n - acc_a, "gain_f1": fn["f1"] - fa["f1"],
        "single_beats_majority": bool(acc_a > base + 0.02),
        # Luat quyet dinh - ghi TRUOC khi chay
        "context_adds_info": bool(acc_n - acc_a > 0.03),
        "features_informative": bool(acc_a > base + 0.05),
    }


def channel_information_knn(src, train_ids, val_ids, cls: str,
                            cfg: RayConfig = RayConfig(), atlas=None,
                            subsets=None, k_nb: int = 8, n_use: int = 4,
                            k_knn: int = 15, rays_per_case: int = 4000,
                            seed: int = 0, verbose: bool = True) -> dict:
    """So NHIEU bo kenh cung luc - dung hinh hoc MOT LAN roi cat kenh.

    VI SAO CAN: ban `raw_context_knn` dau tien chay tren run P2 = I2 = (mri, sdf), tuc
    BO SOT `grad` - ma M8 do duoc grad dong gop +0.018 occ-Dice, gap 4.5 lan sdf (+0.004).
    Ket luan "dac trung tia khong tach duoc" khi do chi dung cho (mri, sdf), khong dung
    cho dac trung tia NOI CHUNG. Ham nay sua lo hong do.

    Dung hinh hoc mot lan voi du 3 kenh (mri, grad, sdf) roi CAT theo chi so kenh =>
    4 bo kenh chi ton 1 luot build thay vi 4.
    """
    from .experiment import RunConfig
    run = RunConfig(cls=cls, surface="S0", domain="D1", inputs="I6", heads="H1")
    CH = {"mri": 0, "grad": 1, "sdf": 2}
    if subsets is None:
        subsets = {"mri": ("mri",), "mri+grad": ("mri", "grad"),
                   "mri+sdf": ("mri", "sdf"), "mri+grad+sdf": ("mri", "grad", "sdf")}

    def collect(ids, sd):
        Xs, Xn, ys = [], [], []
        it = ids
        if verbose:
            try:
                from tqdm.auto import tqdm
                it = tqdm(ids, desc="build", leave=False)
            except Exception:
                pass
        for i, cid in enumerate(it):
            out = mvp.build_case(run, src, cid, cfg, atlas)
            if out is None:
                continue
            X, occ, pres, verts, _ = out
            if len(X) == 0:
                continue
            th = core.ray_stats(occ, cfg)["thickness"]
            sel = np.flatnonzero(th <= 1.0)             # chi tia KHO
            if len(sel) == 0:
                continue
            rng = np.random.default_rng(sd + i)
            if len(sel) > rays_per_case:
                sel = rng.choice(sel, rays_per_case, replace=False)
            nb = surface_neighbors(verts, k_nb)
            Xs.append(X[sel])                            # [n, K, 3]
            Xn.append(X[nb[sel, :n_use]])                # [n, n_use, K, 3]
            ys.append(pres[sel].astype(np.int8))
        if not Xs:
            raise ValueError("khong thu duoc tia nao")
        return np.concatenate(Xs), np.concatenate(Xn), np.concatenate(ys)

    Xa, Xa_nb, ya = collect(train_ids, seed)
    Xb, Xb_nb, yb = collect(val_ids, seed + 500)
    yb_b = yb.astype(bool)
    base = float(max(yb.mean(), 1 - yb.mean()))

    def flat(Xc, Xc_nb, idx, with_nb):
        s = Xc[..., idx].reshape(len(Xc), -1)
        if not with_nb:
            return s
        return np.concatenate(
            [s] + [Xc_nb[:, i][..., idx].reshape(len(Xc), -1) for i in range(n_use)], axis=1)

    res = {}
    for name, chans in subsets.items():
        idx = [CH[c] for c in chans]
        row = {}
        for tag, with_nb in (("single", False), ("with_nb", True)):
            p = _knn_presence(flat(Xa, Xa_nb, idx, with_nb), ya,
                              flat(Xb, Xb_nb, idx, with_nb), k_knn)
            f1 = metrics.presence_f1(yb_b, p)
            row[tag] = {"acc": float((p == yb_b).mean()), "f1": f1["f1"],
                        "recall_present": f1["recall"], "precision": f1["precision"],
                        "absent_recall": f1["absent_recall"]}
        row["lift_vs_majority"] = row["single"]["acc"] - base
        row["gain_from_neighbors"] = row["with_nb"]["acc"] - row["single"]["acc"]
        res[name] = row

    best = max(res, key=lambda n: res[n]["single"]["acc"])
    return {
        "n_train_rays": int(len(ya)), "n_val_rays": int(len(yb)),
        "present_rate_val": float(yb.mean()), "majority_baseline": base,
        "by_channels": res, "best_subset": best,
        "best_lift": res[best]["lift_vs_majority"],
        "best_gain_nb": max(v["gain_from_neighbors"] for v in res.values()),
        # Luat quyet dinh - ghi TRUOC khi chay
        "any_features_informative": bool(
            max(v["lift_vs_majority"] for v in res.values()) > 0.05),
        "neighbors_add_info": bool(
            max(v["gain_from_neighbors"] for v in res.values()) > 0.03),
    }


def summarize_surface_context(rows, baseline_thin_mm: float = None) -> dict:
    """Tong hop nhieu ca + ap luat quyet dinh (ghi TRUOC khi chay)."""
    rows = [r for r in rows if r]
    if not rows:
        return {"n": 0}
    mean = lambda f: float(np.nanmean([f(r) for r in rows]))
    alphas = sorted(rows[0]["sweep"], key=float)
    sweep = {a: {"thin_err_mm": mean(lambda r: r["sweep"][a]["thin_err_mm"]),
                 "presence_f1": mean(lambda r: r["sweep"][a]["presence_f1"]),
                 "absent_recall": mean(lambda r: r["sweep"][a]["absent_recall"]),
                 "n_pred_vox": mean(lambda r: r["sweep"][a].get("n_pred_vox", np.nan))}
             for a in alphas}
    base = sweep[alphas[0]]["thin_err_mm"]                  # alpha=0 = khong lam muot
    # NaN = lam muot xoa sach du doan (n_pred_vox=0) => KHONG duoc chon lam "best"
    finite = [a for a in alphas if np.isfinite(sweep[a]["thin_err_mm"])]
    # Khong alpha nao do duoc (model du doan rong) => best_alpha = None, khong bia ra
    best_a = min(finite, key=lambda a: sweep[a]["thin_err_mm"]) if finite else None
    gain = (base - sweep[best_a]["thin_err_mm"]) if best_a is not None else np.nan

    nb_correct = mean(lambda r: r["fp_iso"]["nb_correct_around_fp"])
    out = {
        "n": len(rows),
        "coherence_lift": mean(lambda r: r["coherence"]["lift"]),
        "nb_absent_given_absent": mean(lambda r: r["coherence"]["nb_absent_given_absent"]),
        "base_absent_rate": mean(lambda r: r["coherence"]["base_absent_rate"]),
        "nb_oracle_f1": mean(lambda r: r["nb_oracle"]["f1"]),
        "nb_oracle_absent_recall": mean(lambda r: r["nb_oracle"]["absent_recall"]),
        "fp_nb_correct": nb_correct,
        "sweep": sweep, "best_alpha": best_a,
        "smoothing_gain_mm": gain,
        "alphas_wiped_out": [a for a in alphas
                             if not np.isfinite(sweep[a]["thin_err_mm"])],
        # Luat quyet dinh - ghi TRUOC khi chay (xem docstring module)
        "signal_exists": bool(mean(lambda r: r["coherence"]["lift"]) > 1.5),
        "fp_isolated": bool(nb_correct > 0.60),
        "smoothing_helps": bool(np.isfinite(gain) and gain > 0.10),
    }
    if baseline_thin_mm is not None and best_a is not None:
        out["best_vs_baseline_mm"] = sweep[best_a]["thin_err_mm"] - baseline_thin_mm
    elif baseline_thin_mm is not None:
        out["best_vs_baseline_mm"] = float("nan")
    out["verdict"] = (
        "CO HEADROOM - ngu canh be mat dang xay (graph/neighbor pooling)"
        if (out["signal_exists"] and out["fp_isolated"] and out["smoothing_helps"]) else
        "KHONG DU HEADROOM - ngu canh be mat khong cuu duoc failure mode nay")
    return out
