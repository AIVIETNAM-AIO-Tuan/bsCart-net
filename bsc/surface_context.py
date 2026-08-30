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
