"""Sinh hinh cho bao cao 14/09 tu `report_data.py`.

    cd bsc && python docs/make_figs.py

Moi con so nam o `report_data.py`, khong go tay o day. Hinh ra `docs/figs/*.png`.

Bang mau lay tu bo mau tham chieu cua skill dataviz, da chay validator:
3 slot dau (xanh duong / cam / xanh ngoc) dat moi cong CVD o che do all-pairs.
Cot nao cung co nhan so truc tiep, nen khong dua vao mau de doc gia tri.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import report_data as D

OUT = Path(__file__).parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# ---- bang mau (bo tham chieu dataviz, che do sang) -----------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD, CRIT = "#0ca30c", "#d03b3b"
SEQ = LinearSegmentedColormap.from_list(       # thang xanh mot mau, nhat -> dam
    "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "figure.dpi": 150,
})
KL = [f"KL{c}" for c in D.CLASSES]


def _finish(fig, name, note=None):
    if note:
        # -0.06 chu khong phai 0.005: chu thich phai nam DUOI nhan truc, khong de len no
        fig.text(0.008, -0.06, note, fontsize=6.8, color=MUTED, ha="left", va="top")
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("  ->", name)


def _bars(ax, y, w, labels, colors, fmt="{:.3f}", pad=0.004, fs=7.5):
    """Cot ngang, bo goc dau du lieu, nhan so TRUC TIEP (bat buoc: mot mau duoi 3:1)."""
    b = ax.barh(y, w, height=0.62, color=colors, zorder=3)
    for r, v, lab in zip(b, w, labels):
        ax.text(r.get_width() + pad * np.sign(v or 1), r.get_y() + r.get_height() / 2,
                fmt.format(lab), va="center", ha="left" if v >= 0 else "right",
                fontsize=fs, color=INK2)
    return b


# =====================================================================================
def fig1_s6():
    """S6: cot moi bat duoc mat sun, cot cu thi khong."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.5),
                                 gridspec_kw=dict(width_ratios=[1.15, 1]))

    names = [n for n, _, _ in D.S6_RHO][::-1]
    rhos = [r for _, r, _ in D.S6_RHO][::-1]
    fams = [f for _, _, f in D.S6_RHO][::-1]
    cols = [BLUE if f == "moi" else CRIT for f in fams]
    y = np.arange(len(names))
    a1.barh(y, rhos, height=0.62, color=cols, zorder=3)
    for i, (v, n) in enumerate(zip(rhos, names)):
        a1.text(v + (0.012 if v >= 0 else -0.012), i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5, color=INK2)
    a1.set_yticks(y, names, fontsize=7.6)
    a1.axvline(0, color=BASELINE, lw=1)
    a1.set_xlim(-0.10, 0.52)
    a1.set_xlabel("Spearman $\\rho$ voi KL")
    a1.set_title("a. Tuong quan voi do KL, 1229 ca", loc="left")
    a1.xaxis.grid(True, zorder=0)
    a1.set_axisbelow(True)
    h = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=CRIT)]
    a1.legend(h, ["Ho be mat S6 (moi)", "Bang S3 (cu)"], loc="lower right",
              frameon=False, fontsize=7.5, labelcolor=INK2)

    for (lab, vals), c in zip(D.S6_FCL_BY_KL.items(), SERIES):
        a2.plot(D.CLASSES, vals, marker="o", ms=6, lw=2, color=c, label=lab, zorder=3)
        a2.text(D.CLASSES[-1] + 0.09, vals[-1], f"{vals[-1]:.2f}%", color=c,
                fontsize=7.5, va="center", fontweight="bold")
    a2.set_xticks(D.CLASSES, KL)
    a2.set_xlim(-0.25, 4.75)
    a2.set_ylabel("% footprint bi tro (trung vi)")
    a2.set_title("b. Mat sun toan be day tang theo KL", loc="left")
    a2.yaxis.grid(True, zorder=0)
    a2.set_axisbelow(True)
    a2.legend(frameon=False, fontsize=7.5, loc="upper left", labelcolor=INK2)

    fig.suptitle("Hinh 1 — S6: biomarker neo be mat xuong do duoc mat sun, cot cu thi khong",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.04)
    _finish(fig, "fig1_s6_fcl.png",
            "Nguon: biomarker_s6_fcl.ipynb muc 5, 1229 ca. Cot cu `denuded_ratio_tibial` chi dat 0.105; "
            "cot moi manh gap ~4 lan.")


def fig2_s7_qwk():
    """S7: QWK theo model, ba bo dac trung. Out-of-fold n=1229."""
    fs_show = ["legacy_s3", "s6_all", "s6_all_plus_radiomics"]
    models = D.S7_MODELS
    keys = ["A", "A2", "B", "C", "D", "E"]
    x = np.arange(len(models))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.2, 4.1))
    for j, (fs, c) in enumerate(zip(fs_show, SERIES)):
        vals = [D.S7_QWK[fs][k] for k in keys]
        off = (j - 1) * (w + 0.015)
        ax.bar(x + off, vals, w, color=c, label=D.S7_FS_LABEL[fs], zorder=3)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.8, color=INK2,
                    rotation=90, va="bottom")
    ax.set_xticks(x, [D.S7_MODEL_SHORT[m] for m in models], fontsize=8)
    ax.set_ylim(0.5, 0.85)
    ax.set_ylabel("QWK (out-of-fold, n=1229)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    # Legend RA NGOAI truc: dat trong se de len hai nhan nhom ben duoi
    ax.legend(frameon=False, fontsize=8, ncol=3, labelcolor=INK2,
              loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.axvspan(-0.5, 1.5, color="#f0efec", zorder=0)
    ax.text(0.5, 0.822, "danh danh\n(khong dung thu tu)", ha="center", fontsize=7.5, color=MUTED)
    ax.text(3.5, 0.822, "co dung thu tu (ordinal)", ha="center", fontsize=7.5, color=MUTED)
    fig.suptitle("Hinh 2 — S7: xu ly KL nhu thang thu tu hon coi la nam lop roi rac",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.09)
    _finish(fig, "fig2_s7_qwk.png",
            "Nguon: biomarker_s7_ordinal.ipynb muc 4. Trung binh 3 seed, 15 fold chia theo subject. "
            "Do lech chuan qua seed 0.002-0.015.")


def _confusion(ax, cm, title, sub=None):
    cm = np.asarray(cm, float)
    row = cm.sum(1, keepdims=True)
    norm = cm / np.maximum(row, 1)
    ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            v = norm[i, j]
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if v > 0.45 else INK2,
                    fontweight="bold" if i == j else "normal")
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 5, 5, fill=False, ec=BASELINE, lw=1))
    for i in range(5):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=GOOD, lw=1.6))
    ax.set_xticks(range(5), KL, fontsize=8)
    ax.set_yticks(range(5), KL, fontsize=8)
    ax.set_xlabel("KL doan", fontsize=8.5)
    ax.set_ylabel("KL that", fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=9.5, pad=22)
    if sub:
        # transAxes, KHONG phai transData: truc imshow lat nguoc nen y am se nhay len tren title
        ax.text(0, 1.02, sub, fontsize=7.6, color=MUTED, ha="left", va="bottom",
                transform=ax.transAxes)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def fig3_s7_confusion():
    """S7: ma tran nham lan cua hai model ordinal tot nhat, n=1229."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    _confusion(axes[0], D.S7_CONFUSION["B_xgb_frankhall"], "B — Frank-Hall",
               "QWK 0.776  ·  MAE 0.533  ·  macro F1 54.5%")
    _confusion(axes[1], D.S7_CONFUSION["D_mlp_ordinal_only"], "D — MLP loss nguong",
               "QWK 0.774  ·  MAE 0.547  ·  macro F1 54.3%")
    fig.suptitle("Hinh 3 — S7: ma tran nham lan, out-of-fold tren ca 1229 ca",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.10)
    _finish(fig, "fig3_s7_confusion.png",
            "Nguon: biomarker_s7_ordinal.ipynb muc 5, bo dac trung day du (926 cot), seed 0. "
            "O to dam = ty le trong hang; so la dem ca. Vien xanh = doan dung.")


def fig4_s7_f1():
    """S7: F1 theo lop - lo ra KL1 la lop yeu o moi model."""
    models = ["A_xgb_softmax", "B_xgb_frankhall", "D_mlp_ordinal_only"]
    lab = ["A — XGB softmax (nen)", "B — Frank-Hall", "D — MLP loss nguong"]
    x = np.arange(5)
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    for j, (m, c, l) in enumerate(zip(models, SERIES, lab)):
        off = (j - 1) * (w + 0.02)
        v = D.S7_F1[m]
        ax.bar(x + off, v, w, color=c, label=f"{l}  (macro {D.S7_MACRO_F1[m]:.1f}%)", zorder=3)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 0.7, f"{vi:.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, KL)
    ax.set_ylim(0, 86)
    ax.set_ylabel("F1 theo lop (%)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    # Legend RA NGOAI: dat trong truc thi de len cot hoac len chu thich KL1
    ax.legend(frameon=False, fontsize=7.8, ncol=3, labelcolor=INK2,
              loc="lower left", bbox_to_anchor=(0, 1.0))
    # Nhan gon NGAY TREN nhom KL1, khong dung mui ten cat ngang bieu do
    ax.text(1, 52, "lop yeu nhat\no MOI model", ha="center", va="bottom", fontsize=7.6,
            color=CRIT, fontweight="bold", linespacing=1.4)
    ax.plot([1], [50], marker="v", ms=7, color=CRIT)
    fig.suptitle("Hinh 4 — S7: F1 theo tung do KL, bo dac trung day du",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.10)
    _finish(fig, "fig4_s7_f1.png",
            "Nguon: biomarker_s7_ordinal.ipynb muc 5b, seed 0, out-of-fold n=1229. KL1 nghia la "
            "'nghi ngo hep khe khop' - chinh bac si doc phim cung dong thuan kem nhat o muc nay.")


def fig5_s8_threshold():
    """S8: do nguong doi precision KL4 lay recall KL4 - tai lap 6/6."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=False)
    labs = [f"{m}\n{fs}" for m, fs, *_ in D.S8_THR_EFFECT]
    for ax, i0, i1, title, ylab in [
            (a1, 2, 3, "a. Recall KL4 — tang o ca 6/6", "recall KL4"),
            (a2, 4, 5, "b. Precision KL4 — giam o ca 6/6", "precision KL4")]:
        # Nhan chi ve khi cach nhan da ve >= 0.022, neu khong chung de len nhau (panel b co
        # bon gia tri nam trong dai 0.83-0.88).
        done_lo, done_hi = [], []
        for row in sorted(D.S8_THR_EFFECT, key=lambda r: r[i0]):
            lo, hi = row[i0], row[i1]
            ax.plot([0, 1], [lo, hi], color=GOOD if hi > lo else CRIT, lw=1.8, zorder=3,
                    marker="o", ms=7, mfc=SURFACE, mew=1.8)
            if all(abs(lo - v) >= 0.022 for v in done_lo):
                ax.text(-0.06, lo, f"{lo:.2f}", ha="right", va="center", fontsize=7, color=MUTED)
                done_lo.append(lo)
            if all(abs(hi - v) >= 0.022 for v in done_hi):
                ax.text(1.06, hi, f"{hi:.2f}", ha="left", va="center", fontsize=7, color=INK2)
                done_hi.append(hi)
        ax.set_xticks([0, 1], ["vach 0.5\nco dinh", "nguong\ndo duoc"], fontsize=8)
        ax.set_xlim(-0.34, 1.34)
        ax.set_ylim(0.30, 1.0)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
    mr0 = np.mean([r[2] for r in D.S8_THR_EFFECT]); mr1 = np.mean([r[3] for r in D.S8_THR_EFFECT])
    mp0 = np.mean([r[4] for r in D.S8_THR_EFFECT]); mp1 = np.mean([r[5] for r in D.S8_THR_EFFECT])
    a1.text(0.5, 0.335, f"trung binh  {mr0:.3f} → {mr1:.3f}", ha="center", fontsize=8,
            color=GOOD, fontweight="bold")
    a2.text(0.5, 0.335, f"trung binh  {mp0:.3f} → {mp1:.3f}", ha="center", fontsize=8,
            color=CRIT, fontweight="bold")
    fig.suptitle("Hinh 6 — S8: do nguong doi precision KL4 lay recall KL4, tai lap o moi cap",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.02)
    _finish(fig, "fig6_s8_threshold.png",
            "Nguon: biomarker_s8_holdout.ipynb muc 4b, n_test=246. Sau cap = 3 model (B, D, E) x 2 bo "
            "dac trung co radiomics. Co che tai lap 6/6; muc loi tren QWK gop thi KHONG.")


def fig6_s8_confusion():
    """S8: model co QWK cao nhat, va tac dong cua doi quy tac quyet dinh."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    _confusion(axes[0], D.S8_CONFUSION["C_xgb_reg_cutpoints"], "C — diem cat toi uu QWK",
               "QWK 0.803  ·  recall KL3 48.3%  ·  recall KL4 80.8%")
    _confusion(axes[1], D.S8_CONFUSION["C_xgb_reg_cutpoints@quantile"], "C — diem cat phan vi",
               "QWK 0.790  ·  recall KL3 55.2%  ·  recall KL4 69.2%")
    fig.suptitle("Hinh 5 — S8: cung mot bo hoi quy, chi doi quy tac dat diem cat",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.10)
    _finish(fig, "fig5_s8_confusion.png",
            "Nguon: biomarker_s8_holdout.ipynb muc 6, tap test co dinh n=246. Diem cat toi uu QWK noi "
            "rong hai bin ngoai cung (KL3 48.3% -> KL4 80.8%); diem cat phan vi tra lai can bang.")


def fig7_s6_usage():
    """S8 muc 4d: cot S6 co thuc su duoc classifier dung khong."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 3.4),
                                 gridspec_kw=dict(width_ratios=[1, 1.2], wspace=0.42))
    hos = list(D.S8_S6_USAGE)
    y = np.arange(len(hos))[::-1]
    kept = [D.S8_S6_USAGE[h][2] for h in hos]
    gain = [D.S8_S6_USAGE[h][3] for h in hos]
    per = [g / k for g, k in zip(gain, kept)]
    a1.barh(y, gain, height=0.5, color=[BLUE, AQUA, ORANGE], zorder=3)
    for yi, g, k, p in zip(y, gain, kept, per):
        # MOT text hai dong, khong phai hai text canh nhau: hai text se de len bar ben canh
        a1.text(g + 2.0, yi, f"{g:.1f}%\n{k} cot · {p:.2f}%/cot", va="center", ha="left",
                fontsize=7.4, color=INK2, linespacing=1.5)
    a1.set_yticks(y, hos, fontsize=8.5)
    a1.set_xlim(0, 122)
    a1.set_xlabel("Ty trong gain cua XGB (%)")
    a1.set_title("a. Cot S6 chiem 15.5% gain voi 12% so cot", loc="left")
    a1.xaxis.grid(True, zorder=0)
    a1.set_axisbelow(True)

    names = [n for n, _ in D.S8_S6_TOP][::-1]
    vals = [v for _, v in D.S8_S6_TOP][::-1]
    yy = np.arange(len(names))
    a2.barh(yy, vals, height=0.6, color=BLUE, zorder=3)
    for i, v in enumerate(vals):
        a2.text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=7.5, color=INK2)
    a2.set_yticks(yy, names, fontsize=7.6)
    a2.set_xlim(0, 3.1)
    a2.set_xlabel("Ty trong gain (%)")
    a2.set_title(f"b. Cot S6 manh nhat dung HANG {D.S8_S6_TOP_RANK} tren 926 cot", loc="left")
    a2.xaxis.grid(True, zorder=0)
    a2.set_axisbelow(True)
    a2.get_yticklabels()[-1].set_color(BLUE)
    a2.get_yticklabels()[-1].set_fontweight("bold")

    fig.suptitle("Hinh 7 — Cot S6 co thuc su duoc dung khong: co, va dung nhieu",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.04)
    _finish(fig, "fig7_s6_usage.png",
            "Nguon: biomarker_s8_holdout.ipynb muc 4d, bo 926 cot. Khong cot S6 nao roi vao trang thai "
            "'song sot LASSO nhung gain 0' — ca 13 cot song sot deu duoc cay tach tren chung.")


if __name__ == "__main__":
    print("sinh hinh vao", OUT)
    fig1_s6(); fig2_s7_qwk(); fig3_s7_confusion(); fig4_s7_f1()
    fig5_s8_threshold(); fig6_s8_confusion(); fig7_s6_usage()
    print("xong.")
