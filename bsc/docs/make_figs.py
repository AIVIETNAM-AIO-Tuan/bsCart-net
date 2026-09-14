"""Sinh hinh cho bao cao 14/09 tu `report_data.py`.

    cd bsc && python docs/make_figs.py

Moi con so nam o `report_data.py`, khong go tay o day. Hinh ra `docs/figs/*.png`.

Bang mau lay tu bo mau tham chieu cua skill dataviz, da chay validator:
3 slot dau (xanh duong / cam / xanh ngoc) dat moi cong CVD o che do all-pairs.
Cot nao cung co nhan so truc tiep, nen khong dua vao mau de doc gia tri.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

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


def vn(x, nd=2, sign=False):
    """Số theo cách viết tiếng Việt: dấu phẩy thập phân. Dùng cho MỌI số vẽ lên hình."""
    return f"{x:{'+' if sign else ''}.{nd}f}".replace(".", ",")


_VN_TICK = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def _finish(fig, name, note=None):
    # Nhãn trục do matplotlib sinh ra mặc định dùng dấu chấm - đổi hết sang dấu phẩy,
    # nếu không trong cùng một hình sẽ có chỗ "0,5" chỗ "0.487".
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labs = [t.get_text() for t in axis.get_ticklabels()]
            if labs and not any(t and not re.fullmatch(r"-?[\d.,]+", t) for t in labs):
                axis.set_major_formatter(_VN_TICK)
    if note:
        # -0.06 chu khong phai 0.005: chu thich phai nam DUOI nhan truc, khong de len no
        fig.text(0.008, -0.06, note, fontsize=6.8, color=MUTED, ha="left", va="top")
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("  ->", name)


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
        a1.text(v + (0.012 if v >= 0 else -0.012), i, vn(v, 3, sign=True), va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5, color=INK2)
    a1.set_yticks(y, names, fontsize=7.6)
    a1.axvline(0, color=BASELINE, lw=1)
    a1.set_xlim(-0.10, 0.52)
    a1.set_xlabel("Spearman $\\rho$ với KL")
    a1.set_title("a. Tương quan với độ KL, 1229 ca", loc="left")
    a1.xaxis.grid(True, zorder=0)
    a1.set_axisbelow(True)
    h = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=CRIT)]
    a1.legend(h, ["Họ bề mặt S6 (mới)", "Bảng S3 (cũ)"], loc="lower right",
              frameon=False, fontsize=7.5, labelcolor=INK2)

    for (lab, vals), c in zip(D.S6_FCL_BY_KL.items(), SERIES):
        a2.plot(D.CLASSES, vals, marker="o", ms=6, lw=2, color=c, label=lab, zorder=3)
        a2.text(D.CLASSES[-1] + 0.09, vals[-1], vn(vals[-1]) + "%", color=c,
                fontsize=7.5, va="center", fontweight="bold")
    a2.set_xticks(D.CLASSES, KL)
    a2.set_xlim(-0.25, 4.75)
    a2.set_ylabel("% footprint bị trơ (trung vị)")
    a2.set_title("b. Mất sụn toàn bề dày tăng theo KL", loc="left")
    a2.yaxis.grid(True, zorder=0)
    a2.set_axisbelow(True)
    a2.legend(frameon=False, fontsize=7.5, loc="upper left", labelcolor=INK2)

    fig.suptitle("Hình 1 — S6: biomarker neo bề mặt xương đo được mất sụn, cột cũ thì không",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.04)
    _finish(fig, "fig1_s6_fcl.png",
            "Nguồn: biomarker_s6_fcl.ipynb mục 5, 1229 ca. Cột cũ `denuded_ratio_tibial` chỉ đạt 0,105; "
            "cột mới mạnh gấp khoảng 4 lần.")


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
            ax.text(xi, v + 0.008, vn(v, 3), ha="center", fontsize=6.8, color=INK2,
                    rotation=90, va="bottom")
    ax.set_xticks(x, [D.S7_MODEL_SHORT[m] for m in models], fontsize=8)
    ax.set_ylim(0.5, 0.85)
    ax.set_ylabel("QWK (out-of-fold, n = 1229)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    # Legend RA NGOAI truc: dat trong se de len hai nhan nhom ben duoi
    ax.legend(frameon=False, fontsize=8, ncol=3, labelcolor=INK2,
              loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.axvspan(-0.5, 1.5, color="#f0efec", zorder=0)
    ax.text(0.5, 0.822, "danh định\n(không dùng thứ tự)", ha="center", fontsize=7.5, color=MUTED)
    ax.text(3.5, 0.822, "có dùng thứ tự (ordinal)", ha="center", fontsize=7.5, color=MUTED)
    fig.suptitle("Hình 2 — S7: xử lý KL như thang thứ tự hơn coi là năm lớp rời rạc",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.09)
    _finish(fig, "fig2_s7_qwk.png",
            "Nguồn: biomarker_s7_ordinal.ipynb mục 4. Trung bình 3 seed, 15 fold chia theo subject. "
            "Độ lệch chuẩn qua seed 0,002–0,015.")


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
    ax.set_xlabel("KL đoán", fontsize=8.5)
    ax.set_ylabel("KL thật", fontsize=8.5)
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
               "QWK 0,776  ·  MAE 0,533  ·  macro F1 54,5%")
    _confusion(axes[1], D.S7_CONFUSION["D_mlp_ordinal_only"], "D — MLP loss ngưỡng",
               "QWK 0,774  ·  MAE 0,547  ·  macro F1 54,3%")
    fig.suptitle("Hình 3 — S7: ma trận nhầm lẫn, out-of-fold trên cả 1229 ca",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.10)
    _finish(fig, "fig3_s7_confusion.png",
            "Nguồn: biomarker_s7_ordinal.ipynb mục 5, bộ đặc trưng đầy đủ (926 cột), seed 0. "
            "Ô tô đậm theo tỉ lệ trong hàng; số là đếm ca. Viền xanh là đoán đúng.")


def fig4_s7_f1():
    """S7: F1 theo lop - lo ra KL1 la lop yeu o moi model."""
    models = ["A_xgb_softmax", "B_xgb_frankhall", "D_mlp_ordinal_only"]
    lab = ["A — XGB softmax (nền)", "B — Frank-Hall", "D — MLP loss ngưỡng"]
    x = np.arange(5)
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    for j, (m, c, l) in enumerate(zip(models, SERIES, lab)):
        off = (j - 1) * (w + 0.02)
        v = D.S7_F1[m]
        ax.bar(x + off, v, w, color=c, label=f"{l}  (macro {vn(D.S7_MACRO_F1[m], 1)}%)", zorder=3)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 0.7, f"{vi:.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, KL)
    ax.set_ylim(0, 86)
    ax.set_ylabel("F1 theo lớp (%)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    # Legend RA NGOAI: dat trong truc thi de len cot hoac len chu thich KL1
    ax.legend(frameon=False, fontsize=7.8, ncol=3, labelcolor=INK2,
              loc="lower left", bbox_to_anchor=(0, 1.0))
    # Nhan gon NGAY TREN nhom KL1, khong dung mui ten cat ngang bieu do
    ax.text(1, 52, "lớp yếu nhất\nở MỌI model", ha="center", va="bottom", fontsize=7.6,
            color=CRIT, fontweight="bold", linespacing=1.4)
    ax.plot([1], [50], marker="v", ms=7, color=CRIT)
    fig.suptitle("Hình 4 — S7: F1 theo từng độ KL, bộ đặc trưng đầy đủ",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.10)
    _finish(fig, "fig4_s7_f1.png",
            "Nguồn: biomarker_s7_ordinal.ipynb mục 5b, seed 0, out-of-fold n = 1229. KL1 nghĩa là "
            "“nghi ngờ hẹp khe khớp” — chính bác sĩ đọc phim cũng đồng thuận kém nhất ở mức này.")


def fig5_s8_threshold():
    """S8: do nguong doi precision KL4 lay recall KL4 - tai lap 6/6."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=False)
    labs = [f"{m}\n{fs}" for m, fs, *_ in D.S8_THR_EFFECT]
    for ax, i0, i1, title, ylab in [
            (a1, 2, 3, "a. Recall KL4 — tăng ở cả 6/6", "recall KL4"),
            (a2, 4, 5, "b. Precision KL4 — giảm ở cả 6/6", "precision KL4")]:
        # Nhan chi ve khi cach nhan da ve >= 0.022, neu khong chung de len nhau (panel b co
        # bon gia tri nam trong dai 0.83-0.88).
        done_lo, done_hi = [], []
        for row in sorted(D.S8_THR_EFFECT, key=lambda r: r[i0]):
            lo, hi = row[i0], row[i1]
            ax.plot([0, 1], [lo, hi], color=GOOD if hi > lo else CRIT, lw=1.8, zorder=3,
                    marker="o", ms=7, mfc=SURFACE, mew=1.8)
            if all(abs(lo - v) >= 0.022 for v in done_lo):
                ax.text(-0.06, lo, vn(lo), ha="right", va="center", fontsize=7, color=MUTED)
                done_lo.append(lo)
            if all(abs(hi - v) >= 0.022 for v in done_hi):
                ax.text(1.06, hi, vn(hi), ha="left", va="center", fontsize=7, color=INK2)
                done_hi.append(hi)
        ax.set_xticks([0, 1], ["vạch 0,5\ncố định", "ngưỡng\ndò được"], fontsize=8)
        ax.set_xlim(-0.34, 1.34)
        ax.set_ylim(0.30, 1.0)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
    mr0 = np.mean([r[2] for r in D.S8_THR_EFFECT]); mr1 = np.mean([r[3] for r in D.S8_THR_EFFECT])
    mp0 = np.mean([r[4] for r in D.S8_THR_EFFECT]); mp1 = np.mean([r[5] for r in D.S8_THR_EFFECT])
    a1.text(0.5, 0.335, f"trung bình  {vn(mr0, 3)} → {vn(mr1, 3)}", ha="center", fontsize=8,
            color=GOOD, fontweight="bold")
    a2.text(0.5, 0.335, f"trung bình  {vn(mp0, 3)} → {vn(mp1, 3)}", ha="center", fontsize=8,
            color=CRIT, fontweight="bold")
    fig.suptitle("Hình 6 — S8: dò ngưỡng đổi precision KL4 lấy recall KL4, tái lập ở mọi cặp",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.02)
    _finish(fig, "fig6_s8_threshold.png",
            "Nguồn: biomarker_s8_holdout.ipynb mục 4b, n_test = 246. Sáu cặp = 3 model (B, D, E) × 2 bộ "
            "đặc trưng có radiomics. Cơ chế tái lập 6/6; mức lợi trên QWK gộp thì KHÔNG.")


def fig6_s8_confusion():
    """S8: model co QWK cao nhat, va tac dong cua doi quy tac quyet dinh."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    _confusion(axes[0], D.S8_CONFUSION["C_xgb_reg_cutpoints"], "C — điểm cắt tối ưu QWK",
               "QWK 0,803  ·  recall KL3 48,3%  ·  recall KL4 80,8%")
    _confusion(axes[1], D.S8_CONFUSION["C_xgb_reg_cutpoints@quantile"], "C — điểm cắt phân vị",
               "QWK 0,790  ·  recall KL3 55,2%  ·  recall KL4 69,2%")
    fig.suptitle("Hình 5 — S8: cùng một bộ hồi quy, chỉ đổi quy tắc đặt điểm cắt",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.10)
    _finish(fig, "fig5_s8_confusion.png",
            "Nguồn: biomarker_s8_holdout.ipynb mục 6, tập test cố định n = 246. Điểm cắt tối ưu QWK nới "
            "rộng hai bin ngoài cùng, bóp KL3 còn 48,3% để đẩy KL4 lên 80,8%; điểm cắt phân vị trả lại cân bằng.")


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
        a1.text(g + 2.0, yi, f"{vn(g, 1)}%\n{k} cột · {vn(p)}%/cột", va="center", ha="left",
                fontsize=7.4, color=INK2, linespacing=1.5)
    a1.set_yticks(y, hos, fontsize=8.5)
    a1.set_xlim(0, 122)
    a1.set_xlabel("Tỉ trọng gain của XGB (%)")
    a1.set_title("a. Cột S6 chiếm 15,5% gain với 12% số cột", loc="left")
    a1.xaxis.grid(True, zorder=0)
    a1.set_axisbelow(True)

    names = [n for n, _ in D.S8_S6_TOP][::-1]
    vals = [v for _, v in D.S8_S6_TOP][::-1]
    yy = np.arange(len(names))
    a2.barh(yy, vals, height=0.6, color=BLUE, zorder=3)
    for i, v in enumerate(vals):
        a2.text(v + 0.05, i, vn(v) + "%", va="center", fontsize=7.5, color=INK2)
    a2.set_yticks(yy, names, fontsize=7.6)
    a2.set_xlim(0, 3.1)
    a2.set_xlabel("Tỉ trọng gain (%)")
    a2.set_title(f"b. Cột S6 mạnh nhất đứng HẠNG {D.S8_S6_TOP_RANK} trên 926 cột", loc="left")
    a2.xaxis.grid(True, zorder=0)
    a2.set_axisbelow(True)
    a2.get_yticklabels()[-1].set_color(BLUE)
    a2.get_yticklabels()[-1].set_fontweight("bold")

    fig.suptitle("Hình 7 — Cột S6 có thực sự được dùng không: có, và dùng nhiều",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.04)
    _finish(fig, "fig7_s6_usage.png",
            "Nguồn: biomarker_s8_holdout.ipynb mục 4d, bộ 926 cột. Không cột S6 nào rơi vào trạng thái "
            "“sống sót LASSO nhưng gain 0” — cả 13 cột sống sót đều được cây tách trên chúng.")


if __name__ == "__main__":
    print("sinh hinh vao", OUT)
    fig1_s6(); fig2_s7_qwk(); fig3_s7_confusion(); fig4_s7_f1()
    fig5_s8_threshold(); fig6_s8_confusion(); fig7_s6_usage()
    print("xong.")
