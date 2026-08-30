"""RayEncoder1D - mo hinh MVP Stage 1 (plan doc §3.4 Step 7).

Bien phan doan sun thanh bai toan 1D doc chieu day: moi node be mat xuong cho MOT tia,
mo hinh doc [K mau x C kenh] doc tia roi tra occupancy [K] + presence [1].

KIEN TRUC - bam sat plan doc §3.4 Step 7, KHONG them gi
-------------------------------------------------------
    Input K x C
    -> Conv1D block
    -> Dilated Conv1D block
    -> Dilated Conv1D block
    -> Depth-wise attention (pooling co trong so)
    -> Occupancy head: K x 1
    -> Presence head:   1 x 1

MVP phai DON GIAN toi da de con truy duoc nguon goc cua bat ky cai thien nao (§3).
Khong graph, khong cross-surface, khong implicit field - do la Stage 2/3.

VI SAO CO PRESENCE HEAD RIENG
-----------------------------
M0 do duoc: bin `absent` sai gap ~4 lan moi bin khac (0.99mm femoral / 0.83mm med_tib
so voi ~0.21-0.25mm). Do la diem yeu cu the cua ResEnc ma bieu dien nay nham toi.
Xem M0_gate1_results_and_decision.md §4.3.

RANG BUOC BAO CAO (M0 §6) - KHONG DUOC QUEN
--------------------------------------------
Muc tieu §3.7 quy ve ASSD TONG chi la 0.0058mm (femoral) / 0.0089mm (med_tib), NHO
ngang ngua sai khac giua hai implementation metric (eps = 0.0033 / 0.0077mm).
=> KHONG BAO GIO bao cao ASSD tong cho hieu ung nay. Chi bao cao o mau so VUNG MONG
   hoac presence F1 / thickness MAE.

MAT CAN BANG - ly do dung focal cho occupancy
---------------------------------------------
Tia trai tu -1mm den +6mm (7mm), sun day dien hinh 2-3mm => ~60-70% mau doc tia la
nen. Them cac tia `absent` (occupancy toan 0). BCE thuong se hoc "doan 0" rat nhanh.
Mac dinh focal_gamma=1.0 va pos_weight tu dong; ha ve 0 neu muon BCE thuan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import core
from .core import SPACING, RayConfig


# --------------------------------------------------------------- kien truc

def _same_padding(kernel_size: int, dilation: int) -> int:
    """Padding giu nguyen do dai K. Chi dung cho kernel LE."""
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size phai le de giu K, nhan duoc {kernel_size}")
    return dilation * (kernel_size - 1) // 2


class _ConvBlock(nn.Module):
    """Conv1d -> GroupNorm -> GELU.

    Dung GroupNorm chu khong BatchNorm: test M5 (overfit 2-4 ca) chay batch rat nho,
    BatchNorm se lech thong ke giua train/eval va lam M5 bao dong gia.
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 5, dilation: int = 1,
                 n_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation,
                              padding=_same_padding(kernel_size, dilation))
        self.norm = nn.GroupNorm(min(n_groups, c_out), c_out)
        self.act = nn.GELU()

    def forward(self, x):                      # [B, C, K] -> [B, C_out, K]
        return self.act(self.norm(self.conv(x)))


class RayEncoder1D(nn.Module):
    """Encoder 1D doc tia. Vao [B, K, C] -> (occ_logits [B, K], pres_logits [B]).

    `width` 64 la du cho MVP: tin hieu doc tia la mot profile 1D don gian (nen - sun -
    khe khop). Tang width truoc khi M5/M6 dau la toi uu hoa som.
    """

    def __init__(self, in_channels: int = 3, width: int = 64, kernel_size: int = 5,
                 dilations: tuple = (1, 2, 4), with_presence: bool = True):
        """`with_presence` = truc H cua ma tran: False => H0 (chi occupancy), True => H1.

        H0 phai KHONG dung presence head that su - khong chi la bo qua dau ra cua no.
        Neu van dung head do, gradient cua no van chay vao than mang => H0 khong con la
        H0, va cap so sanh H0-vs-H1 (§10.2) tro nen vo nghia.
        """
        super().__init__()
        self.in_channels = in_channels
        self.with_presence = with_presence
        self.stem = _ConvBlock(in_channels, width, kernel_size, dilation=dilations[0])
        self.blocks = nn.ModuleList(
            [_ConvBlock(width, width, kernel_size, dilation=d) for d in dilations[1:]]
        )
        self.occ_head = nn.Conv1d(width, 1, 1)      # K x 1
        if with_presence:
            # Depth-wise attention: hoc xem do sau nao quyet dinh "co sun hay khong".
            # Pooling co trong so thay vi mean - mean bi loang boi phan nen chiem da so.
            self.attn = nn.Conv1d(width, 1, 1)
            self.pres_head = nn.Linear(width, 1)    # 1 x 1

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"can [B, K, C], nhan duoc {tuple(x.shape)}")
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"kenh cuoi phai = in_channels={self.in_channels}, nhan {x.shape[-1]}. "
                f"Dung thu tu [B, K, C] (khong phai [B, C, K])."
            )
        h = x.transpose(1, 2)                       # [B, C, K]
        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h)

        occ_logits = self.occ_head(h).squeeze(1)    # [B, K]
        if not self.with_presence:
            return occ_logits, None                 # H0

        w = torch.softmax(self.attn(h), dim=-1)     # [B, 1, K]
        z = (h * w).sum(dim=-1)                     # [B, width]
        pres_logits = self.pres_head(z).squeeze(-1)  # [B]
        return occ_logits, pres_logits


# ------------------------------------------------------------------- loss

def focal_bce_with_logits(logits, target, gamma: float = 1.0, pos_weight=None,
                          reduction: str = "mean"):
    """BCE co modulation focal. gamma=0 => BCE thuan.

    Vi sao can: ~60-70% mau doc tia la nen, chua ke tia absent toan 0. BCE thuan hoi
    tu ve "doan 0" roi dung o do.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none")
    if gamma > 0:
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        bce = bce * (1 - p_t).clamp_min(1e-6) ** gamma
    if reduction == "mean":
        return bce.mean()
    if reduction == "sum":
        return bce.sum()
    return bce


@dataclass
class LossWeights:
    """Plan doc §3.4 Step 7: L = L_occupancy + lambda_p * L_presence.

    PHASE B (review §8): mac dinh GIU NGUYEN hanh vi canonical. Cac lever duoi day mac
    dinh = 0/False nen `LossWeights()` cho ket qua Y HET truoc khi them - co test khoa.

    absent_fp_weight : nhan them cho occupancy loss tren TIA ABSENT. M0 §8.3 do duoc 82%
                       khoi luong loi vung mong la FP o bin absent => day la lever nham
                       THANG vao thu pham. 0 = tat.
    per_ray_norm     : chuan hoa occupancy loss theo TUNG TIA truoc khi trung binh. Khong
                       co no, tia sun DAY (nhieu o duong) chi phoi gradient => recall 81%
                       day vs 37% mong (review §8).
    """
    lambda_presence: float = 0.5
    focal_gamma: float = 1.0
    auto_pos_weight: bool = True     # can bang theo ty le duong thuc te trong batch
    absent_fp_weight: float = 0.0    # >0 => phat them FP tren tia absent
    per_ray_norm: bool = False       # True => moi tia dong gop nhu nhau


def ray_loss(occ_logits, pres_logits, occ_target, pres_target,
             w: LossWeights = LossWeights()):
    """Tra (loss_tong, dict cac thanh phan de log).

    occ_target  [B, K] float 0/1
    pres_target [B]    float 0/1

    LUU Y THIET KE: occupancy loss tinh tren MOI tia, ke ca tia absent (target toan 0).
    Do la GIAM SAT HOP LE - "khong co sun o day" la thong tin ta muon mo hinh hoc.
    Khac voi §4.4 Step 7 (khong ep BOUNDARY ton tai khi presence=0) - o day khong co
    boundary head nen khong vuong.
    """
    occ_target = occ_target.float()

    pw_occ = None
    if w.auto_pos_weight:
        # pos_weight = n_neg / n_pos, chan tren de khong no khi lop duong qua hiem
        pos = occ_target.mean().clamp(1e-4, 1 - 1e-4)
        pw_occ = ((1 - pos) / pos).clamp(max=50.0).to(occ_logits.device)

    if w.absent_fp_weight > 0 or w.per_ray_norm:
        # Duong di PHASE B: giu loss theo tung o de con trong so hoa duoc
        cell = focal_bce_with_logits(occ_logits, occ_target, w.focal_gamma, pw_occ,
                                     reduction="none")                    # [B, K]
        if w.absent_fp_weight > 0:
            # Tia ABSENT (target toan 0): moi o duong doan ra deu la FP - phat manh hon.
            is_absent = (occ_target.sum(dim=1) == 0).float().unsqueeze(1)  # [B, 1]
            cell = cell * (1.0 + w.absent_fp_weight * is_absent)
        l_occ = cell.mean(dim=1).mean() if w.per_ray_norm else cell.mean()
    else:
        l_occ = focal_bce_with_logits(occ_logits, occ_target, w.focal_gamma, pw_occ)

    # H0 (truc H = "occupancy only"): KHONG co so hang presence. `pres_logits=None` la
    # tin hieu tuong minh tu RayEncoder1D(with_presence=False).
    if pres_logits is None:
        return l_occ, {"loss": l_occ.detach().item(), "occ": l_occ.detach().item(),
                       "pres": float("nan")}

    pres_target = pres_target.float()
    pw_pres = None
    if w.auto_pos_weight:
        pos_p = pres_target.mean().clamp(1e-4, 1 - 1e-4)
        pw_pres = ((1 - pos_p) / pos_p).clamp(max=50.0).to(pres_logits.device)

    l_pres = focal_bce_with_logits(pres_logits, pres_target, w.focal_gamma, pw_pres)
    total = l_occ + w.lambda_presence * l_pres
    # detach truoc khi doi sang float: dict nay chi de LOG, khong duoc giu graph
    parts = {"loss": total.detach().item(), "occ": l_occ.detach().item(),
             "pres": l_pres.detach().item()}
    return total, parts


# ---------------------------------------------------- cau noi core.py -> tensor

def mri_gradient(mri: np.ndarray, spacing=SPACING) -> np.ndarray:
    """Do lon gradient MRI (don vi/mm). Kenh 2 cua §3.2.

    np.gradient(*spacing) tra dao ham theo MM - khong phai theo voxel. Duoi anisotropy
    (0.70 vs 0.3646mm) bo qua cho nay se lam thanh phan z bi thoi phong ~2x.
    """
    gz, gy, gx = np.gradient(np.asarray(mri, np.float32), *spacing)
    return np.sqrt(gz ** 2 + gy ** 2 + gx ** 2).astype(np.float32)


#: Bo kenh dinh nghia o `bsc/experiment.py` (truc I cua ma tran chuan hoa).
#: KHONG dinh nghia lai o day - ban truoc da lech khoi plan (P2/P3 sai kenh, P2-P5
#: dung Oracle thay vi Atlas). Xem p3_m8d_standardization_decision_vi.md §6-§7.
from .experiment import INPUT as INPUT_CHANNELS  # noqa: E402  (tien cho notebook)


def build_case_rays(mri: np.ndarray, bone_mask: np.ndarray, cart_mask: np.ndarray,
                    spacing=SPACING, cfg: RayConfig = RayConfig(),
                    domain: "np.ndarray | None" = None, domain_mode: str = "all",
                    opposing_bone_mask: "np.ndarray | None" = None,
                    normalize: bool = True,
                    channels: tuple = ("mri", "grad", "sdf"),
                    extra_vols: "dict | None" = None,
                    direction: str = "normal",
                    jitter_s_mm: float = 0.0, jitter_theta_deg: float = 0.0,
                    seed: int = 0):
    """Mot ca -> (X [N,K,C], occ [N,K], presence [N], verts, dirs).

    channels    : thu tu kenh, xem M8_CHANNELS. "prob" lay tu extra_vols["prob"].
    extra_vols  : the tich phu (vd {"prob": resenc_cart_prob}) cho M8-D / P3.
    direction   : M6 - "normal" | "axial" | "random" | "tangent".
    jitter_*    : M7 - nhieu loan be mat/phap tuyen TRUOC khi phong tia.
    domain      : mask bool [N_verts] gioi han tia (oracle_domain / joint_facing_domain).
    normalize   : z-score cuong do MRI TUNG CA - bat buoc vi MRI khong co don vi tuyet
                  doi; thieu buoc nay mo hinh se hoc theo bias tung ca.

    THU TU QUAN TRONG: jitter (M7) ap len be mat GT truoc, roi MOI doi huong (M6).
    Nguoc lai se xoay mot phap tuyen da bi nhieu, tron lan hai hieu ung.

    LUU Y: `occ` luon lay theo dung tia da dung (ke ca huong doi chung). Do la CHU Y -
    M6 hoi "doc theo huong nay co du thong tin khong", nen target phai theo cung tia.
    """
    sdf, verts, normals = core.bone_geometry(bone_mask, spacing, cfg)

    if jitter_s_mm > 0 or jitter_theta_deg > 0:
        verts, normals = core.jitter_surface(verts, normals, jitter_s_mm,
                                             jitter_theta_deg, seed)
    dirs = core.direction_field(normals, direction, seed)

    mri = np.asarray(mri, np.float32)
    if normalize:
        mu, sd = float(mri.mean()), float(mri.std())
        mri = (mri - mu) / (sd + 1e-6)

    pool = {"mri": mri, "grad": mri_gradient(mri, spacing), "sdf": sdf}
    if extra_vols:
        pool.update({k: np.asarray(v, np.float32) for k, v in extra_vols.items()})
    missing = [c for c in channels if c not in pool]
    if missing:
        raise ValueError(f"thieu kenh {missing}; co san {sorted(pool)}. "
                         f"Kenh 'prob' phai truyen qua extra_vols.")
    vols = {c: pool[c] for c in channels}       # dict giu thu tu => thu tu kenh on dinh

    X = core.sample_rays(vols, verts, dirs, cfg, spacing)
    occ = core.occupancy_target(cart_mask, verts, dirs, cfg, spacing)

    # Mien khop (§3.4 Step 4):
    #   "oracle"     = D0, tu vung bam sun GT + vanh no. Vanh KHONG phai trang tri -
    #                  thieu no thi domain khong chua tia absent nao => presence head
    #                  khong co negative => khong kiem duoc luan diem ve sun MAT.
    #   "fold_atlas" = D1. Hien dung joint_facing_domain (proxy HINH HOC: node co phap
    #                  tuyen huong vao khe khop), CHUA phai atlas quan the theo fold.
    #                  Proxy nay khong the ro ri nhan test ve mat CAU TRUC, va theo
    #                  core.py no chi can la TAP CHA hao phong - precision la viec cua
    #                  presence head. Can `opposing_bone_mask`.
    if domain is None:
        if domain_mode == "oracle":
            domain = core.oracle_domain(verts, occ, cfg)
        elif domain_mode == "fold_atlas":
            if opposing_bone_mask is None:
                raise ValueError(
                    "domain_mode='fold_atlas' can `opposing_bone_mask` (vd xuong chay "
                    "khi lop la femoral_cart) de dung mien huong vao khe khop.")
            sdf_opp = core.signed_distance(opposing_bone_mask, spacing,
                                           smooth_mm=cfg.smooth_mm)
            domain = core.joint_facing_domain(verts, dirs, sdf_opp, spacing)
        elif domain_mode != "all":
            raise ValueError(f"domain_mode khong hop le: {domain_mode!r} "
                             f"(chon 'all'|'oracle'|'fold_atlas')")

    if domain is not None:
        domain = np.asarray(domain, bool)
        X, occ, verts, dirs = X[domain], occ[domain], verts[domain], dirs[domain]

    presence = core.ray_stats(occ, cfg)["presence"]
    return X, occ, presence.astype(np.uint8), verts, dirs


def build_dataset(case_ids, loader, spacing=SPACING, cfg: RayConfig = RayConfig(),
                  domain_mode: str = "oracle", rays_per_case: int = 20000,
                  min_absent_frac: float = 0.0, seed: int = 0, verbose: bool = False,
                  **build_kw):
    """Nhieu ca -> (X, occ, presence) da ghep. `loader(case_id) -> (mri, bone, cart)`.

    `loader` la callable de module nay KHONG phu thuoc nibabel/Colab (test chay local).

    rays_per_case   : lay mau con SO TIA moi ca. core.py da do: subsample tu do khi
                      TRAIN la hop le (tia i.i.d., khong can phu kin be mat); chi khi
                      INFERENCE moi bat buoc dung FULL marching-cubes density.
    min_absent_frac : dam bao it nhat ti le nay la tia ABSENT. Mac dinh 0 = giu nguyen
                      tien nghiem THAT. Chi tang khi presence head khong hoc duoc -
                      lam lech tien nghiem se pha hieu chuan presence (plan §4.7 MM6).
    domain_mode     : "oracle" (MVP-A) | "all". MVP-B atlas truyen qua `domain=`.
    """
    Xs, occs, press = [], [], []
    for i, cid in enumerate(case_ids):
        mri, bone, cart = loader(cid)
        X, occ, pres, _, _ = build_case_rays(
            mri, bone, cart, spacing, cfg, seed=seed + i,
            domain_mode=domain_mode, **build_kw)
        if len(X) == 0:
            if verbose:
                print(f"  {cid}: 0 tia - bo qua")
            continue

        if rays_per_case and len(X) > rays_per_case:
            rng = np.random.default_rng(seed + i)
            absent = np.flatnonzero(pres == 0)
            n_min = int(min_absent_frac * rays_per_case)
            if n_min and len(absent) >= n_min:
                keep_a = rng.choice(absent, n_min, replace=False)
                rest = np.setdiff1d(np.arange(len(X)), keep_a, assume_unique=False)
                keep_r = rng.choice(rest, rays_per_case - n_min, replace=False)
                idx = np.concatenate([keep_a, keep_r])
            else:
                idx = rng.choice(len(X), rays_per_case, replace=False)
            X, occ, pres = X[idx], occ[idx], pres[idx]

        Xs.append(X); occs.append(occ); press.append(pres)
        if verbose:
            print(f"  {cid}: {len(X)} tia, absent {100*(pres==0).mean():.1f}%")

    if not Xs:
        raise ValueError("Khong dung duoc tia nao - kiem loader/domain_mode.")
    return (np.concatenate(Xs), np.concatenate(occs), np.concatenate(press))


# ------------------------------------------------------------------ train

def fit(model: nn.Module, X, occ, presence, epochs: int = 30, batch_size: int = 4096,
        lr: float = 3e-4, w: LossWeights = LossWeights(), device: str = "cpu",
        seed: int = 0, verbose: bool = False, progress: bool = True):
    """Vong train toi gian. Tra list dict lich su moi epoch.

    Co tinh giu don gian: khong scheduler, khong augment, khong early stop. MVP can
    tai lap duoc va debug duoc, khong can toi uu (§3).
    """
    torch.manual_seed(seed)
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    Xt = torch.as_tensor(np.asarray(X, np.float32), device=device)
    ot = torch.as_tensor(np.asarray(occ, np.float32), device=device)
    pt = torch.as_tensor(np.asarray(presence, np.float32), device=device)

    n = Xt.shape[0]
    g = torch.Generator().manual_seed(seed)
    history = []
    epoch_iter = range(epochs)
    if progress:
        try:
            from tqdm.auto import tqdm
            epoch_iter = tqdm(epoch_iter, desc="fit", leave=False)
        except Exception:
            pass
    for ep in epoch_iter:
        perm = torch.randperm(n, generator=g).to(device)
        agg, nb = {"loss": 0.0, "occ": 0.0, "pres": 0.0}, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            occ_logits, pres_logits = model(Xt[idx])
            loss, parts = ray_loss(occ_logits, pres_logits, ot[idx], pt[idx], w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for k in agg:
                agg[k] += parts[k]
            nb += 1
        rec = {k: v / max(nb, 1) for k, v in agg.items()} | {"epoch": ep}
        history.append(rec)
        if verbose:
            print(f"ep {ep:3d}  loss {rec['loss']:.4f}  occ {rec['occ']:.4f}  "
                  f"pres {rec['pres']:.4f}")
    return history


@torch.no_grad()
def predict_rays(model: nn.Module, X, batch_size: int = 8192, device: str = "cpu"):
    """Tra (occ_prob [N,K], pres_prob [N] hoac None neu H0)."""
    model = model.to(device).eval()
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=device)
    occ, pres = [], []
    for i in range(0, Xt.shape[0], batch_size):
        o, p = model(Xt[i:i + batch_size])
        occ.append(torch.sigmoid(o).cpu().numpy())
        if p is not None:
            pres.append(torch.sigmoid(p).cpu().numpy())
    return np.concatenate(occ, 0), (np.concatenate(pres, 0) if pres else None)


def reconstruct_volume(occ_prob, verts, normals, shape, cfg: RayConfig = RayConfig(),
                       spacing=SPACING, presence_prob=None, presence_thr: float = 0.5,
                       presence_gate: str = "hard"):
    """Tia -> the tich xac suat (plan doc Step 8), qua core.splat_rays.

    presence_gate (PHASE B, review §8):
      "hard" - tia duoi nguong bi ep ve 0 (mac dinh, = hanh vi canonical)
      "soft" - nhan occupancy voi xac suat presence (khong cat cung)
      "none" - bo qua presence hoan toan

    VI SAO CAN "soft"/"none": sweep nguong (§8.5) cho thay hard-gate nang nguong lam loi
    TANG - vi no vua chan FP absent vua chan luon tia sun MONG that (bien FP thanh FN).
    Gate mem cho phep tia khong chac chan dong gop MOT PHAN thay vi mat trang.

    LUU Y (core.py): dung FULL marching-cubes density khi inference. Subsample chi
    danh cho train - o mat do thua (h=1.0mm) round-trip mat 20% voxel sun.
    """
    occ_prob = np.asarray(occ_prob, np.float32)
    if presence_prob is not None and presence_gate != "none":
        p = np.asarray(presence_prob, np.float32)
        if presence_gate == "hard":
            occ_prob = occ_prob * (p >= presence_thr)[:, None]
        elif presence_gate == "soft":
            occ_prob = occ_prob * p[:, None]
        else:
            raise ValueError(f"presence_gate khong hop le: {presence_gate!r} "
                             f"(chon 'hard'|'soft'|'none')")
    return core.splat_rays(occ_prob, verts, normals, shape, cfg, spacing)
