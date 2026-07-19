"""Atlas quan the vung khop, xay RIENG trong tung training fold (D1).

DUNG THEO SPEC, khong thay the
------------------------------
plan §3.4 Step 4 (MVP-B): "Construct the articular domain from the training cases
inside each fold."
plan §2.2: "A bone-surface atlas must be built separately inside each training fold.
Validation and test cartilage masks must never be used to construct the articular atlas."

=> Atlas CHI duoc nhin mask sun cua ca TRAIN trong fold do. `build_articular_atlas`
   ghi lai danh sach ca da dung (`case_ids`) de audit duoc dieu nay, va
   `assert_no_leak` chan tan goc viec ca val/test lot vao.

TUONG UNG GIUA CAC CA - lam sao so sanh node giua hai dau goi khac nhau?
------------------------------------------------------------------------
plan §4.4 Step 2: "Progress from simple to complex: 1. Landmark-based rigid or affine
registration... Begin with fixed registration before adding learnable deformation."

Ta o muc 1: chuan hoa TRUC-GOC (tinh tien + ty le theo tung truc anh), khong xoay.
Moi node ve toa do ~[-1,1]^3, atlas la xac suat "node nay co sun bam" tren luoi do.

DA THU PCA VA BAC BO - ghi lai de khong ai lam lai
---------------------------------------------------
Ban dau dung truc chinh PCA + co dinh dau bang do lech (skew). Do bang phantom:
KHONG DUNG DUOC. Tri rieng gan nhau (dau xuong that chi cho ty so ~1.05) => huong truc
tuy y; va vat the doi xung qua mot mat phang co skew = 0 tren moi truc NAM TRONG mat
phang do => dau truc tuy y. Loi cau trong/ngoai gan doi xung nen dinh nay dung ca voi
xuong that. Hai ca giong het nhau van ra toa do lat nguoc => atlas gop thanh chao.
Chi tiet o docstring SurfaceFrame.

HAN CHE DA BIET (phai ghi trong bai bao)
----------------------------------------
* Khong bu duoc xoay giua cac benh nhan (khac tu the chup).
* Dau goi TRAI va PHAI doi xung guong => phai xay atlas rieng moi ben, hoac lat guong
  ve mot ben truoc. Dung `build_articular_atlas(side_key=...)` de tach.
* Dang ky affine cung nay tho hon non-rigid. Theo core.py, atlas chi can la TAP CHA HAO
  PHONG thien ve recall - precision la viec cua presence head. Nen tho la chap nhan duoc
  o MVP; nang cap len non-rigid la viec cua Stage 2 (§4.4 Step 2 muc 2-4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import core
from .core import SPACING, RayConfig


# ------------------------------------------------------- khung toa do be mat

#: Be rong toi thieu (mm) tren MOI truc de khung co nghia
EXTENT_MIN_MM = 5.0


@dataclass(frozen=True)
class SurfaceFrame:
    """Khung chuan hoa TRUC-GOC: tam + nua-be-rong theo tung truc anh (z, y, x).

    VI SAO KHONG XOAY THEO PCA (da thu va bac bo)
    ---------------------------------------------
    Ban dau khung nay dung truc chinh PCA + co dinh dau bang do lech (skew). Do bang
    phantom: KHONG DUNG DUOC.
      * Tri rieng gan nhau => huong truc tuy y (hinh cau la truong hop cuc doan, nhung
        dau xuong that cung cho axis_sep chi ~1.05).
      * Vat the doi xung qua mot mat phang co skew = 0 tren moi truc NAM TRONG mat phang
        do => dau truc tuy y. Loi cau/lat cau trong-ngoai gan doi xung nen dinh nay
        dung ca voi xuong that, khong chi voi phantom.
    Hai ca giong het nhau van co the ra toa do lat nguoc => atlas gop lai thanh chao.

    Cach dung hien tai: GIU NGUYEN truc anh, chi chuan hoa tam + ty le. Hop le vi ca
    dataset cung mot protocol, da resample ve cung spacing va cung quy uoc huong, nen
    truc anh VON DA mang nghia giai phau. Day chinh la muc 1 cua plan §4.4 Step 2
    ("rigid or affine registration"), gioi han o tinh tien + ty le theo truc.

    HAN CHE (phai ghi trong bai bao)
    --------------------------------
    * Khong bu duoc xoay giua cac benh nhan (tu the chup khac nhau).
    * Dau goi TRAI/PHAI doi xung guong => phai xay atlas rieng moi ben, hoac lat guong
      truoc. Dung `side_key` cua ArticularAtlas de tach.
    * Nang cap len dang ky non-rigid la viec cua Stage 2 (§4.4 Step 2 muc 2-4).
    """
    centroid: np.ndarray      # [3] mm, tam bao cua be mat
    scale: np.ndarray         # [3] mm, nua be rong theo tung truc
    extent: np.ndarray        # [3] mm, be rong day du (chan doan)

    @property
    def is_degenerate(self) -> bool:
        """Be mat det tren mot truc nao do => chuan hoa truc do vo nghia."""
        return bool((self.extent < EXTENT_MIN_MM).any())


def fit_frame(verts: np.ndarray, strict: bool = False, q: float = 2.0) -> SurfaceFrame:
    """Khung chuan hoa truc-goc tu dinh be mat.

    Dung PHAN VI (q% va 100-q%) thay vi min/max: mot vai dinh lac o mat cat FOV du de
    keo lech bao va lam sai chuan hoa cho ca ca.
    """
    v = np.asarray(verts, np.float64)
    lo = np.percentile(v, q, axis=0)
    hi = np.percentile(v, 100.0 - q, axis=0)
    extent = hi - lo
    centroid = 0.5 * (lo + hi)
    scale = np.maximum(0.5 * extent, 1e-6)

    frame = SurfaceFrame(centroid.astype(np.float32), scale.astype(np.float32),
                         extent.astype(np.float32))
    if strict and frame.is_degenerate:
        raise ValueError(
            f"Khung be mat SUY BIEN: be rong {frame.extent} mm, can >= {EXTENT_MIN_MM}mm "
            f"tren moi truc. Kiem lai mask xuong (co the rong hoac chi vai lat cat)."
        )
    return frame


def to_normalized(verts: np.ndarray, frame: SurfaceFrame) -> np.ndarray:
    """Dinh (mm) -> toa do chuan hoa ~[-1, 1] theo bao cua chinh be mat do."""
    return (np.asarray(verts, np.float32) - frame.centroid) / frame.scale


# ---------------------------------------------------------------- xay atlas

@dataclass
class ArticularAtlas:
    """Xac suat "co sun bam" tren luoi toa do chuan hoa."""
    prob: np.ndarray          # [B,B,B] trong [0,1]; NaN = khong co du lieu
    n_bins: int
    lo: float
    hi: float
    case_ids: tuple           # AUDIT: dung ca nao de xay (§2.2)
    fold: "int | None" = None
    side_key: "str | None" = None

    def _idx(self, u: np.ndarray) -> np.ndarray:
        t = (u - self.lo) / (self.hi - self.lo)
        i = np.floor(t * self.n_bins).astype(np.int64)
        return np.clip(i, 0, self.n_bins - 1)

    def query(self, verts: np.ndarray, frame: SurfaceFrame) -> np.ndarray:
        """Xac suat vung khop tai moi node cua mot ca MOI. NaN -> 0."""
        i = self._idx(to_normalized(verts, frame))
        p = self.prob[i[:, 0], i[:, 1], i[:, 2]]
        return np.nan_to_num(p, nan=0.0)


def assert_no_leak(atlas: ArticularAtlas, held_out_ids) -> None:
    """§2.2: mask sun cua ca val/test KHONG duoc dung de xay atlas.

    Goi ham nay TRUOC khi dung atlas tren tap held-out. No bien mot vi pham im lang
    thanh loi ngay tai cho.
    """
    bad = sorted(set(atlas.case_ids) & set(held_out_ids))
    if bad:
        raise ValueError(
            f"RO RI ATLAS: {len(bad)} ca held-out da duoc dung khi xay atlas "
            f"(vd {bad[:5]}). Vi pham plan §2.2 - xay lai atlas chi tu ca TRAIN cua fold."
        )


def build_articular_atlas(case_ids, loader, spacing=SPACING, cfg: RayConfig = RayConfig(),
                          n_bins: int = 24, lo: float = -1.2, hi: float = 1.2,
                          min_count: int = 3, fold: "int | None" = None,
                          side_key: "str | None" = None,
                          verbose: bool = False) -> ArticularAtlas:
    """Xay atlas tu ca TRAIN cua MOT fold.

    `loader(case_id) -> (mri, bone_mask, cart_mask)` - cung giao dien voi model.build_dataset.
    Chi dung `bone_mask` + `cart_mask`; mri bi bo qua.

    min_count : o luoi co it hon ngan nay quan sat thi de NaN (khong du bang chung).
                Tranh viec mot ca ca biet tu minh tao ra mot vung atlas.
    """
    acc = np.zeros((n_bins,) * 3, np.float64)     # so lan node o o nay CO sun bam
    cnt = np.zeros((n_bins,) * 3, np.float64)     # tong so node roi vao o nay
    used = []

    for cid in case_ids:
        _, bone, cart = loader(cid)
        if not bone.any() or not cart.any():
            if verbose:
                print(f"  {cid}: thieu xuong/sun - bo qua")
            continue
        _, verts, normals = core.bone_geometry(bone, spacing, cfg)
        occ = core.occupancy_target(cart, verts, normals, cfg, spacing)
        artic = core.oracle_domain(verts, occ, cfg)      # node co sun bam (+ vanh)

        u = to_normalized(verts, fit_frame(verts))
        t = np.clip((u - lo) / (hi - lo), 0.0, 1.0 - 1e-9)
        i = (t * n_bins).astype(np.int64)
        flat = np.ravel_multi_index((i[:, 0], i[:, 1], i[:, 2]), (n_bins,) * 3)

        cnt += np.bincount(flat, minlength=cnt.size).reshape(cnt.shape)
        acc += np.bincount(flat, weights=artic.astype(np.float64),
                           minlength=acc.size).reshape(acc.shape)
        used.append(cid)
        if verbose:
            print(f"  {cid}: {artic.sum()}/{len(verts)} node vung khop")

    if not used:
        raise ValueError("Khong ca nao dung duoc de xay atlas.")

    prob = np.full_like(acc, np.nan)
    ok = cnt >= min_count
    prob[ok] = acc[ok] / cnt[ok]
    return ArticularAtlas(prob.astype(np.float32), n_bins, lo, hi,
                          tuple(used), fold, side_key)


def atlas_domain(verts: np.ndarray, atlas: ArticularAtlas, thr: float = 0.2) -> np.ndarray:
    """Mien khop cho MOT ca, tra tu atlas -> mask bool [N].

    `thr` thap co CHU Y: atlas la BO SINH UNG VIEN THIEN VE RECALL (xem core.py) -
    bo sot node co sun se lam mat luon kha nang du doan o do, con thua node thi
    presence head loc duoc. Nen tha thua hon thieu.
    """
    return atlas.query(verts, fit_frame(verts)) >= thr
