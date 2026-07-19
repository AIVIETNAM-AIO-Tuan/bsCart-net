"""Lop hinh hoc cho he toa do neo vao be mat xuong.

SDF -> be mat -> phap tuyen -> tia -> occupancy -> splat nguoc ve voxel.

Chi dung scipy + skimage + numpy. KHONG can trimesh/vtk/pyvista/open3d.
Da xac minh chay duoc: distance_transform_edt(sampling=), marching_cubes(spacing=)
(tra vertex normals mien phi), map_coordinates (trilinear), cKDTree.

BA CAM BAY ANISOTROPY (spacing that: [0.70, 0.3646, 0.3646] mm)
---------------------------------------------------------------
1. EDT phai truyen `sampling=spacing` moi ra SDF dung mm. Thieu -> SDF theo voxel,
   sai ~2x giua truc z va in-plane.
2. Gaussian sigma phai doi mm -> voxel THEO TUNG TRUC: sigma_vox = sigma_mm / spacing.
   Truyen thang sigma_mm se lam mo qua tay theo z (0.70mm/voxel vs 0.3646 in-plane).
   Viec doi don vi nay LA THAT va co test khoa lai (test_smoothing_sigma_is_mm_not_voxels).
   NHUNG: gia thuyet "sigma=1.0mm lam M3 truot" KHONG TAI LAP duoc - phantom cau tron
   cho M3 = 100% o moi sigma (0.25/0.5/1.0). Giu sigma=0.5mm lam mac dinh THAN TRONG,
   nhung dung coi no la knob quyet dinh cho toi khi kiem lai TREN XUONG THAT o Phase 2
   (xuong that gap ghenh hon cau tron nhieu).
3. marching_cubes(spacing=) tra verts theo MM, nhung map_coordinates lam viec o
   INDEX space => phai chia lai cho spacing truoc khi sample: `pts_mm / spacing`.

HAI QUYET DINH DA DO, DUNG DAO NGUOC
------------------------------------
* Phap tuyen skimage huong XUONG gradient = VAO TRONG xuong. Phai DAO DAU (xem
  extract_surface). Sai dau -> tia ban vao trong xuong, occupancy toan 0. Bug nay
  IM LANG: model van train binh thuong, chi la khong hoc duoc gi. Test M3
  (check_normals) ton tai de bat dung no.
* KHONG remesh/subsample cho "dong deu" khi reconstruct (Step 2 cua plan doc).
  Do duoc: node spacing h=0.4/0.6/1.0mm -> round-trip Dice 0.980/0.957/0.811;
  o 1.0mm mat 20% voxel sun. Bat doi xung dung: subsample TU DO khi train (tia
  i.i.d., khong can phu kin); dung FULL marching-cubes density khi inference/splat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

# Spacing native cua OAI-ZIB / Dataset020, thu tu truc (z, y, x)
# z = through-plane 0.70mm, y/x = in-plane 0.3646mm
SPACING = (0.70, 0.3646, 0.3646)

# Nhan union 8-class cua Dataset020
LABELS = {
    "femoral_bone": 1, "femoral_cart": 2, "tibial_bone": 3, "med_tib_cart": 4,
    "lat_tib_cart": 5, "med_meniscus": 6, "lat_meniscus": 7, "patellar_cart": 8,
}

# Bin do day GT - stratifier CHINH (§2.3). Tinh truc tiep tu mask, khong can KL grade.
THICKNESS_EDGES = (0.5, 1.0, 2.0)
THICKNESS_NAMES = ("absent", "<=0.5mm", "<=1.0mm", "<=2.0mm", ">2.0mm")


@dataclass(frozen=True)
class RayConfig:
    """Tham so lay mau tia. Mac dinh = gia tri DA XAC MINH bang phantom."""
    k: int = 64              # so mau/tia. Do duoc K=48/64/128 -> Dice .987/.987/.988
    d_min: float = -1.0      # mm, am = vao trong xuong (bat lop giao dien sun-xuong)
    d_max: float = 6.0       # mm, du cho sun day nhat + khe khop
    smooth_mm: float = 0.5   # sigma lam muot SDF. 0.5 dau M3, 1.0 truot. Dung tang.

    @property
    def depths(self) -> np.ndarray:
        """Toa do do sau (mm), shape [K]."""
        return np.linspace(self.d_min, self.d_max, self.k).astype(np.float32)

    @property
    def step(self) -> float:
        """Khoang cach 2 mau lien tiep (mm). Dung doi occupancy -> do day."""
        return (self.d_max - self.d_min) / (self.k - 1)


# ---------------------------------------------------------------- SDF & be mat

def signed_distance(mask: np.ndarray, spacing=SPACING, smooth_mm: float = 0.0) -> np.ndarray:
    """Khoang cach co dau toi bien mask, don vi MM.

    Quy uoc (plan doc §3.4 Step 1): AM ben trong, DUONG ben ngoai.
    `sampling=spacing` bat buoc duoi anisotropy (cam bay 1).
    """
    m = np.asarray(mask).astype(bool)
    if not m.any():
        # Mask rong: moi diem "vo cung xa" ben ngoai. Tra gia tri huu han lon de
        # downstream (marching_cubes) khong no NaN.
        return np.full(m.shape, 1e6, np.float32)
    if m.all():
        return np.full(m.shape, -1e6, np.float32)

    d_out = distance_transform_edt(~m, sampling=spacing)
    d_in = distance_transform_edt(m, sampling=spacing)
    sdf = (d_out - d_in).astype(np.float32)

    if smooth_mm > 0:
        # CAM BAY 2: doi mm -> voxel theo TUNG truc
        sdf = gaussian_filter(sdf, sigma=[smooth_mm / s for s in spacing])
    return sdf


def extract_surface(sdf: np.ndarray, spacing=SPACING):
    """Muc 0 cua SDF -> (verts_mm [N,3], faces [F,3], normals_outward [N,3]).

    `spacing=` dat verts theo MM (khong phai index).

    DAO DAU: skimage tra vertex normals huong xuong gradient. SDF cua ta AM ben
    trong => gradient chi RA ngoai => normals skimage chi VAO TRONG xuong. Ta dao
    dau de co phap tuyen huong ra (plan doc §3.4 Step 3). Neu ai do "sua" cho nay,
    check_normals se bat duoc.
    """
    verts, faces, normals, _ = marching_cubes(sdf, level=0.0, spacing=spacing)
    n = -normals.astype(np.float32)                      # <-- huong RA ngoai
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
    return verts.astype(np.float32), faces, n


def check_normals(sdf: np.ndarray, verts: np.ndarray, normals: np.ndarray,
                  spacing=SPACING, eps_mm: float = 0.4):
    """Test M3 - phap tuyen co that su huong RA ngoai khong (plan doc §3.5 M3).

    Kiem tra  phi(s - eps*n) < 0 < phi(s + eps*n).
    Cong: >99.5% dung huong. Do tren phantom: 99.89% (smooth 0.5mm).

    Tra (frac_ok, ok_mask). Node sai thuong nam o mat cat FOV - loai chung.
    """
    inner = sample_volume(sdf, verts - eps_mm * normals, spacing, order=1)
    outer = sample_volume(sdf, verts + eps_mm * normals, spacing, order=1)
    ok = (inner < 0) & (outer > 0)
    return float(ok.mean()), ok


def bone_geometry(bone_mask: np.ndarray, spacing=SPACING, cfg: "RayConfig | None" = None):
    """Tien ich: mask xuong -> (sdf, verts, normals). Duong di chuan cua moi phase."""
    cfg = cfg or RayConfig()
    sdf = signed_distance(bone_mask, spacing, smooth_mm=cfg.smooth_mm)
    verts, _, normals = extract_surface(sdf, spacing)
    return sdf, verts, normals


# ---------------------------------------------------------- Lay mau & tia

def sample_volume(vol: np.ndarray, pts_mm: np.ndarray, spacing=SPACING,
                  order: int = 1, cval: float = 0.0) -> np.ndarray:
    """Noi suy `vol` tai cac diem theo MM. Tra shape = pts_mm.shape[:-1].

    CAM BAY 3: map_coordinates o INDEX space => chia cho spacing.
    order=1 = trilinear. cval cho diem roi ngoai FOV.
    """
    pts = np.asarray(pts_mm, np.float32)
    idx = (pts / np.asarray(spacing, np.float32)).reshape(-1, 3).T
    out = map_coordinates(vol.astype(np.float32), idx, order=order,
                          mode="constant", cval=cval)
    return out.reshape(pts.shape[:-1])


def ray_points(verts: np.ndarray, normals: np.ndarray, cfg: RayConfig = RayConfig()) -> np.ndarray:
    """Diem doc tia: r_i(d_k) = s_i + d_k * n_i  ->  [N, K, 3] mm."""
    return verts[:, None, :] + cfg.depths[None, :, None] * normals[:, None, :]


def sample_rays(vols: dict, verts, normals, cfg: RayConfig = RayConfig(),
                spacing=SPACING) -> np.ndarray:
    """Lay mau nhieu the tich doc tia -> X [N, K, C]. Thu tu kenh = thu tu key.

    Kenh MVP (plan doc §3.2): {"mri":..., "grad":..., "sdf":...} - superset cua M8-C.
    """
    pts = ray_points(verts, normals, cfg)
    chans = [sample_volume(v, pts, spacing, order=1) for v in vols.values()]
    return np.stack(chans, axis=-1).astype(np.float32)


def occupancy_target(cart_mask: np.ndarray, verts, normals,
                     cfg: RayConfig = RayConfig(), spacing=SPACING) -> np.ndarray:
    """Chieu mask sun GT vao toa do tia -> occupancy [N, K] uint8 (plan doc Step 6).

    Trilinear roi nguong 0.5: tuong duong "tam mau nam trong sun", va la phep toan
    ma splat_rays dao nguoc lai duoc (nen round-trip gan nhu khong mat gi).
    """
    pts = ray_points(verts, normals, cfg)
    v = sample_volume(cart_mask.astype(np.float32), pts, spacing, order=1)
    return (v > 0.5).astype(np.uint8)


def ray_stats(occ: np.ndarray, cfg: RayConfig = RayConfig()) -> dict:
    """Thong ke moi tia tu occupancy (plan doc Step 6 + test M4). Cac mang [N]:

      presence     bool  - tia co cham sun khong
      thickness    mm    - TONG so mau bat * step (tong, khong phai khoang dau tien:
                           tong moi la do day that khi tia cat nhieu khoang)
      first, last  mm    - giao dau/cuoi (NaN neu vang sun)
      n_intervals  int   - so khoang lien tuc; M4 dem ty le <=1
    """
    occ = np.asarray(occ).astype(bool)
    n, k = occ.shape
    d = cfg.depths

    presence = occ.any(axis=1)
    thickness = (occ.sum(axis=1) * cfg.step).astype(np.float32)

    first = np.full(n, np.nan, np.float32)
    last = np.full(n, np.nan, np.float32)
    if presence.any():
        idx = np.arange(k)
        fi = np.where(occ, idx[None, :], k).min(axis=1)
        li = np.where(occ, idx[None, :], -1).max(axis=1)
        first[presence] = d[fi[presence]]
        last[presence] = d[li[presence]]

    # Dem khoang = so lan chuyen 0->1 doc truc do sau
    pad = np.zeros((n, 1), bool)
    n_intervals = (np.diff(np.concatenate([pad, occ], 1).astype(np.int8), axis=1) == 1).sum(axis=1)

    return {"presence": presence, "thickness": thickness, "first": first,
            "last": last, "n_intervals": n_intervals.astype(np.int16)}


def single_interval_ratio(occ: np.ndarray) -> float:
    """Test M4 (plan doc §3.5): ty le tia CO sun ma chi co <=1 khoang lien tuc.

    >95% => bieu dien theo bien (two-boundary regression) kha thi.
    <90% => giu occupancy decoder, chua dung two-boundary.
    Do tren phantom: 0.998. Thuc te se thap hon o ria sun / vung cong cao,
    noi meniscus va sun chay chen vao.
    """
    st = ray_stats(occ)
    m = st["presence"]
    if not m.any():
        return float("nan")
    return float((st["n_intervals"][m] <= 1).mean())


# --------------------------------------------------- Dung lai the tich (splat)

def splat_rays(values: np.ndarray, verts, normals, shape, cfg: RayConfig = RayConfig(),
               spacing=SPACING, wgt_eps: float = 1e-3) -> np.ndarray:
    """Tia -> the tich xac suat, bang trilinear splat (plan doc Step 8).

    Dung adjoint chinh xac cua trilinear interpolation: moi mau rai trong so vao 8
    voxel lan can roi chuan hoa theo tong trong so. Nho vay splat la nghich dao
    (nghia binh phuong toi thieu) cua occupancy_target => round-trip gan nhu khong
    mat gi. Do duoc: ASSD 0.011mm, Dice 0.987 - thap hon phan thuong ~20x. Day la
    ly do M2 la THU TUC, khong phai cong chan.

    Hieu nang: np.bincount thay np.add.at - do duoc 7.0s -> ~0.4s.

    Voxel khong tia nao cham (wgt == 0) => ngoai dai tia => tra 0 = background, dung
    ve ngu nghia. O FULL marching-cubes density do duoc 0% voxel sun bi bo sot; o mat
    do thua (h=1.0mm) mat 20% - do la ly do KHONG remesh khi inference.
    """
    shape = tuple(int(s) for s in shape)
    pts = ray_points(verts, normals, cfg)                          # [N,K,3] mm
    idx = (pts / np.asarray(spacing, np.float32)).reshape(-1, 3)   # -> index space
    val = np.asarray(values, np.float32).reshape(-1)

    lo = np.floor(idx).astype(np.int64)
    frac = idx - lo

    acc = np.zeros(int(np.prod(shape)), np.float64)
    wgt = np.zeros(int(np.prod(shape)), np.float64)
    dims = np.asarray(shape)

    for corner in range(8):
        off = np.array([(corner >> 2) & 1, (corner >> 1) & 1, corner & 1], np.int64)
        pos = lo + off
        # Trong so trilinear: gan goc nao thi trong so goc do cao
        w = np.prod(np.where(off == 1, frac, 1.0 - frac), axis=1)

        valid = np.all((pos >= 0) & (pos < dims), axis=1) & (w > 0)
        if not valid.any():
            continue
        p, ww = pos[valid], w[valid]
        flat = np.ravel_multi_index((p[:, 0], p[:, 1], p[:, 2]), shape)
        acc += np.bincount(flat, weights=ww * val[valid], minlength=acc.size)
        wgt += np.bincount(flat, weights=ww, minlength=wgt.size)

    out = np.zeros_like(acc)
    hit = wgt > wgt_eps
    out[hit] = acc[hit] / wgt[hit]
    return out.reshape(shape).astype(np.float32)


def roundtrip(cart_mask: np.ndarray, bone_mask: np.ndarray, spacing=SPACING,
              cfg: RayConfig = RayConfig()) -> np.ndarray:
    """Test M2: mask sun GT -> toa do tia -> tro lai voxel (plan doc §3.5 M2).

    Do TRAN ma phep doi toa do + rasterization ap len. Neu round-trip te thi dung
    train model.

    DO THUC TE TREN PHANTOM (bsc/tests/test_core.py::test_m2_thin_ceiling):

        do day sun     M2 Dice    M2 ASSD
        1.6mm          0.9995     0.001mm
        0.5mm          0.925      0.030mm
        0.4mm          0.848      0.065mm

    => TRAN ROUND-TRIP SUP O VUNG SUN MONG - dung cai vung ma gia thuyet nham toi.
       Voi sun 0.4mm, phep doi toa do TU NO da mat 15% Dice du du doan hoan hao.

    => Cong M2 cua plan doc (Dice >= 0.97) se TRUOT o sun cuc mong. DAT CONG TREN
       ASSD, KHONG PHAI DICE: ASSD van 0.001-0.065mm o moi che do, duoi xa muc tieu
       0.1mm. Dice tren cau truc day ~1 voxel nhay den muc tan nhan.

    Dieu nay CUNG CO quyet dinh cua ke hoach: metric BIEN phai la chinh, khong phai
    Dice. Va no khong doi ket luan "khong co san roi rac hoa" - mot du doan khop
    hoan hao mask GT van cho ASSD = 0; san that la NHIEU CHU THICH.
    """
    _, verts, normals = bone_geometry(bone_mask, spacing, cfg)
    occ = occupancy_target(cart_mask, verts, normals, cfg, spacing)
    rec = splat_rays(occ.astype(np.float32), verts, normals, cart_mask.shape, cfg, spacing)
    return rec


# ------------------------------------------------------ Truy van theo be mat

def nearest_surface_value(verts: np.ndarray, node_values: np.ndarray,
                          query_pts_mm: np.ndarray, max_dist_mm: float = np.inf,
                          fill: float = np.nan) -> np.ndarray:
    """Gan gia tri cua node be mat gan nhat cho moi diem truy van (cKDTree).

    Dung cho M0 theo HAI chieu:
      - gan do day GT cho tung voxel sun GT;
      - gan do day cua node GT GAN NHAT cho voxel be mat PRED.
    Chieu thu hai la cach bin `absent` duoc lap day (no khong co be mat GT rieng),
    va la ly do chan doan M0 hoat dong duoc o OA nang.
    """
    tree = cKDTree(verts)
    q = np.asarray(query_pts_mm, np.float32).reshape(-1, 3)
    d, i = tree.query(q, distance_upper_bound=max_dist_mm)
    out = np.full(i.shape, fill, np.float32)
    ok = np.isfinite(d) & (i < len(verts))
    out[ok] = np.asarray(node_values, np.float32)[i[ok]]
    return out.reshape(np.asarray(query_pts_mm).shape[:-1])


def voxel_centers_mm(mask: np.ndarray, spacing=SPACING) -> np.ndarray:
    """Toa do tam (mm) cua cac voxel bat trong mask -> [M, 3]."""
    zyx = np.argwhere(np.asarray(mask).astype(bool))
    return (zyx * np.asarray(spacing, np.float32)).astype(np.float32)


# ------------------------------------------------------- Mien khop (articular)

def oracle_domain(verts: np.ndarray, occ: np.ndarray, cfg: RayConfig = RayConfig(),
                  rim_mm: float = 5.0, present_max_mm: float = 3.0) -> np.ndarray:
    """MVP-A: mien khop tu vung bam cua sun GT, CO vanh no (plan doc §3.4 Step 4).

    1. Node "present": tia cham sun trong d in [0, present_max_mm].
    2. No them rim_mm quanh chung.
    3. Domain = present hop vanh no.

    Buoc 2 KHONG phai trang tri. Thieu no, domain khong chua tia absent nao =>
    presence head khong co negative => khong the kiem chung luan diem ve sun MAT,
    tuc chinh gia thuyet. Plan doc mo ta MVP-A thieu chi tiet nay.
    """
    d = cfg.depths
    band = (d >= 0) & (d <= present_max_mm)
    present = np.asarray(occ).astype(bool)[:, band].any(axis=1)
    if not present.any():
        return present
    dist, _ = cKDTree(verts[present]).query(verts, distance_upper_bound=rim_mm)
    return present | np.isfinite(dist)


def joint_facing_domain(verts: np.ndarray, normals: np.ndarray, sdf_opposing: np.ndarray,
                        spacing=SPACING, march_mm: float = 15.0,
                        n_steps: int = 30) -> np.ndarray:
    """MVP-B "B-geom": articular ~ node co phap tuyen huong VAO khe khop.

    Voi moi node, di doc +n va kiem tra co tien lai gan xuong doi dien khong.
    Thuan numpy, khong fit, khong fold-specific => KHONG THE ro ri nhan test ve mat
    CAU TRUC (§2.2 thoa man tu dong, khong can ky luat).

    Reframe quan trong: atlas la BO SINH UNG VIEN THIEN VE RECALL - precision la viec
    cua presence head. Nen no chi can la TAP CHA HAO PHONG, khong can chinh xac.
    Dieu do bien item kho nhat cua MVP thanh kha thi.

    Han che: bo sot trochlea vi Dataset001 khong co xuong banh che => Stage 1 gioi han
    o sun dui VUNG CHIU LUC, va phai noi ro dieu do trong bai bao.
    """
    steps = np.linspace(0.0, march_mm, n_steps, dtype=np.float32)
    pts = verts[:, None, :] + steps[None, :, None] * normals[:, None, :]
    along = sample_volume(sdf_opposing, pts, spacing, order=1, cval=1e6)
    closest = along.min(axis=1)
    return (closest < along[:, 0]) & (closest < march_mm)


# ------------------------------------- Nhieu loan huong tia (M6) & be mat (M7)

def tangent_frame(normals: np.ndarray):
    """Hai vector tiep tuyen truc chuan cho moi phap tuyen -> (t1 [N,3], t2 [N,3]).

    Chon vector mo dau KHONG song song voi n (lay truc it trung nhat) roi Gram-Schmidt.
    Dung cho M6 (tia tiep tuyen) va M7 (xoay phap tuyen trong mat phang tiep tuyen).
    """
    n = np.asarray(normals, np.float32)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
    # Truc nao it song song voi n nhat thi lam vector mo dau
    seed = np.zeros_like(n)
    seed[np.arange(len(n)), np.argmin(np.abs(n), axis=1)] = 1.0
    t1 = np.cross(n, seed)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-8
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2, axis=1, keepdims=True) + 1e-8
    return t1.astype(np.float32), t2.astype(np.float32)


def direction_field(normals: np.ndarray, mode: str = "normal", seed: int = 0) -> np.ndarray:
    """Test M6 (plan doc §3.5) - sinh truong huong tia DOI CHUNG.

    mode:
      "normal"  - phap tuyen be mat (nhanh that su cua gia thuyet)
      "axial"   - mot huong co dinh theo truc z (mo phong "cat lat" thuan tuy)
      "random"  - huong ngau nhien tung node
      "tangent" - tiep tuyen be mat (vuong goc phap tuyen)

    VI SAO CAN: neu mo hinh doc theo huong tuy y cung tot ngang phap tuyen, thi loi ich
    den tu "them mot mang nua", KHONG phai tu he toa do. M6 la test phan bac chinh cua
    Stage 1 - thieu no, moi ket qua duong tinh deu khong quy duoc cho gia thuyet.
    """
    n = np.asarray(normals, np.float32)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(seed)

    if mode == "normal":
        return n
    if mode == "axial":
        # Huong z co dinh; giu dau theo phap tuyen de tia khong ban thang vao xuong
        ax = np.zeros_like(n)
        ax[:, 0] = np.where(n[:, 0] >= 0, 1.0, -1.0)
        return ax
    if mode == "random":
        v = rng.normal(size=n.shape).astype(np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    if mode == "tangent":
        t1, t2 = tangent_frame(n)
        phi = rng.uniform(0, 2 * np.pi, len(n)).astype(np.float32)[:, None]
        t = np.cos(phi) * t1 + np.sin(phi) * t2
        return (t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    raise ValueError(f"mode khong hop le: {mode!r} "
                     f"(chon normal/axial/random/tangent)")


def jitter_surface(verts: np.ndarray, normals: np.ndarray, delta_s_mm: float = 0.0,
                   delta_theta_deg: float = 0.0, seed: int = 0):
    """Test M7 (plan doc §3.5) - nhieu loan be mat + phap tuyen -> (verts', normals').

    delta_s_mm      : do lech vi tri DOC PHAP TUYEN, Gaussian std = delta_s_mm.
                      Chon doc phap tuyen vi sai so phan doan xuong chu yeu theo huong do
                      (day/mong vo xuong), khong phai truot doc be mat.
    delta_theta_deg : goc xoay phap tuyen trong mat phang tiep tuyen, Gaussian std.

    Muc dich: do do nhay TRUOC khi chuyen sang be mat xuong DU DOAN (P5). Neu mo hinh
    sup o 0.5mm jitter thi khong the ky vong no song sot voi xuong du doan that.
    """
    v = np.asarray(verts, np.float32)
    n = np.asarray(normals, np.float32)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(seed)

    if delta_s_mm > 0:
        v = v + n * rng.normal(0.0, delta_s_mm, len(v)).astype(np.float32)[:, None]

    if delta_theta_deg > 0:
        t1, t2 = tangent_frame(n)
        th = np.deg2rad(rng.normal(0.0, delta_theta_deg, len(n))).astype(np.float32)[:, None]
        phi = rng.uniform(0, 2 * np.pi, len(n)).astype(np.float32)[:, None]
        t = np.cos(phi) * t1 + np.sin(phi) * t2      # truc xoay ngau nhien trong mat tiep tuyen
        n = np.cos(th) * n + np.sin(th) * t
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-8

    return v.astype(np.float32), n.astype(np.float32)


def assign_thickness_bin(thickness_mm: np.ndarray) -> np.ndarray:
    """Do day (mm) -> chi so bin. 0=absent, 1=<=0.5, 2=<=1.0, 3=<=2.0, 4=>2.0.

    Xem THICKNESS_NAMES. Day la stratifier CHINH cua Stage 1 (§2.3) - tinh truc tiep
    tu mask GT nen khong phu thuoc vao viec co lay duoc KL grade hay khong.
    """
    t = np.asarray(thickness_mm, np.float32)
    out = np.digitize(t, THICKNESS_EDGES, right=True) + 1
    out[t <= 0] = 0
    return out.astype(np.int8)
