"""M0 - Chan doan HEADROOM. CONG CHAN CHINH cua Stage 1 (GATE 1).

KHONG co trong plan doc. Day la thu quan trong nhat cua Stage 1 - no hoi dung cau
hoi ma ROI cascade da quen hoi: PHAN THUONG LON CO NAO, truoc khi xay bat cu thu gi.

ROI cascade da dot mot chu ky vi bat dau xay (crop, train ResEnc-L 250ep) roi moi do
xem localization co giup khong - va cau tra loi la KHONG (Dice +-0.003 du bbox hoan
hao). M0 do phan thuong TRUOC, bang ~1h CPU, khong GPU.

Y TUONG
-------
He toa do neo vao xuong hua cai thien o BIEN va DO DAY, dac biet noi sun MONG / MAT.
Cau hoi: loi cua ResEnc (baseline B0) co THUC SU nam nhieu o vung mong/mat khong?
Neu loi tap trung o sun DAY (noi ResEnc da tot), thi bieu dien theo tia khong the
giup nhieu, du no hoan hao o vung mong.

CACH DO
-------
1. Truong do day GT: SDF xuong -> node be mat -> tia -> occupancy GT -> do day/node.
2. Gan do day cho MOI voxel be mat GT (node gan nhat) va cho MOI voxel be mat PRED
   (node GT gan nhat - day la cach bin `absent` duoc lap day, vi no khong co be mat
   GT rieng; chinh chieu nay lam chan doan hoat dong o OA nang).
3. Bin do day {absent, <=0.5, <=1.0, <=2.0, >2.0}mm. Voi moi bin: N, khoang cach
   trung binh, TONG khoang cach (= "error mass"), % tong error mass.
4. Counterfactual "phan thuong": neu triet tieu toan bo loi o {absent, <=0.5, <=1.0},
   ASSD/surface-Dice cai thien bao nhieu? Do la tran tren cua thu gia thuyet co the
   dat duoc - paired per-case, bootstrap CI.

GATE 1 (dung cung)
------------------
  TIEP     : dASSD_prize >= 0.08mm  VA  bin (thin+absent) >= 35% error mass
  DOI PHAM VI : dASSD_prize in [0.04, 0.08) -> viet lai muc tieu quanh presence/thickness
  DUNG     : dASSD_prize < 0.04mm  HOAC  bac thang z chiem uu the (M0c)

error mass la thu dang ke, KHONG phai dien tich. Vung mong kho bat tuong xung nen
ty trong error mass phai vuot ty trong dien tich - do la cai M0 do.
"""

from __future__ import annotations

import numpy as np

from . import core, metrics
from .core import RayConfig, SPACING, THICKNESS_NAMES


def gt_thickness_per_node(bone_mask, cart_mask, spacing=SPACING, cfg=RayConfig()):
    """Do day sun GT tai moi node be mat xuong. Tra (verts, normals, thickness_mm)."""
    _, verts, normals = core.bone_geometry(bone_mask, spacing, cfg)
    occ = core.occupancy_target(cart_mask, verts, normals, cfg, spacing)
    st = core.ray_stats(occ, cfg)
    return verts, normals, st["thickness"]


def error_mass_by_thickness(gt_cart, pred_cart, bone_mask, spacing=SPACING,
                            cfg=RayConfig(), thickness=None) -> dict:
    """Phan ra khoi luong loi be mat cua ResEnc theo bin do day GT.

    Hai chieu (doi xung, giong ASSD):
      - voxel be mat GT -> khoang cach toi be mat PRED, bin theo do day cua CHINH no
        (khoi luong duoi-phan-doan / false-negative)
      - voxel be mat PRED -> khoang cach toi be mat GT, bin theo do day cua node GT
        GAN NHAT (khoi luong thua / false-positive; day la cach bin `absent` co du lieu)

    `thickness`: tuple (verts, normals, thick_node) tu gt_thickness_per_node. Neu None,
    tu tinh. Truyen vao de KHONG dung marching-cubes hai lan khi goi kem prize (M0 loop).

    Tra dict: per_bin[name] = {n, mean_dist_mm, error_mass_mm, frac_error_mass},
    kem tong.
    """
    if thickness is None:
        verts, _, thick_node = gt_thickness_per_node(bone_mask, cart_mask=gt_cart,
                                                     spacing=spacing, cfg=cfg)
    else:
        verts, _, thick_node = thickness

    sg = metrics.surface_mask(gt_cart)
    sp_ = metrics.surface_mask(pred_cart)
    from scipy.ndimage import distance_transform_edt
    d_to_pred = distance_transform_edt(~sp_, sampling=spacing) if sp_.any() else \
        np.full(gt_cart.shape, np.nan, np.float32)
    d_to_gt = distance_transform_edt(~sg, sampling=spacing) if sg.any() else \
        np.full(gt_cart.shape, np.nan, np.float32)

    # Toa do mm cua voxel be mat -> do day node GT gan nhat
    gt_surf_pts = core.voxel_centers_mm(sg, spacing)
    pr_surf_pts = core.voxel_centers_mm(sp_, spacing)

    thick_gt_surf = core.nearest_surface_value(verts, thick_node, gt_surf_pts)
    thick_pr_surf = core.nearest_surface_value(verts, thick_node, pr_surf_pts)

    dist_gt = d_to_pred[sg]                     # GT -> PRED
    dist_pr = d_to_gt[sp_]                       # PRED -> GT

    # Gop hai chieu: khoang cach + do day tuong ung
    all_dist = np.concatenate([dist_gt, dist_pr])
    all_thick = np.concatenate([thick_gt_surf, thick_pr_surf])
    ok = np.isfinite(all_dist) & np.isfinite(all_thick)
    all_dist, all_thick = all_dist[ok], all_thick[ok]

    bins = core.assign_thickness_bin(all_thick)
    total_mass = float(all_dist.sum()) or 1.0

    per_bin = {}
    for bi, name in enumerate(THICKNESS_NAMES):
        sel = bins == bi
        d = all_dist[sel]
        per_bin[name] = {
            "n": int(sel.sum()),
            "mean_dist_mm": float(d.mean()) if d.size else 0.0,
            "error_mass_mm": float(d.sum()),
            "frac_error_mass": float(d.sum()) / total_mass,
        }
    return {"per_bin": per_bin, "total_error_mass_mm": float(all_dist.sum()),
            "n_surface_voxels": int(all_dist.size)}


def prize_counterfactual(gt_cart, pred_cart, bone_mask, spacing=SPACING, cfg=RayConfig(),
                         prize_bins=("absent", "<=0.5mm", "<=1.0mm"), thickness=None) -> dict:
    """Neu triet tieu loi o cac bin `prize_bins`, ASSD/surface-Dice cai thien bao nhieu?

    Day la TRAN TREN cua thu bieu dien theo tia co the dat - vi no chi giup o cac bin
    mong/vang. Tinh ca-level de vao paired bootstrap.

    `thickness`: tuple (verts, normals, thick_node) tu gt_thickness_per_node. Neu None,
    tu tinh. Truyen vao de KHONG dung marching-cubes hai lan khi goi kem error_mass.

    Cach: bo cac voxel be mat thuoc bin phan thuong ra khoi tinh ASSD (coi nhu da
    dat hoan hao o do), roi tinh lai ASSD tren phan con lai.
    """
    if thickness is None:
        verts, _, thick_node = gt_thickness_per_node(bone_mask, cart_mask=gt_cart,
                                                     spacing=spacing, cfg=cfg)
    else:
        verts, _, thick_node = thickness
    sg = metrics.surface_mask(gt_cart)
    sp_ = metrics.surface_mask(pred_cart)
    if not sg.any() or not sp_.any():
        return {"assd_base": np.nan, "assd_prize": np.nan, "d_assd_prize": np.nan,
                "sdice05_base": np.nan, "sdice05_prize": np.nan}

    from scipy.ndimage import distance_transform_edt
    d_to_pred = distance_transform_edt(~sp_, sampling=spacing)
    d_to_gt = distance_transform_edt(~sg, sampling=spacing)

    thick_gt = core.nearest_surface_value(verts, thick_node,
                                          core.voxel_centers_mm(sg, spacing))
    thick_pr = core.nearest_surface_value(verts, thick_node,
                                          core.voxel_centers_mm(sp_, spacing))

    dist_gt, dist_pr = d_to_pred[sg], d_to_gt[sp_]
    bins_gt = core.assign_thickness_bin(thick_gt)
    bins_pr = core.assign_thickness_bin(thick_pr)
    prize_idx = {THICKNESS_NAMES.index(b) for b in prize_bins}

    keep_gt = ~np.isin(bins_gt, list(prize_idx))
    keep_pr = ~np.isin(bins_pr, list(prize_idx))

    base_all = np.concatenate([dist_gt, dist_pr])
    prize_all = np.concatenate([dist_gt[keep_gt], dist_pr[keep_pr]])
    base_all = base_all[np.isfinite(base_all)]
    prize_all = prize_all[np.isfinite(prize_all)]

    assd_base = float(base_all.mean())
    # Voxel bin phan thuong duoc coi nhu khop hoan hao (khoang cach 0)
    n_prize = base_all.size - prize_all.size
    assd_prize = float(prize_all.sum() / base_all.size) if base_all.size else np.nan

    return {
        "assd_base_mm": assd_base,
        "assd_prize_mm": assd_prize,
        "d_assd_prize_mm": assd_base - assd_prize,
        "sdice05_base": float((base_all <= 0.5).mean()),
        "sdice05_prize": float((np.concatenate([prize_all, np.zeros(n_prize)]) <= 0.5).mean()),
        "n_prize_voxels": int(n_prize),
        "n_total_voxels": int(base_all.size),
    }


def gt_staircase_z_vs_inplane(gt_cart, spacing=SPACING) -> dict:
    """M0c - do "bac thang" bien sun GT theo z vs in-plane.

    OAI-ZIB duoc ve tay theo tung lat cat axial. Neu bien GT tu mang nhieu luong tu
    hoa ~0.70mm theo z, do la thanh phan KHONG THE loai bo cua con so ASSD 0.21mm -
    khong kien truc nao xoa duoc. Neu no chiem uu the, gia thuyet nhu dang viet la
    khong kiem chung duoc tren GT nay => DUNG (nhanh Gate 1).

    Cach do don gian, dieu kien du: so do go ghe cua be mat GT theo huong z so voi
    in-plane. Do bang do lech gradient chuan hoa cua SDF sun theo tung truc tren cac
    voxel be mat. Bac thang z => thanh phan gradient z nho bat thuong tren mat gan nhu
    thang dung theo z (vi bien bi ep vao mat phang lat cat).
    """
    sdf = core.signed_distance(gt_cart, spacing, smooth_mm=0.0)
    surf = metrics.surface_mask(gt_cart)
    if surf.sum() < 100:
        return {"z_roughness": np.nan, "inplane_roughness": np.nan, "ratio": np.nan}

    # Gradient SDF theo tung truc (don vi mm/mm). Tren be mat, |grad| ~ 1.
    gz, gy, gx = np.gradient(sdf, *spacing)
    zc = np.abs(gz[surf])
    ic = np.abs(gy[surf]) + np.abs(gx[surf])
    # Do "nham" = do lech quanh gia tri lang cua thanh phan phap tuyen theo tung huong
    return {
        "z_grad_std": float(zc.std()),
        "inplane_grad_std": float(ic.std()),
        "z_over_inplane_std": float(zc.std() / (ic.std() + 1e-8)),
        "n_surface": int(surf.sum()),
    }


def evaluate_gate1(prize_stats_per_case: list, mass_stats_per_case: list,
                   n_boot: int = 10000, seed: int = 0) -> dict:
    """Tong hop M0 tren nhieu ca -> quyet dinh Gate 1.

    prize_stats_per_case: list dict tu prize_counterfactual (moi ca)
    mass_stats_per_case:  list dict tu error_mass_by_thickness (moi ca)
    """
    d_assd = np.array([p["d_assd_prize_mm"] for p in prize_stats_per_case], float)
    # Bootstrap CI cua phan thuong (paired: day la cai thien noi bo tung ca)
    zeros = np.zeros_like(d_assd)
    boot = metrics.paired_bootstrap(zeros, d_assd, n_boot=n_boot, seed=seed)

    # Ty trong error mass o bin thin+absent, trung binh qua cac ca
    thin_names = ("absent", "<=0.5mm", "<=1.0mm")
    thin_frac = []
    for m in mass_stats_per_case:
        f = sum(m["per_bin"][n]["frac_error_mass"] for n in thin_names)
        thin_frac.append(f)
    thin_frac = np.array(thin_frac, float)

    d_mean = float(np.nanmean(d_assd))
    thin_mean = float(np.nanmean(thin_frac))

    if d_mean >= 0.08 and thin_mean >= 0.35:
        decision = "PROCEED"
    elif d_mean >= 0.04:
        decision = "RESCOPE"
    else:
        decision = "STOP"

    return {
        "decision": decision,
        "d_assd_prize_mean_mm": d_mean,
        "d_assd_prize_ci": (boot["ci_low"], boot["ci_high"]),
        "thin_absent_error_mass_frac_mean": thin_mean,
        "n_cases": len(prize_stats_per_case),
        "rule": ("PROCEED nếu dASSD>=0.08 & thin>=0.35 | RESCOPE nếu dASSD>=0.04 | "
                 "else STOP"),
    }
