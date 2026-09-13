"""Biomarker hinh thai tu mask 8-class: legacy (S3, voxel) + neo be mat xuong (S6, FCL).

MOT DINH NGHIA cho moi biomarker - notebook chi goi, KHONG copy-paste.

SPACING LA THAM SO BAT BUOC (khong co default)
-----------------------------------------------
io_utils ghi nhan mask that doc bang nibabel/SimpleITK cho spacing (0.3646, 0.3646, 0.70)
- truc 0.70mm nam CUOI, khac core.SPACING. Moi ham o day nhan `spacing` theo DUNG thu tu
truc cua mang truyen vao; notebook lay spacing tu header, khong bao gio gia dinh.

HAI HO BIOMARKER
----------------
1. legacy_biomarkers  : dinh nghia Y HET biomarker_s3 (vol / thickness proxy / denuded_ratio /
                        extrusion). Giu de bang moi la SUPERSET cua bang cu, doi chieu 1:1.
2. surface_biomarkers : do tren luoi marching-cubes cua be mat xuong (3D that, khong theo lat):
     - do day sun doc PHAP TUYEN tai tung dinh (tia, cung co che voi core.py)
     - footprint (tAB) = closing TRAC DIA cua vung co sun -> lap lo BEN TRONG mang sun
     - FCL (= dAB) = dinh trong footprint ma do day = 0
     - ThC.tAB (mat sun tinh = 0), ThC.cAB, phan tram mong, so o mat sun
   Thuat ngu Eckstein/Wirth: tAB total area of subchondral bone, cAB cartilage-covered,
   dAB denuded. FCL trong MOAKS ~ dAB%.

VI SAO KHONG DUNG denuded_ratio CU CHO FCL
------------------------------------------
Mau so cua no la TOAN BO be mat xuong trong FOV (than xuong, mat cat FOV...) nen ty le bi
chi phoi boi giai phau/FOV, khong phai ton thuong. Va thickness proxy cu = the tich / dien
tich tiep xuc: mat sun lam ca tu so lan mau so cung giam => "mu" voi FCL. Ca hai dieu nay
duoc chung minh bang phantom trong tests/test_biomarkers.py.

HAN CHE DA BIET (ghi khi bao cao)
---------------------------------
* Footprint bang closing chi bat duoc mat sun DUOC BAO QUANH boi sun con lai. Mat sun o
  ria mang hoac ca khoang tro trui KHONG duoc dem => FCL la CAN DUOI. Ban kinh closing la
  tham so nhay (test_fcl_closing_radius_sensitivity) - bao cao gia tri dung.
* Khong co nhan gai xuong: gai xuong nam trong mask xuong; mu sun gai xuong co the bi gan
  nhan sun => footprint bi keo ra ria. Kiem bang diem gai xuong X-quang (notebook S6 §6).
* Khong co xuong banh che trong mask => khong do duoc FCL sun banh che.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage.measure import marching_cubes

# Nhan union 8-class cua Dataset020 (id -> ten), cung thu tu voi biomarker_s3
LABELS = {1: "femoral_bone", 2: "femoral_cart", 3: "tibial_bone", 4: "med_tib_cart",
          5: "lat_tib_cart", 6: "med_meniscus", 7: "lat_meniscus", 8: "patellar_cart"}

#: Tien to cot feature. S4 chon feature bang startswith(LEGACY_PREFIXES); S7 dung ca hai.
LEGACY_PREFIXES = ("vol_", "thickness_", "denuded_", "extrusion_")
SURFACE_PREFIXES = ("tab_", "cab_", "fcl_", "thc_", "thickp05_", "thin_")
FEATURE_PREFIXES = LEGACY_PREFIXES + SURFACE_PREFIXES

#: Khoang (compartment) cua ho surface. fem = ca sun dui; mt/lt = mam chay trong/ngoai.
COMPARTMENTS = ("fem", "fem_med", "fem_lat", "mt", "lt")


def _check(arr, spacing):
    arr = np.asarray(arr)
    spacing = np.asarray(spacing, np.float64).reshape(-1)
    if arr.ndim != 3:
        raise ValueError(f"mask phai 3D, nhan shape {arr.shape}")
    if spacing.shape != (3,) or (spacing <= 0).any():
        raise ValueError(f"spacing phai la 3 so duong theo thu tu truc cua mang, nhan {spacing}")
    return arr, spacing


# =====================================================================================
# 1. LEGACY - port nguyen ven tu biomarker_s3 (cell 6/8/10). KHONG doi dinh nghia o day:
#    doi o day = bang v2 khong con doi chieu duoc voi bang cu.
# =====================================================================================

def voxel_vol_mm3(spacing) -> float:
    return float(spacing[0] * spacing[1] * spacing[2])


def label_volume_mm3(arr, spacing, lab) -> float:
    return int((np.asarray(arr) == lab).sum()) * voxel_vol_mm3(spacing)


def bone_surface_voxels(bone_bin) -> np.ndarray:
    """Voxel bien cua xuong (erosion 6-lan-can, ngoai FOV coi la nen)."""
    bone_bin = np.asarray(bone_bin, bool)
    return bone_bin & ~ndi.binary_erosion(bone_bin)


def contact_and_surface_area(bone_bin, cart_bin, spacing):
    """(dien tich tiep xuc sun-xuong, tong dien tich be mat xuong) mm^2 - xap xi dem voxel.

    Moi voxel bien duoc gan dien tich = tich 2 canh NHO NHAT bat ke huong mat => phu thuoc
    huong (xem CLAUDE.md thao luan). Giu nguyen de khop S3.
    """
    surf = bone_surface_voxels(bone_bin)
    contact = surf & ndi.binary_dilation(np.asarray(cart_bin, bool))
    s = sorted(spacing)
    face_area = s[0] * s[1]
    return float(contact.sum()) * face_area, float(surf.sum()) * face_area


def cartilage_thickness_proxy(vol_mm3, contact_mm2) -> float:
    return vol_mm3 / contact_mm2 if contact_mm2 > 0 else np.nan


def denuded_ratio(bone_bin, cart_bin) -> float:
    """1 - (bien xuong cham sun) / (TOAN BO bien xuong). Mau so gom ca than xuong/mat cat FOV."""
    surf = bone_surface_voxels(bone_bin)
    if surf.sum() == 0:
        return np.nan
    contact = surf & ndi.binary_dilation(np.asarray(cart_bin, bool))
    return 1.0 - contact.sum() / surf.sum()


def split_axis_by_centroids(mask_a, mask_b) -> int:
    ca = np.array(ndi.center_of_mass(mask_a)) if mask_a.sum() else np.zeros(3)
    cb = np.array(ndi.center_of_mass(mask_b)) if mask_b.sum() else np.zeros(3)
    return int(np.argmax(np.abs(ca - cb)))


def meniscus_extrusion_ratio(arr, meniscus_lab, tibial_bone_lab=3) -> float:
    """Phan hinh chieu sun chem nam ngoai 'bong' xuong chay / tong hinh chieu sun chem.

    Chieu 3D -> 2D doc truc co khoang cach trong tam lon nhat giua 2 cau truc.
    """
    arr = np.asarray(arr)
    men = arr == meniscus_lab
    tib = arr == tibial_bone_lab
    if men.sum() == 0 or tib.sum() == 0:
        return np.nan
    axis = split_axis_by_centroids(tib, men)
    tib_foot = tib.any(axis=axis)
    men_foot = men.any(axis=axis)
    if men_foot.sum() == 0:
        return np.nan
    outside = men_foot & ~tib_foot
    return float(outside.sum()) / float(men_foot.sum())


def legacy_biomarkers(arr, spacing) -> dict:
    """Dung bo cot cua biomarker_s3 (biomarker_table.csv), cung thu tu, cung dinh nghia."""
    arr, spacing = _check(arr, spacing)
    row = {}
    for lab, name in LABELS.items():
        row[f"vol_{name}_mm3"] = label_volume_mm3(arr, spacing, lab)

    fem_bone, tib_bone = arr == 1, arr == 3
    fem_cart, med_tib, lat_tib = arr == 2, arr == 4, arr == 5

    ct, _ = contact_and_surface_area(fem_bone, fem_cart, spacing)
    row["thickness_femoral_mm"] = cartilage_thickness_proxy(row["vol_femoral_cart_mm3"], ct)
    row["denuded_ratio_femoral"] = denuded_ratio(fem_bone, fem_cart)

    ct, _ = contact_and_surface_area(tib_bone, med_tib, spacing)
    row["thickness_med_tib_mm"] = cartilage_thickness_proxy(row["vol_med_tib_cart_mm3"], ct)
    ct, _ = contact_and_surface_area(tib_bone, lat_tib, spacing)
    row["thickness_lat_tib_mm"] = cartilage_thickness_proxy(row["vol_lat_tib_cart_mm3"], ct)
    row["denuded_ratio_tibial"] = denuded_ratio(tib_bone, (med_tib | lat_tib))

    row["extrusion_med_meniscus"] = meniscus_extrusion_ratio(arr, 6, tibial_bone_lab=3)
    row["extrusion_lat_meniscus"] = meniscus_extrusion_ratio(arr, 7, tibial_bone_lab=3)
    return row


# =====================================================================================
# 2. SURFACE - luoi be mat xuong
# =====================================================================================

def _sample(vol_f32: np.ndarray, pts_mm: np.ndarray, spacing) -> np.ndarray:
    """Trilinear tai diem theo MM (map_coordinates o INDEX space => chia spacing)."""
    pts = np.asarray(pts_mm, np.float32)
    idx = (pts / np.asarray(spacing, np.float32)).reshape(-1, 3).T
    out = map_coordinates(vol_f32, idx, order=1, mode="constant", cval=0.0)
    return out.reshape(pts.shape[:-1])


def _empty_mesh():
    return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64),
            np.zeros((0, 3), np.float32), dict(n_verts=0, flip_frac=np.nan))


def bone_mesh(bone_bin, spacing, smooth_mm: float = 0.5, eps_mm: float = 0.8):
    """Mask xuong -> (verts_mm [N,3], faces [F,3], normals_out [N,3], info).

    Marching cubes muc 0.5 tren mask lam muot Gaussian (sigma doi mm -> voxel TUNG TRUC).
    Phap tuyen duoc DINH HUONG BANG DU LIEU: lay mau the tich muot tai v +/- eps*n, phia
    nao co gia tri thap hon la phia ngoai. Khong dua vao quy uoc dau cua skimage (core.py
    da ghi nhan quy uoc do de nham). info["flip_frac"] ~0 hoac ~1 la binh thuong; ~0.5 la
    co gi do sai (cau truc mong hon 2*eps).
    """
    bone_bin, spacing = _check(bone_bin, spacing)
    vol = bone_bin.astype(np.float32)
    if smooth_mm > 0:
        vol = gaussian_filter(vol, sigma=[smooth_mm / s for s in spacing])
    if not (vol.max() > 0.5 and vol.min() < 0.5):
        return _empty_mesh()
    verts, faces, normals, _ = marching_cubes(vol, level=0.5,
                                              spacing=tuple(float(s) for s in spacing))
    n = normals.astype(np.float32)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
    vp = _sample(vol, verts + eps_mm * n, spacing)
    vm = _sample(vol, verts - eps_mm * n, spacing)
    flip = vp > vm
    n[flip] *= -1.0
    info = dict(n_verts=int(len(verts)), flip_frac=float(flip.mean()))
    return verts.astype(np.float32), faces.astype(np.int64), n, info


def vertex_areas(verts, faces) -> np.ndarray:
    """Dien tich barycentric moi dinh (mm^2): 1/3 dien tich cac tam giac ke."""
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces, np.int64)
    if len(f) == 0:
        return np.zeros(len(v))
    fa = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    areas = np.zeros(len(v))
    np.add.at(areas, f.ravel(), np.repeat(fa / 3.0, 3))
    return areas


def mesh_graph(verts, faces):
    """Do thi canh cua luoi (csr doi xung), trong so = do dai canh mm. Dung cho trac dia."""
    f = np.asarray(faces, np.int64)
    n = len(verts)
    if len(f) == 0:
        return coo_matrix((n, n)).tocsr()
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    e = e[e[:, 0] != e[:, 1]]
    e = np.unique(e, axis=0)                      # moi canh dung 1 lan (2 tam giac ke)
    w = np.linalg.norm(np.asarray(verts, np.float64)[e[:, 0]] - np.asarray(verts, np.float64)[e[:, 1]], axis=1)
    w = np.maximum(w, 1e-6)                       # csgraph coi 0 tuong minh la canh; tranh nhap nhang
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    return coo_matrix((np.concatenate([w, w]), (rows, cols)), shape=(n, n)).tocsr()


def geodesic_dilate(graph, seed, radius_mm: float) -> np.ndarray:
    """Tap dinh cach tap `seed` <= radius_mm theo duong trac dia tren luoi."""
    seed = np.asarray(seed, bool)
    idx = np.flatnonzero(seed)
    if idx.size == 0 or radius_mm <= 0:
        return seed.copy()
    d = dijkstra(graph, directed=False, indices=idx, min_only=True, limit=float(radius_mm))
    return np.isfinite(d) | seed


def geodesic_closing(graph, mask, radius_mm: float) -> np.ndarray:
    """closing = erode(dilate(mask)) tren luoi => lap lo co duong kinh < ~2*radius.

    Khac closing voxel 3D: tap 2D mong trong 3D khong bao gio duoc lap lo boi closing 3D
    (qua cau ban kinh R quanh tam lo luon thoat khoi ong dilate). Tren luoi thi dung.
    O mep luoi (mat cat FOV) khong co "ngoai" => khong bi an mon - dung y.
    """
    mask = np.asarray(mask, bool)
    dil = geodesic_dilate(graph, mask, radius_mm)
    comp = ~dil
    if not comp.any():
        return dil
    near_comp = geodesic_dilate(graph, comp, radius_mm)
    return (dil & ~near_comp) | mask


def mesh_components(graph, mask):
    """Thanh phan lien thong cua tap dinh `mask` -> (labels [N] (-1 ngoai mask), n)."""
    mask = np.asarray(mask, bool)
    idx = np.flatnonzero(mask)
    lab = np.full(len(mask), -1, np.int64)
    if idx.size == 0:
        return lab, 0
    sub = graph[idx][:, idx]
    n, l = connected_components(sub, directed=False)
    lab[idx] = l
    return lab, int(n)


def drop_small_islands(graph, mask, areas, min_area_mm2: float) -> np.ndarray:
    """Bo thanh phan lien thong co dien tich < min_area_mm2 (nhieu nhan roi rac)."""
    mask = np.asarray(mask, bool)
    if min_area_mm2 <= 0 or not mask.any():
        return mask.copy()
    lab, n = mesh_components(graph, mask)
    comp_area = np.bincount(lab[mask], weights=np.asarray(areas)[mask], minlength=n)
    out = mask.copy()
    out[mask] = (comp_area >= min_area_mm2)[lab[mask]]
    return out


def thickness_along_normals(cart_bin, verts, normals, spacing, step_mm: float = 0.1,
                            max_mm: float = 6.0, gap_mm: float = 1.0, chunk: int = 50000):
    """Do day sun tai moi dinh be mat xuong, doc phap tuyen huong ra -> (thick_mm [N], first_mm [N]).

    Tia: d = 0..max_mm buoc step_mm, sun = trilinear(mask) > 0.5 (cung phep voi
    core.occupancy_target). Do day = DOAN LIEN TUC DAU TIEN (khong phai tong moi doan):
    doan sau co the la sun cua xuong doi dien/khoang khac. Neu diem cham dau tien cach
    xuong > gap_mm => coi la KHONG co sun tai dinh do (thick = 0, first = NaN).
    """
    verts = np.asarray(verts, np.float32)
    normals = np.asarray(normals, np.float32)
    n_v = len(verts)
    depths = np.arange(0.0, max_mm + 1e-6, step_mm, dtype=np.float32)
    k = len(depths)
    cart = np.asarray(cart_bin).astype(np.float32)
    thick = np.zeros(n_v, np.float32)
    first = np.full(n_v, np.nan, np.float32)
    ar = np.arange(k)[None, :]
    for s in range(0, n_v, chunk):
        v, n = verts[s:s + chunk], normals[s:s + chunk]
        pts = v[:, None, :] + depths[None, :, None] * n[:, None, :]
        occ = _sample(cart, pts, spacing) > 0.5                       # [n, K]
        any_hit = occ.any(axis=1)
        fi = np.where(occ, ar, k).min(axis=1)                          # chi so mau dau tien co sun
        after = np.where((~occ) & (ar > fi[:, None]), ar, k).min(axis=1)  # mau KHONG sun dau tien sau fi
        run = (after - fi) * step_mm
        d_first = depths[np.minimum(fi, k - 1)]
        ok = any_hit & (d_first <= gap_mm)
        thick[s:s + chunk] = np.where(ok, run, 0.0)
        first[s:s + chunk] = np.where(ok, d_first, np.nan)
    return thick, first


def medial_lateral_axis(arr, spacing, med_lab: int = 4, lat_lab: int = 5):
    """(vector don vi TRONG->NGOAI theo mm, diem giua) tu trong tam 2 sun chay. (None, None) neu thieu."""
    arr, spacing = _check(arr, spacing)
    med, lat = arr == med_lab, arr == lat_lab
    if not med.any() or not lat.any():
        return None, None
    cm = np.array(ndi.center_of_mass(med)) * spacing
    cl = np.array(ndi.center_of_mass(lat)) * spacing
    axis = cl - cm
    nrm = np.linalg.norm(axis)
    if nrm < 1e-6:
        return None, None
    return (axis / nrm).astype(np.float32), (0.5 * (cm + cl)).astype(np.float32)


def side_of(verts, axis, mid) -> np.ndarray:
    """<0 phia TRONG (medial), >0 phia NGOAI (lateral)."""
    return (np.asarray(verts, np.float32) - mid) @ axis


def weighted_percentile(values, weights, q: float) -> float:
    v = np.asarray(values, np.float64)
    w = np.asarray(weights, np.float64)
    if v.size == 0 or w.sum() <= 0:
        return np.nan
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, q / 100.0 * cw[-1])])


def compartment_metrics(name: str, thick, areas, footprint, graph,
                        thin_edges=(0.5, 1.0), min_defect_mm2: float = 5.0) -> dict:
    """Bo metric cho MOT khoang. Moi cot mang ten khoang o giua: <metric>_<name>_<don vi>."""
    thick = np.asarray(thick, np.float32)
    areas = np.asarray(areas, np.float64)
    fp = np.asarray(footprint, bool)
    nan = np.nan
    out = {f"tab_{name}_mm2": nan, f"cab_{name}_mm2": nan, f"fcl_{name}_mm2": nan,
           f"fcl_{name}_pct": nan, f"thc_tab_{name}_mm": nan, f"thc_cab_{name}_mm": nan,
           f"thickp05_{name}_mm": nan}
    for e in thin_edges:
        out[f"thin_le{int(round(e * 10)):02d}_{name}_pct"] = nan
    out[f"fcl_{name}_ndef"] = nan
    out[f"fcl_{name}_maxdef_mm2"] = nan

    tab = areas[fp].sum()
    if tab <= 0:
        return out
    covered = fp & (thick > 0)
    fcl = fp & ~(thick > 0)
    cab = areas[covered].sum()
    dab = areas[fcl].sum()
    out[f"tab_{name}_mm2"] = float(tab)
    out[f"cab_{name}_mm2"] = float(cab)
    out[f"fcl_{name}_mm2"] = float(dab)
    out[f"fcl_{name}_pct"] = float(100.0 * dab / tab)
    out[f"thc_tab_{name}_mm"] = float((thick[fp] * areas[fp]).sum() / tab)
    out[f"thc_cab_{name}_mm"] = float((thick[covered] * areas[covered]).sum() / cab) if cab > 0 else nan
    out[f"thickp05_{name}_mm"] = weighted_percentile(thick[covered], areas[covered], 5.0) if cab > 0 else nan
    for e in thin_edges:
        thin = covered & (thick <= e)
        out[f"thin_le{int(round(e * 10)):02d}_{name}_pct"] = float(100.0 * areas[thin].sum() / tab)

    lab, n = mesh_components(graph, fcl)
    if n > 0:
        comp_area = np.bincount(lab[fcl], weights=areas[fcl], minlength=n)
        big = comp_area[comp_area >= min_defect_mm2]
        out[f"fcl_{name}_ndef"] = float(len(big))
        out[f"fcl_{name}_maxdef_mm2"] = float(big.max()) if len(big) else 0.0
    else:
        out[f"fcl_{name}_ndef"] = 0.0
        out[f"fcl_{name}_maxdef_mm2"] = 0.0
    return out


def _bone_surface_pack(arr, bone_lab, spacing, smooth_mm):
    verts, faces, normals, info = bone_mesh(arr == bone_lab, spacing, smooth_mm=smooth_mm)
    if len(verts) == 0:
        return None
    return dict(verts=verts, faces=faces, normals=normals, info=info,
                areas=vertex_areas(verts, faces), graph=mesh_graph(verts, faces))


def surface_biomarkers(arr, spacing, close_mm: float = 8.0, step_mm: float = 0.1,
                       max_mm: float = 6.0, gap_mm: float = 1.0, smooth_mm: float = 0.5,
                       min_island_mm2: float = 100.0, min_defect_mm2: float = 5.0,
                       return_surfaces: bool = False):
    """Ho biomarker be mat cho 1 ca (mask 8-class). Tra dict cot; them dict be mat neu yeu cau.

    Xuong dui (1) + sun dui (2): footprint closing RIENG tung nua trong/ngoai (chia boi mat
    phang vuong goc truc trong-ngoai, suy tu trong tam 2 sun chay) de closing khong bac cau
    qua hom lien loi cau. `fem` = hop 2 nua. Thieu nhan 4/5 => chi co `fem`.
    Xuong chay (3) + sun chay trong (4) / ngoai (5): do day rieng tung nhan, footprint rieng,
    FCL cua khoang = footprint khoang do ma KHONG co sun nao (4 hay 5).
    """
    arr, spacing = _check(arr, spacing)
    out = {}
    surfaces = {}
    axis, mid = medial_lateral_axis(arr, spacing)
    out["qc_axis_ok"] = float(axis is not None)
    kw = dict(step_mm=step_mm, max_mm=max_mm, gap_mm=gap_mm)

    # ------------------------------------------------------------------ xuong dui
    fem = _bone_surface_pack(arr, 1, spacing, smooth_mm)
    out["qc_nverts_fem"] = float(fem["info"]["n_verts"]) if fem else 0.0
    out["qc_flipfrac_fem"] = fem["info"]["flip_frac"] if fem else np.nan
    if fem:
        thick, _ = thickness_along_normals(arr == 2, fem["verts"], fem["normals"], spacing, **kw)
        covered = drop_small_islands(fem["graph"], thick > 0, fem["areas"], min_island_mm2)
        if axis is not None:
            med = side_of(fem["verts"], axis, mid) < 0
            fp_med = geodesic_closing(fem["graph"], covered & med, close_mm) & med
            fp_lat = geodesic_closing(fem["graph"], covered & ~med, close_mm) & ~med
            fp = fp_med | fp_lat
            out.update(compartment_metrics("fem_med", thick, fem["areas"], fp_med, fem["graph"],
                                           min_defect_mm2=min_defect_mm2))
            out.update(compartment_metrics("fem_lat", thick, fem["areas"], fp_lat, fem["graph"],
                                           min_defect_mm2=min_defect_mm2))
        else:
            fp = geodesic_closing(fem["graph"], covered, close_mm)
            out.update(compartment_metrics("fem_med", thick, fem["areas"], np.zeros_like(fp), fem["graph"]))
            out.update(compartment_metrics("fem_lat", thick, fem["areas"], np.zeros_like(fp), fem["graph"]))
        out.update(compartment_metrics("fem", thick, fem["areas"], fp, fem["graph"],
                                       min_defect_mm2=min_defect_mm2))
        surfaces["fem"] = dict(verts=fem["verts"], faces=fem["faces"], thick=thick,
                               footprint=fp, fcl=fp & ~(thick > 0))
    else:
        for c in ("fem", "fem_med", "fem_lat"):
            out.update(compartment_metrics(c, np.zeros(0), np.zeros(0), np.zeros(0, bool), None))

    # ----------------------------------------------------------------- xuong chay
    tib = _bone_surface_pack(arr, 3, spacing, smooth_mm)
    out["qc_nverts_tib"] = float(tib["info"]["n_verts"]) if tib else 0.0
    out["qc_flipfrac_tib"] = tib["info"]["flip_frac"] if tib else np.nan
    if tib:
        t_m, _ = thickness_along_normals(arr == 4, tib["verts"], tib["normals"], spacing, **kw)
        t_l, _ = thickness_along_normals(arr == 5, tib["verts"], tib["normals"], spacing, **kw)
        t_any = np.maximum(t_m, t_l)
        cov_m = drop_small_islands(tib["graph"], t_m > 0, tib["areas"], min_island_mm2)
        cov_l = drop_small_islands(tib["graph"], t_l > 0, tib["areas"], min_island_mm2)
        fp_m = geodesic_closing(tib["graph"], cov_m, close_mm)
        fp_l = geodesic_closing(tib["graph"], cov_l, close_mm)
        both = fp_m & fp_l
        if both.any():
            # Chong lan gan gai chay: chia theo mat phang trong/ngoai neu co, khong thi uu tien trong
            if axis is not None:
                s = side_of(tib["verts"], axis, mid)
                fp_m &= ~(both & (s > 0))
                fp_l &= ~(both & (s <= 0))
            else:
                fp_l &= ~both
        out.update(compartment_metrics("mt", t_any, tib["areas"], fp_m, tib["graph"],
                                       min_defect_mm2=min_defect_mm2))
        out.update(compartment_metrics("lt", t_any, tib["areas"], fp_l, tib["graph"],
                                       min_defect_mm2=min_defect_mm2))
        surfaces["tib"] = dict(verts=tib["verts"], faces=tib["faces"], thick=t_any,
                               footprint=fp_m | fp_l, fcl=(fp_m | fp_l) & ~(t_any > 0))
    else:
        for c in ("mt", "lt"):
            out.update(compartment_metrics(c, np.zeros(0), np.zeros(0), np.zeros(0, bool), None))

    return (out, surfaces) if return_surfaces else out


def all_biomarkers(arr, spacing, return_surfaces: bool = False, **surface_kw):
    """legacy + surface trong MOT dict (cot legacy truoc, giu thu tu S3)."""
    row = legacy_biomarkers(arr, spacing)
    if return_surfaces:
        s, surf = surface_biomarkers(arr, spacing, return_surfaces=True, **surface_kw)
        row.update(s)
        return row, surf
    row.update(surface_biomarkers(arr, spacing, **surface_kw))
    return row


def paint_vertices(verts, mask, shape, spacing) -> np.ndarray:
    """Danh dau voxel gan nhat cua cac dinh `mask` -> bool [shape]. Chi de QC/ve."""
    out = np.zeros(tuple(int(s) for s in shape), bool)
    v = np.asarray(verts, np.float32)[np.asarray(mask, bool)]
    if len(v) == 0:
        return out
    idx = np.rint(v / np.asarray(spacing, np.float32)).astype(np.int64)
    idx = np.clip(idx, 0, np.asarray(out.shape) - 1)
    out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return out
