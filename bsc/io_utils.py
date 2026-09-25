"""I/O + tien ich Colab. Chay TREN COLAB (co nibabel); may local thieu no.

Giu module nay mong: notebook chi nen goi vao code da test, khong chua logic.
Cac ham day KHONG co unit test vi phu thuoc nibabel + du lieu that; chung duoc
kiem gian tiep qua Gate 0 (doi chieu bang da cong bo).

TRUC & SPACING - DOC KY, DAY LA CHO DE SAI IM LANG
---------------------------------------------------
load_nii() dao truc mang (transpose 2,1,0) VA dao spacing tuong ung, nen mang va
spacing no tra ve LUON KHOP NHAU. Do la dam bao duy nhat.

NO KHONG dam bao ket qua khop `core.SPACING = (0.70, 0.3646, 0.3646)`.
Do bang du lieu that (Dataset001_KneeOA, 2026-07-19): load_nii tra ve
`(0.3646, 0.3646, 0.70)` - truc 0.70mm nam CUOI, khong phai dau. Ban docstring cu
khang dinh nguoc lai va da SAI.

=> LUON dung `spacing` do load_nii tra ve. KHONG BAO GIO dua vao mac dinh
   `spacing=SPACING` cua cac ham trong core/metrics/headroom khi lam viec voi du lieu
   that: neu spacing that lat truc, ban se ap anisotropy vao SAI TRUC. EDT, gaussian
   sigma va marching_cubes deu nhan spacing => sai lan ra toan bo hinh hoc, ma khong
   ham nao bao loi.
"""

from __future__ import annotations

import glob
import os
import posixpath
import zipfile

import numpy as np


# ------------------------------------------------------------- I/O anh y te

def load_nii(path: str):
    """Doc .nii.gz -> (array [Z,Y,X], spacing (z,y,x) mm).

    Dung nibabel (co san tren Colab). Chuyen ve thu tu truc (Z,Y,X) de khop
    SPACING va core.py. KHONG gia dinh spacing - doc tu header.
    """
    import nibabel as nib
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj)                 # (X, Y, Z) theo nib
    if arr.ndim == 4 and arr.shape[3] == 1:          # vai bo chuyen DICOM ghi (X,Y,Z,1)
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"{path}: can khoi 3D, nhan shape {arr.shape}")
    arr = np.transpose(arr, (2, 1, 0))               # -> (Z, Y, X)
    zooms = img.header.get_zooms()[:3]               # (x, y, z)
    spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    return arr, spacing


def save_nii(array, path: str, spacing=(0.70, 0.3646, 0.3646), reference: str = None):
    """Ghi mang [Z,Y,X] ra .nii.gz. Neu co `reference`, muon affine tu do."""
    import nibabel as nib
    arr = np.transpose(np.asarray(array), (2, 1, 0))  # (Z,Y,X) -> (X,Y,Z)
    if reference:
        ref = nib.load(reference)
        img = nib.Nifti1Image(arr.astype(np.float32), ref.affine, ref.header)
    else:
        aff = np.diag([spacing[2], spacing[1], spacing[0], 1.0])
        img = nib.Nifti1Image(arr.astype(np.float32), aff)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    nib.save(img, path)


# ------------------------------------------------------ tach lop (cartilage)

def class_mask(label_vol, class_name: str):
    """Tach mot lop tu the tich nhan da gop (theo core.LABELS)."""
    from .core import LABELS
    return np.asarray(label_vol) == LABELS[class_name]


# --------------------------------------------------- giai nen baseline tu Drive

def unzip_multipart(zip_glob: str, dest: str) -> int:
    """Giai nen cac zip nhieu phan (Google Takeout) vao dest.

    15 GB checkpoint dang o dang <name>-*.zip nhieu phan. Moi phan la zip doc lap
    duoc; giai tat ca vao cung dest. Tra so file da giai.
    """
    os.makedirs(dest, exist_ok=True)
    n = 0
    for zp in sorted(glob.glob(zip_glob)):
        try:
            with zipfile.ZipFile(zp) as z:
                z.extractall(dest)
                n += len(z.namelist())
        except zipfile.BadZipFile:
            print(f"  bo qua (khong doc doc lap duoc): {os.path.basename(zp)}")
    return n


def _is_colab_local(p: str) -> bool:
    return p.startswith("/content/") and not p.startswith("/content/drive/")


def assert_drive_first(path):
    """Bat loi ghi SAN PHAM vao /content/ RAM Colab. Tra lai path neu hop le.

    Xem track._assert_drive_first - lap lai o day de notebook goi truc tiep.
    Kiem ca abspath (duong tuong doi tren Colab nam trong /content) LAN chuoi POSIX da chuan
    hoa (de test chay duoc tren Windows, noi abspath('/content/x') thanh 'C:\\content\\x').
    /content/input_cache (bo dem DOC du lieu dau vao, ngoai le duoc duyet 24/09/2026) KHONG
    duoc dung cho san pham - no cung bi chan o day.
    """
    s = os.fspath(path)
    raw = posixpath.normpath(s.replace("\\", "/"))
    if _is_colab_local(os.path.abspath(s)) or _is_colab_local(raw):
        raise ValueError(
            f"'{s}' nam trong /content/ RAM Colab - MAT khi ngat phien. "
            f"Ghi vao /content/drive/MyDrive/bsc/. Day la loi da lam mat 544 prediction."
        )
    return path


def resolve_path(p, remaps=(), must_exist: bool = True):
    """Tim duong ton tai cho `p`: chinh no, roi thay tien to theo `remaps` [(cu, moi), ...].

    Dung cho `dess_path` cua manifest da cu (vd nnUNet_raw chuyen vao MyDrive/OAI_seg/).
    Thu theo thu tu; tien to phai khop NGUYEN thanh phan duong (ket thuc bang '/').
    must_exist=False => tra None khi khong thay (de audit dem); mac dinh => FileNotFoundError.
    """
    s = os.fspath(p)
    tried = [s]
    if os.path.exists(s):
        return s
    for old, new in remaps:
        old_d = old.rstrip("/") + "/"
        if s.startswith(old_d):
            cand = new.rstrip("/") + "/" + s[len(old_d):]
            tried.append(cand)
            if os.path.exists(cand):
                return cand
    if must_exist:
        raise FileNotFoundError(f"khong thay {s}; da thu {tried}")
    return None


def list_cases(images_dir: str, suffix: str = "_0000.nii.gz") -> list:
    """Liet ke case_id tu thu muc imagesTr/imagesTs (bo hau to kenh)."""
    return sorted(os.path.basename(p)[: -len(suffix)]
                  for p in glob.glob(os.path.join(images_dir, f"*{suffix}")))
