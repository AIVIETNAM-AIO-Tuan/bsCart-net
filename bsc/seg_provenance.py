"""Provenance mo hinh segmentation nnU-Net doc THANG tu cac zip "Download" cua Google Drive (khong giai nen).

Drive tai mot thu muc lon ve thanh nhieu zip `<ten>-<thoi diem>-1-00k.zip`; moi file nam tron trong DUNG MOT zip.
Chi doc central directory: ten ca val cua tung fold (thu muc `fold_k/validation/`), CRC32 + kich thuoc tung file
(dau van tay ca bo trong so, khong phai doc 1,1 GB moi checkpoint). sha256 doc stream khi can.

Ten ca -> subject: so OAI 7 chu so (`9602703_V00_R`) hoac `oaizib_XXX` qua subInfo chinh thuc
(`oaizib.load_subinfo`). Ban provenance cua d020 150 epoch da ghim o `splits/d020_150ep_provenance.json`.
Phan khoi phuc fold trung y tuong voi `make_splits.py` (da ghim `splits/splits_zib_v1.json`, test kiem hai ban trung
nhau); module nay them gop nhieu zip, CRC32/sha256 checkpoint va map subject de audit phoi nhiem.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile

_SUBJ_RE = re.compile(r"(?<!\d)(9\d{6})(?!\d)")
_OAIZIB_RE = re.compile(r"oaizib_(\d+)")
_VAL_RE = re.compile(r"fold_(\d+)/validation/([^/]+)\.nii\.gz$")


def zip_members(zip_paths, prefix) -> dict:
    """{member bat dau bang prefix: {zip, crc32, bytes}} gop tu nhieu zip.

    Cung member o hai zip: CRC + kich thuoc trung thi coi la ban sao, khac thi ValueError.
    """
    out = {}
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as z:
            for i in z.infolist():
                if i.is_dir() or not i.filename.startswith(prefix):
                    continue
                rec = dict(zip=os.fspath(zp), crc32=f"{i.CRC:08x}", bytes=i.file_size)
                old = out.setdefault(i.filename, rec)
                if (old["crc32"], old["bytes"]) != (rec["crc32"], rec["bytes"]):
                    raise ValueError(f"{i.filename} khac nhau giua {old['zip']} va {zp}")
    return out


def members_md5(members) -> str:
    """md5 cua cac dong 'ten,crc32,bytes' sap theo ten - dau van tay ca bo file (vd. cua mot trainer_dir)."""
    rows = [f"{n},{members[n]['crc32']},{members[n]['bytes']}" for n in sorted(members)]
    return hashlib.md5("\n".join(rows).encode()).hexdigest()


def read_member(members, name) -> bytes:
    """Noi dung mot member nho (json, log); zipfile tu kiem CRC khi doc het."""
    with zipfile.ZipFile(members[name]["zip"]) as z:
        return z.read(name)


def sha256_member(members, name, chunk=1 << 24) -> str:
    """sha256 mot member doc stream (checkpoint ~1,1 GB), khong giai nen ra dia."""
    h = hashlib.sha256()
    with zipfile.ZipFile(members[name]["zip"]) as z, z.open(name) as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def num_training(members, prefix) -> int:
    """numTraining trong `<prefix>dataset.json` (so ca cua dataset train mo hinh)."""
    return int(json.loads(read_member(members, f"{prefix}dataset.json"))["numTraining"])


def val_folds(members, prefix) -> dict:
    """{ca: fold} tu `<prefix>fold_k/validation/<ca>.nii.gz`; mot ca o hai fold -> ValueError."""
    out = {}
    for name in members:
        if not name.startswith(prefix):
            continue
        m = _VAL_RE.fullmatch(name[len(prefix):])
        if m is None:
            continue
        fold, case = int(m.group(1)), m.group(2)
        if out.setdefault(case, fold) != fold:
            raise ValueError(f"ca {case} la val cua fold {out[case]} va fold {fold}")
    return out


def roles(val_fold, folds_used) -> dict:
    """{ca: 'train'|'val'} cho mo hinh ensemble cac fold trong folds_used.

    'train' khi IT NHAT mot fold duoc dung da train tren ca (ca la val cua fold khac); 'val' khi ca chi la val
    cua fold duy nhat duoc dung. Ensemble du 5 fold -> moi ca deu 'train'.
    """
    used = set(folds_used)
    if not used:
        raise ValueError("folds_used rong")
    return {c: "train" if used - {f} else "val" for c, f in val_fold.items()}


def case_subject(name, cmt2subj):
    """Ten ca -> subject OAI (chuoi 7 so): so 9xxxxxx trong ten, hoac oaizib_XXX qua {cmt_id: subject}; khong ra -> None."""
    m = _SUBJ_RE.search(str(name))
    if m:
        return m.group(1)
    m = _OAIZIB_RE.search(str(name))
    return cmt2subj.get(f"{int(m.group(1)):03d}") if m else None
