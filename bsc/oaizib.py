"""OAIZIB-CM (HuggingFace `YongchengYAO/OAIZIB-CM`): metadata + tai anh/nhan TRUC TIEP len Drive.

Anh OAI-ZIB cua cohort truoc day chi duoc tai vao /content (mat khi het phien) nen khong con tren
Drive (kiem 27/09/2026). Dataset cong khai (CC-BY-NC-4.0; bat buoc trich CartiMorph,
doi:10.1016/j.media.2023.103035) nen tai lai duoc bat cu luc nao, TUNG FILE trong zip (doc tu xa
qua HfFileSystem), khong phai tai ca goi imagesTr.zip 10 GB / imagesTs.zip 2,5 GB.

`info/subInfo_{train,test}.xlsx`: SubjectID, CMT-ID (so trong ten oaizib_XXX), Path OAI
("0.C.2/<subject>/<ngay>/<series>" - tien to 0 = baseline V00), MRBarCode (8 so cuoi = so series =
barcode cua npz M3T), KneeSide (1 = phai, 2 = trai), KLGrade.
Doi chieu 27/09/2026 voi zip M3T: 507/507 khop barcode, subject 100%, ca 507 goi phai, ca 507 baseline.
"""

from __future__ import annotations

import os
import re
import zipfile

from .io_utils import assert_drive_first

REPO_ID = "YongchengYAO/OAIZIB-CM"
_CMT_RE = re.compile(r"oaizib_(\d+)")
_SIDE = {1: "R", 2: "L"}


def cmt_id(path):
    """'.../oaizib_488_0000.nii.gz' -> '488'; khong co -> None."""
    m = _CMT_RE.search(str(path))
    return f"{int(m.group(1)):03d}" if m else None


def visit_of(oai_path) -> str:
    """Thu muc OAI '0.C.2/...' -> 'V00', '3.C.2/...' -> 'V03' (so dau = ma lan kham)."""
    head = str(oai_path).split("/")[0]
    if not re.fullmatch(r"\d\.[A-Z]\.\d", head):
        raise ValueError(f"khong nhan ra thu muc lan kham OAI: {oai_path!r}")
    return f"V{int(head[0]):02d}"


def _info_file(info_dir, name) -> str:
    for p in (os.path.join(info_dir, name), os.path.join(info_dir, "info", name)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"khong thay {name} trong {info_dir} (hay {info_dir}/info)")


def load_subinfo(info_dir):
    """subInfo_train + subInfo_test -> cmt_id, subject, barcode (8 so), side, visit, kl, split.

    Kiem: 8 so cuoi MRBarCode == so series cuoi Path; cmt_id va barcode duy nhat; side 1/2.
    """
    import pandas as pd
    from .m3t_train import norm_subject
    frames = []
    for split in ("train", "test"):
        d = pd.read_excel(_info_file(info_dir, f"subInfo_{split}.xlsx"))
        need = {"SubjectID", "CMT-ID", "Path", "MRBarCode", "KneeSide", "KLGrade"}
        if need - set(d.columns):
            raise ValueError(f"subInfo_{split}.xlsx thieu cot {sorted(need - set(d.columns))}")
        path = d["Path"].astype(str)
        frames.append(pd.DataFrame(dict(
            cmt_id=d["CMT-ID"].astype(int).map(lambda x: f"{x:03d}"),
            subject=d["SubjectID"].map(norm_subject),
            barcode=d["MRBarCode"].astype("int64").astype(str).str[-8:],
            path_barcode=path.str.split("/").str[-1],
            side=d["KneeSide"].map(_SIDE),
            visit=path.map(visit_of),
            kl=d["KLGrade"],
            split=split)))
    s = pd.concat(frames, ignore_index=True)
    if (s["barcode"] != s["path_barcode"]).any():
        raise ValueError("MRBarCode khong khop so series trong Path")
    if s["side"].isna().any():
        raise ValueError("KneeSide ngoai {1, 2}")
    for col in ("cmt_id", "barcode"):
        if s[col].duplicated().any():
            raise ValueError(f"{col} trung trong subInfo")
    return s.drop(columns="path_barcode")


def _remote_zip(zip_name, repo_id=REPO_ID):
    from huggingface_hub import HfFileSystem
    return HfFileSystem().open(f"datasets/{repo_id}/{zip_name}", "rb")


def fetch_members(zip_name, members, dest_dir, opener=None, log=print) -> list:
    """Tai TUNG member cua mot zip trong dataset ve dest_dir/<member> (tren Drive).

    File da co -> bo qua (khong ghi de, khong mo zip tu xa neu tat ca da co). Ghi .part roi doi ten nen
    ton tai = da tai du; zipfile kiem CRC khi doc het member. `opener(zip_name)` -> file-like doc duoc (test truyen zip cuc bo).
    """
    dest_dir = os.fspath(dest_dir)
    assert_drive_first(dest_dir)
    out = [os.path.join(dest_dir, *m.split("/")) for m in members]
    todo = [(m, t) for m, t in zip(members, out) if not os.path.exists(t)]
    if not todo:                               # file ghi qua .part -> ton tai = da tai du; khong can mang
        log(f"{zip_name}: {len(members)} file da co san trong {dest_dir}")
        return out
    n_new = 0
    with (opener or _remote_zip)(zip_name) as f, zipfile.ZipFile(f) as z:
        infos = {i.filename: i for i in z.infolist()}
        missing = [m for m, _ in todo if m not in infos]
        if missing:
            raise KeyError(f"{zip_name} khong co {missing[:3]} (tong {len(missing)})")
        for m, target in todo:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(m) as src, open(target + ".part", "wb") as dst:
                while True:
                    b = src.read(1 << 22)
                    if not b:
                        break
                    dst.write(b)
            os.replace(target + ".part", target)
            n_new += 1
    log(f"{zip_name}: {len(members)} file, tai moi {n_new}, da co {len(members) - n_new} -> {dest_dir}")
    return out


def fetch_info(dest_dir, opener=None, log=print) -> str:
    """Tai info.zip (subInfo, kneeSideInfo, README) ve dest_dir/info; tra duong thu muc info."""
    names = ["info/subInfo_train.xlsx", "info/subInfo_test.xlsx", "info/kneeSideInfo.csv", "info/README.md"]
    fetch_members("info.zip", names, dest_dir, opener=opener, log=log)
    return os.path.join(os.fspath(dest_dir), "info")


def image_members(dess_paths) -> dict:
    """{zip: [member]} cho cac duong anh '.../imagesTs/oaizib_488_0000.nii.gz' cua manifest."""
    out = {}
    for p in dess_paths:
        parts = str(p).replace("\\", "/").split("/")
        folder, name = parts[-2], parts[-1]
        if folder not in ("imagesTr", "imagesTs") or cmt_id(name) is None:
            raise ValueError(f"khong phai anh OAIZIB-CM: {p}")
        out.setdefault(f"{folder}.zip", []).append(f"{folder}/{name}")
    return out
