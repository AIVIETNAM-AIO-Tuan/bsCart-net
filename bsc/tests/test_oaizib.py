"""Test bsc/oaizib.py - khong goi mang: zip cuc bo thay cho HuggingFace.

    python -m pytest bsc/tests/test_oaizib.py -v
"""

from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from bsc import m3t_train as T
from bsc import oaizib as OZ


def test_cmt_id_and_visit():
    assert OZ.cmt_id("/x/Dataset001_KneeOA/imagesTs/oaizib_488_0000.nii.gz") == "488"
    assert OZ.cmt_id("oaizib_5.nii.gz") == "005" and OZ.cmt_id("/x/9000099_DESS_right.nii.gz") is None
    assert OZ.visit_of("0.E.1/9005075/20050926/10593811") == "V00"
    assert OZ.visit_of("3.C.2/9005075/20070101/10000001") == "V03"
    with pytest.raises(ValueError):
        OZ.visit_of("abc/9005075")


def _subinfo(tmp_path, rows_train, rows_test):
    cols = ["SubjectID", "CMT-ID", "Path", "MRBarCode", "KneeSide", "KLGrade", "Gender", "Age", "BMI"]
    (tmp_path / "info").mkdir(exist_ok=True)
    pd.DataFrame(rows_train, columns=cols).to_excel(tmp_path / "info" / "subInfo_train.xlsx", index=False)
    pd.DataFrame(rows_test, columns=cols).to_excel(tmp_path / "info" / "subInfo_test.xlsx", index=False)


def test_load_subinfo_parses_barcode_side_visit(tmp_path):
    _subinfo(tmp_path,
             [[9005075, 1, "0.E.1/9005075/20050926/10593811", 16610593811, 1, 0.0, 1, 47, 39.2]],
             [[9545822, 405, "0.C.2/9545822/20050614/10559012", 16610559012, 1, 2.0, 2, 72, 28.6]])
    s = OZ.load_subinfo(tmp_path)
    assert s.cmt_id.tolist() == ["001", "405"] and s.barcode.tolist() == ["10593811", "10559012"]
    assert s.subject.tolist() == ["9005075", "9545822"] and set(s.side) == {"R"} and set(s.visit) == {"V00"}
    assert s.split.tolist() == ["train", "test"]
    _subinfo(tmp_path,
             [[9005075, 1, "0.E.1/9005075/20050926/10593811", 16610593899, 1, 0.0, 1, 47, 39.2]], [])
    with pytest.raises(ValueError, match="MRBarCode"):
        OZ.load_subinfo(tmp_path)


def _local_zip(tmp_path, members):
    zp = tmp_path / "remote.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return lambda zip_name: open(zp, "rb")


def test_fetch_members_writes_skips_existing_and_rejects_missing(tmp_path):
    opener = _local_zip(tmp_path, {"imagesTs/": b"", "imagesTs/oaizib_488_0000.nii.gz": b"abc",
                                   "imagesTs/oaizib_489_0000.nii.gz": b"defg"})
    dest = tmp_path / "drive" / "OAIZIB-CM"
    msgs = []
    out = OZ.fetch_members("imagesTs.zip", ["imagesTs/oaizib_488_0000.nii.gz"], dest, opener=opener,
                           log=msgs.append)
    assert open(out[0], "rb").read() == b"abc" and "tai moi 1" in msgs[-1]
    def no_network(zip_name):
        raise AssertionError("khong duoc mo zip tu xa khi moi file da co")
    OZ.fetch_members("imagesTs.zip", ["imagesTs/oaizib_488_0000.nii.gz"], dest, opener=no_network, log=msgs.append)
    assert "da co san" in msgs[-1]                                        # da co -> khong ghi lai, khong mang
    out2 = OZ.fetch_members("imagesTs.zip", ["imagesTs/oaizib_488_0000.nii.gz", "imagesTs/oaizib_489_0000.nii.gz"],
                            dest, opener=opener, log=msgs.append)
    assert "tai moi 1" in msgs[-1] and open(out2[1], "rb").read() == b"defg"
    with pytest.raises(KeyError, match="khong co"):
        OZ.fetch_members("imagesTs.zip", ["imagesTs/oaizib_999_0000.nii.gz"], dest, opener=opener)
    with pytest.raises(ValueError, match="RAM Colab"):
        OZ.fetch_members("imagesTs.zip", [], "/content/input_cache/x", opener=opener)


def test_image_members_groups_by_zip():
    got = OZ.image_members(["/c/nnUNet_raw/Dataset001_KneeOA/imagesTs/oaizib_488_0000.nii.gz",
                            "/c/nnUNet_raw/Dataset001_KneeOA/imagesTr/oaizib_215_0000.nii.gz"])
    assert got == {"imagesTs.zip": ["imagesTs/oaizib_488_0000.nii.gz"],
                   "imagesTr.zip": ["imagesTr/oaizib_215_0000.nii.gz"]}
    with pytest.raises(ValueError):
        OZ.image_members(["/c/OAI_DESS_Right_NIfTI/9000099/9000099_DESS_right.nii.gz"])


def test_exposure_table_uses_mr_barcode_before_dess_path():
    idx = pd.DataFrame(dict(npz_name=["9005075_10593811_RIGHT.npz", "9005075_10593805_LEFT.npz"],
                            subject=["9005075", "9005075"], side=["R", "L"], barcode=["10593811", "10593805"],
                            member=["a", "b"]))
    lab = T.load_labels(pd.DataFrame(dict(id=[9005075, 9005075], side=["RIGHT", "LEFT"],
                                          mri_path=["9005075_10593811_RIGHT.npz", "9005075_10593805_LEFT.npz"],
                                          kl_grade=[0, 1], xray_path=["x", "x"], subset=["train", "train"])))
    man = pd.DataFrame([dict(case_id="9005075_V00_R", subject=9005075, side="R", visit="V00",
                             dess_path="/c/Dataset001_KneeOA/imagesTr/oaizib_001_0000.nii.gz", mr_barcode="10593811"),
                        dict(case_id="k2", subject=9005075, side="R", visit="V00",
                             dess_path="/c/Dataset001_KneeOA/imagesTr/oaizib_001_0000.nii.gz", mr_barcode=float("nan"))])
    ex = T.exposure_table(man, lab, idx).set_index("case_id")
    assert ex.loc["9005075_V00_R", "match"] == "barcode"
    assert ex.loc["9005075_V00_R", "npz_name"] == "9005075_10593811_RIGHT.npz"
    assert not ex.loc["9005075_V00_R", "side_conflict"]
    assert ex.loc["k2", "match"] == "subject_side"                         # NaN -> van theo subject+ben
