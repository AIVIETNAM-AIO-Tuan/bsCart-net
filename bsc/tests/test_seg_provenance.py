"""Test bsc/seg_provenance.py - zip tong hop thay cho zip "Download" cua Drive.

    python -m pytest bsc/tests/test_seg_provenance.py -v
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections import Counter

import pytest

from bsc import seg_provenance as SP

P = "Dataset020_KneeUnion/nnUNetTrainer_150epochs__P__3d_fullres/"
PROV_JSON = os.path.join(os.path.dirname(__file__), "..", "splits", "d020_150ep_provenance.json")


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return path


def _model_zips(tmp_path):
    """Hai zip nhu Drive chia: ca val cua fold 0/1 o zip 1, fold 2 + checkpoint o zip 2, them trainer khac."""
    z1 = _zip(tmp_path / "D-1-001.zip", {
        P + "dataset.json": json.dumps({"numTraining": 5}),
        P + "fold_0/validation/9000001_V00_R.nii.gz": b"a",
        P + "fold_0/validation/oaizib_007.nii.gz": b"b",
        P + "fold_1/validation/9000001_V01_R.nii.gz": b"c",
        P + "fold_1/validation/summary.json": b"{}",
        "Dataset020_KneeUnion/nnUNetTrainer_250epochs__P__3d_fullres/fold_0/validation/9999999_V00_L.nii.gz": b"x"})
    z2 = _zip(tmp_path / "D-1-002.zip", {
        P + "fold_2/validation/9000003_V00_L.nii.gz": b"d",
        P + "fold_2/validation/oaizib_101.nii.gz": b"e",
        P + "fold_0/checkpoint_final.pth": b"trong-so-fold-0" * 100})
    return [z1, z2]


def test_zip_members_merges_parts_and_filters_prefix(tmp_path):
    m = SP.zip_members(_model_zips(tmp_path), P)
    assert len(m) == 8 and all(n.startswith(P) for n in m)
    assert m[P + "fold_0/checkpoint_final.pth"]["zip"].endswith("D-1-002.zip")
    ck = zipfile.ZipFile(tmp_path / "D-1-002.zip").getinfo(P + "fold_0/checkpoint_final.pth")
    assert m[P + "fold_0/checkpoint_final.pth"]["crc32"] == f"{ck.CRC:08x}"
    assert m[P + "fold_0/checkpoint_final.pth"]["bytes"] == 1500


def test_zip_members_duplicate_copy_ok_conflict_raises(tmp_path):
    zs = _model_zips(tmp_path)
    same = _zip(tmp_path / "D-1-003.zip", {P + "fold_2/validation/oaizib_101.nii.gz": b"e"})
    assert SP.zip_members(zs + [same], P) == SP.zip_members(zs, P)
    other = _zip(tmp_path / "D-1-004.zip", {P + "fold_2/validation/oaizib_101.nii.gz": b"KHAC"})
    with pytest.raises(ValueError, match="khac nhau"):
        SP.zip_members(zs + [other], P)


def test_members_md5_order_free_and_content_sensitive(tmp_path):
    zs = _model_zips(tmp_path)
    a = SP.members_md5(SP.zip_members(zs, P))
    assert a == SP.members_md5(SP.zip_members(zs[::-1], P))
    (tmp_path / "v2").mkdir()
    zs2 = _model_zips(tmp_path / "v2")
    _zip(zs2[1], {P + "fold_2/validation/9000003_V00_L.nii.gz": b"d",
                  P + "fold_2/validation/oaizib_101.nii.gz": b"e",
                  P + "fold_0/checkpoint_final.pth": b"trong-so-KHAC-0" * 100})
    assert SP.members_md5(SP.zip_members(zs2, P)) != a


def test_val_folds_roles_and_num_training(tmp_path):
    m = SP.zip_members(_model_zips(tmp_path), P)
    vf = SP.val_folds(m, P)
    assert vf == {"9000001_V00_R": 0, "oaizib_007": 0, "9000001_V01_R": 1, "9000003_V00_L": 2, "oaizib_101": 2}
    assert SP.num_training(m, P) == 5
    assert set(SP.roles(vf, [0, 1, 2]).values()) == {"train"}           # ensemble: moi ca deu da train
    r0 = SP.roles(vf, [0])
    assert {c for c, r in r0.items() if r == "val"} == {"9000001_V00_R", "oaizib_007"}
    assert r0["9000003_V00_L"] == "train"
    with pytest.raises(ValueError):
        SP.roles(vf, [])


def test_val_folds_case_in_two_folds_raises(tmp_path):
    z = _zip(tmp_path / "x.zip", {P + "fold_0/validation/9000001_V00_R.nii.gz": b"a",
                                  P + "fold_3/validation/9000001_V00_R.nii.gz": b"a"})
    with pytest.raises(ValueError, match="fold 0 va fold 3"):
        SP.val_folds(SP.zip_members([z], P), P)


def test_sha256_and_read_member(tmp_path):
    m = SP.zip_members(_model_zips(tmp_path), P)
    assert SP.sha256_member(m, P + "fold_0/checkpoint_final.pth", chunk=7) == \
        hashlib.sha256(b"trong-so-fold-0" * 100).hexdigest()
    assert json.loads(SP.read_member(m, P + "dataset.json")) == {"numTraining": 5}


def test_case_subject():
    cmt = {"007": "9100007", "101": "9100101"}
    assert SP.case_subject("9602703_V00_R", cmt) == "9602703"
    assert SP.case_subject("oaizib_007", cmt) == "9100007" and SP.case_subject("oaizib_7_0000.nii.gz", cmt) == "9100007"
    assert SP.case_subject("oaizib_555", cmt) is None and SP.case_subject("abc_V00", cmt) is None
    assert SP.case_subject("166910559012_x", cmt) is None           # barcode dai khong bi cat thanh subject


def test_pinned_d020_provenance_is_consistent():
    prov = json.load(open(PROV_JSON, encoding="utf-8"))
    vf = prov["val_fold_of_case"]
    assert len(vf) == prov["n_cases"] == prov["dataset_json_numTraining"] == 544
    assert Counter(vf.values()) == {0: 109, 1: 109, 2: 109, 3: 109, 4: 108}
    assert set(SP.roles(vf, prov["folds"]).values()) == {"train"}
    assert all(SP.case_subject(c, {}) or c.startswith("oaizib_") for c in vf)
    for k in prov["folds"]:
        for ck in ("checkpoint_final.pth", "checkpoint_best.pth"):
            rec = prov["checkpoints"][f"fold_{k}/{ck}"]
            assert len(rec["sha256"]) == 64 and len(rec["crc32"]) == 8 and rec["bytes"] > 10 ** 9
    assert len(prov["members_md5"]) == 32
    # make_splits.py da khoi phuc cung fold tu cung thu muc validation/ -> hai ban ghim phai trung nhau
    zib = json.load(open(os.path.join(os.path.dirname(PROV_JSON), "splits_zib_v1.json"), encoding="utf-8"))
    assert zib["fold_of"] == vf
    for name, rec in prov["oaizib_test_subjects_in_d020"]["cases"].items():
        assert all(vf[c] == f and SP.case_subject(c, {}) == rec["subject"] for c, f in rec["d020_cases"].items())
