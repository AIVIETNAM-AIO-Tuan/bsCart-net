"""Test bsc/m3t_train.py - CPU, du lieu tong hop, cau hinh M3T thu nho.

    python -m pytest bsc/tests/test_m3t_train.py -v
"""

from __future__ import annotations

import io
import json
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bsc import io_utils, m3t
from bsc import m3t_train as T
from bsc import ordinal as ORD

TINY = dict(C3d=4, N=4, emb_dim=16, C2d=8, target_size=12)
SHAPE = (6, 8, 8)
REPO = Path(__file__).resolve().parents[1]


def cfg(**kw):
    base = dict(seed=7, batch_size=2, lr=1e-3, max_epochs=3, min_epochs=2, patience=5, min_delta=0.005,
                window=2, num_workers=0, allow_tf32=False, cudnn_benchmark=False, augment="none",
                model=TINY)
    return T.M3TTrainConfig(**{**base, **kw})


def _npz_bytes(arr):
    b = io.BytesIO()
    np.savez_compressed(b, data=arr)
    return b.getvalue()


def make_zip(path, knees, shape=SHAPE, seed=0, extra=()):
    """knees: [(subject, barcode, 'LEFT'|'RIGHT')]. Moi khoi khac nhau, uint16 nhu that."""
    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("SAG_3D_DESS_v2_full/", "")
        zf.writestr("SAG_3D_DESS_v2_full/label.csv", "id,side\n")
        for s, b, side in knees:
            zf.writestr(f"SAG_3D_DESS_v2_full/MRI_Numpy/{s}_{b}_{side}.npz",
                        _npz_bytes(rng.integers(0, 500, size=shape).astype(np.uint16)))
        for name, data in extra:
            zf.writestr(name, data)
    return path


def labels_frame(rows):
    return pd.DataFrame(rows, columns=["id", "side", "mri_path", "kl_grade", "xray_path", "subset"])


# ------------------------------------------------------------ 6. zip + reader

def test_npz_index_reader_and_pickle_before_first_read(tmp_path):
    zp = make_zip(tmp_path / "z.zip", [("9000001", "10000001", "LEFT"), ("9000001", "10000002", "RIGHT"),
                                       ("9000002", "10000003", "RIGHT")])
    idx = T.npz_index(zp)
    assert idx["npz_name"].tolist() == ["9000001_10000001_LEFT.npz", "9000001_10000002_RIGHT.npz",
                                        "9000002_10000003_RIGHT.npz"]
    assert idx["side"].tolist() == ["L", "R", "R"] and idx["subject"].tolist()[0] == "9000001"
    assert len(T.index_md5(idx)) == 32
    reader = T.ZipNpzReader(zp)
    fresh = pickle.loads(pickle.dumps(reader))                  # pickle TRUOC lan doc dau
    a = fresh.read(idx.member[0])
    assert a.shape == SHAPE and a.dtype == np.uint16
    b = reader.read(idx.member[0])
    assert np.array_equal(a, b)
    again = pickle.loads(pickle.dumps(reader))                  # SAU khi da mo handle
    assert again._zf is None and np.array_equal(again.read(idx.member[0]), a)


def test_npz_index_rejects_unexpected_npz_names(tmp_path):
    zp = make_zip(tmp_path / "z.zip", [("9000001", "10000001", "LEFT")],
                  extra=[("SAG_3D_DESS_v2_full/MRI_Numpy/oops.npz", _npz_bytes(np.zeros(SHAPE)))])
    with pytest.raises(ValueError, match="khong dung mau"):
        T.npz_index(zp)


def test_stage_input_copies_verifies_crc_and_guards_the_cache_root(tmp_path):
    src = make_zip(tmp_path / "src.zip", [(f"90000{i:02d}", f"100000{i:02d}", "LEFT") for i in range(12)])
    cache = tmp_path / "cache"
    dst = T.stage_input(src, cache_root=cache, n_check=5, min_free_gb=0.0, log=lambda *_: None)
    assert os.path.getsize(dst) == os.path.getsize(src)
    assert T.stage_input(src, cache_root=cache, n_check=5, min_free_gb=0.0, log=lambda *_: None) == dst
    with pytest.raises(ValueError, match="byte"):
        T.stage_input(src, cache_root=cache, expected_size=1, min_free_gb=0.0)
    with pytest.raises(ValueError, match="input_cache"):
        T.stage_input(src, cache_root="/content/other_dir")
    with pytest.raises(ValueError, match="input_cache"):
        T.stage_input(src, cache_root="/content")
    # hong du lieu mot npz => CRC bat duoc
    raw = bytearray(Path(dst).read_bytes())
    with zipfile.ZipFile(dst) as zf:
        info = [i for i in zf.infolist() if i.filename.endswith(".npz")][3]
    off = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra) + 10
    raw[off] ^= 0xFF
    bad = tmp_path / "bad" / "src.zip"
    bad.parent.mkdir()
    bad.write_bytes(bytes(raw))
    with pytest.raises(Exception):
        T.verify_zip_members(bad, n_check=12)


# ------------------------------------------------------------ 7. nhan + pool

def _labels():
    rows = []
    for i, subset in enumerate(["train"] * 6 + ["val"] * 2 + ["test"] * 2):
        s = f"90000{i:02d}"
        rows.append((int(s), "RIGHT", f"{s}_1000{i:02d}12_RIGHT.npz", i % 5, "x", subset))
        rows.append((int(s), "LEFT", f"{s}_1000{i:02d}05_LEFT.npz", (i + 1) % 5, "x", subset))
    return T.load_labels(labels_frame(rows))


def test_load_labels_normalizes_and_validates():
    lab = _labels()
    assert lab["subject"].iloc[0] == "9000000" and set(lab["side"]) == {"L", "R"}
    bad = labels_frame([(9000001, "LEFT", "9000001_10000105_LEFT.npz", 1, "x", "train"),
                        (9000001, "RIGHT", "9000001_10000112_RIGHT.npz", 1, "x", "test")])
    with pytest.raises(ValueError, match="nhieu subset"):
        T.load_labels(bad)
    wrong = labels_frame([(9000001, "LEFT", "9000002_10000105_LEFT.npz", 1, "x", "train")])
    with pytest.raises(ValueError, match="khong khop"):
        T.load_labels(wrong)


def test_build_pool_drops_both_knees_in_every_subset_and_normalizes_ids():
    lab = _labels()
    excl = [9000000, "9000006.0", " 9000009 "]                   # train, val, test; int/float/khoang trang
    pool = T.build_pool(lab, excl)
    assert len(pool) == len(lab) - 6
    assert not ({"9000000", "9000006", "9000009"} & set(pool["subject"]))
    T.assert_subject_disjoint(pool, excl, "pool vs cohort")
    with pytest.raises(AssertionError, match="subject chung"):
        T.assert_subject_disjoint(lab, excl, "lab")
    with pytest.raises(ValueError, match="7 chu so"):
        T.build_pool(lab, ["oaizib_001"])
    assert T.pool_md5(pool) != T.pool_md5(lab) and T.pool_md5(pool) == T.pool_md5(pool.iloc[::-1])


# ------------------------------------------------------------ 8. phoi nhiem

def test_exposure_table_barcode_first_then_subject_side_with_flags():
    lab = _labels()
    idx = pd.DataFrame(dict(npz_name=lab["npz_name"], subject=lab["subject"], side=lab["side"]))
    idx["barcode"] = idx["npz_name"].str.split("_").str[1]
    idx["member"] = "SAG/" + idx["npz_name"]
    extra = dict(npz_name="9000003_20000005_LEFT.npz", subject="9000003", side="L", barcode="20000005",
                 member="SAG/x")                                  # goi L cua 9000003 co 2 lan chup
    idx = pd.concat([idx, pd.DataFrame([extra])], ignore_index=True)
    man = pd.DataFrame([
        dict(case_id="a", subject="9000001", side="R", visit="V00", dess_path="/d/OAI_DESS/10000112.nii.gz"),
        dict(case_id="b", subject="9000002", side="L", visit="V00", dess_path="/d/nnUNet_raw/9000002_V00_L_0000.nii.gz"),
        dict(case_id="c", subject="9000003", side="L", visit="V01", dess_path="/d/x/9000003_V01_L.nii.gz"),
        dict(case_id="d", subject="9000004", side="L", visit="V00", dess_path="/d/20040909/10000412.nii.gz"),
        dict(case_id="e", subject="9999999", side="R", visit="V00", dess_path="/d/oaizib_001_0000.nii.gz"),
        dict(case_id="f", subject="oaizib", side="R", visit="V00", dess_path=""),
    ])
    ex = T.exposure_table(man, lab, idx).set_index("case_id")
    assert ex.loc["a", "match"] == "barcode" and ex.loc["a", "npz_name"] == "9000001_10000112_RIGHT.npz"
    assert not ex.loc["a", "side_conflict"] and ex.loc["a", "m3t_subset"] == "train"
    assert ex.loc["b", "match"] == "subject_side" and not ex.loc["b", "ambiguous"]
    assert ex.loc["c", "match"] == "subject_side" and ex.loc["c", "ambiguous"]
    assert ex.loc["c", "npz_name"] == "9000003_10000305_LEFT.npz"          # ban co nhan trong CSV
    # ngay 20040909 KHONG phai barcode; 10000412 la barcode goi PHAI => ben trong manifest sai
    assert ex.loc["d", "match"] == "barcode" and ex.loc["d", "side_conflict"]
    assert ex.loc["e", "match"] == "none" and ex.loc["e", "m3t_subset"] == "absent"
    assert ex.loc["f", "m3t_subset"] == "invalid_subject"
    assert ex.loc["a", "knee_in_csv"] and ex.loc["a", "m3t_kl"] == 1


# ------------------------------------------------------------ 14. chon epoch + dung som

def _hist(q):
    return [dict(epoch=i, val_qwk=v) for i, v in enumerate(q)]


def test_select_epoch_prefers_plateau_over_spike_and_ignores_downstream_keys():
    q = [0.5, 0.52, 0.9, 0.55, 0.56, 0.74, 0.75, 0.75, 0.76, 0.75, 0.6]
    sel = T.select_epoch(_hist(q), window=5)
    assert sel["epoch"] == 9 and sel["window"] == [5, 6, 7, 8, 9]
    assert T.select_epoch(_hist(q[:5]), window=5)["epoch"] == 4
    noisy = [dict(h, s7_qwk=np.random.default_rng(i).random(), s8=1.0) for i, h in enumerate(_hist(q))]
    assert T.select_epoch(noisy, window=5) == sel                       # diem downstream KHONG anh huong
    tie = T.select_epoch(_hist([0.6, 0.7, 0.6, 0.7, 0.6, 0.7]), window=2)
    assert tie["epoch"] == 1                                            # hoa => t nho nhat
    with pytest.raises(ValueError):
        T.select_epoch(_hist([0.5, 0.6]), window=5)
    with pytest.raises(ValueError, match="lien tuc"):
        T.select_epoch([dict(epoch=0, val_qwk=0.5), dict(epoch=2, val_qwk=0.6)], window=1)


def test_should_stop_rules():
    c = cfg(max_epochs=100, min_epochs=10, patience=5, min_delta=0.005, window=1)
    assert T.should_stop(_hist([0.5] * 9), c) == (False, "")             # chua du min_epochs
    stop, why = T.should_stop(_hist([0.5] * 10), c)
    assert stop and "khong vuot" in why                                  # 9 epoch khong cai thien
    rising = list(np.linspace(0.5, 0.6, 12))                             # ~0.009/epoch > min_delta
    assert not T.should_stop(_hist(rising), c)[0]
    slow = [0.5 + 0.004 * i for i in range(12)]                          # tich luy vuot min_delta sau 2 epoch
    assert not T.should_stop(_hist(slow), c)[0]
    assert T.should_stop(_hist([0.5 + 0.01 * i for i in range(100)]), c)[0]   # du max_epochs


# ------------------------------------------------------------ 15. fit + chay tiep bit-doi-bit

class _MemReader:
    def __init__(self, vols):
        self.vols = vols

    def read(self, member):
        return self.vols[member]


def _aug(x):
    """Tang cuong gia dung RNG cua torch (nhu torchio) - de thay seed co phu tang cuong khong."""
    return (x.astype(np.float32) * (1.0 + 0.2 * torch.rand(1).item())) / 500.0


def _data(n=8, seed=0):
    rng = np.random.default_rng(seed)
    vols = {f"m{i}": rng.integers(0, 500, SHAPE).astype(np.uint16) for i in range(n)}
    df = pd.DataFrame(dict(member=list(vols), kl_grade=[i % 5 for i in range(n)]))
    reader = _MemReader(vols)
    return (T.NpzKLDataset(df, reader, transform=_aug, expect_shape=SHAPE),
            T.NpzKLDataset(df.iloc[:4], reader, transform=None, expect_shape=SHAPE))


def _fresh(seed=0):
    torch.manual_seed(seed)
    return m3t.build_m3t(**TINY)


def test_two_epochs_equal_one_epoch_plus_resume_bit_for_bit(tmp_path):
    tr, va = _data()
    c = cfg(max_epochs=2, min_epochs=2)
    log = lambda *_: None
    h_full = T.fit(_fresh(), tr, va, tmp_path / "full", c, "pool", device="cpu", log=log)
    T.fit(_fresh(), tr, va, tmp_path / "split", c, "pool", device="cpu", max_new_epochs=1, log=log)
    assert len(torch.load(tmp_path / "split" / "last.pt", weights_only=True)["history"]) == 1
    h_split = T.fit(_fresh(99), tr, va, tmp_path / "split", c, "pool", device="cpu", log=log)
    strip = lambda h: [{k: v for k, v in r.items() if k != "seconds"} for r in h]
    assert strip(h_full) == strip(h_split)
    a = torch.load(tmp_path / "full" / "epoch_001.pt", weights_only=True)
    b = torch.load(tmp_path / "split" / "epoch_001.pt", weights_only=True)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert (tmp_path / "split" / "last_prev.pt").exists()
    # khac seed => khac (seed thuc su dieu khien tang cuong/tron)
    h_other = T.fit(_fresh(), tr, va, tmp_path / "other", cfg(max_epochs=2, min_epochs=2, seed=8), "pool",
                    device="cpu", log=log)
    assert strip(h_other) != strip(h_full)


def test_resume_refuses_changed_config_or_pool(tmp_path):
    tr, va = _data()
    log = lambda *_: None
    T.fit(_fresh(), tr, va, tmp_path / "r", cfg(), "pool", device="cpu", max_new_epochs=1, log=log)
    with pytest.raises(ValueError, match="cfg_hash"):
        T.fit(_fresh(), tr, va, tmp_path / "r", cfg(lr=2e-3), "pool", device="cpu", log=log)
    with pytest.raises(ValueError, match="pool_md5"):
        T.fit(_fresh(), tr, va, tmp_path / "r", cfg(), "pool_khac", device="cpu", log=log)
    with pytest.raises(FileExistsError):                                 # run_config.json ghi mot lan
        T.write_once_json(dict(khac=1), tmp_path / "r" / "run_config.json")


def test_stop_file_and_checkpoint_fallback(tmp_path):
    tr, va = _data()
    log = lambda *_: None
    run = tmp_path / "s"
    T.fit(_fresh(), tr, va, run, cfg(max_epochs=3, min_epochs=3), "pool", device="cpu", max_new_epochs=2, log=log)
    (run / "STOP").write_text("")
    h = T.fit(_fresh(), tr, va, run, cfg(max_epochs=3, min_epochs=3), "pool", device="cpu", log=log)
    assert len(h) == 2
    (run / "last.pt").write_bytes(b"hong")                               # mat phien giua luc ghi
    msgs = []
    state, name = T.load_checkpoint(run, log=msgs.append)
    assert name == "last_prev.pt" and state["epoch"] == 0 and "CANH BAO" in msgs[0]
    (run / "last_prev.pt").write_bytes(b"hong")                          # ca hai hong => KHONG train lai
    (run / "STOP").unlink()
    with pytest.raises(RuntimeError, match="KHONG train lai"):
        T.fit(_fresh(), tr, va, run, cfg(max_epochs=3, min_epochs=3), "pool", device="cpu", log=log)
    assert (run / "epoch_001.pt").exists()


def test_checkpoint_versions_are_plain_str():
    v = T.runtime_versions()
    assert all(type(x) is str for x in v.values())
    with pytest.raises(TypeError, match="weights_only"):
        T._assert_plain(dict(v=torch.__version__))                      # TorchVersion la lop con cua str


def test_checkpoint_state_must_be_weights_only_loadable(tmp_path):
    with pytest.raises(TypeError, match="weights_only"):
        T.save_checkpoint(tmp_path, dict(history=[dict(val_qwk=np.float64(0.5))]))
    T.save_checkpoint(tmp_path, dict(epoch=0, x=torch.zeros(2), history=[dict(val_qwk=0.5)]))
    assert torch.load(tmp_path / "last.pt", weights_only=True)["history"][0]["val_qwk"] == 0.5


# ------------------------------------------------------------ 16. Drive-first

@pytest.mark.parametrize("p", ["/content/input_cache/m3t_cls.csv", "/content/out.csv",
                               "/content/drive/../x.csv", "/content/input_cache"])
def test_drive_first_blocks_colab_local_including_the_input_cache(p):
    with pytest.raises(ValueError, match="RAM Colab"):
        io_utils.assert_drive_first(p)
    with pytest.raises(ValueError, match="RAM Colab"):
        T.write_once_csv(pd.DataFrame(dict(a=[1])), p)


def test_drive_first_allows_drive_and_fit_refuses_cache_run_dir(tmp_path):
    assert io_utils.assert_drive_first("/content/drive/MyDrive/x.csv") == "/content/drive/MyDrive/x.csv"
    tr, va = _data()
    with pytest.raises(ValueError, match="RAM Colab"):
        T.fit(_fresh(), tr, va, "/content/input_cache/run", cfg(), "pool", device="cpu")


def test_resolve_path_remaps_old_prefixes(tmp_path):
    new = tmp_path / "OAI_seg" / "nnUNet_raw"
    new.mkdir(parents=True)
    (new / "a.nii.gz").write_text("x")
    old = str(tmp_path / "nnUNet_raw").replace("\\", "/")
    p = old + "/a.nii.gz"
    got = io_utils.resolve_path(p, [(old, str(new).replace("\\", "/"))])
    assert os.path.exists(got)
    assert io_utils.resolve_path(p, [], must_exist=False) is None
    with pytest.raises(FileNotFoundError, match="da thu"):
        io_utils.resolve_path(p, [(old + "_x", str(new))])


def test_write_once_is_idempotent_for_identical_content(tmp_path):
    df = pd.DataFrame(dict(case_id=["a", "b"], v=[0.1, 0.2]))
    p = tmp_path / "t.csv"
    T.write_once_csv(df, p)
    T.write_once_csv(df.copy(), p)                                       # giong het => bo qua
    with pytest.raises(FileExistsError, match="KHAC"):
        T.write_once_csv(df.assign(v=[0.1, 0.3]), p)


# ------------------------------------------------------------ 17-18. config + evaluate

def test_repo_config_is_complete_and_matches_the_plan():
    c, raw = T.load_config(REPO / "configs" / "m3t_s9.json")
    assert (c.batch_size, c.lr, c.max_epochs, c.min_epochs, c.patience, c.min_delta, c.window) == \
        (2, 1e-4, 300, 60, 30, 0.005, 5)
    assert c.model == {} and c.augment in T.AUGMENTS and len(c.hash()) == 16
    g = raw["gates"]
    assert g["image"] == dict(r_median=0.99, r_p01=0.97, margin=0.1, slope_lo=0.95, slope_hi=1.05)
    assert g["cls"] == dict(nn_self=0.98, dist_ratio=0.25, head_agree=0.95, d_ekl=0.1)
    assert set(g["image"]) == set(m3t.IMAGE_GATE_KEYS) and set(g["cls"]) == set(m3t.CLS_GATE_KEYS)
    assert g["g0"] == dict(n_test=1636, correct=1073, tol=2) and g["pool_kl4_train_min"] == 84
    ev = raw["evaluation"]
    assert ev["primary"] == dict(model="B_xgb_frankhall", a="m3t_only__nosel", b="s6_all_plus_m3t__nosel",
                                 n_cols_a=128, n_cols_b=198)
    assert ev["min_delta"] == 0.02 and ev["bootstrap"]["n_boot"] == 10000 and ev["cv"]["seeds"] == [0, 1, 2]
    assert raw["inputs"]["torchio_version"] == "1.2.1"


def test_load_config_rejects_missing_or_unknown_fields(tmp_path):
    raw = json.loads((REPO / "configs" / "m3t_s9.json").read_text(encoding="utf-8"))
    raw["train"]["dropout"] = 0.2
    (tmp_path / "a.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="thua"):
        T.load_config(tmp_path / "a.json")
    del raw["train"]["dropout"], raw["train"]["lr"]
    (tmp_path / "b.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="thieu"):
        T.load_config(tmp_path / "b.json")
    with pytest.raises(ValueError):
        cfg(window=10, min_epochs=5)


def test_evaluate_metrics_match_ordinal_module():
    _, va = _data(n=10)
    model = _fresh().eval()
    loader = torch.utils.data.DataLoader(va, batch_size=3)
    r = T.evaluate(model, loader, "cpu", return_outputs=True)
    assert r["qwk"] == ORD.qwk(r["y"], r["pred"], 5)
    assert r["macro_recall"] == pytest.approx(ORD.macro_recall(r["y"], r["pred"], 5))
    assert r["acc"] == pytest.approx(float((r["y"] == r["pred"]).mean())) and r["n"] == 4
    assert not model.training


def test_train_transform_is_the_original_cell8_pipeline():
    tio = pytest.importorskip("torchio")
    t = T.train_transform()
    names = [type(x).__name__ for x in t.transforms]
    assert names == ["Compose", "OneOf", "RescaleIntensity"]
    torch.manual_seed(0)
    out = np.asarray(t(np.random.default_rng(0).integers(0, 500, (1, 12, 16, 16)).astype(np.uint16)))
    assert out.shape == (1, 12, 16, 16) and out.dtype == np.float32
    assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)
    assert T.runtime_versions()["torchio"] == tio.__version__
