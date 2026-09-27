"""Test bsc/m3t_eval.py - du lieu tong hop, khong can Drive.

    python -m pytest bsc/tests/test_m3t_eval.py -v
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from bsc import m3t as M3T
from bsc import m3t_eval as ME
from bsc import ordinal as ORD

CFG_JSON = os.path.join(os.path.dirname(__file__), "..", "configs", "m3t_s9.json")
LEG = [f"vol_{i}" for i in range(15)]
SURF = [f"fcl_{i}" for i in range(55)]
RAD = [f"rad_{i}" for i in range(30)]
CLS = M3T.cls_cols()


def _cls_csv(tmp_path, n=6, h="a" * 64, epoch=94, dup=False, const=False, name="m3t_cls_v2.csv"):
    rng = np.random.default_rng(0)
    ids = [f"900000{i}_V00_R" for i in range(n)]
    if dup:
        ids[1] = ids[0]
    x = rng.normal(size=(n, M3T.CLS_DIM))
    if const:
        x[:, 5] = 1.0
    df = pd.DataFrame(x, columns=CLS)
    df.insert(0, "case_id", ids)
    for k in range(5):
        df[f"{M3T.LOGIT_PREFIX}{k}"] = rng.normal(size=n)
    df["prov_weights_hash"] = h
    df["prov_epoch"] = epoch
    df.to_csv(tmp_path / name, index=False)
    meta = dict(main_epoch=94, weights_hash={"90": "b" * 64, "94": "a" * 64}, known_leaky=["c" * 64])
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path / name, tmp_path / "meta.json"


def test_load_cls_checks_provenance(tmp_path):
    p, m = _cls_csv(tmp_path)
    cls = ME.load_cls(p, m)
    assert len(cls) == 6 and cls.attrs["weights_hash"] == "a" * 64 and cls.attrs["epoch"] == 94
    assert [c for c in cls.columns if M3T.is_cls_col(c)] == CLS and "prov_weights_hash" not in cls.columns
    with pytest.raises(ValueError, match="weights_hash"):            # CSV cua epoch 94 doc nhu epoch 90
        ME.load_cls(p, m, epoch=90)
    for kw, msg in ((dict(dup=True), "trung"), (dict(const=True), "hang so")):
        p2, m2 = _cls_csv(tmp_path, **kw, name="x.csv")
        with pytest.raises(ValueError, match=msg):
            ME.load_cls(p2, m2)
    p3, m3 = _cls_csv(tmp_path, h="c" * 64, name="y.csv")
    meta = json.loads(m3.read_text())
    meta["weights_hash"]["94"] = "c" * 64
    m3.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="RO RI"):
        ME.load_cls(p3, m3)
    leaky = sorted(M3T.LEAKY_HASHES)[0]
    p4, m4 = _cls_csv(tmp_path, h=leaky, name="z.csv")
    meta = json.loads(m4.read_text())
    meta["weights_hash"]["94"] = leaky
    m4.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="RO RI"):
        ME.load_cls(p4, m4)


def test_attach_cls_keeps_order_no_impute(tmp_path):
    p, m = _cls_csv(tmp_path)
    cls = ME.load_cls(p, m)
    df = pd.DataFrame(dict(case_id=["x_missing"] + cls.case_id.tolist()[::-1], KL=range(7)))
    out, has, missing = ME.attach_cls(df, cls, expected_n=6)
    assert out.case_id.tolist() == df.case_id.tolist() and missing == ["x_missing"]
    assert has.tolist() == [False] + [True] * 6 and out.loc[0, CLS].isna().all()
    np.testing.assert_array_equal(out.loc[1:, CLS].to_numpy(), cls.loc[::-1, CLS].to_numpy())
    with pytest.raises(AssertionError, match="can 7"):
        ME.attach_cls(df, cls, expected_n=7)
    with pytest.raises(Exception):                                    # case_id trung -> one_to_one vo
        ME.attach_cls(pd.concat([df, df.iloc[[1]]]), cls)


def test_feature_sets_primary_pair_and_selection_rule():
    fs = ME.m3t_feature_sets(LEG, SURF, RAD, CLS)
    assert list(fs) == ["rad_only", "m3t_only", "legacy_plus_m3t", "s6_all_plus_m3t",
                        "s6_all_plus_radiomics_plus_m3t", "m3t_only__nosel", "legacy_plus_m3t__nosel",
                        "s6_all_plus_m3t__nosel"]
    a, b = fs["m3t_only__nosel"], fs["s6_all_plus_m3t__nosel"]
    assert (len(a), len(b)) == (128, 198) and [c for c in b if M3T.is_cls_col(c)] == a
    assert not ME.needs_selection("s6_all_plus_m3t__nosel", 198) and ME.needs_selection("s6_all_plus_m3t", 198)
    assert ME.needs_selection("m3t_only", 128) and not ME.needs_selection("s6_all", 70)
    no_rad = ME.m3t_feature_sets(LEG, SURF, [], CLS)
    assert "rad_only" not in no_rad and "s6_all_plus_radiomics_plus_m3t" not in no_rad
    with pytest.raises(ValueError):
        ME.m3t_feature_sets(LEG, SURF, RAD, ["vol_0"])


def test_bio_mask_and_family_treat_cls_as_image():
    cols = LEG[:2] + SURF[:2] + RAD[:2] + CLS[:2]
    assert ME.bio_mask(cols).tolist() == [True] * 4 + [False] * 4
    assert not ME.is_image_col("m3tlogit_0") and not ME.is_image_col("prov_weights_hash")
    fam = [ME.family(c, ("fcl_",)) for c in cols]
    assert fam == ["legacy"] * 2 + ["S6_surface"] * 2 + ["radiomics"] * 2 + ["m3t"] * 2


def test_subset_folds_keeps_membership():
    rng = np.random.default_rng(1)
    n = 40
    folds = []
    for s in (0, 1):
        perm = rng.permutation(n)
        for f in range(4):
            te = np.sort(perm[f::4])
            folds.append((s, f, np.setdiff1d(np.arange(n), te), te))
    keep = np.ones(n, bool)
    keep[[3, 17, 29]] = False
    sub = ME.subset_folds(folds, keep)
    old_ids = np.flatnonzero(keep)                                    # chi so moi -> chi so cu
    for (s, f, tr, te), (s2, f2, tr2, te2) in zip(folds, sub):
        assert (s, f) == (s2, f2)
        assert set(old_ids[te2]) == set(te) - {3, 17, 29} and set(old_ids[tr2]) == set(tr) - {3, 17, 29}
    bad = [(0, 0, np.arange(20), np.arange(20, 40)), (0, 1, np.arange(20, 40), np.arange(10, 30))]
    with pytest.raises(AssertionError):
        ME.subset_folds(bad, np.ones(n, bool))


def test_head_pred(tmp_path):
    p, m = _cls_csv(tmp_path)
    cls = ME.load_cls(p, m)
    lg = cls[[f"{M3T.LOGIT_PREFIX}{k}" for k in range(5)]].to_numpy()
    assert ME.head_pred(cls, [0, 1, 2, 3, 4]).tolist() == lg.argmax(1).tolist()
    with pytest.raises(ValueError):
        ME.head_pred(cls, [1, 2, 3, 4])


def test_registered_pairs_from_real_config_use_known_names():
    ev = json.load(open(CFG_JSON, encoding="utf-8"))["evaluation"]
    pairs = ME.registered_pairs(ev)
    assert pairs[0] == dict(a="m3t_only__nosel", b="s6_all_plus_m3t__nosel",
                            question=pairs[0]["question"], primary=True)
    assert sum(p["primary"] for p in pairs) == 1 and len(pairs) == 1 + len(ev["secondary"])
    known = set(ME.m3t_feature_sets(LEG, SURF, RAD, CLS)) | {"s6_all", "s6_all_plus_radiomics", ME.HEAD}
    assert {p["a"] for p in pairs} | {p["b"] for p in pairs} <= known
    fs = ME.m3t_feature_sets(LEG, SURF, RAD, CLS)
    assert (len(fs[ev["primary"]["a"]]), len(fs[ev["primary"]["b"]])) == (ev["primary"]["n_cols_a"],
                                                                         ev["primary"]["n_cols_b"])


def test_compare_pair_levels_and_metrics():
    rng = np.random.default_rng(3)
    n = 400
    y = rng.integers(0, 5, n)
    groups = np.repeat(np.arange(n // 2), 2)
    noisy = np.clip(y[:, None] + rng.integers(-2, 3, (n, 3)), 0, 4)
    same = ME.compare_pair(y, noisy, noisy, groups, 5, n_boot=200)
    assert same["delta"] == 0 and same["ci_low"] == same["ci_high"] == 0 and same["level"] == "chua_du"
    perfect = np.repeat(y[:, None], 3, axis=1)
    better = ME.compare_pair(y, noisy, perfect, groups, 5, n_boot=200)
    assert better["level"] == "vuot_nguong" and better["metrics_b"]["qwk"] == 1.0
    assert better["d_metrics"]["mae"] < 0 and len(better["delta_per_seed"]) == 3
    assert np.isclose(better["delta"], np.mean(better["delta_per_seed"]), atol=1e-6)
    head = np.clip(y + rng.integers(-1, 2, n), 0, 4)                  # nhanh 1 cot (head) lap qua 3 seed
    r = ME.compare_pair(y, np.repeat(head[:, None], 3, axis=1), noisy, groups, 5, n_boot=200)
    assert r["level"] in ME.ORD.DELTA_LEVELS
    with pytest.raises(ValueError):
        ME.stack_seeds({0: y, 1: np.full(n, -1)}, [0, 1])
    assert ME.stack_seeds({0: y, 1: y}, [1, 0]).shape == (n, 2)


def test_parallel_map_same_result_any_n_jobs():
    rng = np.random.default_rng(5)
    tasks = [(rng.integers(0, 5, 300), rng.integers(0, 5, 300), 5) for _ in range(4)]
    seq = ME.parallel_map(ORD.qwk, tasks, n_jobs=1)
    par = ME.parallel_map(ORD.qwk, tasks, n_jobs=2)
    assert seq == par == [ORD.qwk(*t) for t in tasks]
