"""Test phan khong can nibabel cua bsc/io_utils.py: OutDir (Drive lam mat thu muc vua tao).

    python -m pytest bsc/tests/test_io_utils.py -v
"""

from __future__ import annotations

import json
import pickle
import shutil

import numpy as np
import pandas as pd
import pytest

from bsc import io_utils as IO


def test_outdir_rewrites_everything_when_folder_vanishes(tmp_path):
    logs = []
    out = IO.OutDir(tmp_path / "run", log=logs.append)
    df = pd.DataFrame(dict(a=[1, 2], b=[0.5, np.nan]))
    out.csv(df, "t.csv", index=False)
    out.json(dict(x=np.int64(3), y=np.arange(2)), "c.json", indent=2)
    out.pickle({"k": [1, 2]}, "m.pkl")
    before = {n: (tmp_path / "run" / n).read_bytes() for n in ("t.csv", "c.json", "m.pkl")}
    shutil.rmtree(tmp_path / "run")                                   # Drive lam mat thu muc
    out.csv(df.assign(c=1), "u.csv", index=False)
    assert out.recreated == 1 and "bien mat" in logs[0]
    for n, b in before.items():
        assert (tmp_path / "run" / n).read_bytes() == b                 # ghi lai y het tu bo nho
    assert json.loads((tmp_path / "run" / "c.json").read_text()) == dict(x=3, y=[0, 1])
    assert pickle.loads((tmp_path / "run" / "m.pkl").read_bytes()) == {"k": [1, 2]}
    assert out.verify() == ["t.csv", "c.json", "m.pkl", "u.csv"]


def test_outdir_verify_restores_missing_file_and_fails_loudly(tmp_path):
    out = IO.OutDir(tmp_path / "run", log=lambda *_: None)
    out.csv(pd.DataFrame(dict(a=[1])), "t.csv", index=False)
    (tmp_path / "run" / "t.csv").unlink()
    assert out.verify() == ["t.csv"] and (tmp_path / "run" / "t.csv").exists()
    shutil.rmtree(tmp_path / "run")
    assert out.verify() == ["t.csv"] and out.recreated == 1
    out.put("ghost.txt", lambda p: None)                                # writer khong ghi gi -> phai loi
    with pytest.raises(FileNotFoundError, match="ghost.txt"):
        out.verify()


def test_outdir_rejects_colab_local():
    with pytest.raises(ValueError):
        IO.OutDir("/content/s8_out")
