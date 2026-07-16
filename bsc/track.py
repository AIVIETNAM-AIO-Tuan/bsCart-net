"""Theo doi thi nghiem (§7 cua plan doc) - Drive-first, khong ghi vao /content/.

Moi run ghi vao BSC_ROOT/runs/<exp_id>/:
    config.json    - toan bo cau hinh + git sha + timestamp (do ben ngoai truyen vao)
    metrics.csv    - metric PER-CASE (khong chi trung binh: §2.4 doi so sanh ghep cap)
    <artifact>     - checkpoint, qc/, ...

exp_id theo mau plan doc §7, vd:
    MVP_FC_GTsurf_Ray64_MRI-SDF_Fold0_Seed1

LUU Y VE MOI TRUONG (khong co Date.now/random trong workflow, nhung o day la Python
thuong nen dung binh thuong). Van tranh gan timestamp ben trong ham thuan de ket qua
tai lap - truyen `now` tu ngoai vao.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess

# Mac dinh Drive-first. Notebook set BSC_ROOT trong cell config dau tien.
DEFAULT_ROOT = "/content/drive/MyDrive/bsc"


def git_sha(repo_dir: str = ".") -> str:
    """SHA commit hien tai (§7). Tra 'nogit' neu khong trong repo."""
    try:
        out = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        sha = out.stdout.strip()
        dirty = subprocess.run(["git", "-C", repo_dir, "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def make_exp_id(stage: str, cls: str, surface: str, ray_k: int, inputs: str,
                fold: int, seed: int) -> str:
    """Sinh exp_id theo mau plan doc §7. Vd:
        make_exp_id("MVP","FC","GTsurf",64,"MRI-SDF",0,1)
        -> "MVP_FC_GTsurf_Ray64_MRI-SDF_Fold0_Seed1"
    """
    return f"{stage}_{cls}_{surface}_Ray{ray_k}_{inputs}_Fold{fold}_Seed{seed}"


def _assert_drive_first(path: str) -> None:
    """Bat loi ghi vao /content/ (khong phai /content/drive/).

    Day la hoi quy truc tiep cua loi da lam mat 544 prediction: chung o
    /content/pred_* (RAM Colab, mat khi ngat phien) thay vi /content/drive/.
    """
    ap = os.path.abspath(path)
    if ap.startswith("/content/") and not ap.startswith("/content/drive/"):
        raise ValueError(
            f"Duong ghi '{path}' nam trong /content/ RAM Colab - se MAT khi ngat "
            f"phien. Ghi vao /content/drive/MyDrive/bsc/ (BSC_ROOT). Day la loi da "
            f"tung lam mat 544 prediction."
        )


class Run:
    """Mot run thi nghiem. Ghi config + metric per-case vao BSC_ROOT/runs/<exp_id>/."""

    def __init__(self, exp_id: str, config: dict, root: str = DEFAULT_ROOT,
                 repo_dir: str = ".", now: "str | None" = None):
        self.exp_id = exp_id
        self.dir = os.path.join(root, "runs", exp_id)
        _assert_drive_first(self.dir)
        os.makedirs(os.path.join(self.dir, "qc"), exist_ok=True)

        self.config = {
            "exp_id": exp_id,
            "git_sha": git_sha(repo_dir),         # §7
            "timestamp": now,                     # truyen tu ngoai de tai lap
            **config,
        }
        with open(os.path.join(self.dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        self._rows: list[dict] = []

    def log_case(self, case_id: str, cls: str, **metrics) -> None:
        """Ghi metric cho MOT (ca, lop). Per-case la bat buoc cho §2.4."""
        self._rows.append({"case_id": case_id, "class": cls, **metrics})

    def save_metrics(self) -> str:
        """Ghi metrics.csv. Tra duong dan."""
        if not self._rows:
            raise RuntimeError("Chua co case nao duoc log")
        keys = list({k for r in self._rows for k in r})
        head = ["case_id", "class"] + [k for k in keys if k not in ("case_id", "class")]
        p = os.path.join(self.dir, "metrics.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=head)
            w.writeheader()
            w.writerows(self._rows)
        return p

    def artifact_path(self, name: str) -> str:
        """Duong dan artifact duoi run dir (vd 'ckpt.pt'), da kiem Drive-first."""
        p = os.path.join(self.dir, name)
        _assert_drive_first(p)
        return p
