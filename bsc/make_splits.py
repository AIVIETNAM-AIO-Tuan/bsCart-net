"""Khoi phuc + ghim data splits cho Stage 1 (bone-surface coordinate).

Vi sao file nay ton tai
-----------------------
nnUNet sinh `splits_final.json` trong /content/nnUNet_preprocessed moi phien Colab
roi vut di => khong co gi duoc ghim. Ke hoach §2.2 doi splits co dinh cho MOI thi nghiem.

May man: ban 150ep da train du 5 fold va moi case duoc predict boi dung fold ma no
lam validation. Nen fold assignment khoi phuc duoc 100% tu cay thu muc:
    Dataset020_KneeUnion/nnUNetTrainer_150epochs__.../fold_<k>/validation/<case>.nii.gz

RO RI DA XAC NHAN
-----------------
nnUNet chia KFold thuan theo TEN CASE, khong biet den benh nhan. iMorphics dat ten
`<barcode>_<visit>_<side>` (vd 9007827_V00_L) va moi dau goi co 2 timepoint V00/V01.
=> 51/70 dau goi co V00 va V01 o HAI FOLD KHAC NHAU: cung mot dau goi, cung benh nhan,
   chup cach nhau ~1 nam, xuat hien dong thoi o train va val.
=> So cross-validation BI THOI PHONG. Vi pham truc tiep §2.2.

Kiem che duoc: iMorphics train/test CO tach theo benh nhan (70 bn x2 = 140 train,
18 bn x2 = 36 test), nen so *test* da bao cao (ZIB Ts 103, iMorph test 36) van SACH
va dung lam baseline duoc. Chi so CV la khong dung duoc.

File nay sinh 2 split:
  * splits_zib_v1.json       : fold GOC khoi phuc (de tai lap B0/B1 y het)
  * splits_zib_v1_fixed.json : da sua ro ri, group theo (barcode, side) - dung cho
                               MOI training moi va cho atlas fold-specific (§2.2)

Stage 1 chi dung OAI-ZIB (nguon DUY NHAT co GT xuong), nen ro ri iMorphics khong
truc tiep dau doc Stage 1 - nhung splits van phai ghim va phai dung.

Chay:
    python bsc_splits.py --results-dir nnUNet_results --out splits/
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import zipfile

# Ten case iMorphics: <barcode>_<visit>_<side>, vd 9007827_V00_L
IMORPH_RE = re.compile(r"^(?P<barcode>\d{7})_(?P<visit>V\d\d)_(?P<side>[LR])$")
# Ten case OAI-ZIB: oaizib_XXX
ZIB_RE = re.compile(r"^oaizib_\d+$")

TRAINER_150 = "nnUNetTrainer_150epochs"


def _iter_zip_members(results_dir: str, dataset: str):
    """Doc ten file thang tu zip, khong giai nen (15 GB, khong can bung ra)."""
    for zp in sorted(glob.glob(os.path.join(results_dir, f"{dataset}-*.zip"))):
        try:
            with zipfile.ZipFile(zp) as z:
                for n in z.namelist():
                    yield n
        except zipfile.BadZipFile:
            # Google Takeout chia nhieu phan; phan khong doc doc lap duoc thi bo qua
            continue


def _iter_fs_members(results_dir: str, dataset: str):
    """Doc tu cay thu muc da giai nen (neu co)."""
    root = os.path.join(results_dir, dataset)
    if not os.path.isdir(root):
        return
    for dirpath, _, files in os.walk(root):
        for f in files:
            rel = os.path.relpath(os.path.join(dirpath, f), results_dir)
            yield rel.replace(os.sep, "/")


def recover_fold_assignment(results_dir: str, dataset: str = "Dataset020_KneeUnion",
                            trainer_prefix: str = TRAINER_150) -> dict[str, int]:
    """case_id -> fold, khoi phuc tu <trainer>/fold_<k>/validation/<case>.nii.gz."""
    fold_of: dict[str, int] = {}
    members = list(_iter_zip_members(results_dir, dataset)) + \
              list(_iter_fs_members(results_dir, dataset))
    for n in members:
        p = n.split("/")
        # <dataset>/<trainer>/fold_<k>/validation/<case>.nii.gz
        if len(p) < 5 or not n.endswith(".nii.gz"):
            continue
        if not p[1].startswith(trainer_prefix) or p[3] != "validation":
            continue
        m = re.match(r"^fold_(\d+)$", p[2])
        if not m:
            continue
        fold_of[p[4][: -len(".nii.gz")]] = int(m.group(1))
    return fold_of


def patient_key(case: str) -> str:
    """Khoa gom nhom o muc benh nhan.

    iMorphics: (barcode, side) - hai timepoint V00/V01 cua CUNG mot dau goi phai
               o cung fold. Dung barcode+side chu khong chi barcode: hai dau goi
               trai/phai cua cung nguoi la hai mau doc lap ve giai phau, nhung
               V00/V01 cua CUNG mot dau goi thi khong.
    OAI-ZIB:   ten `oaizib_XXX` khong lo danh tinh benh nhan => moi case mot khoa.
               (Can info.zip tu HF de xac minh 404 train / 103 test co trung benh
               nhan khong - task Phase 0.6.)
    """
    m = IMORPH_RE.match(case)
    if m:
        return f"imorph:{m['barcode']}_{m['side']}"
    return f"case:{case}"


def audit_leak(fold_of: dict[str, int]) -> dict:
    """Dem so nhom benh nhan bi chia qua nhieu fold."""
    groups: dict[str, set[int]] = collections.defaultdict(set)
    members: dict[str, list[str]] = collections.defaultdict(list)
    for c, k in fold_of.items():
        groups[patient_key(c)].add(k)
        members[patient_key(c)].append(c)
    multi = {g: f for g, f in groups.items() if len(members[g]) > 1}
    leaked = {g: sorted(f) for g, f in multi.items() if len(f) > 1}
    return {
        "n_cases": len(fold_of),
        "n_groups": len(groups),
        "n_groups_multi_case": len(multi),
        "n_groups_leaked": len(leaked),
        "leaked_examples": {
            g: {c: fold_of[c] for c in sorted(members[g])}
            for g in sorted(leaked)[:10]
        },
    }


def regroup_no_leak(fold_of: dict[str, int], n_folds: int = 5, seed: int = 0) -> dict[str, int]:
    """Gan lai fold o muc NHOM BENH NHAN, giu kich thuoc fold can bang.

    Deterministic: sort theo hash cua khoa nhom (khong dung random state toan cuc),
    roi rai greedy vao fold nho nhat. Cung input => cung output, moi may.
    """
    members: dict[str, list[str]] = collections.defaultdict(list)
    for c in fold_of:
        members[patient_key(c)].append(c)

    def h(g: str) -> str:
        return hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()

    # Nhom lon truoc => can bang tot hon; hash de pha the deterministic
    order = sorted(members, key=lambda g: (-len(members[g]), h(g)))
    sizes = [0] * n_folds
    out: dict[str, int] = {}
    for g in order:
        k = min(range(n_folds), key=lambda i: (sizes[i], i))
        for c in members[g]:
            out[c] = k
        sizes[k] += len(members[g])
    return out


def to_nnunet_splits(fold_of: dict[str, int], n_folds: int = 5) -> list[dict]:
    """Doi sang dung format nnUNet: [{'train': [...], 'val': [...]}, ...]."""
    cases = sorted(fold_of)
    return [
        {
            "train": [c for c in cases if fold_of[c] != k],
            "val": [c for c in cases if fold_of[c] == k],
        }
        for k in range(n_folds)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="nnUNet_results")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    fold_of = recover_fold_assignment(args.results_dir)
    if not fold_of:
        raise SystemExit(
            f"Khong khoi phuc duoc fold nao tu {args.results_dir}. "
            f"Can Dataset020_KneeUnion/{TRAINER_150}__*/fold_*/validation/*.nii.gz"
        )

    sizes = collections.Counter(fold_of.values())
    n_zib = sum(1 for c in fold_of if ZIB_RE.match(c))
    n_imo = sum(1 for c in fold_of if IMORPH_RE.match(c))
    print(f"Khoi phuc {len(fold_of)} case  (ZIB {n_zib} / iMorph {n_imo})")
    print(f"Fold sizes goc: {dict(sorted(sizes.items()))}")

    before = audit_leak(fold_of)
    print(f"\n[RO RI - split GOC]")
    print(f"  nhom benh nhan co >1 case : {before['n_groups_multi_case']}")
    print(f"  !! bi chia qua nhieu fold : {before['n_groups_leaked']}")
    for g, m in list(before["leaked_examples"].items())[:5]:
        print(f"     {g}: " + ", ".join(f"{c}->f{k}" for c, k in m.items()))

    fixed = regroup_no_leak(fold_of, args.n_folds)
    after = audit_leak(fixed)
    print(f"\n[RO RI - split DA SUA]")
    print(f"  !! bi chia qua nhieu fold : {after['n_groups_leaked']}  (phai = 0)")
    print(f"  Fold sizes moi: {dict(sorted(collections.Counter(fixed.values()).items()))}")
    assert after["n_groups_leaked"] == 0, "Regroup that bai - van con ro ri"

    os.makedirs(args.out, exist_ok=True)
    for name, fo, audit in [
        ("splits_zib_v1.json", fold_of, before),
        ("splits_zib_v1_fixed.json", fixed, after),
    ]:
        payload = {
            "_comment": (
                "IMMUTABLE. Sinh boi bsc_splits.py. splits_zib_v1 = fold GOC khoi phuc tu "
                "Dataset020 150ep (tai lap B0/B1). splits_zib_v1_fixed = da sua ro ri "
                "V00/V01, dung cho MOI training moi va atlas fold-specific (§2.2)."
            ),
            "n_folds": args.n_folds,
            "leak_audit": audit,
            "fold_of": fo,
            "splits": to_nnunet_splits(fo, args.n_folds),
        }
        p = os.path.join(args.out, name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        digest = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
        print(f"\nGhi {p}\n  sha256[:16] = {digest}")


if __name__ == "__main__":
    main()
