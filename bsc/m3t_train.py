"""Huan luyen lai M3T TU DAU tren kho du lieu cua no TRU moi subject cua cohort 1325 ca.

Ly do: M3T goc (knee_testing_v3.ipynb) train tren 2880 subject OAI, trong do co subject cua
cohort (44/70 subject iMorphics nam o tap train cua no) => CLS cua trong so cu da mang nhan KL
cua cohort. Xem docs/ke_hoach_m3t_cls.md muc 2-4.

CONG THUC = BAN GOC (cell 8, 20, 21) + MOT cho lech co chu y
  giu nguyen : kien truc, tang cuong torchio (cung thu tu - noise tinh tren cuong do THO),
               CE khong trong so, Adam lr 1e-4, batch 2, khong scheduler
  lech       : chon epoch theo trung binh truot TRAILING 5 epoch cua val QWK (val acc 1 epoch
               dao dong +-0.016 => chon theo 1 epoch la duoi nhieu; bai toan sau la thu tu)
  dung som   : khong cai thien min_delta tren diem truot trong `patience` epoch, trong
               [min_epochs, max_epochs]
Gia tri that nam MOT cho: configs/m3t_s9.json['train'] (dang ky truoc). Dataclass khong co
gia tri mac dinh de khong co ban thu hai cua cong thuc.

TAI LAP / CHAY TIEP QUA NHIEU PHIEN COLAB
  seed epoch e = SeedSequence([seed, e]) cho random/numpy/torch + generator cua DataLoader
  (thu tu tron VA seed worker => tang cuong torchio, vi torchio dung RNG cua torch).
  last.pt ghi nguyen tu + last_prev.pt du phong, chi tensor + kieu Python thuan (nap duoc
  weights_only=True); epoch_XXX.pt moi epoch (cua so do nhay checkpoint). Tu choi chay tiep
  neu cfg_hash, pool_md5 hoac phien ban torchio khac.
  CPU: hai epoch lien = mot epoch + chay tiep, bit-doi-bit (test). GPU voi cudnn.benchmark:
  cung chuoi du lieu/tang cuong nhung so hoc KHONG bit-doi-bit - gioi han da biet.

NGOAI LE /content (duyet 24/09/2026): `stage_input` la ham DUY NHAT ghi vao /content, chi vao
/content/input_cache, chi du lieu dau vao. Moi san pham qua io_utils.assert_drive_first (chan
ca /content/input_cache).
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import platform
import posixpath
import random
import re
import shutil
import time
import warnings
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import m3t as M3T
from . import ordinal as ORD
from .io_utils import assert_drive_first

CACHE_ROOT = "/content/input_cache"
KL_CLASSES = 5
SUBSETS = ("train", "val", "test")
_NPZ_RE = re.compile(r"(?:^|/)(\d+)_(\d{8})_(LEFT|RIGHT)\.npz$")
_BARCODE_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")
_SIDE = {"L": "L", "LEFT": "L", "R": "R", "RIGHT": "R"}
LAST, PREV = "last.pt", "last_prev.pt"


# ============================================================ chuan hoa ID

def norm_subject(x, strict: bool = True):
    """ID subject OAI -> chuoi 7 chu so. '9000099.0' (doc CSV thanh float) va khoang trang duoc sua."""
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    if re.fullmatch(r"\d{7}", s):
        return s
    if strict:
        raise ValueError(f"subject OAI can 7 chu so, nhan {x!r}")
    return None


def norm_side(x, strict: bool = True):
    s = str(x).strip().upper()
    if s in _SIDE:
        return _SIDE[s]
    if strict:
        raise ValueError(f"ben goi can L/R/LEFT/RIGHT, nhan {x!r}")
    return None


# ============================================================ zip npz

def parse_npz_name(name):
    """'.../{id}_{barcode8}_{LEFT|RIGHT}.npz' -> (subject, barcode, 'L'|'R')."""
    m = _NPZ_RE.search(str(name))
    if not m:
        raise ValueError(f"ten npz khong dung mau {{id}}_{{barcode8}}_{{LEFT|RIGHT}}.npz: {name!r}")
    return norm_subject(m.group(1)), m.group(2), _SIDE[m.group(3)]


def npz_index(zip_path):
    """Muc luc npz cua zip (chi doc central directory): member, npz_name, subject, barcode, side,
    file_size, crc. Ten npz sai mau => loi (khong bo qua im lang)."""
    import pandas as pd
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for i in zf.infolist():
            if i.is_dir() or not i.filename.endswith(".npz"):
                continue
            subj, bc, side = parse_npz_name(i.filename)
            rows.append(dict(member=i.filename, npz_name=posixpath.basename(i.filename), subject=subj,
                             barcode=bc, side=side, file_size=i.file_size, crc=i.CRC))
    if not rows:
        raise ValueError(f"{zip_path}: khong co npz")
    df = pd.DataFrame(rows).sort_values("member").reset_index(drop=True)
    for col in ("barcode", "npz_name"):
        if df[col].duplicated().any():
            raise ValueError(f"{col} trung trong zip: {df.loc[df[col].duplicated(), col].tolist()[:3]}")
    return df


def index_md5(index) -> str:
    """Dau van tay NOI DUNG zip tu central directory (ten, CRC, kich thuoc) - khong doc 30 GB."""
    lines = sorted(f"{m},{c},{s}" for m, c, s in zip(index["member"], index["crc"], index["file_size"]))
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


class ZipNpzReader:
    """Doc npz TRONG zip (khong giai nen ra dia). Mo zip lai trong tung process: handle cua
    process cha (truoc fork) dung chung vi tri doc voi worker => doc hong ma khong bao loi."""

    def __init__(self, zip_path):
        self.zip_path = os.fspath(zip_path)
        self._zf = None
        self._pid = None

    def _handle(self):
        if self._zf is None or self._pid != os.getpid():
            self._zf = zipfile.ZipFile(self.zip_path)
            self._pid = os.getpid()
        return self._zf

    def read(self, member) -> np.ndarray:
        raw = self._handle().read(member)            # zipfile kiem CRC khi doc het member
        with np.load(io.BytesIO(raw)) as z:
            return z["data"]

    def close(self):
        if self._zf is not None:
            self._zf.close()
        self._zf, self._pid = None, None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_zf"], d["_pid"] = None, None
        return d


def _check_cache_root(root) -> str:
    raw = posixpath.normpath(os.fspath(root).replace("\\", "/"))
    if (raw == "/content" or raw.startswith("/content/")) and not (
            raw == CACHE_ROOT or raw.startswith(CACHE_ROOT + "/")):
        raise ValueError(f"bo dem dau vao chi duoc nam o {CACHE_ROOT} (ngoai le duoc duyet), nhan {root}")
    return os.fspath(root)


def verify_zip_members(zip_path, n_check: int = 200, seed: int = 0) -> list:
    """Doc DAY DU n_check npz chon tat dinh; zipfile nem BadZipFile neu CRC sai."""
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(i.filename for i in zf.infolist() if i.filename.endswith(".npz"))
        if not names:
            raise ValueError(f"{zip_path}: khong co npz")
        rng = np.random.default_rng(seed)
        pick = sorted(rng.choice(len(names), size=min(n_check, len(names)), replace=False).tolist())
        for k in pick:
            zf.read(names[k])
    return [names[k] for k in pick]


def stage_input(src, cache_root=CACHE_ROOT, expected_size=None, n_check: int = 200,
                min_free_gb: float = 40.0, seed: int = 0, log=print) -> str:
    """Chep zip dau vao vao bo dem DOC cuc bo roi kiem size + CRC n_check member.

    HAM DUY NHAT duoc ghi vao /content (chi /content/input_cache). Da co ban dung size thi
    chi kiem CRC. src da nam trong bo dem (vd tai bang token vao thang day) => chi kiem.
    """
    root = _check_cache_root(cache_root)
    os.makedirs(root, exist_ok=True)
    src = os.fspath(src)
    dst = os.path.join(root, os.path.basename(src))
    size = os.path.getsize(src)
    if expected_size is not None and size != int(expected_size):
        raise ValueError(f"{src}: {size} byte, can {expected_size} - khong phai zip da ghim")
    same = os.path.abspath(src) == os.path.abspath(dst)
    if not same and not (os.path.exists(dst) and os.path.getsize(dst) == size):
        free = shutil.disk_usage(root).free
        need = max(min_free_gb * 1e9, 1.05 * size)
        if free < need:
            raise OSError(f"{root} con {free / 1e9:.1f} GB, can {need / 1e9:.1f} GB")
        t0 = time.time()
        tmp = dst + ".part"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        dt = time.time() - t0
        log(f"da chep {size / 1e9:.1f} GB trong {dt / 60:.1f} phut ({size / 1e6 / max(dt, 1e-9):.0f} MB/s)")
    if os.path.getsize(dst) != size:
        raise OSError(f"{dst}: kich thuoc {os.path.getsize(dst)} khac nguon {size}")
    checked = verify_zip_members(dst, n_check, seed)
    log(f"CRC dung {len(checked)} npz trong {dst}")
    return dst


# ============================================================ nhan, pool, phoi nhiem

def load_labels(src):
    """unified_xray_mri_label.csv -> subject, side, npz_name, kl_grade, subset (chuan hoa + kiem).

    Kiem: dung mot dong moi goi, KL trong 0..4, subset hop le, MOI subject nam o MOT subset
    (split goc chia theo subject), ten npz khop subject/ben cua dong.
    """
    import pandas as pd
    df = pd.read_csv(src) if not isinstance(src, pd.DataFrame) else src.copy()
    need = {"id", "side", "mri_path", "kl_grade", "subset"}
    if need - set(df.columns):
        raise ValueError(f"thieu cot {sorted(need - set(df.columns))}")
    out = pd.DataFrame(dict(subject=df["id"].map(norm_subject), side=df["side"].map(norm_side),
                            npz_name=df["mri_path"].astype(str).map(posixpath.basename),
                            kl_grade=df["kl_grade"].astype(int), subset=df["subset"].astype(str)))
    bad = ~out["subset"].isin(SUBSETS)
    if bad.any():
        raise ValueError(f"subset la: {sorted(out.loc[bad, 'subset'].unique())}")
    if not out["kl_grade"].between(0, KL_CLASSES - 1).all():
        raise ValueError("kl_grade ngoai 0..4")
    if out.duplicated(["subject", "side"]).any():
        raise ValueError("mot goi co nhieu dong")
    per = out.groupby("subject")["subset"].nunique()
    if (per > 1).any():
        raise ValueError(f"{int((per > 1).sum())} subject nam o nhieu subset - split khong theo subject")
    parsed = out["npz_name"].map(parse_npz_name)
    if not ((parsed.str[0] == out["subject"]) & (parsed.str[2] == out["side"])).all():
        raise ValueError("ten npz khong khop subject/ben cua dong")
    return out


def attach_members(df, index):
    """Them cot `member` (duong trong zip) theo npz_name; thieu npz nao => loi."""
    m = df.merge(index[["npz_name", "member"]], on="npz_name", how="left", validate="many_to_one")
    if m["member"].isna().any():
        raise ValueError(f"{int(m['member'].isna().sum())} npz khong co trong zip, "
                         f"vd {m.loc[m['member'].isna(), 'npz_name'].tolist()[:3]}")
    return m


def build_pool(labels, exclude_subjects):
    """Bo MOI dong cua subject bi loai: ca hai goi, moi subset (train/val/test)."""
    excl = {norm_subject(s) for s in exclude_subjects}
    return labels[~labels["subject"].isin(excl)].reset_index(drop=True)


def _subjects(x) -> set:
    if hasattr(x, "columns"):
        x = x["subject"]
    return {norm_subject(s) for s in x}


def assert_subject_disjoint(a, b, what: str = "") -> None:
    common = _subjects(a) & _subjects(b)
    if common:
        raise AssertionError(f"{what}: {len(common)} subject chung, vd {sorted(common)[:5]}")


def pool_md5(pool) -> str:
    """Dau van tay cua pool dung de train (thanh vien + nhan + subset)."""
    cols = ["subject", "side", "npz_name", "kl_grade", "subset"]
    lines = sorted(",".join(str(v) for v in r) for r in pool[cols].itertuples(index=False))
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def exposure_table(manifest, labels, index):
    """Moi ca cohort: goi nao cua M3T tuong ung, va subject do nam o dau trong split goc.

    match  : 'barcode'            - so 8 chu so trong dess_path trung barcode mot npz (CUNG lan chup)
             'barcode_ambiguous'  - trung nhieu barcode
             'subject_side'       - khong co barcode; co npz cung subject + ben (co the KHAC lan kham)
             'subject_other_side' - subject co npz nhung chi o ben kia (ben trong manifest co the sai)
             'none'               - subject khong co npz nao
    ambiguous    : subject_side nhung goi do co >1 npz (nhieu lan kham) - khong biet cai nao.
    side_conflict/subject_conflict: barcode chi toi npz co ben/subject KHAC manifest.
    m3t_subset   : subset cua SUBJECT trong split goc ('absent' neu khong co trong CSV nhan).
    knee_in_csv, m3t_kl: goi (subject, ben) co nhan M3T khong, va KL do.
    """
    import pandas as pd
    need = {"case_id", "subject", "side", "dess_path"}
    if need - set(manifest.columns):
        raise ValueError(f"manifest thieu cot {sorted(need - set(manifest.columns))}")
    by_barcode = index.set_index("barcode")
    by_knee = index.groupby(["subject", "side"])["npz_name"].apply(list).to_dict()
    lab = labels.set_index(["subject", "side"])
    subj_subset = labels.groupby("subject")["subset"].first().to_dict()
    rows = []
    for r in manifest.to_dict("records"):
        subj = norm_subject(r["subject"], strict=False)
        side = norm_side(r["side"], strict=False)
        tokens = list(dict.fromkeys(_BARCODE_RE.findall(str(r.get("dess_path") or ""))))
        hits = [t for t in tokens if t in by_barcode.index]
        out = dict(case_id=str(r["case_id"]), subject=subj, side=side, visit=r.get("visit"),
                   source_dataset=r.get("source_dataset"), match="none", barcode=None, npz_name=None,
                   n_candidates=0, ambiguous=False, side_conflict=False, subject_conflict=False)
        if len(hits) == 1:
            row = by_barcode.loc[hits[0]]
            out.update(match="barcode", barcode=hits[0], npz_name=row["npz_name"], n_candidates=1,
                       side_conflict=side is not None and row["side"] != side,
                       subject_conflict=subj is not None and row["subject"] != subj)
        elif len(hits) > 1:
            out.update(match="barcode_ambiguous", n_candidates=len(hits), ambiguous=True)
        elif subj is not None and side is not None and (subj, side) in by_knee:
            cands = by_knee[(subj, side)]
            in_csv = [c for c in cands if (subj, side) in lab.index and lab.loc[(subj, side), "npz_name"] == c]
            out.update(match="subject_side", n_candidates=len(cands), ambiguous=len(cands) > 1,
                       npz_name=cands[0] if len(cands) == 1 else (in_csv[0] if in_csv else None))
        elif subj is not None and any((subj, s) in by_knee for s in "LR"):
            out.update(match="subject_other_side")
        out["m3t_subset"] = subj_subset.get(subj, "absent") if subj is not None else "invalid_subject"
        knee = (subj, side)
        out["knee_in_csv"] = subj is not None and side is not None and knee in lab.index
        out["m3t_kl"] = int(lab.loc[knee, "kl_grade"]) if out["knee_in_csv"] else None
        rows.append(out)
    return pd.DataFrame(rows)


# ============================================================ dataset + tang cuong

class NpzKLDataset(Dataset):
    """(khoi [1,L,W,H] float32 trong [0,1], nhan KL long). transform None => minmax01 (= val/test
    transform goc); transform torchio nhan mang [1,L,W,H] tho nhu cell 7 goc."""

    def __init__(self, df, reader, transform=None, expect_shape=M3T.INPUT_SHAPE):
        if {"member", "kl_grade"} - set(df.columns):
            raise ValueError("df can cot member, kl_grade (xem attach_members)")
        self.members = df["member"].astype(str).tolist()
        self.labels = df["kl_grade"].astype(int).tolist()
        self.reader, self.transform, self.expect_shape = reader, transform, expect_shape

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        vol = self.reader.read(self.members[i])
        if self.expect_shape is not None and tuple(vol.shape) != tuple(self.expect_shape):
            raise ValueError(f"{self.members[i]}: shape {vol.shape}, can {tuple(self.expect_shape)}")
        if self.transform is None:
            x = M3T.minmax01(vol)[None]
        else:
            x = np.asarray(self.transform(np.expand_dims(vol, 0)), dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(self.labels[i], dtype=torch.long)


def train_transform():
    """Tang cuong NGUYEN VAN cell 8 cua knee_testing_v3.ipynb (torchio 1.2.1 nhu ban goc)."""
    import torchio as tio
    warnings.filterwarnings("ignore", module="torchio")
    spatial_augment = [
        tio.RandomAffine(degrees=15, p=0.5),
        tio.RandomFlip(axes=(0,), flip_probability=0.5),
    ]
    intensity_augment = {
        tio.RandomNoise(): 0.25,
        tio.RandomBiasField(): 0.25,
        tio.RandomBlur(std=(0, 1.5)): 0.25,
        tio.RandomMotion(): 0.25,
    }
    return tio.Compose([
        tio.Compose(spatial_augment, p=1),
        tio.OneOf(intensity_augment, p=0.75),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])


AUGMENTS = {"knee_testing_v3_cell8": train_transform}


def get_augment(name: str):
    if name not in AUGMENTS:
        raise ValueError(f"tang cuong '{name}' khong co trong AUGMENTS {sorted(AUGMENTS)}")
    return AUGMENTS[name]()


# ============================================================ cau hinh

@dataclasses.dataclass(frozen=True)
class M3TTrainConfig:
    seed: int
    batch_size: int
    lr: float
    max_epochs: int
    min_epochs: int
    patience: int
    min_delta: float
    window: int
    num_workers: int
    allow_tf32: bool
    cudnn_benchmark: bool
    augment: str
    model: dict

    def __post_init__(self):
        if not (1 <= self.window <= self.min_epochs <= self.max_epochs):
            raise ValueError("can 1 <= window <= min_epochs <= max_epochs")
        if self.batch_size < 1 or self.lr <= 0 or self.patience < 1 or self.min_delta < 0:
            raise ValueError("batch_size/lr/patience/min_delta khong hop le")
        bad = set(self.model) - set(M3T.M3T_CFG)
        if bad:
            raise ValueError(f"khoa mo hinh la: {sorted(bad)}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def load_config(path):
    """-> (M3TTrainConfig tu khoa 'train', toan bo JSON). Thieu hay thua truong => loi."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    t = raw["train"]
    fields = {f.name for f in dataclasses.fields(M3TTrainConfig)}
    missing, unknown = fields - set(t), set(t) - fields
    if missing or unknown:
        raise ValueError(f"config['train'] thieu {sorted(missing)}, thua {sorted(unknown)}")
    return M3TTrainConfig(**t), raw


def runtime_versions() -> dict:
    """Chuoi str THUAN: torch.__version__ la TorchVersion (lop con cua str) - pickle no vao
    checkpoint lam torch.load(weights_only=True) tu choi nap."""
    v = dict(torch=str(torch.__version__), numpy=str(np.__version__), python=platform.python_version(),
             cuda=str(torch.version.cuda), cudnn=str(torch.backends.cudnn.version()))
    try:
        import torchio
        v["torchio"] = str(torchio.__version__)
    except ImportError:
        v["torchio"] = "absent"
    return v


# ============================================================ train / danh gia

def epoch_seed(base_seed: int, epoch: int) -> int:
    return int(np.random.SeedSequence([int(base_seed), int(epoch)]).generate_state(1, dtype=np.uint32)[0])


def seed_everything(seed: int, generator=None) -> None:
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if generator is not None:
        generator.manual_seed(seed)


def _worker_init(worker_id):
    warnings.filterwarnings("ignore", module="torchio")
    s = torch.initial_seed() % 2 ** 32                  # = base_seed (tu generator) + worker_id
    np.random.seed(s)
    random.seed(s)


def train_one_epoch(model, loader, optimizer, device) -> dict:
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)                  # = nn.CrossEntropyLoss() cua ban goc
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss khong huu han sau {n} mau")
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * x.size(0)
        correct += int((out.argmax(1) == y).sum().item())
        n += int(y.size(0))
    if n == 0:
        raise ValueError("loader train rong")
    return dict(train_loss=loss_sum / n, train_acc=correct / n, n_train=n)


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int = KL_CLASSES, return_outputs: bool = False) -> dict:
    """loss, acc, qwk (ORD.qwk), macro_recall + recall tung lop (ORD.per_class_prf)."""
    was_training = model.training
    model.eval()
    ys, lgs, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += float(F.cross_entropy(out, y, reduction="sum").item())
        ys.append(y.cpu().numpy())
        lgs.append(out.float().cpu().numpy())
    model.train(was_training)
    if not ys:
        raise ValueError("loader danh gia rong")
    y, lg = np.concatenate(ys), np.concatenate(lgs)
    pred = lg.argmax(1)
    prf = ORD.per_class_prf(y, pred, n_classes)
    res = dict(loss=loss_sum / len(y), acc=float((pred == y).mean()), qwk=ORD.qwk(y, pred, n_classes),
               macro_recall=float(prf["recall"].mean()), recall=[float(v) for v in prf["recall"]],
               n=int(len(y)))
    if return_outputs:
        res.update(y=y, pred=pred, logits=lg)
    return res


# ============================================================ chon epoch + dung som

def _val_qwk(history) -> list:
    if [h["epoch"] for h in history] != list(range(len(history))):
        raise ValueError("history phai lien tuc tu epoch 0")
    q = [float(h["val_qwk"]) for h in history]
    if not np.isfinite(q).all():
        raise ValueError("val_qwk co NaN")
    return q


def trailing_means(values, window: int) -> list:
    """m[t] = trung binh values[t-window+1 .. t]; None khi chua du cua so."""
    v = np.asarray(values, np.float64)
    return [None if t < window - 1 else float(v[t - window + 1:t + 1].mean()) for t in range(len(v))]


def select_epoch(history, window: int) -> dict:
    """Cua so TRAILING [t-window+1..t] co trung binh val QWK cao nhat trong cac cua so DU; hoa
    thi t nho nhat. CHI doc 'val_qwk' - khong mot diem downstream nao (S7/S8) duoc anh huong."""
    q = _val_qwk(history)
    if len(q) < window:
        raise ValueError(f"can >= {window} epoch, co {len(q)}")
    sm = trailing_means(q, window)
    t = max(range(window - 1, len(q)), key=lambda i: (sm[i], -i))
    return dict(epoch=int(t), window=list(range(t - window + 1, t + 1)), score=float(sm[t]))


def should_stop(history, cfg) -> tuple:
    """(dung?, ly do). Kieu Keras tren diem truot: cai thien khi > best + min_delta."""
    n = len(history)
    if n >= cfg.max_epochs:
        return True, f"du max_epochs={cfg.max_epochs}"
    if n < cfg.min_epochs:
        return False, ""
    best, wait = -np.inf, 0
    for s in trailing_means(_val_qwk(history), cfg.window):
        if s is None:
            continue
        if s > best + cfg.min_delta:
            best, wait = s, 0
        else:
            wait += 1
    if wait >= cfg.patience:
        return True, f"{wait} epoch khong vuot {best:.4f} + {cfg.min_delta} (diem truot {cfg.window})"
    return False, ""


# ============================================================ checkpoint + ghi mot lan

_PLAIN = (str, int, float, bool, type(None))


def _assert_plain(obj, path="state") -> None:
    """Chi tensor + kieu Python thuan (KIEU CHINH XAC, khong lop con) => torch.load(weights_only=
    True) nap duoc. Chan np.float64 (lop con cua float) va TorchVersion (lop con cua str)."""
    if type(obj) in (dict,) or type(obj).__name__ == "OrderedDict":
        for k, v in obj.items():
            if type(k) not in (str, int):
                raise TypeError(f"{path}: khoa {k!r} kieu {type(k)}")
            _assert_plain(v, f"{path}[{k!r}]")
    elif type(obj) in (list, tuple):
        for i, v in enumerate(obj):
            _assert_plain(v, f"{path}[{i}]")
    elif not (type(obj) in _PLAIN or isinstance(obj, torch.Tensor)):
        raise TypeError(f"{path}: kieu {type(obj)} khong nap duoc voi weights_only=True")


def save_checkpoint(run_dir, state: dict) -> str:
    """last.pt nguyen tu: ghi .tmp -> last.pt cu thanh last_prev.pt -> .tmp thanh last.pt."""
    run_dir = os.fspath(run_dir)
    assert_drive_first(run_dir)
    _assert_plain(state)
    path = os.path.join(run_dir, LAST)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    if os.path.exists(path):
        os.replace(path, os.path.join(run_dir, PREV))
    os.replace(tmp, path)
    return path


def load_checkpoint(run_dir, log=print):
    """-> (state, ten file) hoac (None, None). last.pt hong => thu last_prev.pt (co bao)."""
    for name in (LAST, PREV):
        p = os.path.join(os.fspath(run_dir), name)
        if os.path.exists(p):
            try:
                return torch.load(p, map_location="cpu", weights_only=True), name
            except Exception as e:                        # file ghi do dang khi mat phien
                log(f"CANH BAO: khong nap duoc {name}: {type(e).__name__}: {e}")
    return None, None


def write_once_text(text: str, path) -> str:
    """Ghi mot lan. Da ton tai: noi dung GIONG => bo qua; KHAC => FileExistsError (khong ghi de)."""
    path = os.fspath(path)
    assert_drive_first(path)
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as f:
            if f.read() != text:
                raise FileExistsError(f"{path} da ton tai voi noi dung KHAC - khong ghi de")
        return path
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def write_once_csv(df, path, **kw) -> str:
    return write_once_text(df.to_csv(index=False, lineterminator="\n", **kw), path)


def write_once_json(obj, path) -> str:
    return write_once_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", path)


# ============================================================ fit

def _loaders(train_ds, val_ds, cfg, device, generator):
    pin = str(device).startswith("cuda")
    tr = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    pin_memory=pin, generator=generator, worker_init_fn=_worker_init)
    va = None if val_ds is None else DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=pin, worker_init_fn=_worker_init)
    return tr, va


def _set_backends(cfg) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)
    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)


def fit(model, train_ds, val_ds, run_dir, cfg: M3TTrainConfig, pool_md5: str, device="cuda",
        deadline=None, max_new_epochs=None, log=print) -> list:
    """Train/chay tiep toi khi should_stop, file STOP, het `deadline` (time.time()) hoac du
    `max_new_epochs` trong phien nay. Tra history. Goi lai o phien sau de chay tiep."""
    run_dir = os.fspath(run_dir)
    assert_drive_first(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    _set_backends(cfg)
    versions = runtime_versions()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    state, src = load_checkpoint(run_dir, log)
    if state is None:
        left = [n for n in sorted(os.listdir(run_dir))
                if n in (LAST, PREV) or (n.startswith("epoch_") and n.endswith(".pt"))]
        if left:
            raise RuntimeError(f"{run_dir} co {left[:3]}... nhung khong nap duoc checkpoint nao - "
                               "KHONG train lai tu dau de ghi de; kiem tra file tren Drive")
        history = []
        write_once_json(dict(cfg=cfg.to_dict(), cfg_hash=cfg.hash(), pool_md5=pool_md5, versions=versions),
                        os.path.join(run_dir, "run_config.json"))
    else:
        bad = [k for k, a, b in (("cfg_hash", state["cfg_hash"], cfg.hash()),
                                 ("pool_md5", state["pool_md5"], pool_md5),
                                 ("torchio", state["versions"].get("torchio"), versions["torchio"])) if a != b]
        if bad:
            raise ValueError(f"tu choi chay tiep: {bad} khac checkpoint trong {run_dir}")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = list(state["history"])
        if state["epoch"] != len(history) - 1:
            raise ValueError("checkpoint hong: epoch khong khop do dai history")
        log(f"chay tiep tu epoch {len(history)} ({src})")
    g = torch.Generator()
    train_loader, val_loader = _loaders(train_ds, val_ds, cfg, device, g)
    n_new = 0
    while True:
        stop, why = should_stop(history, cfg)
        if stop:
            log(f"DUNG: {why}")
            break
        if max_new_epochs is not None and n_new >= max_new_epochs:
            log(f"du {max_new_epochs} epoch cho phien nay")
            break
        if os.path.exists(os.path.join(run_dir, "STOP")):
            log("DUNG: co file STOP")
            break
        if deadline is not None and history:
            est = float(np.median([h["seconds"] for h in history[-5:]]))
            if time.time() + 1.1 * est > deadline:
                log(f"DUNG: khong du gio cho mot epoch (~{est / 60:.0f} phut)")
                break
        epoch = len(history)
        seed_everything(epoch_seed(cfg.seed, epoch), g)
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, device)
        ev = evaluate(model, val_loader, device)
        history.append(dict(epoch=epoch, **tr, val_loss=ev["loss"], val_acc=ev["acc"], val_qwk=ev["qwk"],
                            val_macro_recall=ev["macro_recall"], val_recall=ev["recall"],
                            seconds=time.time() - t0))
        sd = model.state_dict()
        torch.save(sd, os.path.join(run_dir, f"epoch_{epoch:03d}.pt"))
        save_checkpoint(run_dir, dict(epoch=epoch, model_state_dict=sd, optimizer_state_dict=optimizer.state_dict(),
                                      history=history, cfg=cfg.to_dict(), cfg_hash=cfg.hash(),
                                      pool_md5=pool_md5, versions=versions))
        n_new += 1
        sm = trailing_means([h["val_qwk"] for h in history], cfg.window)[-1]
        log(f"epoch {epoch}: train loss {tr['train_loss']:.4f} acc {tr['train_acc']:.3f} | val acc "
            f"{ev['acc']:.3f} qwk {ev['qwk']:.3f} truot {'-' if sm is None else f'{sm:.3f}'} | "
            f"{(time.time() - t0) / 60:.1f} phut")
    return history


def benchmark(model, train_ds, val_ds, cfg: M3TTrainConfig, device="cuda", n_steps: int = 100) -> dict:
    """Do thoi gian THAT truoc khi cam ket GPU, tren BAN SAO mo hinh (khong dong vao run that).

    data_s  : moi buoc chi doc + tang cuong (DataLoader, khong mo hinh) - nut that CPU.
    train_s : moi buoc train day du (doc + forward + backward + Adam), co dong bo GPU.
    eval_s  : moi buoc danh gia val.
    """
    import copy
    _set_backends(cfg)
    m = copy.deepcopy(model).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
    g = torch.Generator()
    seed_everything(epoch_seed(cfg.seed, 10 ** 6), g)      # seed rieng, khong trung epoch nao
    tr_loader, va_loader = _loaders(train_ds, val_ds, cfg, device, g)
    n = max(1, min(n_steps, len(tr_loader) - 1))

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    it = iter(tr_loader)
    next(it)                                                 # khoi dong worker
    t = time.time()
    for _ in range(n):
        next(it)
    data_s = (time.time() - t) / n
    del it
    it = iter(tr_loader)
    m.train()

    def step(x, y):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()

    step(*next(it))                                          # khoi dong cudnn.benchmark
    sync()
    t = time.time()
    for _ in range(n):
        step(*next(it))
    sync()
    train_s = (time.time() - t) / n
    del it
    m.eval()
    nv = max(1, min(n, len(va_loader)))
    t = time.time()
    with torch.no_grad():
        for i, (x, _) in enumerate(va_loader):
            if i >= nv:
                break
            m(x.to(device, non_blocking=True))
    sync()
    eval_s = (time.time() - t) / nv
    epoch_min = (train_s * len(tr_loader) + eval_s * len(va_loader)) / 60
    return dict(n_steps=n, data_s_per_step=data_s, train_s_per_step=train_s, eval_s_per_step=eval_s,
                steps_per_epoch=len(tr_loader), est_epoch_min=epoch_min,
                est_hours_min_epochs=epoch_min * cfg.min_epochs / 60,
                est_hours_max_epochs=epoch_min * cfg.max_epochs / 60,
                cpu_bound=bool(data_s > 0.8 * train_s), n_cpu=os.cpu_count())
