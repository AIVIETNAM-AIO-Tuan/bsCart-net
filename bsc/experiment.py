"""Truc cau hinh thi nghiem + run registry cho Stage 1.

Sao y `p3_m8d_standardization_decision_vi.md`. Doc do chot 5 quyet dinh, module nay
thi hanh chung bang code de khong ai lech lai duoc:

  QD1: P3 va M8-D KHONG phai hai thi nghiem doc lap. P3 la RUN; M8-D la VAI TRO
       ablation cua chinh run do.
  QD2: chi train va luu MOT cau hinh canonical - khong tao checkpoint M8-D rieng.
  QD3: dung ENSEMBLE TARGET-CARTILAGE PROBABILITY, khong dung raw logits.
  QD4: moi ablation chi thay DUNG MOT yeu to.
  QD5: P-series = run registry; M-series = nhom phan tich.

VI SAO CAN MODULE RIENG
-----------------------
Truoc do bo kenh duoc viet tay trong model.py va da lech khoi plan (P2/P3 sai kenh,
P2-P5 dung Oracle thay vi Atlas). Dat cac TRUC thanh du lieu + kiem tra tu dong thi
loai duoc ca lop loi do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------- cac truc (§6)

#: S - nguon be mat xuong
SURFACE = {
    "S0": "gt_bone",              # be mat xuong ground-truth
    "S1": "gt_bone_jitter",       # GT + nhieu loan co kiem soat
    "S2": "predicted_bone",       # be mat xuong du doan
}

#: D - mien khop
DOMAIN = {
    "D0": "oracle",               # suy tu vung bam sun GT
    "D1": "fold_atlas",           # atlas quan the theo tung fold
}

#: I - kenh dau vao. "prob" = ENSEMBLE target-cartilage probability (QD3),
#: KHONG phai raw logit. Xem §4 cua doc: probability va logit KHAC nhau.
INPUT = {
    "I0": ("mri",),
    "I1": ("mri", "grad"),
    "I2": ("mri", "sdf"),
    "I3": ("mri", "prob"),
    "I4": ("mri", "feat"),
    "I5": ("mri", "prob", "feat"),
}

#: H - dau ra
HEADS = {
    "H0": ("occupancy",),
    "H1": ("occupancy", "presence"),
}

#: Nguon hop le cho kenh "prob" (§12 doi ghi ro, cam ghi chung chung "coarse prior").
#:
#: QUYET DINH (2026-07-19): MVP dung `resenc_150ep_oof_softmax`.
#: Ly do - doc §5.1 de xuat trung binh 5 fold, nhung tren ca TRAIN/VAL thi 4/5 model da
#: HUAN LUYEN tren chinh ca do => xac suat qua lac quan => RO RI nhan vao kenh dau vao.
#: Mo hinh tia se hoc cach tin kenh do, roi sup khi gap xac suat that o test.
#: Out-of-fold (dung fold ma ca do la validation) sach ve mat nay, doi lai la 1 model
#: chu khong phai ensemble. Chap nhan danh doi do o MVP.
#: `resenc_5fold_mean_softmax` CHI hop le tren tap test da giu kin - dung o Gate 4.
PROB_SOURCES = {
    "resenc_150ep_oof_softmax": "softmax out-of-fold, fold ma ca do lam validation",
    "resenc_5fold_mean_softmax": "trung binh 5 fold - CHI dung cho tap test giu kin",
}
DEFAULT_PROB_SOURCE = "resenc_150ep_oof_softmax"

#: Ma tran chuan hoa §7. P3 = S0-D1-I3-H1 = dung bang M8-D.
PLAN_MATRIX = {
    "P0": ("S0", "D0", "I0", "H0"),
    "P1": ("S0", "D0", "I1", "H0"),
    "P2": ("S0", "D1", "I2", "H1"),
    "P3": ("S0", "D1", "I3", "H1"),
    # P4/P5 dung "best input" - xac dinh bang thuc nghiem, dien luc chay
    "P4": ("S1", "D1", None, "H1"),
    "P5": ("S2", "D1", None, "H1"),
}

#: M8 la NHOM PHAN TICH duoi mot scaffold co dinh S0+D1+H1 (§8), khong phai run rieng.
M8_SCAFFOLD = ("S0", "D1", "H1")
M8_INPUTS = {"M8-A": "I0", "M8-B": "I1", "M8-C": "I2", "M8-D": "I3", "M8-E": "I4"}


def m8_role(surface: str, domain: str, inputs: str, heads: str) -> "str | None":
    """Run nay dong vai tro M8 nao? None neu khong nam trong scaffold M8.

    Nho ham nay ma M8-D duoc SUY RA tu P3 thay vi train lai (QD1, QD2).
    """
    if (surface, domain, heads) != M8_SCAFFOLD:
        return None
    for role, i in M8_INPUTS.items():
        if i == inputs:
            return role
    return None


# ------------------------------------------------------------ cau hinh mot run

@dataclass(frozen=True)
class RunConfig:
    """Mot run cu the. `experiment_id` la ID ky thuat duy nhat (§11)."""
    cls: str                      # "femoral_cart" | "med_tib_cart"
    surface: str = "S0"
    domain: str = "D1"
    inputs: str = "I2"
    heads: str = "H1"
    fold: int = 0
    seed: int = 1
    version: str = "v1"
    plan_alias: "str | None" = None       # "P2", "P3", ... neu khop ma tran
    prob_source: "str | None" = None      # mo ta nguon probability (bat buoc neu I co "prob")
    notes: str = ""

    #: Viet tat lop dung trong ten run
    _ABBR = {"femoral_cart": "FC", "med_tib_cart": "MTC", "lat_tib_cart": "LTC"}

    def __post_init__(self):
        for name, table in (("surface", SURFACE), ("domain", DOMAIN),
                            ("inputs", INPUT), ("heads", HEADS)):
            v = getattr(self, name)
            if v not in table:
                raise ValueError(f"{name}={v!r} khong hop le; chon {sorted(table)}")
        if self.cls not in self._ABBR:
            raise ValueError(f"cls={self.cls!r} khong hop le; chon {sorted(self._ABBR)}")
        # QD3 + §12: neu dung probability thi PHAI ghi ro nguon, khong duoc mo ho
        if "prob" in self.channels:
            if not self.prob_source:
                raise ValueError(
                    f"inputs co kenh 'prob' => phai ghi `prob_source`; chon "
                    f"{sorted(PROB_SOURCES)}. §12: cam ghi chung chung 'coarse prior'."
                )
            if self.prob_source not in PROB_SOURCES:
                raise ValueError(
                    f"prob_source={self.prob_source!r} khong nam trong danh sach da chot; "
                    f"chon {sorted(PROB_SOURCES)}. Them nguon moi thi phai cap nhat "
                    f"PROB_SOURCES kem ly do."
                )

    @property
    def channels(self) -> tuple:
        return INPUT[self.inputs]

    @property
    def with_presence(self) -> bool:
        return "presence" in HEADS[self.heads]

    @property
    def domain_mode(self) -> str:
        return DOMAIN[self.domain]

    @property
    def m8_role(self) -> "str | None":
        return m8_role(self.surface, self.domain, self.inputs, self.heads)

    @property
    def experiment_id(self) -> str:
        """vd MVP_MTC_S0_D1_I3_H1_v1_Fold0_Seed1  (§11)."""
        return (f"MVP_{self._ABBR[self.cls]}_{self.surface}_{self.domain}_"
                f"{self.inputs}_{self.heads}_{self.version}_"
                f"Fold{self.fold}_Seed{self.seed}")

    def to_registry(self, code_commit: str = "", dataset_revision: str = "",
                    extra: "dict | None" = None) -> dict:
        """Ban ghi audit trail theo §12. Luu kem MOI checkpoint."""
        rec = {
            "experiment_id": self.experiment_id,
            "plan_alias": self.plan_alias,
            "ablation_role": self.m8_role,
            "class": self.cls,
            "surface_source": SURFACE[self.surface],
            "articular_domain": DOMAIN[self.domain],
            "inputs": list(self.channels),
            "outputs": list(HEADS[self.heads]),
            "fold": self.fold,
            "seed": self.seed,
            "version": self.version,
            "code_commit": code_commit,
            "dataset_revision": dataset_revision,
            "notes": self.notes,
        }
        if "prob" in self.channels:
            rec["resenc_prior"] = {"representation": "probability",
                                   "source": self.prob_source}
        if extra:
            rec.update(extra)
        return rec


def from_plan(alias: str, cls: str, inputs: "str | None" = None, **kw) -> RunConfig:
    """Dung RunConfig tu ma tran §7. P4/P5 phai truyen `inputs` (best input)."""
    if alias not in PLAN_MATRIX:
        raise ValueError(f"alias {alias!r} khong co; chon {sorted(PLAN_MATRIX)}")
    s, d, i, h = PLAN_MATRIX[alias]
    if i is None:
        if inputs is None:
            raise ValueError(f"{alias} dung 'best input' => phai truyen inputs=...")
        i = inputs
    elif inputs is not None and inputs != i:
        raise ValueError(f"{alias} co dinh inputs={i}, khong the doi thanh {inputs}")
    return RunConfig(cls=cls, surface=s, domain=d, inputs=i, heads=h,
                     plan_alias=alias, **kw)


# ------------------------------------------------- kiem tra ablation hop le (QD4)

def diff_axes(a: RunConfig, b: RunConfig) -> list:
    """Cac truc khac nhau giua hai run."""
    return [n for n in ("surface", "domain", "inputs", "heads")
            if getattr(a, n) != getattr(b, n)]


def assert_single_factor(a: RunConfig, b: RunConfig, axis: str) -> None:
    """QD4: mot cap so sanh ablation chi duoc thay DUNG MOT yeu to.

    §3.2 cua doc: neu doi dong thoi domain + presence + input thi KHONG the quy cai
    thien cho bat ky yeu to nao. Ham nay chan lop loi do ngay tai cho goi.
    """
    d = diff_axes(a, b)
    if d != [axis]:
        raise ValueError(
            f"So sanh khong hop le: muon co lap {axis!r} nhung khac o {d}.\n"
            f"  {a.experiment_id}\n  {b.experiment_id}\n"
            f"Xem §3.2 + QD4: moi ablation chi thay dung mot yeu to."
        )
