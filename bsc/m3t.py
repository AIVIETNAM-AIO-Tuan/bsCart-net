"""M3T (3D CNN + 2D CNN + transformer) - SUY LUAN, chuyen doi anh, dau van tay anh.

Nguon: `knee_testing_v3.ipynb` cell 11-18 (mo hinh), cell 20 (cau hinh that). Cac lop mo hinh
duoi day CHEP NGUYEN VAN, chi them `forward_features` (token CLS) - `forward` = fc(CLS), cung
phep tinh voi ban goc (test_m3t khoa bang so voi ban sao nguyen van cua forward cu).
Ten thuoc tinh giu nguyen => state_dict cua trong so cu nap `strict=True`.

Module nay KHONG import torchio/nibabel: suy luan chi can min-max tung khoi ve [0,1], dung
bang `tio.RescaleIntensity(out_min_max=(0, 1))` cua val/test transform goc.

TRONG SO RO RI
--------------
M3T goc train tren 2880 subject OAI, trong do co subject cua cohort 1325 ca cua minh (44/70
subject iMorphics nam o tap train cua no) => CLS cua trong so do DA MANG nhan KL cua cohort.
`LEAKY_HASHES` = hash NOI DUNG (sha256 tren tensor state_dict) cua ca ba file trong so cu.
`load_m3t` va `extract` tu choi chung tru khi goi ro `allow_leaky=True` - chi duoc dung cho
kiem port (G0) va thiet ke chuyen doi anh (khong dung nhan), KHONG BAO GIO de trich dac trung.

CHUYEN DOI NIfTI -> M3T
-----------------------
npz cua M3T (120,160,160) uint16 sinh bang mot pipeline thuong nguon KHONG ro. Anh cohort la
NIfTI (load_nii -> [Z,Y,X], truc 0.70 mm nam CUOI). Ham o day chi dinh nghia khong gian tim:
huong (hoan vi truc + lat) x kieu resize. Chon phuong an bang tuong quan voxel voi npz cua
CHINH ca do (khop barcode/subject) - khong dung nhan KL. Cong G4.1-G4.3 o notebook S9, nguong
nam MOT cho la configs/m3t_s9.json['gates'] (dang ky truoc), ham o day nhan nguong lam tham so.
"""

from __future__ import annotations

import hashlib
import itertools
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Cau hinh that cua cell 20: M3TModelFull(C3d=32, N=20, emb_dim=128, C2d=128, num_classes=5,
# target_size=160). Doi cau hinh => doi mo hinh => khong nap duoc trong so cu.
M3T_CFG = dict(C3d=32, N=20, emb_dim=128, C2d=128, num_classes=5, target_size=160)
INPUT_SHAPE = (120, 160, 160)
N_PARAMS = 1_160_677
CLS_DIM = 128
CLS_PREFIX = "m3t_"
LOGIT_PREFIX = "m3tlogit_"
PROV_PREFIX = "prov_"

#: sha256 noi dung state_dict (weights_hash) cua trong so M3T train tren split GOC - RO RI.
#: Do tu file cuc bo ngay 25/09/2026 (md5 file trong ngoac):
#:   nnUnet-OAI/best_model.pth   [BEST], acc test 0.6559 = 1073/1636          (e4ca502e)
#:   nnUnet-OAI/checkpoint.pth   'epoch': 248 (0-based), model_state_dict     (a82cfe39)
#:   Downloads/best_model.pth    ban train 113 epoch                          (fa8184af)
#: File ro ri khac (vd ban tai lai tu Drive) co hash moi: notebook S9 ghi them vao
#: `leaky_hashes_seen.json` cua run va kiem ca danh sach do truoc khi trich CLS.
LEAKY_HASHES = frozenset({
    "13b3b7839434133ee03cda4c494911527d7076469e7b42f63a6263e80b2e538d",
    "e41ddb2f349dbd75316accc790389aabafe6a6fd5cfd75430ddabb3bf7aa77e3",
    "d99915f00c5a0466b019551ec07cd2088589ff8626a4c89a287ac5f1f31fe4a5",
})


# ============================================================ mo hinh (cell 11-18, nguyen van)

class D3DBlock(nn.Module):
    """
    3D CNN block from M3T:
    - Two 5x5x5 conv layers
    - BN + ReLU
    - Output: X ∈ (C3d x L x W x H)
    """

    def __init__(self, in_channels=1, C3d=32):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, C3d, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(C3d)

        self.conv2 = nn.Conv3d(C3d, C3d, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm3d(C3d)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x  # shape: (C3d, L, W, H)


class MultiPlaneExtractor(nn.Module):
    def __init__(self, N=20, target_size=None):
        super().__init__()
        self.N = N
        self.target_size = target_size  # single int

    def extract_slices_batch(self, X):
        B, C, L, W, H = X.shape
        device = X.device
        indices_L = torch.linspace(0, L-1, self.N, device=device).long()
        indices_W = torch.linspace(0, W-1, self.N, device=device).long()
        indices_H = torch.linspace(0, H-1, self.N, device=device).long()

        Scor = X[:, :, indices_L, :, :].permute(0, 2, 1, 3, 4)
        Ssag = X[:, :, :, indices_W, :].permute(0, 3, 1, 2, 4)
        Sax  = X[:, :, :, :, indices_H].permute(0, 4, 1, 2, 3)

        if self.target_size is not None:
            Scor = F.interpolate(Scor.reshape(B*self.N, C, Scor.shape[-2], Scor.shape[-1]),
                                size=(self.target_size, self.target_size),
                                mode='bilinear', align_corners=False).reshape(B, self.N, C, self.target_size, self.target_size)
            Ssag = F.interpolate(Ssag.reshape(B*self.N, C, Ssag.shape[-2], Ssag.shape[-1]),
                                size=(self.target_size, self.target_size),
                                mode='bilinear', align_corners=False).reshape(B, self.N, C, self.target_size, self.target_size)
            Sax  = F.interpolate(Sax.reshape(B*self.N, C, Sax.shape[-2], Sax.shape[-1]),
                                size=(self.target_size, self.target_size),
                                mode='bilinear', align_corners=False).reshape(B, self.N, C, self.target_size, self.target_size)

        S = torch.cat([Scor, Ssag, Sax], dim=1)
        return S

    def forward(self, X):
        return self.extract_slices_batch(X)


class D2DBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=128):
        super().__init__()
        # Simple 2-layer CNN
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B*3N, C3d, L, L)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        # Global average pooling -> (B*3N, C2d)
        x = x.mean(dim=[2,3])
        return x


class NonLinearProjection(nn.Module):
    def __init__(self, in_dim=128, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # x: (B*3N, C2d)
        return self.mlp(x)  # (B*3N, d)


class PosPlaneEmbedding(nn.Module):
    def __init__(self, N, d):
        super().__init__()
        self.N = N
        self.d = d

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, d))

        # Positional and plane embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, 3*N + 4, d))
        self.plane_embedding = nn.Parameter(torch.zeros(1, 3*N + 4, d))

    def forward(self, Tcor, Tsag, Tax):
        # Tcor/Tsag/Tax: (B, N, d)
        B = Tcor.shape[0]
        sep = self.sep_token.expand(B, -1, -1)

        # Concatenate: [cls, Tcor, sep, Tsag, sep, Tax, sep]
        Z0 = torch.cat([self.cls_token.expand(B, -1, -1),
                        Tcor, sep,
                        Tsag, sep,
                        Tax, sep], dim=1)  # (B, 3N+4, d)

        # Add embeddings
        Z0 = Z0 + self.pos_embedding + self.plane_embedding
        return Z0


class TransformerBlock(nn.Module):
    def __init__(self, d=128, num_heads=4, num_layers=4, mlp_ratio=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=num_heads, dim_feedforward=d*mlp_ratio, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, 3N+4, d)
        return self.transformer(x)


class M3TModelFull(nn.Module):
    def __init__(self, C3d=32, N=20, emb_dim=128, C2d=128, num_classes=2, target_size=None):
        super().__init__()
        self.d3d = D3DBlock(in_channels=1, C3d=C3d)
        self.extractor = MultiPlaneExtractor(N=N, target_size=target_size)
        self.d2d = D2DBlock(in_channels=C3d, out_channels=C2d)
        self.projection = NonLinearProjection(in_dim=C2d, out_dim=emb_dim)
        self.pos_plane = PosPlaneEmbedding(N=N, d=emb_dim)
        self.transformer = TransformerBlock(d=emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes)
        self.N = N

    def forward_features(self, x):
        """Token CLS (B, emb_dim) - than cua forward goc tu dau toi `Z[:, 0, :]`."""
        B = x.shape[0]
        X = self.d3d(x)                     # (B, C3d, L, W, H)
        S = self.extractor(X)               # (B, 3N, C3d, L, L)

        # Merge batch and slices -> (B*3N, C3d, L, L)
        B, T, C, L_s, _ = S.shape
        S_flat = S.reshape(B*T, C, L_s, L_s)

        K = self.d2d(S_flat)                # (B*3N, C2d)
        T_proj = self.projection(K)         # (B*3N, d)

        # Split planes
        T_proj = T_proj.reshape(B, 3*self.N, -1)
        Tcor = T_proj[:, :self.N, :]
        Tsag = T_proj[:, self.N:2*self.N, :]
        Tax  = T_proj[:, 2*self.N:, :]

        # Add CLS, SEP, embeddings
        Z0 = self.pos_plane(Tcor, Tsag, Tax) # (B, 3N+4, d)

        # Transformer
        Z = self.transformer(Z0)             # (B, 3N+4, d)

        # Classification token
        cls_token = Z[:, 0, :]               # (B, d)
        return cls_token

    def forward(self, x):
        cls_token = self.forward_features(x)
        logits = self.fc(cls_token)          # (B, num_classes)
        return logits


# ============================================================ dung / nap / hash

def build_m3t(**overrides) -> M3TModelFull:
    """M3T voi cau hinh that (M3T_CFG), sua tung khoa qua overrides (test dung cau hinh nho)."""
    unknown = set(overrides) - set(M3T_CFG)
    if unknown:
        raise ValueError(f"khoa cau hinh khong ton tai: {sorted(unknown)}")
    return M3TModelFull(**{**M3T_CFG, **overrides})


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def weights_hash(state_dict) -> str:
    """sha256 NOI DUNG: ten, dtype, shape va byte cua moi tensor, theo ten da sap xep.

    Khac md5 cua file: cung trong so luu theo hai cach (state_dict tho / boc trong
    checkpoint) cho cung hash, nen chan duoc ca ban da boc lai cua trong so ro ri.
    """
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k]
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"state_dict[{k!r}] khong phai tensor: {type(t)}")
        a = t.detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(a.numpy().tobytes())
    return h.hexdigest()


def unwrap_state_dict(obj) -> dict:
    """state_dict tho, hoac dict checkpoint co khoa 'model_state_dict' (cell 21 / m3t_train)."""
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    if not isinstance(obj, dict) or not obj or not all(isinstance(v, torch.Tensor) for v in obj.values()):
        raise ValueError("khong nhan ra state_dict: can dict ten -> tensor hoac {'model_state_dict': ...}")
    return obj


def assert_not_leaky(h: str, allow_leaky: bool = False) -> None:
    if h in LEAKY_HASHES and not allow_leaky:
        raise PermissionError(
            f"trong so {h[:12]} la M3T train tren split GOC (ro ri nhan KL cua cohort). "
            "Chi dung cho kiem port / thiet ke chuyen doi, goi voi allow_leaky=True."
        )


def load_m3t(path, device="cpu", allow_leaky: bool = False, **overrides):
    """Nap trong so (weights_only=True, strict) -> (model.eval(), weights_hash).

    Gan `model.weights_hash` de `extract` kiem lai; trong so ro ri bi chan tru khi
    allow_leaky=True.
    """
    obj = torch.load(path, map_location="cpu", weights_only=True)
    sd = unwrap_state_dict(obj)
    h = weights_hash(sd)
    assert_not_leaky(h, allow_leaky)
    model = build_m3t(**overrides)
    model.load_state_dict(sd, strict=True)
    model.weights_hash = h
    return model.to(device).eval(), h


# ============================================================ tien xu ly + trich CLS

def minmax01(vol) -> np.ndarray:
    """Min-max CA KHOI ve [0,1], float32 - bang tio.RescaleIntensity(out_min_max=(0,1)).

    Cung thu tu phep tinh voi torchio (mang float32, tru/chia tai cho bang so float64).
    Khac torchio o MOT ca bien: khoi hang -> 0 (torchio canh bao va tra khoi tho).
    NaN/inf => loi (anh hong khong duoc lot vao CLS).
    """
    a = np.array(vol, dtype=np.float32, copy=True)
    if a.size == 0:
        raise ValueError("khoi rong")
    if not np.isfinite(a).all():
        raise ValueError("khoi co NaN/inf")
    lo, hi = np.float64(a.min()), np.float64(a.max())
    if hi <= lo:
        return np.zeros_like(a)
    a -= lo
    a /= (hi - lo)
    return a


def extract(model: M3TModelFull, arrays, device="cpu", bs: int = 2, allow_leaky: bool = False,
            expect_shape=INPUT_SHAPE, progress=None):
    """Khoi THO (chua chuan hoa) -> (cls [N, emb] float32, logits [N, K] float32).

    `arrays`: iterable cac mang 3D (co the la generator doc lan luot tu zip). Tu ap minmax01
    - khong truyen anh da chuan hoa hai lan. Mo hinh phai o eval() (BatchNorm dung thong ke
    chay; o train() CLS phu thuoc ca batch).
    """
    if model.training:
        raise RuntimeError("model dang o train() - goi model.eval() truoc khi trich CLS")
    assert_not_leaky(getattr(model, "weights_hash", ""), allow_leaky)
    cls_out, logit_out, batch = [], [], []

    def flush():
        x = torch.from_numpy(np.stack(batch)[:, None]).to(device)
        with torch.no_grad():
            f = model.forward_features(x)
            lg = model.fc(f)
        cls_out.append(f.float().cpu().numpy())
        logit_out.append(lg.float().cpu().numpy())
        batch.clear()

    n = 0
    for vol in arrays:
        v = minmax01(vol)
        if expect_shape is not None and tuple(v.shape) != tuple(expect_shape):
            raise ValueError(f"khoi thu {n} shape {v.shape}, can {tuple(expect_shape)}")
        batch.append(v)
        n += 1
        if len(batch) == bs:
            flush()
            if progress is not None:
                progress(n)
    if batch:
        flush()
    if not cls_out:
        raise ValueError("khong co khoi nao")
    return np.concatenate(cls_out), np.concatenate(logit_out)


def softmax_np(logits) -> np.ndarray:
    z = np.asarray(logits, np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_kl(logits) -> np.ndarray:
    """E[KL] = sum_k k * softmax_k (dung cho cong G4.3)."""
    p = softmax_np(logits)
    return p @ np.arange(p.shape[1], dtype=np.float64)


# ============================================================ bang CLS

_CLS_RE = re.compile(rf"^{CLS_PREFIX}\d{{3}}$")
_LOGIT_RE = re.compile(rf"^{LOGIT_PREFIX}\d$")


def cls_cols(dim: int = CLS_DIM) -> list:
    return [f"{CLS_PREFIX}{i:03d}" for i in range(dim)]


def is_cls_col(c: str) -> bool:
    """Cot dac trung CLS (m3t_000..m3t_127). KHONG khop logit hay cot provenance."""
    return bool(_CLS_RE.match(str(c)))


def is_logit_col(c: str) -> bool:
    return bool(_LOGIT_RE.match(str(c)))


def cls_frame(ids, cls, logits, prov: dict):
    """DataFrame: case_id, m3t_000.., m3tlogit_0.., prov_<k> (hang so moi dong).

    Chi `m3t_\\d{3}` la dac trung. Provenance mang tien to prov_ de khong bao gio bi chon
    nham theo tien to "m3t_". Kiem: case_id duy nhat, khong NaN, khong chieu hang.
    """
    import pandas as pd
    ids = [str(i) for i in ids]
    cls = np.asarray(cls, np.float64)
    logits = np.asarray(logits, np.float64)
    if cls.ndim != 2 or logits.ndim != 2 or not (len(ids) == len(cls) == len(logits)):
        raise ValueError(f"shape khong khop: ids {len(ids)}, cls {cls.shape}, logits {logits.shape}")
    if len(set(ids)) != len(ids):
        raise ValueError("case_id trung")
    if not (np.isfinite(cls).all() and np.isfinite(logits).all()):
        raise ValueError("CLS/logit co NaN/inf")
    if len(ids) > 1 and (cls.std(axis=0) == 0).any():
        raise ValueError(f"{int((cls.std(axis=0) == 0).sum())} chieu CLS hang so")
    df = pd.DataFrame({"case_id": ids})
    feat = pd.DataFrame(cls, columns=cls_cols(cls.shape[1]))
    lg = pd.DataFrame(logits, columns=[f"{LOGIT_PREFIX}{k}" for k in range(logits.shape[1])])
    meta = pd.DataFrame({f"{PROV_PREFIX}{k}": [v] * len(ids) for k, v in prov.items()})
    return pd.concat([df, feat, lg, meta], axis=1)


# ============================================================ chuyen doi NIfTI -> M3T

def _orient_name(perm, flips) -> str:
    return "p" + "".join(map(str, perm)) + "_f" + "".join("1" if a in flips else "0" for a in range(3))


#: 48 huong = 6 hoan vi truc x 8 to hop lat. out = flip(transpose(src, perm), flips);
#: truc i cua out = truc perm[i] cua src.
ORIENTATIONS = {
    _orient_name(p, f): (tuple(p), tuple(f))
    for p in itertools.permutations(range(3))
    for k in range(4) for f in itertools.combinations(range(3), k)
}


def slice_axis(spacing, tol: float = 0.05) -> int:
    """Truc co spacing LON NHAT (0.70 mm cua DESS). Loi neu khong duy nhat."""
    s = np.asarray(spacing, np.float64)
    ax = int(np.argmax(s))
    if (np.abs(s - s[ax]) < tol).sum() != 1:
        raise ValueError(f"khong xac dinh duoc truc lat cat duy nhat tu spacing {tuple(s)}")
    return ax


def orientations_for(src_slice_axis: int, dst_axis: int = 0) -> dict:
    """16 huong dua truc lat cat cua src ve truc `dst_axis` cua M3T (2 hoan vi x 8 lat)."""
    return {k: v for k, v in ORIENTATIONS.items() if v[0][dst_axis] == src_slice_axis}


def dst_axis_of(orient: str, src_slice_axis: int) -> int:
    """Truc M3T ma truc lat cat cua src roi vao duoi huong `orient`."""
    return list(ORIENTATIONS[orient][0]).index(int(src_slice_axis))


def reorient(vol, orient) -> np.ndarray:
    perm, flips = ORIENTATIONS[orient] if isinstance(orient, str) else orient
    out = np.transpose(np.asarray(vol), perm)
    return np.flip(out, axis=flips) if flips else out


RESIZE_METHODS = ("trilinear", "trilinear_ac", "area", "nearest", "zoom1")


def resize3d(vol, shape, method: str = "trilinear") -> np.ndarray:
    """Resize khoi 3D ve `shape` (float32). Moi kieu la mot gia thuyet ve pipeline thuong nguon."""
    shape = tuple(int(s) for s in shape)
    a = np.asarray(vol, np.float32)
    if a.ndim != 3 or len(shape) != 3:
        raise ValueError(f"can khoi 3D va shape 3 truc, nhan {a.shape} -> {shape}")
    if method == "zoom1":
        from scipy.ndimage import zoom
        out = zoom(a, [t / s for t, s in zip(shape, a.shape)], order=1, grid_mode=False)
        if out.shape != shape:                      # zoom lam tron kich thuoc -> cat/dem ve dung
            out = out[:shape[0], :shape[1], :shape[2]]
            out = np.pad(out, [(0, t - s) for t, s in zip(shape, out.shape)], mode="edge")
        return out.astype(np.float32)
    x = torch.from_numpy(np.ascontiguousarray(a))[None, None]
    if method == "trilinear":
        y = F.interpolate(x, size=shape, mode="trilinear", align_corners=False)
    elif method == "trilinear_ac":
        y = F.interpolate(x, size=shape, mode="trilinear", align_corners=True)
    elif method == "area":
        y = F.interpolate(x, size=shape, mode="area")
    elif method == "nearest":
        y = F.interpolate(x, size=shape, mode="nearest")
    else:
        raise ValueError(f"kieu resize khong biet: {method} (co {RESIZE_METHODS})")
    return y[0, 0].numpy()


def spec_name(orient: str, method: str) -> str:
    return f"{orient}|{method}"


def parse_spec(spec: str):
    orient, method = spec.split("|")
    if orient not in ORIENTATIONS or method not in RESIZE_METHODS:
        raise ValueError(f"spec khong hop le: {spec}")
    return orient, method


def nifti_to_m3t(arr, spacing, spec: str, shape=INPUT_SHAPE, dst_axis: int = 0) -> np.ndarray:
    """Mang NIfTI [Z,Y,X] (tu io_utils.load_nii) -> khoi M3T `shape` theo spec 'huong|resize'.

    `spacing` de kiem: truc lat cat (0.70 mm) phai la truc ma spec dua ve truc `dst_axis` cua
    M3T (mac dinh 0; notebook S9 xac nhan bang do rong 48 huong o muc 5).
    """
    orient, method = parse_spec(spec)
    perm, _ = ORIENTATIONS[orient]
    ax = slice_axis(spacing)
    if perm[dst_axis] != ax:
        raise ValueError(f"spec {spec} dua truc {perm[dst_axis]} ve truc {dst_axis} nhung truc lat cat "
                         f"la {ax} (spacing {tuple(spacing)})")
    return resize3d(reorient(arr, orient), shape, method)


def convert_case(path, spec: str, dst_axis: int = 0, shape=INPUT_SHAPE) -> np.ndarray:
    """Doc NIfTI (io_utils.load_nii, can nibabel - co tren Colab) roi nifti_to_m3t."""
    from .io_utils import load_nii
    arr, spacing = load_nii(str(path))
    return nifti_to_m3t(arr, spacing, spec, shape=shape, dst_axis=dst_axis)


def _centered(vol):
    """(vector minmax01 da tru trung binh, float64; tong binh phuong)."""
    x = minmax01(vol).astype(np.float64).ravel()
    x -= x.mean()
    return x, float((x * x).sum())


def _corr(x, vx, y, vy):
    if vx == 0 or vy == 0:
        return 0.0, 0.0
    cxy = float(x @ y)
    return cxy / np.sqrt(vx * vy), cxy / vy


def volume_corr(a, b):
    """(r Pearson, he so goc cua a hoi quy theo b) sau minmax01 ca hai khoi, cung shape."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape khac nhau {a.shape} vs {b.shape}")
    r, slope = _corr(*_centered(a), *_centered(b))
    return float(r), float(slope)


def search_conversion(pairs, orients=None, methods=RESIZE_METHODS, shape=INPUT_SHAPE, dst_axis: int = 0):
    """Dò (huong x resize) tren cac cap (case_id, mang NIfTI, spacing, npz[, tag]) -> DataFrame dai.

    `pairs` la iterable (generator doc tung ca - khong giu 300 khoi NIfTI trong RAM). `tag` tuy
    chon (vd ben goi cua npz dem so) di theo vao cot 'tag' - xem pick_best_tag.
    `orients` None => 16 huong dua truc lat cat cua TUNG ca ve `dst_axis`; `orients` = list ten
    (vd list(ORIENTATIONS) cho do rong 48 huong). Moi hoan vi chi resize
    MOT lan roi lat tren ket qua: lat giao hoan voi resize tuyen tinh/area; voi 'nearest' chi
    xap xi - cong cuoi phai do lai spec da chon bang nifti_to_m3t. npz chuan hoa MOT lan/cap.
    """
    import pandas as pd
    rows = []
    for item in pairs:
        case_id, arr, spacing, npz = item[:4]
        tag = item[4] if len(item) > 4 else None
        if tuple(np.shape(npz)) != tuple(shape):
            raise ValueError(f"{case_id}: npz shape {np.shape(npz)} != {tuple(shape)}")
        y, vy = _centered(npz)
        src_ax = slice_axis(spacing)
        cand = orientations_for(src_ax, dst_axis) if orients is None else \
            {k: ORIENTATIONS[k] for k in orients}
        by_perm = {}
        for name, (perm, flips) in cand.items():
            by_perm.setdefault(perm, []).append((name, flips))
        for perm, items in by_perm.items():
            base = np.transpose(np.asarray(arr), perm)
            for method in methods:
                r0 = resize3d(base, shape, method)
                for name, flips in items:
                    v = np.flip(r0, axis=flips) if flips else r0
                    r, slope = _corr(*_centered(v), y, vy)
                    rows.append(dict(case_id=case_id, tag=tag, src_axis=src_ax, orient=name,
                                     method=method, r=float(r), slope=float(slope)))
    # du cot ca khi rong (khong co cap nao doc duoc) -> CSV van co header, merge/groupby khong vo
    return pd.DataFrame(rows, columns=["case_id", "tag", "src_axis", "orient", "method", "r", "slope"])


def pick_best_tag(df):
    """Moi ca co nhieu cap (vd npz goi TRAI va PHAI khi ben trong manifest chua kiem): giu tag
    co r lon nhat (qua moi spec). -> (bang da loc, bang phan quyet case_id/best_tag/r_best/r_other).

    Luu y: chon tag theo max-r lam margin cua cong G4.1 lac quan hon mot chut - bao cao kem.
    """
    import pandas as pd
    best = df.groupby(["case_id", "tag"])["r"].max().reset_index()
    rows = []
    for cid, g in best.groupby("case_id"):
        g = g.sort_values("r", ascending=False)
        rows.append(dict(case_id=cid, best_tag=g.iloc[0]["tag"], r_best=float(g.iloc[0]["r"]),
                         r_other=float(g.iloc[1]["r"]) if len(g) > 1 else np.nan))
    verdict = pd.DataFrame(rows)
    keep = df.merge(verdict[["case_id", "best_tag"]], on="case_id")
    keep = keep[keep["tag"] == keep["best_tag"]].drop(columns="best_tag").reset_index(drop=True)
    return keep, verdict


def summarize_conversion(df):
    """Moi spec: n, r trung vi, phan vi 1%, slope trung vi; sap theo r trung vi giam dan.

    `margin` cua spec dung dau = r_median cua no tru r_median cua spec tot nhat co HUONG
    KHAC (khong phai chi khac kieu resize - hai kieu resize cung huong gan nhu trung nhau).
    """
    g = df.groupby(["orient", "method"])
    s = g.agg(n=("r", "size"), r_median=("r", "median"), r_p01=("r", lambda v: np.quantile(v, 0.01)),
              slope_median=("slope", "median")).reset_index()
    s = s.sort_values(["r_median", "r_p01"], ascending=False).reset_index(drop=True)
    s["spec"] = [spec_name(o, m) for o, m in zip(s.orient, s.method)]
    top = s.iloc[0]
    other = s[s.orient != top.orient]
    s["margin"] = np.nan
    s.loc[0, "margin"] = float(top.r_median - other.r_median.iloc[0]) if len(other) else np.nan
    return s


IMAGE_GATE_KEYS = ("r_median", "r_p01", "margin", "slope_lo", "slope_hi")
CLS_GATE_KEYS = ("nn_self", "dist_ratio", "head_agree", "d_ekl")


def _need(th: dict, keys) -> None:
    missing = [k for k in keys if k not in th]
    if missing:
        raise KeyError(f"thieu nguong {missing} - nguong nam MOT cho: configs/m3t_s9.json['gates']")


def image_gate(summary, th: dict) -> dict:
    """G4.1 tren spec dung dau cua summarize_conversion. `th` = config['gates']['image']."""
    _need(th, IMAGE_GATE_KEYS)
    top = summary.iloc[0]
    checks = dict(r_median=top.r_median >= th["r_median"], r_p01=top.r_p01 >= th["r_p01"],
                  margin=bool(np.nan_to_num(top.margin, nan=-1.0) >= th["margin"]),
                  slope=th["slope_lo"] <= top.slope_median <= th["slope_hi"])
    return dict(spec=top.spec, passed=bool(all(checks.values())), checks={k: bool(v) for k, v in checks.items()},
                r_median=float(top.r_median), r_p01=float(top.r_p01), margin=float(np.nan_to_num(top.margin, nan=-1.0)),
                slope_median=float(top.slope_median), n=int(top.n))


def conversion_gate(cls_a, cls_b, logit_a, logit_b, th: dict) -> dict:
    """G4.2 + G4.3: a = tu anh CHUYEN DOI, b = tu npz cua CUNG ca (cung thu tu dong).

    `th` = config['gates']['cls'].

    nn_self   : ty le ca ma lang gieng gan nhat (Euclid) cua a_i trong {b_j} la b_i.
    dist_ratio: trung vi |a_i - b_i| / trung vi |a_i - b_j| (i != j). Cosine don thuan la
                phep thu yeu vi CLS ra sau LayerNorm (moi vector gan cung chuan).
    head_agree: argmax logit trung nhau. d_ekl: trung binh |E_a[KL] - E_b[KL]|.
    """
    _need(th, CLS_GATE_KEYS)
    a, b = np.asarray(cls_a, np.float64), np.asarray(cls_b, np.float64)
    la, lb = np.asarray(logit_a, np.float64), np.asarray(logit_b, np.float64)
    if a.shape != b.shape or la.shape != lb.shape or len(a) != len(la) or len(a) < 2:
        raise ValueError(f"shape khong khop / qua it ca: {a.shape} {b.shape} {la.shape} {lb.shape}")
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))       # [n, n]
    n = len(a)
    nn_self = float((d.argmin(axis=1) == np.arange(n)).mean())
    off = d[~np.eye(n, dtype=bool)]
    ratio = float(np.median(np.diag(d)) / np.median(off)) if np.median(off) > 0 else np.inf
    agree = float((la.argmax(1) == lb.argmax(1)).mean())
    d_ekl = float(np.abs(expected_kl(la) - expected_kl(lb)).mean())
    checks = dict(nn_self=nn_self >= th["nn_self"], dist_ratio=ratio <= th["dist_ratio"],
                  head_agree=agree >= th["head_agree"], d_ekl=d_ekl <= th["d_ekl"])
    return dict(passed=bool(all(checks.values())), checks={k: bool(v) for k, v in checks.items()},
                nn_self=nn_self, dist_ratio=ratio, head_agree=agree, d_ekl=d_ekl, n=int(n))


# ============================================================ dau van tay anh (kiem trung)

FP_SHAPE = (12, 16, 16)


def fingerprint(vol, shape=FP_SHAPE) -> np.ndarray:
    """Vector [prod(shape)] chuan hoa (trung binh 0, chuan 1) cua khoi M3T da thu nho (area).

    Hai anh cung mot lan chup (npz vs NIfTI da chuyen doi) cho tuong quan ~1; hai goi khac
    nhau thap hon ro. Dung cung minmax01 voi dau vao mo hinh.
    """
    v = resize3d(minmax01(vol), shape, "area").astype(np.float64).ravel()
    v -= v.mean()
    nrm = np.linalg.norm(v)
    if nrm == 0:
        raise ValueError("khoi hang - khong co dau van tay")
    return (v / nrm).astype(np.float32)


def max_corr(q, bank):
    """(r lon nhat, chi so) cua fingerprint q voi tung dong cua bank [M, D] (da chuan hoa)."""
    bank = np.asarray(bank, np.float32)
    if bank.ndim != 2 or bank.shape[1] != len(q) or len(bank) == 0:
        raise ValueError(f"bank {bank.shape} khong khop q {np.shape(q)}")
    r = bank @ np.asarray(q, np.float32)
    i = int(np.argmax(r))
    return float(r[i]), i
