"""Kiem chung RayEncoder1D tren phantom - chay CPU, khong can du lieu that.

Trong tam la **M5 (plan doc §3.5)**: mo hinh phai gan nhu thuoc long tap tia rat nho.
M5 that bai => co loi o INPUT / TARGET / KIEN TRUC / LOSS / RECONSTRUCTION. Day la
cai bay re tien nhat de bat cac bug im lang truoc khi dot GPU tren du lieu that.

Bug im lang M5 nham toi (da co tien le trong repo nay):
  - phap tuyen sai dau => tia ban vao trong xuong => occupancy toan 0, model van
    "train binh thuong" ma khong hoc duoc gi (xem core.extract_surface).
  - nham thu tu truc [B,K,C] vs [B,C,K] => conv truot tren kenh thay vi do sau.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bsc import core, model
from bsc.core import RayConfig

SP = core.SPACING


def _shell_phantom(shape=(40, 100, 100), spacing=SP, r_mm=12.0, t_mm=2.0,
                   gap_frac=0.25, noise=0.05, seed=0):
    """Vo sun quanh cau xuong, CO mot vung mat sun hoan toan (test M1/M5).

    Vung mat sun la bat buoc: khong co no thi presence head khong co negative va
    ta khong kiem duoc luan diem ve sun MAT - tuc chinh gia thuyet.
    """
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(*[np.arange(s) * sp for s, sp in zip(shape, spacing)],
                             indexing="ij")
    c = [s * sp / 2 for s, sp in zip(shape, spacing)]
    r = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)

    bone = r <= r_mm
    cart = (r > r_mm) & (r <= r_mm + t_mm)
    # Xoa sun o mot chom cau => vung absent that su
    cart &= ~(xx > c[2] + (1 - gap_frac) * r_mm)

    # MRI gia: xuong toi, sun sang, nen trung binh + nhieu
    mri = np.full(shape, 0.3, np.float32)
    mri[bone] = 0.1
    mri[cart] = 0.9
    mri += rng.normal(0, noise, shape).astype(np.float32)
    return mri, bone, cart


def test_forward_shapes_and_channel_order():
    """Vao [B,K,C] -> occ [B,K], pres [B]. Sai thu tu truc phai BAO LOI, khong im lang."""
    cfg = RayConfig()
    net = model.RayEncoder1D(in_channels=3)
    x = torch.randn(7, cfg.k, 3)
    occ, pres = net(x)
    assert occ.shape == (7, cfg.k)
    assert pres.shape == (7,)

    # [B,C,K] nham cho => phai no ngay (K=64 != in_channels=3)
    with pytest.raises(ValueError):
        net(torch.randn(7, 3, cfg.k))


def test_h0_has_no_presence_head_at_all():
    """Truc H: H0 phai KHONG CO presence head, khong phai co ma bo qua dau ra.

    Neu head van ton tai, gradient cua no van chay vao than mang => H0 khong con la H0
    va cap so sanh H0-vs-H1 (§10.2) vo nghia. Ma tran §7 dat P0/P1 la H0.
    """
    net0 = model.RayEncoder1D(in_channels=3, with_presence=False)
    assert not hasattr(net0, "pres_head") and not hasattr(net0, "attn")

    occ, pres = net0(torch.randn(5, RayConfig().k, 3))
    assert occ.shape == (5, RayConfig().k)
    assert pres is None, "H0 phai tra pres_logits=None"

    net1 = model.RayEncoder1D(in_channels=3, with_presence=True)
    assert hasattr(net1, "pres_head")
    assert net1(torch.randn(5, RayConfig().k, 3))[1] is not None

    # H0 phai it tham so hon H1 (khong dung attn + pres_head)
    n0 = sum(p.numel() for p in net0.parameters())
    n1 = sum(p.numel() for p in net1.parameters())
    assert n0 < n1


def test_h0_loss_has_no_presence_term():
    """H0: loss = occupancy THUAN. Truyen pres_logits=None thi khong duoc cong gi them."""
    occ_l = torch.randn(16, 64, requires_grad=True)
    occ_t = (torch.rand(16, 64) > 0.7).float()
    pres_t = (torch.rand(16) > 0.5).float()

    l0, p0 = model.ray_loss(occ_l, None, occ_t, pres_t)
    assert np.isnan(p0["pres"]), "H0 khong duoc co so hang presence"
    assert np.isclose(p0["loss"], p0["occ"]), "H0: loss phai bang dung phan occupancy"

    l1, p1 = model.ray_loss(occ_l, torch.randn(16), occ_t, pres_t)
    assert p1["loss"] > p1["occ"], "H1 phai cong them presence"

    # H0 co the backward binh thuong
    l0.backward()


def test_h0_train_and_predict_end_to_end():
    """H0 phai train va predict duoc tron ven (P0/P1 dung cau hinh nay)."""
    mri, bone, cart = _shell_phantom()
    X, occ, pres, _, _ = model.build_case_rays(mri, bone, cart, SP, RayConfig(),
                                               domain_mode="oracle")
    idx = np.random.default_rng(0).choice(len(X), min(600, len(X)), replace=False)
    net = model.RayEncoder1D(in_channels=X.shape[-1], with_presence=False)
    hist = model.fit(net, X[idx], occ[idx], pres[idx], epochs=25, batch_size=256, lr=3e-3)
    assert hist[-1]["loss"] < hist[0]["loss"]

    op, pp = model.predict_rays(net, X[idx])
    assert op.shape == (len(idx), RayConfig().k)
    assert pp is None, "H0 khong tra presence"


def test_conv_preserves_depth_for_all_dilations():
    """Padding phai giu nguyen K o moi dilation - neu khong occupancy lech do sau."""
    for d in (1, 2, 4, 8):
        blk = model._ConvBlock(3, 16, kernel_size=5, dilation=d)
        out = blk(torch.randn(2, 3, 64))
        assert out.shape[-1] == 64, f"dilation={d} lam doi K"


def test_build_case_rays_shapes_and_absence():
    """Cau noi core->tensor: dung so kenh, va phantom PHAI co ca tia absent."""
    mri, bone, cart = _shell_phantom()
    cfg = RayConfig()
    X, occ, pres, verts, normals = model.build_case_rays(mri, bone, cart, SP, cfg)

    assert X.shape[1:] == (cfg.k, 3), f"can [N,K,3], nhan {X.shape}"
    assert occ.shape == (X.shape[0], cfg.k)
    assert pres.shape == (X.shape[0],)
    assert np.isfinite(X).all(), "X co NaN/Inf - kiem sample_volume cval"

    # Phai co CA HAI lop, neu khong M5/presence vo nghia
    assert pres.mean() > 0.05, "khong co tia nao cham sun - phap tuyen sai dau?"
    assert pres.mean() < 0.95, "khong co tia absent - phantom thieu vung mat sun"


def test_m5_tiny_set_overfitting():
    """M5 (plan doc §3.5) - mo hinh phai gan nhu thuoc long mot tap tia nho.

    Cong (plan): occupancy Dice train rat cao + loss giam ro. Neu TRUOT, dung tim
    hyperparameter - hay tim bug o input/target/kien truc/loss.
    """
    mri, bone, cart = _shell_phantom()
    cfg = RayConfig()
    X, occ, pres, _, _ = model.build_case_rays(mri, bone, cart, SP, cfg)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(600, len(X)), replace=False)
    X, occ, pres = X[idx], occ[idx], pres[idx]

    net = model.RayEncoder1D(in_channels=3, width=64)
    hist = model.fit(net, X, occ, pres, epochs=60, batch_size=256, lr=3e-3, seed=0)

    assert hist[-1]["loss"] < hist[0]["loss"] * 0.5, (
        f"loss khong giam du: {hist[0]['loss']:.4f} -> {hist[-1]['loss']:.4f}")

    occ_p, pres_p = model.predict_rays(net, X)
    pred = (occ_p > 0.5)
    tgt = occ.astype(bool)
    dice = 2 * (pred & tgt).sum() / (pred.sum() + tgt.sum() + 1e-8)
    assert dice > 0.90, f"M5 truot: occupancy Dice train chi {dice:.3f} (can >0.90)"

    acc = ((pres_p > 0.5) == pres.astype(bool)).mean()
    assert acc > 0.90, f"presence head khong thuoc duoc tap nho: acc {acc:.3f}"


def test_presence_head_separates_absent_rays():
    """Presence phai TACH duoc tia absent - day la luan diem chinh cua gia thuyet.

    M0 do duoc bin `absent` sai gap ~4 lan cac bin khac; neu presence head khong tach
    duoc thi bieu dien nay khong giai quyet duoc diem yeu no nham toi.
    """
    mri, bone, cart = _shell_phantom()
    cfg = RayConfig()
    X, occ, pres, _, _ = model.build_case_rays(mri, bone, cart, SP, cfg)

    rng = np.random.default_rng(1)
    idx = rng.choice(len(X), size=min(800, len(X)), replace=False)
    X, occ, pres = X[idx], occ[idx], pres[idx]

    net = model.RayEncoder1D(in_channels=3)
    model.fit(net, X, occ, pres, epochs=60, batch_size=256, lr=3e-3, seed=0)
    _, pres_p = model.predict_rays(net, X)

    # Xac suat trung binh tren tia CO sun phai cao han han tia absent
    has, no = pres.astype(bool), ~pres.astype(bool)
    assert pres_p[has].mean() - pres_p[no].mean() > 0.3, (
        f"presence khong tach duoc: co-sun {pres_p[has].mean():.3f} vs "
        f"absent {pres_p[no].mean():.3f}")


def test_reconstruct_volume_roundtrip_with_perfect_prediction():
    """Neu du doan = GT occupancy, the tich dung lai phai khop mask GT.

    Chot buoc 8 (reconstruction) KHONG phai nguon mat mat. Neu test nay truot thi
    moi so do downstream deu vo nghia, bat ke model tot den dau.
    """
    mri, bone, cart = _shell_phantom()
    cfg = RayConfig()
    X, occ, pres, verts, normals = model.build_case_rays(mri, bone, cart, SP, cfg)

    rec = model.reconstruct_volume(occ.astype(np.float32), verts, normals,
                                   cart.shape, cfg, SP)
    pred = rec > 0.5
    dice = 2 * (pred & cart).sum() / (pred.sum() + cart.sum() + 1e-8)
    assert dice > 0.90, f"round-trip reconstruction chi dat Dice {dice:.3f}"


def test_presence_gating_removes_rays():
    """presence_prob duoi nguong phai ep occupancy ve 0 truoc khi splat."""
    mri, bone, cart = _shell_phantom()
    cfg = RayConfig()
    X, occ, pres, verts, normals = model.build_case_rays(mri, bone, cart, SP, cfg)

    all_off = np.zeros(len(X), np.float32)      # moi tia bi coi la absent
    rec = model.reconstruct_volume(occ.astype(np.float32), verts, normals, cart.shape,
                                   cfg, SP, presence_prob=all_off, presence_thr=0.5)
    assert rec.max() == 0.0, "presence gating khong chan duoc tia"


def test_m4_single_interval_ratio_on_phantom():
    """M4 (plan doc §3.5) - ty le tia co <=1 khoang sun lien tuc.

    >95% => bieu dien theo bien kha thi. Phantom vo cau phai gan 1.0; con so THAT
    phai do tren xuong that (ria sun / vung cong cao / meniscus chen vao).
    """
    _, bone, cart = _shell_phantom()
    cfg = RayConfig()
    _, verts, normals = core.bone_geometry(bone, SP, cfg)
    occ = core.occupancy_target(cart, verts, normals, cfg, SP)
    r = core.single_interval_ratio(occ)
    assert r > 0.95, f"M4 phantom chi dat {r:.3f}"


def test_m6_direction_fields_are_valid():
    """M6 - moi truong huong phai la vector don vi; tiep tuyen phai VUONG GOC phap tuyen."""
    _, bone, _ = _shell_phantom()
    _, verts, normals = core.bone_geometry(bone, SP)

    for mode in ("normal", "axial", "random", "tangent"):
        d = core.direction_field(normals, mode, seed=0)
        assert d.shape == normals.shape
        norms = np.linalg.norm(d, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), f"{mode}: khong phai vector don vi"

    tan = core.direction_field(normals, "tangent", seed=0)
    dot = np.abs((tan * normals).sum(axis=1))
    assert dot.max() < 1e-3, f"tiep tuyen khong vuong goc phap tuyen (max|dot|={dot.max():.2e})"

    with pytest.raises(ValueError):
        core.direction_field(normals, "khong-ton-tai")


def test_m6_normal_direction_beats_tangent_on_profile_structure():
    """M6 - tia PHAP TUYEN phai cho profile 1D co cau truc hon tia TIEP TUYEN.

    Doc theo phap tuyen, tia cat NGANG lop sun => mot khoang lien tuc gon. Doc theo
    tiep tuyen, tia truot DOC theo lop sun => khoang dai/vun, mat cau truc "1D theo
    chieu day" ma ca gia thuyet dua vao.

    Day la phien ban RE (khong can train) cua M6. Ban day du (train tung huong roi so
    boundary/thickness) chay tren du lieu that o notebook bsc_02.
    """
    _, bone, cart = _shell_phantom()
    cfg = RayConfig()
    _, verts, normals = core.bone_geometry(bone, SP, cfg)

    r = {}
    for mode in ("normal", "tangent"):
        d = core.direction_field(normals, mode, seed=0)
        occ = core.occupancy_target(cart, verts, d, cfg, SP)
        r[mode] = core.single_interval_ratio(occ)

    assert r["normal"] > r["tangent"], (
        f"phap tuyen khong hon tiep tuyen ve cau truc profile: "
        f"normal {r['normal']:.3f} vs tangent {r['tangent']:.3f}")


def test_m7_jitter_is_identity_at_zero_and_perturbs_otherwise():
    """M7 - jitter=0 phai KHONG doi gi (neu khong, moi diem chuan deu troi)."""
    _, bone, _ = _shell_phantom()
    _, verts, normals = core.bone_geometry(bone, SP)

    v0, n0 = core.jitter_surface(verts, normals, 0.0, 0.0, seed=0)
    assert np.allclose(v0, verts) and np.allclose(n0, normals)

    v1, n1 = core.jitter_surface(verts, normals, delta_s_mm=0.5, seed=0)
    disp = np.linalg.norm(v1 - verts, axis=1)
    assert 0.2 < disp.mean() < 1.0, f"do lech vi tri bat thuong: {disp.mean():.3f}mm"
    assert np.allclose(n1, normals), "delta_s khong duoc lam doi phap tuyen"

    _, n2 = core.jitter_surface(verts, normals, delta_theta_deg=10.0, seed=0)
    ang = np.degrees(np.arccos(np.clip((n2 * normals).sum(1), -1, 1)))
    assert 3.0 < ang.mean() < 20.0, f"goc xoay bat thuong: {ang.mean():.1f} do"
    assert np.allclose(np.linalg.norm(n2, axis=1), 1.0, atol=1e-4)


def test_m8_channel_selection():
    """Truc I (experiment.INPUT) - dung so kenh, dung THU TU; thieu 'prob' phai bao loi."""
    from bsc import experiment as X_
    mri, bone, cart = _shell_phantom()
    cfg = RayConfig()

    for key, n_ch in [("I0", 1), ("I1", 2), ("I2", 2)]:
        X, *_ = model.build_case_rays(mri, bone, cart, SP, cfg,
                                      channels=X_.INPUT[key])
        assert X.shape[-1] == n_ch, f"{key}: can {n_ch} kenh, nhan {X.shape[-1]}"

    # I3 co 'prob' nhung khong truyen extra_vols => phai no ngay, khong im lang
    with pytest.raises(ValueError, match="prob"):
        model.build_case_rays(mri, bone, cart, SP, cfg, channels=X_.INPUT["I3"])

    prob = cart.astype(np.float32) * 0.8
    X, *_ = model.build_case_rays(mri, bone, cart, SP, cfg,
                                  channels=X_.INPUT["I3"],
                                  extra_vols={"prob": prob})
    assert X.shape[-1] == 2


def test_atlas_domain_needs_opposing_bone():
    """D1 (fold_atlas) can xuong doi dien; thieu thi phai bao loi ro rang."""
    mri, bone, cart = _shell_phantom()
    cfg = RayConfig()

    with pytest.raises(ValueError, match="opposing_bone_mask"):
        model.build_case_rays(mri, bone, cart, SP, cfg, domain_mode="fold_atlas")

    # Xuong "doi dien" gia: mot khoi lech sang mot phia
    opp = np.zeros_like(bone)
    opp[:, :, -12:] = True
    X, occ, pres, _, _ = model.build_case_rays(mri, bone, cart, SP, cfg,
                                               domain_mode="fold_atlas",
                                               opposing_bone_mask=opp)
    assert len(X) > 0, "mien atlas rong - proxy hinh hoc khong chon duoc node nao"
    assert len(X) < len(core.bone_geometry(bone, SP, cfg)[1]), "atlas phai LOC bot node"


def test_build_dataset_multi_case_and_subsampling():
    """Ghep nhieu ca + lay mau con. Loader la callable => module khong can nibabel."""
    phantoms = {f"c{i}": _shell_phantom(gap_frac=0.2 + 0.1 * i, seed=i) for i in range(3)}

    def loader(cid):
        mri, bone, cart = phantoms[cid]
        return mri, bone, cart

    X, occ, pres = model.build_dataset(
        list(phantoms), loader, SP, RayConfig(),
        domain_mode="oracle", rays_per_case=500, seed=0)

    assert X.shape[0] == occ.shape[0] == pres.shape[0]
    assert X.shape[0] <= 3 * 500
    assert X.shape[-1] == 3
    assert 0.0 < pres.mean() < 1.0, "phai co ca tia co sun lan absent"


def test_oracle_domain_contains_absent_rays():
    """MVP-A: vanh no phai keo duoc tia ABSENT vao domain.

    Thieu vanh => domain toan tia co sun => presence head khong co negative => khong
    kiem duoc luan diem chinh cua gia thuyet. Test nay khoa lai hanh vi do.
    """
    mri, bone, cart = _shell_phantom()
    cfg = RayConfig()
    _, _, pres, _, _ = model.build_case_rays(mri, bone, cart, SP, cfg,
                                             domain_mode="oracle")
    assert (pres == 0).sum() > 0, "oracle domain khong chua tia absent nao"
    assert (pres == 1).sum() > 0, "oracle domain khong chua tia co sun nao"

    with pytest.raises(ValueError):
        model.build_case_rays(mri, bone, cart, SP, cfg, domain_mode="sai")


def test_presence_f1_and_thickness_mae():
    """Metric hinh thai §2.3 - do la thu se bao cao THAY cho ASSD tong (M0 §6)."""
    from bsc import metrics as M
    t = np.array([1, 1, 0, 0, 1, 0], bool)
    p = np.array([1, 0, 0, 1, 1, 0], bool)
    r = M.presence_f1(t, p)
    assert r["n_present"] == 3 and r["n_absent"] == 3
    assert 0 < r["f1"] < 1
    assert np.isclose(r["absent_recall"], 2 / 3)

    a = np.array([0.0, 1.0, 2.0]); b = np.array([0.5, 1.0, 1.0])
    m = M.thickness_mae(a, b)
    assert np.isclose(m["mae_mm"], 0.5) and np.isclose(m["bias_mm"], -1 / 6)
    # only_present bo tia absent => chi con 2 mau
    assert M.thickness_mae(a, b, only_present=True)["n"] == 2


def test_thin_region_boundary_error_is_the_37_denominator():
    """Ham quyet dinh §3.7: loi bien TB trong vung mong (KHONG phai ASSD tong)."""
    from bsc import headroom as H
    mri, bone, cart = _shell_phantom()
    pred = cart.copy()
    pred[:, :, cart.shape[2] // 2:] = False          # bo mot nua => tao loi

    r = H.thin_region_boundary_error(cart, pred, bone, SP, RayConfig())
    assert r["thin_n"] > 0, "khong co voxel nao roi vao bin mong"
    assert np.isfinite(r["thin_mean_err_mm"])
    assert r["thin_mean_err_mm"] > 0
    # Mau so vung mong phai KHAC mau so toan be mat - do la ca diem cua ham nay
    assert not np.isclose(r["thin_mean_err_mm"], r["all_mean_err_mm"])


def test_loss_components_and_focal_reduces_to_bce():
    """gamma=0 phai bang BCE thuan - khoa lai de khong ai 'sua' focal thanh mac dinh cung."""
    logits = torch.randn(32, 64)
    target = (torch.rand(32, 64) > 0.7).float()
    focal0 = model.focal_bce_with_logits(logits, target, gamma=0.0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    assert torch.allclose(focal0, bce, atol=1e-6)

    occ_l, pres_l = torch.randn(32, 64), torch.randn(32)
    occ_t, pres_t = (torch.rand(32, 64) > 0.7).float(), (torch.rand(32) > 0.5).float()
    total, parts = model.ray_loss(occ_l, pres_l, occ_t, pres_t)
    assert set(parts) == {"loss", "occ", "pres"}
    assert np.isfinite(parts["loss"]) and parts["loss"] > 0
