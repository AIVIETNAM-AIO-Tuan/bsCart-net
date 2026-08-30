"""Khoa cac quyet dinh trong p3_m8d_standardization_decision_vi.md bang test.

Tai lieu do co the bi quen; test thi khong. Moi test o day tuong ung mot quyet dinh.
"""

from __future__ import annotations

import pytest

from bsc import experiment as X


def test_qd1_p3_va_m8d_la_cung_mot_cau_hinh():
    """QD1: P3 va M8-D KHONG phai hai thi nghiem doc lap."""
    p3 = X.from_plan("P3", "med_tib_cart", prob_source="resenc_150ep_oof_softmax")
    assert (p3.surface, p3.domain, p3.inputs, p3.heads) == ("S0", "D1", "I3", "H1")
    assert p3.m8_role == "M8-D", "P3 phai TU DONG mang vai tro M8-D"
    assert p3.plan_alias == "P3"


def test_qd2_khong_can_run_rieng_cho_m8d():
    """QD2: M8-D duoc SUY RA tu scaffold, khong phai mot alias run rieng."""
    assert "M8-D" not in X.PLAN_MATRIX, "M8-D khong duoc la run trong registry"
    # Moi vai tro M8 deu suy duoc tu (S0, D1, I*, H1)
    for role, i in X.M8_INPUTS.items():
        assert X.m8_role("S0", "D1", i, "H1") == role
    # Ngoai scaffold thi khong mang vai tro M8 nao
    assert X.m8_role("S0", "D0", "I3", "H1") is None      # sai domain
    assert X.m8_role("S2", "D1", "I3", "H1") is None      # sai surface
    assert X.m8_role("S0", "D1", "I3", "H0") is None      # sai heads


def test_qd3_kenh_prob_bat_buoc_ghi_nguon():
    """QD3 + §12: dung probability thi phai ghi ro nguon, cam mo ho."""
    with pytest.raises(ValueError, match="prob_source"):
        X.RunConfig(cls="med_tib_cart", inputs="I3")     # thieu prob_source

    ok = X.RunConfig(cls="med_tib_cart", inputs="I3",
                     prob_source="resenc_150ep_oof_softmax")
    reg = ok.to_registry()
    assert reg["resenc_prior"]["representation"] == "probability"
    assert reg["resenc_prior"]["source"] == "resenc_150ep_oof_softmax"
    # Kenh khong co prob thi khong bat buoc
    assert X.RunConfig(cls="med_tib_cart", inputs="I2").to_registry().get("resenc_prior") is None


def test_prob_source_phai_nam_trong_danh_sach_da_chot():
    """Nguon tu bia => bao loi. Them nguon moi phai cap nhat PROB_SOURCES kem ly do."""
    with pytest.raises(ValueError, match="khong nam trong danh sach"):
        X.RunConfig(cls="med_tib_cart", inputs="I3", prob_source="tu_bia")

    # Quyet dinh 2026-07-19: MVP dung OOF (5-fold mean ro ri tren ca train/val)
    assert X.DEFAULT_PROB_SOURCE == "resenc_150ep_oof_softmax"
    assert "CHI dung cho tap test" in X.PROB_SOURCES["resenc_5fold_mean_softmax"]


def test_qd4_ablation_chi_thay_mot_yeu_to():
    """QD4: cap so sanh phai co lap dung mot truc."""
    base = X.RunConfig(cls="med_tib_cart", surface="S0", domain="D1", inputs="I0", heads="H1")
    prob = X.RunConfig(cls="med_tib_cart", surface="S0", domain="D1", inputs="I3",
                       heads="H1", prob_source="resenc_150ep_oof_softmax")
    X.assert_single_factor(base, prob, "inputs")         # hop le

    # Doi dong thoi domain + inputs => phai bao loi (§3.2)
    bad = X.RunConfig(cls="med_tib_cart", surface="S0", domain="D0", inputs="I3",
                      heads="H1", prob_source=X.DEFAULT_PROB_SOURCE)
    with pytest.raises(ValueError, match="chi thay dung mot yeu to"):
        X.assert_single_factor(base, bad, "inputs")


def test_qd5_ma_tran_ke_hoach_khop_tai_lieu_muc_7():
    """QD5 + §7: P0/P1 dung Oracle; P2 tro di dung Atlas + presence head."""
    assert X.PLAN_MATRIX["P0"] == ("S0", "D0", "I0", "H0")
    assert X.PLAN_MATRIX["P1"] == ("S0", "D0", "I1", "H0")
    assert X.PLAN_MATRIX["P2"] == ("S0", "D1", "I2", "H1")
    assert X.PLAN_MATRIX["P3"] == ("S0", "D1", "I3", "H1")

    for alias in ("P2", "P3"):
        s, d, i, h = X.PLAN_MATRIX[alias]
        assert d == "D1", f"{alias} phai dung Atlas, khong phai Oracle"
        assert h == "H1", f"{alias} phai co presence head"
    for alias in ("P0", "P1"):
        assert X.PLAN_MATRIX[alias][1] == "D0"
        assert X.PLAN_MATRIX[alias][3] == "H0"


def test_experiment_id_theo_quy_tac_muc_11():
    """§11: ten run phai the hien dung cau hinh thuc te."""
    r = X.from_plan("P3", "med_tib_cart", fold=0, seed=1,
                    prob_source="resenc_150ep_oof_softmax")
    assert r.experiment_id == "MVP_MTC_S0_D1_I3_H1_v1_Fold0_Seed1"

    r2 = X.RunConfig(cls="femoral_cart", surface="S2", domain="D1", inputs="I2",
                     heads="H1", fold=3, seed=7, version="v2")
    assert r2.experiment_id == "MVP_FC_S2_D1_I2_H1_v2_Fold3_Seed7"


def test_p4_p5_bat_buoc_khai_bao_best_input():
    """P4/P5 dung 'best input' => khong duoc de mo ho, phai truyen tuong minh."""
    with pytest.raises(ValueError, match="best input"):
        X.from_plan("P4", "med_tib_cart")

    r = X.from_plan("P5", "med_tib_cart", inputs="I2")
    assert r.surface == "S2" and r.domain == "D1" and r.heads == "H1"

    # Khong duoc doi input cua alias da co dinh
    with pytest.raises(ValueError, match="co dinh"):
        X.from_plan("P2", "med_tib_cart", inputs="I3")


def test_registry_ghi_du_truong_audit_muc_12():
    r = X.from_plan("P3", "femoral_cart", prob_source="resenc_5fold_mean_softmax",
                    notes="pilot")
    reg = r.to_registry(code_commit="abc123", dataset_revision="d001-v1")
    for k in ("experiment_id", "plan_alias", "ablation_role", "surface_source",
              "articular_domain", "inputs", "outputs", "fold", "seed",
              "code_commit", "dataset_revision"):
        assert k in reg, f"thieu truong audit {k}"
    assert reg["ablation_role"] == "M8-D"
    assert reg["articular_domain"] == "fold_atlas"
