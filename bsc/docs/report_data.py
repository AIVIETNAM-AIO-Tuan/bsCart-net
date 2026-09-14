"""SO LIEU DA DO - moi con so trong bao cao 14/09 deu lay tu day.

SUA O DAY, dung sua trong make_figs.py. Chay lai `python docs/make_figs.py` de sinh lai hinh.

Nguon cua tung khoi duoc ghi ngay tren no: ten notebook + muc + co mau. Khong co con so nao
duoc go tay ma khong co nguon.
"""

# =====================================================================================
# COHORT - chung cho S6, S7, S8
# =====================================================================================
COHORT = dict(
    n_case=1229, n_subject=1215,
    kl_counts=[284, 233, 295, 311, 106],          # KL0..KL4
    n_external_reserved=96,                        # OAIZIB-CM danh rieng, KHONG dung o day
    sources={"new_oai1000_balanced_no_gt": 766, "partial_gt_oaizib_ai_meniscus_patella": 375,
             "gt_real_imorphics": 72, "ai_full_8class": 16},
    holdout=dict(n_train=983, n_test=246, n_test_kl=[55, 50, 57, 58, 26]),
)
CLASSES = [0, 1, 2, 3, 4]

# =====================================================================================
# S6 - biomarker neo be mat xuong (biomarker_s6_fcl.ipynb)
# =====================================================================================
S6 = dict(
    n_done=1229, n_error=0, sec_per_case=15.7, table_shape=(1229, 88),
    legacy_max_diff_vs_s3=1.16e-10,               # muc 4: doi chieu 15 cot cu
    params=dict(close_mm=8.0, step_mm=0.1, max_mm=6.0, gap_mm=1.0, smooth_mm=0.5,
                min_island_mm2=100.0, min_defect_mm2=5.0),
)

# muc 5: Spearman rho voi KL, 1229 ca. "moi" = ho be mat S6, "cu" = bang S3.
S6_RHO = [
    # (ten cot, rho, ho)
    ("fcl_fem_maxdef_mm2",    0.433, "moi"),
    ("fcl_mt_ndef",           0.424, "moi"),
    ("fcl_fem_pct",           0.423, "moi"),
    ("fcl_mt_pct",            0.384, "moi"),
    ("fcl_lt_pct",            0.371, "moi"),
    ("thin_le10_mt_pct",      0.162, "moi"),
    ("denuded_ratio_tibial",  0.105, "cu"),
    ("thickness_med_tib_mm", -0.031, "cu"),
]

# muc 5: trung vi phan tram footprint bi tro, theo KL
S6_FCL_BY_KL = {
    "Sụn đùi":        [1.110, 1.073, 1.248, 1.628, 2.995],
    "Mâm chày trong": [0.856, 0.804, 0.966, 1.185, 2.336],
    "Mâm chày ngoài": [0.659, 0.654, 0.735, 0.984, 1.397],
}

# muc 5 + Event.md B3: trung vi dien tich footprint mm2 - phep kiem an toan
S6_TAB_BY_KL = {"tab_fem": {0: 5936, 4: 6882}, "tab_mt": {0: 1269, 4: 1364},
                "tab_lt": {0: 1143, 4: 1227}}

# =====================================================================================
# S7 - cross-validation 15 fold (biomarker_s7_ordinal.ipynb)
# Out-of-fold gop tren ca 1229 ca, trung binh qua 3 seed. StratifiedGroupKFold theo subject.
# =====================================================================================
S7 = dict(n_combo=540, minutes=41.3, n_splits=5, seeds=[0, 1, 2], n_folds=15)

S7_MODELS = ["A_xgb_softmax", "A2_mlp_softmax", "B_xgb_frankhall",
             "C_xgb_reg_cutpoints", "D_mlp_ordinal_only", "E_mlp_slide_multitask"]
S7_MODEL_SHORT = {"A_xgb_softmax": "A\nXGB softmax", "A2_mlp_softmax": "A2\nMLP softmax",
                  "B_xgb_frankhall": "B\nFrank-Hall", "C_xgb_reg_cutpoints": "C\nhồi quy + cắt",
                  "D_mlp_ordinal_only": "D\nMLP ordinal", "E_mlp_slide_multitask": "E\nMLP slide"}

# muc 4, bang "out-of-fold gop": QWK trung binh qua 3 seed
S7_QWK = {
    "legacy_s3":             dict(A=0.554, A2=0.622, B=0.580, C=0.599, D=0.640, E=0.644),
    "s6_surface_only":       dict(A=0.564, A2=0.624, B=0.600, C=0.631, D=0.666, E=0.660),
    "s6_all":                dict(A=0.636, A2=0.669, B=0.664, C=0.693, D=0.717, E=0.706),
    "s5_radiomics":          dict(A=0.738, A2=0.743, B=0.763, C=0.775, D=0.770, E=0.764),
    "s6_all_plus_radiomics": dict(A=0.750, A2=0.739, B=0.776, C=0.772, D=0.774, E=0.764),
}
S7_MAE = {
    "legacy_s3":             dict(A=0.840, A2=0.774, B=0.777, C=0.874, D=0.725, E=0.716),
    "s6_surface_only":       dict(A=0.834, A2=0.784, B=0.757, C=0.801, D=0.708, E=0.710),
    "s6_all":                dict(A=0.744, A2=0.708, B=0.686, C=0.739, D=0.633, E=0.643),
    "s5_radiomics":          dict(A=0.599, A2=0.599, B=0.555, C=0.604, D=0.549, E=0.554),
    "s6_all_plus_radiomics": dict(A=0.580, A2=0.604, B=0.533, C=0.612, D=0.547, E=0.558),
}
S7_N_FEATURES = {"legacy_s3": 15, "s6_surface_only": 55, "s6_all": 70,
                 "s5_radiomics": 871, "s6_all_plus_radiomics": 926}
S7_FS_LABEL = {"legacy_s3": "Bảng cũ (15 cột)", "s6_surface_only": "Bề mặt S6 (55)",
               "s6_all": "Cũ + S6 (70)", "s5_radiomics": "Cũ + radiomics (871)",
               "s6_all_plus_radiomics": "Tất cả (926)"}

# muc 5b: F1 theo lop (%), bo s6_all_plus_radiomics, seed 0, out-of-fold n=1229
S7_F1 = {
    "A_xgb_softmax":        [64.1, 34.1, 45.1, 61.9, 54.3],
    "A2_mlp_softmax":       [59.8, 31.2, 43.9, 63.9, 58.7],
    "B_xgb_frankhall":      [60.6, 42.7, 47.2, 63.5, 58.4],
    "C_xgb_reg_cutpoints":  [62.2, 41.5, 37.8, 47.5, 56.0],
    "D_mlp_ordinal_only":   [60.4, 36.7, 48.2, 64.4, 61.8],
    "E_mlp_slide_multitask":[61.3, 37.4, 48.5, 64.8, 58.9],
}
S7_MACRO_F1 = {"A_xgb_softmax": 51.9, "A2_mlp_softmax": 51.5, "B_xgb_frankhall": 54.5,
               "C_xgb_reg_cutpoints": 49.0, "D_mlp_ordinal_only": 54.3,
               "E_mlp_slide_multitask": 54.2}

# muc 5: ma tran nham lan, bo s6_all_plus_radiomics, seed 0, out-of-fold n=1229.
# Hang = KL that, cot = KL doan.
S7_CONFUSION = {
    "B_xgb_frankhall": [[152, 98, 31, 3, 0], [47, 115, 67, 4, 0], [17, 75, 145, 56, 2],
                        [2, 18, 75, 198, 18], [0, 0, 2, 52, 52]],
    "D_mlp_ordinal_only": [[160, 81, 42, 1, 0], [62, 89, 71, 11, 0], [21, 64, 149, 60, 1],
                           [3, 17, 57, 200, 34], [0, 1, 4, 38, 63]],
    "E_mlp_slide_multitask": [[159, 83, 38, 4, 0], [51, 91, 80, 11, 0], [23, 61, 150, 60, 1],
                              [2, 19, 52, 206, 32], [0, 0, 4, 44, 58]],
}

# muc 5: thang xac suat - trung binh P(KL>k) theo lop THAT, Frank-Hall, n=1229
S7_P_LADDER = {0: [0.49, 0.19, 0.03, 0.00], 1: [0.73, 0.36, 0.06, 0.00],
               2: [0.90, 0.66, 0.22, 0.01], 3: [0.97, 0.91, 0.67, 0.08],
               4: [1.00, 0.99, 0.94, 0.46]}
S7_THRESHOLD_AUC = [0.872, 0.902, 0.934, 0.952]      # AUC cua P(KL>0..3), Frank-Hall

# muc 6: tach dong gop cua KIEN TRUC khoi dong gop cua LOSS, nho doi chung A2
S7_DECOMP = [
    # (bo dac trung, so cot, A->A2 = doi cay sang mang, A2->D = them loss ordinal)
    ("legacy_s3", 15, +0.068, +0.018),
    ("s6_surface_only", 55, +0.060, +0.042),
    ("s6_all", 70, +0.033, +0.048),
    ("s5_radiomics", 871, +0.005, +0.027),
    ("s6_all_plus_radiomics", 926, -0.011, +0.035),
]

# =====================================================================================
# S8 - holdout mot lan, n_test = 246 (biomarker_s8_holdout.ipynb, ban v2)
# GroupShuffleSplit(test_size=0.2, random_state=42) theo subject - giong bao cao 13/9.
# CHI GIU NHUNG DONG TOT NHAT.
# =====================================================================================
S8 = dict(n_test=246, minutes=3.8, n_decision_rows=95, noise_qwk=0.055)

# muc 5: bang metric cua quy tac MAC DINH, bo s6_all_plus_radiomics
S8_DEFAULT = {
    "A_xgb_softmax":       dict(acc=51.2, qwk=0.750, mae=0.593, off2=9.8),
    "B_xgb_frankhall":     dict(acc=53.3, qwk=0.783, mae=0.528, off2=6.1),
    "C_xgb_reg_cutpoints": dict(acc=54.5, qwk=0.803, mae=0.520, off2=6.1),
    "D_mlp_ordinal_only":  dict(acc=54.5, qwk=0.796, mae=0.516, off2=5.3),
    "E_mlp_slide":         dict(acc=57.7, qwk=0.797, mae=0.500, off2=7.3),
    "I_mlp_fusion":        dict(acc=56.1, qwk=0.788, mae=0.516, off2=6.9),
}

# muc 4b: nhung dong TOT NHAT cua tang quyet dinh
S8_BEST = [
    # (nhan, bo dac trung, qwk, macro_recall, off2, recall_KL4, precision_KL4, ghi chu)
    ("E @ ngưỡng dò",  "Tất cả (926)",        0.814, 0.614, 0.077, 0.769, 0.690, "cao nhất bảng"),
    ("C @ sàn recall", "Tất cả (926)",        0.805, 0.571, 0.057, 0.808, 0.724, ""),
    ("D @ ngưỡng dò",  "Tất cả (926)",        0.805, 0.565, 0.049, 0.731, 0.826, ""),
    ("C @ mặc định",   "Tất cả (926)",        0.803, 0.571, 0.061, 0.808, 0.724, "mặc định tốt nhất"),
    ("B @ ngưỡng dò",  "Cũ + radiomics (871)",0.798, 0.610, 0.089, 0.885, 0.676, ""),
    ("E @ mặc định",   "Tất cả (926)",        0.797, 0.564, 0.073, 0.500, 0.867, ""),
]

# muc 4b: hieu ung do nguong len lop KL4. 6 cap (model x bo dac trung), vach 0.5 -> nguong do.
S8_THR_EFFECT = [
    # (model, bo dac trung, recall co dinh, recall do, precision co dinh, precision do)
    ("B", "Tất cả",         0.423, 0.731, 0.846, 0.704),
    ("D", "Tất cả",         0.538, 0.731, 0.875, 0.826),
    ("E", "Tất cả",         0.500, 0.731, 0.867, 0.731),
    ("B", "Cũ + radiomics", 0.385, 0.885, 0.833, 0.676),
    ("D", "Cũ + radiomics", 0.538, 0.654, 0.875, 0.630),
    ("E", "Cũ + radiomics", 0.538, 0.731, 0.933, 0.655),
]

# muc 6: ma tran nham lan tren tap test n=246, bo s6_all_plus_radiomics
S8_CONFUSION = {
    "A_xgb_softmax": [[35, 11, 9, 0, 0], [15, 9, 21, 5, 0], [6, 10, 28, 13, 0],
                      [2, 2, 10, 42, 2], [0, 0, 0, 14, 12]],
    "C_xgb_reg_cutpoints": [[34, 14, 7, 0, 0], [11, 21, 16, 2, 0], [4, 13, 30, 10, 0],
                            [1, 1, 20, 28, 8], [0, 0, 0, 5, 21]],
    "C_xgb_reg_cutpoints@quantile": [[34, 16, 5, 0, 0], [12, 20, 14, 4, 0], [4, 13, 29, 11, 0],
                                     [1, 4, 15, 32, 6], [0, 0, 0, 8, 18]],
}

# muc 4d: cot S6 co duoc classifier dung khong, bo s6_all_plus_radiomics
S8_S6_USAGE = {
    # ho: (so cot vao, LASSO bo, giu va co gain, ty trong gain %)
    "Bề mặt S6": (55, 42, 13, 15.5),
    "Bảng cũ":   (15, 10, 5, 6.2),
    "Radiomics": (856, 767, 89, 78.4),
}
S8_S6_TOP = [("fcl_fem_maxdef_mm2", 2.467), ("fcl_mt_ndef", 2.180), ("fcl_lt_mm2", 1.828),
             ("thin_le05_mt_pct", 1.225), ("thc_tab_fem_med_mm", 1.200),
             ("tab_fem_med_mm2", 1.104), ("thickp05_fem_lat_mm", 1.005)]
S8_S6_TOP_RANK = 2          # hang cua cot S6 manh nhat trong 926 cot

# muc 4c: khong quy tac nao dat tren CA HAI bo dac trung co radiomics
S8_GATE_RESULT = {"E_mlp_slide@tuned_mr_slack0.02": ("s6_all_plus_radiomics",),
                  "B_xgb_frankhall@tuned_mr_slack0.02": ("s5_radiomics",)}

# muc 7: ket qua am tinh, giu de doi chieu
S8_NEGATIVE = {
    "G (ASL)": "te nhat trong 10 model tren ca 5 bo dac trung, ca QWK lan off-by>=2",
    "F (head MLP sau hon)": "hieu so -0.011 toi +0.026, duoi nguong nhieu",
    "H (nguong tau hoc duoc)": "tau chi dich toi 0.51-0.53, gan nhu khong roi vach 0.5",
    "C2 (trong so theo lop)": "trung binh -0.007, am tren ca hai bo co radiomics",
}
