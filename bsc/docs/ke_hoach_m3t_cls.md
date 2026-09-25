# Kế hoạch: CLS của M3T huấn luyện lại không rò rỉ, thay cho radiomics

> **Trạng thái:** CHƯA THỰC HIỆN — kế hoạch đã chốt các quyết định, chờ lệnh bắt đầu.
> **Ngày lập:** 24/09/2026. **Liên quan:** `Event.md`, `docs/bao_cao_14_09.md`, `notebooks/biomarker_s7_ordinal.ipynb`, `notebooks/biomarker_s8_holdout.ipynb`.

## Context

**Yêu cầu:** dùng một mô hình ảnh để phân loại KL, lấy CLS token của nó thay cho 856 cột
radiomics trong pipeline bảng S7/S8, rồi so xem có thay được không.

**Quyết định của người dùng (24/09/2026):**
1. Dùng **M3T có sẵn** trong `knee_testing_v3.ipynb` (CNN 3D + CNN 2D + transformer, CLS 128 chiều,
   1.160.677 tham số, acc test 0,656 trên 1636 gối).
2. **Đóng băng** làm bộ trích đặc trưng.
3. Vì M3T đã train trên 2880 subject OAI — trong 70 subject iMorphics của mình thì 44 nằm ở tập train
   của nó, 8 ở val — nên CLS của trọng số hiện có đã mang nhãn KL của cohort. **Train lại M3T từ đầu
   trên kho dữ liệu của nó trừ mọi subject có trong cohort**, rồi mới đóng băng.
4. Chạy trên **RTX 6000 qua Colab**.
5. **Cho phép ngoại lệ:** chép zip đầu vào vào `/content/input_cache` làm bộ đệm đọc. Mọi sản phẩm vẫn
   ghi lên Drive.

**Kết quả mong đợi:** một bảng CLS 128 chiều cho 1325 ca (1229 + 96 external), sinh từ một M3T chưa
từng thấy subject nào của cohort; nối vào S8 rồi S7 như họ đặc trưng `m3t_`; một phán quyết đăng ký
trước về việc CLS có thay được radiomics không.

**Kỳ vọng thực tế:** y văn cho thấy ở cỡ mẫu ~1000, embedding đóng băng thường ngang chứ không thắng
radiomics. Nhưng M3T là mô hình **có giám sát trên KL** với kho dữ liệu gấp ~5 lần cohort, khác bản chất
với foundation model tự giám sát, nên cơ hội tốt hơn.

---

## Sự thật đã kiểm, dùng để thiết kế

- **Dữ liệu M3T:** `SAG_3D_DESS_v2_full.zip` 29,5 GB, 9.444 npz `{id}_{barcode8}_{LEFT|RIGHT}.npz`,
  mỗi npz `data` (120,160,160) uint16, đọc thẳng qua `zipfile` + `io.BytesIO`. Nhãn
  `unified_xray_mri_label.csv`: 8.149 gối / 4.116 subject, cột `id, side, mri_path, kl_grade,
  xray_path, subset`. Chia **theo subject** (đã kiểm rời nhau). Gần như chắc chắn chỉ V00. Bản cục bộ
  ở `C:\Users\Admin\Downloads\`; trên Drive qua gdown id `1k4EUOURlK6IxnmlYOV1vVmI4UFwIaym8` và
  `1GTXX5qt89wMY4NuwAJzCiveyx6APR8yL`.
- **Tiền xử lý suy luận:** chỉ min-max từng khối về [0,1] (numpy tái lập softmax ghi lại **chính xác**).
  Tăng cường khi train (torchio): RandomAffine ±15°, lật trục 0, một trong noise/bias/blur/motion p=0,75,
  rồi RescaleIntensity — **giữ đúng thứ tự** vì noise tính trên cường độ thô.
- **Huấn luyện gốc:** CE không trọng số, Adam lr 1e-4, batch 2, không scheduler, không seed, chọn theo val
  acc. Val acc epoch 201–248 dao động 0,636 ± 0,016, tức **chọn theo một epoch là đuổi nhiễu**.
- **Trọng số rò rỉ** `C:\Users\Admin\nnUnet-OAI\best_model.pth` (epoch 246) nạp khớp strict. Chỉ dùng
  cho kiểm port và thiết kế chuyển đổi — **không bao giờ** để trích đặc trưng.
- **Kiến trúc chỉ size-agnostic theo kích thước đầu vào.** Chi phí 2D cố định theo N×target_size², cấu
  hình thật mất 3 s trên CPU kể cả đầu vào 8³ → test phải thu nhỏ **cấu hình**
  (C3d=4, N=4, emb=16, C2d=8, target=12), không chỉ đầu vào. `target_size=None` vỡ với đầu vào không lập
  phương.
- **Cohort của mình:** ảnh DESS đến từ hai đường khác nhau (`OAI_DESS/` cho new_oai1000 và iMorphics;
  HuggingFace `Dataset001_KneeOA` cho OAI-ZIB kể cả 96 external). S1 gán cứng side="R" cho mọi ca OAI-ZIB
  → **side của chúng chưa kiểm**. Một số `dess_path` có thể trỏ đường cũ.
- **Rủi ro KL4:** tập train M3T chỉ có 167 gối KL4; trừ subject của mình có thể mất 40–50%.
- `rad_` **không** bị gán cứng trong `bsc/ordinal.py`; chỉ trong notebook (BIO_MASK, `ho_cua`, danh sách
  tên feature set).
- Đang có thay đổi **chưa commit** ở `bsc/docs/make_figs.py`, `bsc/docs/report_data.py` (sửa Hình 1) —
  việc riêng, **không đụng tới**.

---

## Cách làm

Tôi viết code, test, notebook, push lên `main` (theo quyền đẩy thường trực). **Bạn chạy notebook trên
Colab RTX 6000** và gửi lại output; tôi đọc và làm bước kế.

### 1. `bsc/m3t.py` — suy luận, không import torchio/nibabel

```
M3T_CFG, INPUT_SHAPE=(120,160,160), N_PARAMS=1_160_677, CLS_PREFIX="m3t_", LEAKY_HASHES
D3DBlock, MultiPlaneExtractor, D2DBlock, NonLinearProjection, PosPlaneEmbedding,
TransformerBlock, M3TModelFull            # chép nguyên văn cell 11–18
  + forward_features(x)->(B,128)          # token 0; forward = fc(forward_features)
build_m3t(**overrides)->model
load_m3t(path, device)->(model.eval(), weights_hash)   # state_dict thô hoặc {"model_state_dict"}, strict
minmax01(vol)->float32                    # khối hằng -> 0
extract(model, arrays, device, bs=2)->(cls[N,128], logits[N,5])
cls_frame(ids, cls, logits, prov)->DataFrame ; is_cls_col(c)       # m3t_000..127, m3tlogit_0..4
ORIENTATIONS ; resize3d(vol, shape, method) ; nifti_to_m3t(arr, spacing, spec)->(120,160,160)
volume_corr(a,b)->(r, slope) ; search_conversion(pairs, orients, methods)->DataFrame
conversion_gate(cls_a, cls_b, logit_a, logit_b)->dict
fingerprint(vol) ; max_corr(q, bank)->(r, idx)
```

### 2. `bsc/m3t_train.py` — huấn luyện, torchio import lười

```
parse_npz_name ; npz_index(zip_path)->DataFrame ; ZipNpzReader(zip).read(member)  # mở zip theo từng worker
stage_input(src, cache_root="/content/input_cache")->path   # HÀM DUY NHẤT được ghi /content; kiểm size + CRC 200 member
build_pool(labels, exclude_subjects)->DataFrame ; assert_subject_disjoint(a, b)
exposure_table(manifest, labels, index)->DataFrame          # khớp barcode trước, rồi subject+side, cờ mơ hồ
NpzKLDataset(df, reader, transform) ; train_transform()      # nguyên văn cell 8
M3TTrainConfig (đóng băng, .hash()) ; load_config(json)
train_one_epoch(...)->dict ; evaluate(...)->dict            # metric qua ORD.qwk / ORD.per_class_prf
select_epoch(history, window=5) ; should_stop(history, cfg)
save_checkpoint / load_checkpoint(run_dir) ; fit(model, loaders, run_dir, cfg, deadline)->history
write_once_csv(df, path)                                    # đã tồn tại thì assert bằng nhau, không ghi đè
```

`bsc/io_utils.py`: thêm `resolve_path(p, remaps)` sửa tiền tố `dess_path` cũ.
Cấu hình đăng ký trước: `bsc/configs/m3t_s9.json`.

### 3. Công thức huấn luyện — đăng ký trước, không sửa sau khi thấy kết quả S7/S8

- Kiến trúc, tăng cường, loss, Adam lr 1e-4, **batch 2** giữ nguyên bản gốc.
- **Một chỗ lệch có chủ ý:** chọn epoch theo **trung bình trượt 5 epoch của val QWK** thay vì val acc của
  một epoch, vì val acc dao động ±0,016 và bài toán sau là thứ tự.
- Dừng sớm: không cải thiện 0,005 trong 30 epoch; tối thiểu 60, tối đa 300.
- Seed theo từng epoch (kể cả generator của DataLoader) để chạy tiếp tái lập bit-đối-bit.
- Chạy tiếp qua nhiều phiên: `last.pt` ghi nguyên tử + `last_prev.pt` dự phòng, trạng thái chỉ kiểu
  Python thuần (nạp được `weights_only=True`), `epoch_XXX.pt` mỗi epoch; từ chối chạy tiếp nếu config,
  pool hay phiên bản torchio khác; hạn giờ để không bắt đầu epoch không kịp xong; file `STOP` để dừng êm.
- `num_workers ≥ 6`, `cudnn.benchmark`, fp32 (TF32 nếu GPU hỗ trợ).

### 4. Notebook `notebooks/biomarker_s9_m3t.ipynb` → ghi vào `.../knee_biomarkers_09_09/s9_m3t/`

| Mục | Việc | Cổng |
|---|---|---|
| 0 | Môi trường, in tên GPU và compute capability, nạp `m3t_s9.json`, chặn ghi đè | — |
| 1 | `stage_input`: chép zip vào `/content/input_cache`, kiểm CRC; đo tốc độ đọc Drive vs cục bộ | cần ≥ 40 GB trống |
| 2 | **G0 kiểm port:** trọng số rò rỉ trên đúng 1636 gối test gốc | **1073 ± 2 ca đúng** |
| 3 | Audit: resolve 1325 `dess_path`; phơi nhiễm theo subset × nguồn; tỉ lệ khớp barcode; đồng thuận KL giữa hai nguồn nhãn (chỉ báo cáo) | — |
| 4 | Pool: loại **mọi** subject của 1325 ca (cả 96 external), mọi bên gối, mọi lần khám, ở cả train/val/test của M3T | disjoint; **KL4 pool-train ≥ 84, nếu không thì dừng hỏi bạn** |
| 5 | Thiết kế chuyển đổi NIfTI → M3T bằng trọng số rò rỉ (không dùng nhãn): tìm theo **nguồn × bên gối**, khớp barcode trước, xác định trục 0,70 mm từ spacing, dò đồng thời 16 hướng × 4 kiểu resize | cổng rút gọn trên ~300 ca |
| 6 | Kiểm trùng ảnh: fingerprint mọi npz trong pool với mọi ca cohort đã chuyển đổi | **không trúng ca nào** |
| 7 | **Dừng để đăng ký trước:** commit config + mục Event.md | — |
| 8 | Benchmark 100 bước để đo thời gian thật; rồi `fit()` chạy lại mỗi phiên | epoch 30: val QWK trượt ≥ 0,5 |
| 9 | Chọn epoch; so M3T mới với M3T rò rỉ trên cùng tập test M3T đã thu hẹp | lệch acc và QWK đều ≤ 0,05 |
| 10 | Cổng chuyển đổi cuối (trọng số mới) | xem dưới |
| 11 | Trích CLS → `m3t_cls.csv` + metadata (hash trọng số, md5 pool, spec chuyển đổi, git SHA) | không NaN, case_id duy nhất, không chiều hằng, **hash ∉ LEAKY_HASHES** |
| 12 | Mốc trực tiếp: head M3T trên 246 ca test S8 và trên cả 1229 ca; tùy chọn: M3T rò rỉ tách theo phơi nhiễm để đo độ lạc quan | — |

**Cổng chuyển đổi, xét riêng từng nguồn × bên gối:**
- G4.1 ảnh: tương quan voxel trung vị ≥ 0,99, phân vị 1% ≥ 0,97, hơn phương án nhì ≥ 0,1, hệ số góc
  0,95–1,05.
- G4.2 CLS: láng giềng gần nhất của CLS chuyển đổi là chính CLS npz của nó ở ≥ 98% ca; khoảng cách
  trung vị tới chính nó ≤ 0,25 lần khoảng cách giữa các ca khác nhau. (Cosine đơn thuần là phép thử yếu vì
  CLS ra sau LayerNorm.)
- G4.3 dự đoán: head đồng ý ≥ 95%, |ΔE[KL]| trung bình ≤ 0,1.
- **Qua:** chuyển đổi cả 1325 ca qua **một** đường duy nhất. **Trượt:** chỉ dùng ca có npz, lọc split
  đã ghim mà không đổi thành viên train/test, và **mọi** feature set (kể cả radiomics) chạy trên cùng giao
  đó. Trộn nguồn là nhiễu thật: có npz hay không đi theo lần khám và nguồn, mà nguồn lệch phân bố KL.

### 5. Sửa S8 rồi S7

**S8** (`biomarker_s8_holdout.ipynb`), `PREV_DIR = s8_holdout_v2`, `OUT_DIR = s8_holdout_m3t` có chặn
ghi đè:
- Ô cấu hình: merge `m3t_cls.csv` với `validate="one_to_one"`; assert số dòng không đổi, **phủ 100%**
  (không để median lấp ca thiếu), kiểm provenance.
- Feature set mới: `rad_only`, `m3t_only`, `legacy_plus_m3t`, `s6_all_plus_m3t`,
  `s6_all_plus_radiomics_plus_m3t`, và ba bản `__nosel` cho các set CLS không có radiomics.
- `BIO_MASK` coi **cả** `rad_` và `m3t_` là nhánh ảnh (không thì CLS lặng lẽ rơi vào nhánh bio).
- Vòng lặp: `do_sel = len(cols) > 120 and not fs.endswith("__nosel")`; ghi số chiều CLS sống sót LASSO.
- Giữ assert A/B/C trùng khít v2 trên các set cũ (nếu trượt, so phiên bản thư viện trước khi nghi code).
- `ho_cua` thêm họ `m3t`; thay mọi `list(FEATURE_SETS)[-1]` bằng tên set tường minh.
- Mục mới: các phép so đăng ký trước + dòng head M3T.

**S7** (`biomarker_s7_ordinal.ipynb`), `OUT_DIR = s7_ordinal_m3t` (tên `s7_ordinal_v2` đã dành cho
việc khác), cùng các sửa trên; ô mới assert A/B/C fold QWK trùng `s7_ordinal/cv_fold_metrics.csv` cho 6
set cũ; ô phán quyết.

### 6. Phép so và phán quyết — đăng ký trước

| Câu hỏi | So |
|---|---|
| **Chính:** CLS thay được radiomics không | `s6_all_plus_m3t` vs `s6_all_plus_radiomics` |
| Phụ | `legacy_plus_m3t` vs `s5_radiomics` |
| Đứng riêng | `m3t_only` vs `rad_only` |
| Bổ sung | `s6_all_plus_radiomics_plus_m3t` vs `s6_all_plus_radiomics` |
| Gia tăng | `s6_all_plus_m3t` vs `s6_all` |
| Mô hình ảnh một mình | head M3T |

**Phán quyết trên S7** (S8 chỉ sàng lọc): "CLS thay được radiomics" nếu `ORD.bootstrap_delta` của QWK
trên OOF gộp cho cận dưới 95% > −0,02 ở **≥ 4/6** mô hình. Ngược lại: "không thay được".

### 7. Ghi chép

- `Event.md`: (a) mục ngoại lệ `/content/input_cache` — phạm vi, lý do, rào chắn; (b) mục **đăng ký
  trước** ở mục 7 notebook; (c) mục kết quả. CLAUDE.md giữ nguyên.
- `Progress.md`: cập nhật theo ngày.
- graphify chưa cài → ghi chú, không chặn.

---

## Xác minh

**Cục bộ, trước mỗi push** — `python -m pytest bsc/tests -v` từ `C:\Users\Admin\nnUnet-OAI`, toàn bộ test
cũ vẫn xanh. Test mới `bsc/tests/test_m3t.py`, CPU, cấu hình thu nhỏ, dưới 15 s:
1. Số tham số cấu hình thật = 1.160.677 và shape `pos_embedding` (không chạy forward).
2. `forward` = `fc(forward_features)` chính xác.
3. Cấu hình nhỏ chạy với shape lẻ và khi một trục ngắn hơn N; `target_size=None` báo lỗi với đầu vào không lập phương.
4. Nạp strict state_dict thô và bọc; hash; chặn hash rò rỉ.
5. `minmax01` các ca biên.
6. Zip tổng hợp: index và reader, kể cả pickle trước lần đọc đầu.
7. `build_pool` loại cả gối kia và mọi subset; chuẩn hóa subject ID.
8. Bảng phơi nhiễm: khớp barcode, fallback, cờ mơ hồ.
9. Phantom có hướng và resize biết trước được tìm lại đúng.
10. Shape đầu ra `resize3d`.
11. `volume_corr` với biến đổi tuyến tính.
12. Metric cổng: CLS trùng, lệch hằng, có nhiễu.
13. Fingerprint bắt được bản trùng cài sẵn.
14. `select_epoch` ưu tiên bình nguyên hơn đỉnh nhọn; quy tắc dừng.
15. Hai epoch liền = một epoch + chạy tiếp, bit-đối-bit; config khác thì báo lỗi.
16. Rào Drive-first, kể cả chặn đường bộ đệm cho sản phẩm.
17. Schema CSV và regex tên cột.
18. `evaluate` khớp `ORD.qwk`.

**Trên Colab:** các cổng G0, pool, trùng ảnh, epoch 30, sanity, G4.1–4.3, trích CLS như bảng mục 4; S8
tái lập A/B/C trùng khít v2; S7 tái lập fold QWK của 6 set cũ.

**Commit:** chỉ stage đường dẫn tường minh, kiểm `git diff --cached --name-only` trước mỗi commit, không
đụng `bsc/docs/make_figs.py` và `bsc/docs/report_data.py`. Thứ tự: (1) `m3t.py` + test → (2) `m3t_train.py` + test + config → (3) notebook s9 mục
0–7 → push, bạn chạy tới mục 7 → (4) đăng ký trước → bạn chạy mục 8–12 → (5) sửa S8 → bạn chạy →
(6) sửa S7 → bạn chạy → phán quyết.

## Ngân sách tính toán

| Bước | Thời gian |
|---|---|
| Chép zip, mỗi phiên | 5–12 phút |
| G0, audit, pool | ~20 phút |
| Thiết kế chuyển đổi + kiểm trùng | 30–60 phút |
| Huấn luyện RTX 6000 | **chưa đo được**: nếu là Ada/Blackwell thì nút thắt chuyển sang tăng cường trên CPU (~0,4 s/khối), ước ~10–23 giờ; nếu là Quadro Turing thì có thể 24–50 giờ. **Benchmark 100 bước ở mục 8 thay số ước bằng số đo** trước khi bạn cam kết GPU |
| Cổng cuối + trích CLS | 30–60 phút |
| S8 m3t | ~10 phút CPU |
| S7 m3t | 1,5–2 giờ CPU |

## Rủi ro đã biết

- **KL4 cạn trong pool** → cổng mục 4 dừng để bạn chọn: chấp nhận, thêm 1.295 npz ngoài CSV (vẫn loại
  theo subject), hoặc trọng số lớp.
- **Chuyển đổi trượt** → thu hẹp về giao có npz; mọi set cùng chạy trên giao đó.
- **Side sai của OAI-ZIB** làm hỏng loại theo ID → kiểm trùng ảnh ở mục 6 là lưới an toàn.
- **Nhãn KL hai nguồn có thể lệch** (M3T lấy từ bộ X-quang, mình lấy từ `KXR_SQ_BU00`) → chỉ báo cáo;
  không ảnh hưởng tính sạch vì M3T không bao giờ thấy nhãn của cohort.
- **CLS 128 chiều qua LASSO** có thể bị cắt mạnh → bản `__nosel` là phép nhạy.

## Ngoài phạm vi

Fine-tune M3T trong từng fold; Curia-2 / DINOv2; nhánh ảnh end-to-end; sửa Hình 1 đang dở ở `bsc/docs/make_figs.py` và `bsc/docs/report_data.py`.
