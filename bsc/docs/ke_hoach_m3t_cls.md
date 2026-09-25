# Kế hoạch: cải thiện M3T bằng biomarker từ segmentation

> **Trạng thái:** ĐANG TRIỂN KHAI (25/09/2026) — code bước (1)–(4) đã push: `ordinal.bootstrap_delta`/`classify_delta`,
> `m3t.py`, `m3t_train.py`, `configs/m3t_s9.json` (**status DRAFT**), notebook S9 mục 0–12 và S10 phần audit. Đã chạy
> thử cục bộ trên dữ liệu tổng hợp; **chưa chạy Colab**. Đăng ký trước chỉ hoàn tất khi config chuyển REGISTERED sau
> output S9 mục 0–7. B `__nosel` là phép so chính, δ=0,02 là ngưỡng vận hành. Chỗ triển khai khác chữ: xem cuối file.
> **Ngày lập:** 24/09/2026. **Liên quan:** `Event.md`, `docs/bao_cao_14_09.md`, `notebooks/biomarker_s7_ordinal.ipynb`, `notebooks/biomarker_s8_holdout.ipynb`.

## Context

**Mục tiêu chính (làm rõ ngày 25/09/2026):** dùng **M3T làm mô hình ảnh nền** cho phân loại KL,
đo xem thông tin từ segmentation và biomarker có cải thiện dự đoán của M3T hay không.
Segmentation cung cấp cấu trúc giải phẫu để tính biomarker; CLS 128 chiều cung cấp đặc trưng ảnh
để kết hợp với các biomarker đó trong pipeline S7/S8. So với radiomics là phép đối chiếu phụ.

**Phạm vi hiện tại:** tiếp tục dùng M3T. Các SOTA foundation model là hướng thử sau, khi người dùng
quyết định chuyển hướng; không phải điều kiện để triển khai thí nghiệm hiện tại.

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
từng thấy subject nào của cohort; nối vào S8 rồi S7 như họ đặc trưng `m3t_`; đo giá trị gia tăng
của biomarker khi kết hợp với M3T, trên cùng ca và cùng split.
Tên `external` được giữ để tương thích manifest; 96 ca này là **holdout cùng nguồn OAI-ZIB**,
không phải dữ liệu ngoài độc lập. Biomarker hiện chỉ có cho 1229 ca; nhánh 96 ca cần dựng riêng
ở mục 5b trước khi có thể xác nhận pipeline kết hợp.

**Giả thuyết cần kiểm:** các biomarker hình học và bề mặt có thể bổ sung thông tin cho đặc trưng ảnh
M3T. Cần tách mức lợi do thêm biomarker khỏi mức lợi chỉ do học lại bộ phân loại trên CLS.
Giai đoạn này đóng băng encoder M3T và học bộ phân loại kết hợp; chưa fine-tune encoder end-to-end.

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
- **Độ phủ biomarker:** S6 tìm thấy mask cho 1229 ca, bảng v2 có đúng 1229 dòng; 96 ca giữ lại
  chưa có biomarker S3/S6. Không suy từ manifest 1325 dòng rằng đặc trưng đã phủ đủ 1325 ca.
- **96 ca giữ lại:** KL0/1/2/3/4 = **21/12/22/26/15** theo đối chiếu của người dùng ngày
  25/09/2026; kiểm lại số đếm khi dựng manifest riêng, không chọn mô hình bằng nhãn holdout.
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

### 2b. `bsc/ordinal.py` — bootstrap ghép cặp theo subject và seed

Mở rộng `bootstrap_delta(..., seed=0, alpha=0.05, *, groups=None)`; giữ nguyên các tham số cũ,
giá trị mặc định, chiều hiệu **metric(b) - metric(a)** và schema kết quả.
Thêm `classify_delta(result, min_delta=0.02)` để áp dụng một định nghĩa bốn mức ở mục 6,
dùng chung trong S7/S8 và test các biên; không tự suy mức từ các chuỗi hiển thị trong notebook.

- `y_true`: `(N,)`; hai mảng dự đoán cùng shape `(N,)` hoặc `(N, S)`, trục 1 là seed chia fold.
  `groups`: `(N,)`. Không tự flatten đầu vào hai chiều; sai shape, ID thiếu hoặc dự đoán thiếu thì lỗi.
- Mỗi lượt lấy có hoàn lại G subject trong G subject duy nhất; một subject được lấy k lần thì
  đưa toàn bộ các dòng của nó vào k lần. Dùng cùng chỉ số cho hai nhánh và cho mọi seed.
- Tính lại QWK trên toàn mẫu bootstrap của từng seed, rồi lấy trung bình hiệu qua seed.
  Point estimate cũng là trung bình hiệu QWK qua seed; không gộp các seed thành N×S quan sát.
- Không có `groups`: resample từng ca như cũ. Với đầu vào 1D và `groups=None`, kết quả phải
  khớp hàm cũ khi cùng seed. Dùng metric hiện có, không tạo định nghĩa QWK thứ hai.
- Phân tích chính: **10.000 bootstrap, seed=0, CI percentile hai phía 95%**. Báo thêm hiệu
  từng seed. Cohort chỉ có 14 subject lặp trong 1215 subject nên không kỳ vọng đổi CI nhiều.

Notebook S9 thêm bước kiểm độ chính xác thống kê trước đăng ký: trên dự đoán OOF cũ của B,
so `s6_all` (a) với `s6_all_plus_radiomics` (b), lưu Δ, CI và độ rộng CI. Đây là kiểm tra độ
chính xác có thể đạt, **không dùng độ rộng CI để định nghĩa mức cải thiện có giá trị thực dụng**;
CI của cặp này cũng không dự báo chính xác CI của cặp M3T vì cấu trúc sai số có thể khác.
Chưa tìm thấy CSV dự đoán OOF trong các file repo đã kiểm; chưa tính được CI thật. Nếu artifact cũ chỉ lưu seed 0,
ghi rõ pilot một seed hoặc tái xuất OOF đủ ba seed từ lần chạy có provenance; không nhân bản seed 0.

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
| 3 | Audit: resolve 1325 `dess_path`; phơi nhiễm theo subset × nguồn; tỉ lệ khớp barcode; xác minh subject/bên/lần khám, bảng đồng thuận KL giữa hai nguồn nhãn | dùng để diễn giải so sánh head với downstream; không sửa nhãn theo kết quả |
| 4 | Pool: loại **mọi** subject của 1325 ca (cả 96 external), mọi bên gối, mọi lần khám, ở cả train/val/test của M3T | disjoint; **KL4 pool-train ≥ 84, nếu không thì dừng hỏi bạn** |
| 5 | Thiết kế chuyển đổi NIfTI → M3T bằng trọng số rò rỉ (không dùng nhãn): tìm theo **nguồn × bên gối**, khớp barcode trước, xác định trục 0,70 mm từ spacing, dò đồng thời 16 hướng × 4 kiểu resize | cổng rút gọn trên ~300 ca |
| 6 | Kiểm trùng ảnh: fingerprint mọi npz trong pool với mọi ca cohort đã chuyển đổi | **không trúng ca nào** |
| 7 | Pilot bootstrap trên cặp S7 cũ; **đăng ký trước** toàn bộ giao thức mục 6 và checklist mục 7; commit config + Event.md | không dùng kết quả M3T downstream để chọn giao thức |
| 8 | Benchmark 100 bước để đo thời gian thật; rồi `fit()` chạy lại mỗi phiên | epoch 30: val QWK trượt ≥ 0,5 |
| 9 | Chọn epoch; so M3T mới với M3T rò rỉ trên cùng tập test M3T đã thu hẹp | lệch acc và QWK đều ≤ 0,05 |
| 10 | Cổng chuyển đổi cuối (trọng số mới) | xem dưới |
| 11 | Trích CLS checkpoint chính → `m3t_cls.csv` + metadata; trích thêm riêng từng checkpoint trong cửa sổ 5 epoch đã ghim để kiểm độ nhạy | không NaN, case_id duy nhất, không chiều hằng, **hash ∉ LEAKY_HASHES** |
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
- **Phép so chính:** B (`B_xgb_frankhall`), `s6_all_plus_m3t__nosel` (198 cột) với
  `m3t_only__nosel` (128 cột). Assert hai nhánh giữ cùng danh sách và giá trị 128 cột CLS.
  Không lọc tương quan hay LASSO ở hai nhánh này; bản có chọn đặc trưng là phép nhạy.
- `BIO_MASK` coi **cả** `rad_` và `m3t_` là nhánh ảnh (không thì CLS lặng lẽ rơi vào nhánh bio).
- Vòng lặp: `do_sel = len(cols) > 120 and not fs.endswith("__nosel")`; ghi số chiều CLS sống sót LASSO.
- Giữ assert A/B/C trùng khít v2 trên các set cũ (nếu trượt, so phiên bản thư viện trước khi nghi code).
- `ho_cua` thêm họ `m3t`; thay mọi `list(FEATURE_SETS)[-1]` bằng tên set tường minh.
- Mục mới: các phép so đăng ký trước + dòng head M3T.

**S7** (`biomarker_s7_ordinal.ipynb`), `OUT_DIR = s7_ordinal_m3t` (tên `s7_ordinal_v2` đã dành cho
việc khác), cùng các sửa trên; ô mới assert A/B/C fold QWK trùng `s7_ordinal/cv_fold_metrics.csv` cho 6
set cũ; ô **đóng góp biomarker cho M3T** theo bốn mức ở mục 6. Lưu dự đoán OOF đủ ba seed,
case_id, subject, KL, fold, feature set, model và hash checkpoint để bootstrap ghép cặp được.

### 5b. Dựng và đánh giá 96 ca holdout cùng nguồn

Notebook mới `notebooks/biomarker_s10_oaizib_holdout.ipynb`, thư mục sản phẩm mới
`.../knee_biomarkers_09_09/s10_oaizib_holdout/`. Không ghi thêm vào bảng v2 1229 ca cũ.

1. Ghim manifest 96 ca, xác minh subject/bên/lần khám, ảnh tồn tại và rời subject với 1229 ca.
   Audit danh sách train và validation/chọn checkpoint của **mô hình segmentation thực sự dùng**,
   kể cả mọi fold trong ensemble và các bộ dữ liệu union/iMorphics liên quan. Kiểm cả hai bên gối,
   mọi lần khám. Ghi hash checkpoint và bằng chứng loại subject, không chỉ tin nhãn `test`.
   Nếu có phơi nhiễm hoặc thiếu provenance thì chưa được gọi đây là xác nhận trên ca chưa thấy;
   phải giải quyết checkpoint sạch trước khi đánh giá. Ghi cả việc đã dùng tập này để chọn thiết kế.
2. Chạy segmentation 96 ảnh bằng checkpoint đã khóa; kiểm đủ mask, nhãn, hình học và QC bằng ảnh.
   Mask dùng trích đặc trưng phải là dự đoán AI, không thay bằng GT để cải thiện chất lượng đầu vào.
3. Tính đủ **15 biomarker S3 + 55 biomarker S6** bằng đúng định nghĩa/config của cohort phát triển.
   Tái dùng hàm đã kiểm trong `biomarkers.py` và các bước trích S3/S6; không dựng lại cohort bằng
   notebook S1–S5 cũ vốn đã lỗi thời. Lưu bảng 96 dòng riêng, manifest mask, QC và provenance.
4. Ghép CLS từ M3T sạch theo case_id, `validate="one_to_one"`; assert đủ 96 ca, không dùng imputation
   để che ca thiếu mask/thiếu cả họ đặc trưng. Ca hỏng kỹ thuật được báo rõ, không thay bằng ca khác.
5. Sau khi khóa giao thức: fit lại đúng hai nhánh B chính trên toàn bộ 1229 ca, fit mọi preprocessing
   chỉ trên cohort phát triển, rồi đánh giá holdout một lần. Không dùng holdout để chọn checkpoint,
   ngưỡng phân loại, feature set hoặc tham số; không sửa conversion bằng kết quả KL của holdout.
6. **Radiomics là tùy chọn phụ trên holdout:** nếu tiếp tục phép so radiomics ở đây thì phải trích
   đủ radiomics cho 96 ca bằng cấu hình S5 đã ghim. Thiếu radiomics không chặn phép so chính CLS±bio.

Đăng ký tiêu chí **ủng hộ về chiều**: khi ΔQWK chính trên S7 > 0, ΔQWK holdout > 0 là cùng chiều;
không bắt buộc CI loại 0. Vẫn báo Δ, CI, confusion matrix và metric từng lớp. Δ=0 là không có chiều;
Δ<0 là ngược chiều. Cùng chiều ở n=96 là bằng chứng hỗ trợ yếu, không được nâng mức kết luận S7
hay gọi là xác nhận có ý nghĩa thống kê. Nếu S7 làm tệ hơn và holdout cũng âm thì ghi lặp lại tác hại.
Nếu conversion chỉ cho phép dùng giao có NPZ, giữ nguyên thành viên split và báo lại n/KL thực tế;
không tiếp tục gọi đó là đánh giá đủ 96 ca.

### 6. Đóng góp biomarker cho M3T — phép so chính và bốn mức kết luận

| Câu hỏi | So |
|---|---|
| **Chính:** biomarker có cải thiện nhánh ảnh M3T không | B: `s6_all_plus_m3t__nosel` vs `m3t_only__nosel`, cùng checkpoint và split |
| Giá trị riêng của biomarker bề mặt S6 | B: `s6_all_plus_m3t__nosel` vs `legacy_plus_m3t__nosel` |
| Cải thiện so với mô hình ảnh ban đầu | B: `s6_all_plus_m3t__nosel` vs head M3T đóng băng |
| Đối chứng cho việc học lại bộ phân loại và thích nghi nhãn | B: `m3t_only__nosel` vs head M3T đóng băng |
| Giá trị của ảnh bên cạnh biomarker | B: `s6_all_plus_m3t__nosel` vs `s6_all` |
| Phụ: đối chiếu với pipeline radiomics | `s6_all_plus_m3t` vs `s6_all_plus_radiomics`; `m3t_only` vs `rad_only` |
| Phụ: kết hợp cả ba nguồn đặc trưng | `s6_all_plus_radiomics_plus_m3t` vs `s6_all_plus_m3t` |

**Ghim mô hình chính:** B Frank–Hall, tham số XGB hiện có của S7, seed XGB=42, ép đơn điệu và
giải mã đếm ở 0,5 như baseline; không dò lại ngưỡng. S7 giữ 5 fold × seed chia `[0,1,2]` theo subject.
Pin phiên bản thư viện và môi trường; seed cố định không bảo đảm bit-đối-bit khi đổi thư viện/phần cứng.
Lý do chọn B: dẫn đầu bộ đầy đủ trong kết quả S7 cũ (QWK 0,776); lựa chọn trước khi có kết quả M3T.

**Estimand:** Δ = trung bình qua ba seed của `QWK(CLS+bio) - QWK(CLS)` trên OOF gộp từng seed.
Ghim **δ = +0,02 QWK** làm ngưỡng vận hành cho mức lợi đáng theo đuổi khi thêm nhánh biomarker.
Đây là quyết định thực dụng của thí nghiệm, chưa phải ngưỡng có ý nghĩa lâm sàng đã được xác thực.
Pilot CI cũ kiểm độ chính xác; không nâng δ theo độ rộng CI, không sửa δ sau khi thấy M3T.
Nếu CI rộng, chấp nhận kết luận chưa đủ bằng chứng thay vì thay mục tiêu. Xem phân biệt giữa độ
chính xác và mức hiệu quả đáng quan tâm trong [Lakens, Sample Size Justification](https://lakens.github.io/statistical_inferences/08-samplesizejustification.html).

Gọi `[L,U]` là CI 95% của Δ theo mục 2b. Bốn mức, dùng bất đẳng thức nghiêm:

| Mức | Điều kiện | Diễn giải |
|---|---|---|
| Cải thiện vượt ngưỡng | `L > δ` | Có bằng chứng mức lợi lớn hơn +0,02 QWK |
| Cải thiện dưới ngưỡng | `L > 0` và `U < δ` | Có bằng chứng mức lợi dương nhưng nhỏ hơn +0,02 |
| Làm tệ hơn | `U < 0` | Có bằng chứng Δ âm |
| Chưa đủ bằng chứng để phân mức | Mọi trường hợp còn lại, kể cả chạm biên | Báo rõ bất định nằm ở dấu hay ở độ lớn |

Trường hợp `0 < L <= δ <= U` đã có bằng chứng cải thiện, nhưng chưa xác định được có vượt δ;
ghi rõ điều này trong mức cuối, không viết thành “chưa thấy cải thiện”. Luôn công bố Δ và CI gốc.
Báo thêm MAE, tỷ lệ lệch ≥2 bậc, recall/precision KL4; nếu các metric này xấu đi thì nêu trade-off,
không diễn giải mức QWK thành cải thiện mọi mặt. Các mô hình khác và cặp có LASSO là phân tích phụ.

**Diễn giải head:** head M3T học KL từ bộ X-quang, downstream học nhãn `KXR_SQ_BU00`.
Audit đồng thuận nhãn phải xác minh cùng subject/bên/lần khám, báo ma trận nhầm lẫn và chiều lệch
theo lớp/nguồn. Lợi ích của `m3t_only__nosel` so với head có thể gồm thích nghi nguồn nhãn,
không chỉ học bộ phân loại tốt hơn. Phép so chính dùng cùng nhãn cho cả hai nhánh nên kiểm soát
được yếu tố này; không sửa nhãn theo dự đoán hoặc chọn riêng ca đồng thuận để làm kết quả chính.

**Độ nhạy checkpoint:** chọn cửa sổ trượt trailing 5 epoch `[t-4,...,t]` có trung bình val QWK
cao nhất trong các cửa sổ đầy đủ; hòa thì chọn t nhỏ nhất. Checkpoint chính là epoch t; không
chọn lại bằng S7/S8. Trích CLS riêng ở đủ 5 checkpoint và chạy lại chỉ phép so B `__nosel`,
giữ fold/config. Báo Δ từng checkpoint, min/max và số checkpoint cùng chiều; không chọn checkpoint
tốt nhất, không gộp 5 checkpoint thành 5 mẫu độc lập. Đây là độ nhạy với thời điểm dừng trong
**một** lần train, không thay cho đo biến động giữa các seed train encoder. CI OOF cũng chưa đo nó.

**Giới hạn phạm vi:** chỉ kiểm **ghép muộn** biomarker dạng bảng với CLS đóng băng. Kết quả âm
không bác bỏ việc segmentation giúp M3T qua mask làm kênh ảnh, crop vùng khớp hay tương tác trong
encoder; các cách đó chưa được thử ở đây. Không suy rằng thay đổi Dice tự nó gây thay đổi KL.
Nhắc crop là giới hạn suy luận, không mở lại ROI cascade segmentation vốn đã có kết quả âm.

S8 dùng kiểm pipeline theo cấu hình đã ghim; nếu dùng kết quả S8 để chọn phương án thì S7 vẫn là
đánh giá nội bộ trên cùng cohort, không phải xác nhận độc lập. Chuẩn bị và đánh giá 96 ca holdout
theo mục 5b, với mức khẳng định chỉ là hỗ trợ về chiều trong cùng nguồn dữ liệu.

Luật cũ “CLS thay radiomics nếu ≥4/6 mô hình có cận dưới ΔQWK > −0,02” không còn là tiêu chí
thành công chính của dự án.

### 7. Ghi chép

- `Event.md`: (a) ngoại lệ `/content/input_cache` — phạm vi, lý do, rào chắn; (b) **đăng ký trước**
  ở mục 7 notebook; (c) kết quả. Checklist đăng ký: mục tiêu ghép muộn; B và toàn bộ XGB/decode;
  hai set `__nosel` với danh sách 128/198 cột; fold/seed/case_id; δ=0,02 và bốn mức kết luận;
  schema bootstrap `(N,S)`, groups, 10.000 draw, seed 0, CI 95%; pilot CI cũ và giới hạn của nó;
  checkpoint chính/cửa sổ độ nhạy; audit nhãn; nhánh mask→S3/S6 của 96 ca, audit segmentation,
  tiêu chí cùng chiều; điều kiện chạy radiomics phụ; provenance/hash/config và quy tắc giao NPZ.
  Ghi rõ code nào mới là kế hoạch, phép kiểm nào đã chạy; không ghi đăng ký đã hoàn tất trước
  khi config và split thực tế được commit. CLAUDE.md giữ nguyên.
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

**Bootstrap — bổ sung `bsc/tests/test_ordinal.py`:** tương thích chính xác lời gọi 1D cũ;
resample cả cụm có nhiều ca và cả cụm được lấy lặp; cùng draw qua seed/nhánh; tính lại QWK từng
seed rồi lấy trung bình; lặp một cột seed không làm CI hẹp giả; đảo a/b đảo dấu Δ và CI; cùng
dự đoán cho Δ=CI=0; shape/group thiếu/NaN sai thì lỗi. Test bảng bốn mức gồm CI cắt 0, cắt δ
và chạm biên; checkpoint chính không phụ thuộc bất kỳ score downstream nào.

**Trên Colab:** các cổng G0, pool, trùng ảnh, epoch 30, sanity, G4.1–4.3, trích CLS như bảng mục 4; S8
tái lập A/B/C trùng khít v2; S7 tái lập fold QWK của 6 set cũ, xuất OOF đủ seed; kiểm hai nhánh
chính có cùng CLS và đúng 128/198 cột; nhánh holdout đủ mask/S3/S6/CLS, QC/provenance và rời subject.

**Commit:** chỉ stage đường dẫn tường minh, kiểm `git diff --cached --name-only` trước mỗi commit, không
đụng `bsc/docs/make_figs.py` và `bsc/docs/report_data.py`. Thứ tự: (1) mở rộng bootstrap + test,
xuất/pilot OOF cũ → (2) `m3t.py` + test → (3) `m3t_train.py` + test + config → (4) notebook S9
mục 0–7 và S10 phần audit/chuẩn bị 96 ca → chạy audit, pilot → (5) commit đăng ký trước/config
thực tế → (6) chạy S9 mục 8–12, dựng mask/S3/S6 holdout → (7) sửa/chạy S8 rồi S7 với B `__nosel`,
bốn mức kết luận và độ nhạy checkpoint → (8) khóa mô hình, chạy S10 đánh giá holdout một lần.

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
| Bootstrap 10.000 draw × 3 seed | Chưa đo; benchmark với OOF thật trước khi chạy đầy đủ |
| Trích CLS 4 checkpoint bổ sung + chạy lại cặp B chính | Chưa đo; không chạy lại toàn bộ ma trận mô hình |
| Audit + segmentation 96 ca + S3/S6 + đánh giá S10 | Chưa đo; benchmark inference/trích biomarker, cộng riêng vào ngân sách |
| Radiomics 96 ca (nếu chạy phép so phụ) | Chưa đo; không là điều kiện chạy phép so chính |

## Rủi ro đã biết

- **KL4 cạn trong pool** → cổng mục 4 dừng để bạn chọn: chấp nhận, thêm 1.295 npz ngoài CSV (vẫn loại
  theo subject), hoặc trọng số lớp.
- **Chuyển đổi trượt** → thu hẹp về giao có npz; mọi set cùng chạy trên giao đó.
- **Side sai của OAI-ZIB** làm hỏng ghép ảnh/nhãn/conversion; loại đúng theo subject vẫn phải
  loại được cả hai bên. Kiểm trùng ảnh ở mục 6 là lớp kiểm bổ sung, không thay audit subject.
- **Nhãn KL hai nguồn có thể lệch** (M3T lấy từ bộ X-quang, mình lấy từ `KXR_SQ_BU00`) →
  dùng audit để diễn giải lợi ích của downstream so với head theo mục 6. Loại subject vẫn giữ
  đánh giá M3T sạch nhưng không xóa khác biệt định nghĩa/nguồn nhãn.
- **CLS 128 chiều qua LASSO** có thể bị cắt khác nhau khi thêm biomarker → `__nosel` là chính,
  LASSO chỉ là phép nhạy.
- **Holdout chưa có biomarker và chưa audit segmentation** → phải hoàn thành mục 5b; n=96 và
  cùng nguồn OAI-ZIB giới hạn sức thống kê lẫn mức khái quát của kết quả.

## Ngoài phạm vi

Fine-tune M3T trong từng fold; Curia-2 / DINOv2; nhánh ảnh end-to-end; sửa Hình 1 đang dở ở `bsc/docs/make_figs.py` và `bsc/docs/report_data.py`.

## Ghi chú triển khai (25/09/2026) — chỗ code khác chữ trong kế hoạch

Không chỗ nào đổi phép so chính, δ, bootstrap hay quy tắc chọn checkpoint. Các chỗ chi tiết hóa:

- **Kiểu resize: 5, không phải 4** (`trilinear`, `trilinear_ac`, `area`, `nearest`, `zoom1`) — tập lớn hơn chỉ làm cổng
  margin khó hơn.
- **"Nguồn" của cổng chuyển đổi = họ file NIfTI** (`OAI_DESS`, `D012_imorphics`, `D001_oaizib`, … theo đường dẫn),
  vì công cụ ghi NIfTI mới quyết định hướng trục; `source_dataset` trộn nhiều họ.
- **Thêm bước A dò rộng 48 hướng** trên 12 ca chắc cùng lần chụp để **kiểm** giả định "trục 0,70 mm → trục 0 của M3T"
  thay vì mặc nhiên; bước B dò 16 hướng theo trục tìm được (`dst_axis`).
- **Bên gối OAI-ZIB xác định bằng ảnh:** so với npz của **cả hai** gối của subject (`pick_best_tag` ở mục 5, dấu vân
  tay ở mục 6 cho mọi ca); barcode chỉ gối kia thì dùng bên của barcode (`side_img`).
- **Ngưỡng dấu vân tay 0,90**, hiệu chỉnh trên npz thật trước khi thấy bất kỳ kết quả cohort nào (khác subject: max
  0,852 trên 3000 npz; cùng gối khác lần chụp: trung vị 0,957). Trúng thì dừng để xem, không tự loại.
- **Pilot bootstrap một seed** vì `s7_ordinal` cũ chỉ lưu OOF seed 0 — đúng phương án kế hoạch cho phép.
- **Mục 11 lưu thêm logit head của M3T rò rỉ** (`m3t_leaky_head.csv`, **không có CLS**) chỉ để đo độ lạc quan ở mục 12.
- **Nhóm họ × bên có < 2 ca thiết kế** thì cổng CLS trượt ("không kiểm được") thay vì báo lỗi.
- `fit` **không bao giờ train lại từ đầu** khi thư mục run có checkpoint mà không nạp được — dừng để kiểm file.

## Cập nhật 26/09/2026 — cổng KL4 trượt, chuyển sang pool v2

Chạy S9 trên Colab: loại 1.308 subject cohort thì pool (chỉ CSV unified) còn 5.772/8.149 gối (−29%), nhưng KL4 mất
60% (244 → 97; **train 167 → 65 < 84**) — cohort lấy mẫu cân bằng KL đã lấy phần lớn gối KL4 của M3T.
Kiểm thêm: `label.csv` **bên trong zip** có nhãn cho **cả 9.444 npz** (trùng KL 99,35% với CSV unified trên 8.149 gối
chung; 53 gối lệch, chủ yếu KL0↔1), tức có thêm 1.295 npz / 701 subject, trong đó 74 gối KL4. Phép chia
train/val/test đi kèm zip **không** chia theo subject (2.151 subject ở nhiều tập) nên không dùng.

**Quyết định của người dùng (26/09):** pool v2 = CSV unified + 1.295 npz đó, nhãn từ `label.csv` (md5 ghim trong
config), tập con gán theo subject (subject đã có giữ tập cũ; subject mới vào train), vẫn loại mọi subject cohort.
Nếu v2 vẫn dưới 84 thì **chấp nhận con số thật, không đổi loss**. Pool v1 (`pool.csv`) giữ nguyên làm hồ sơ.
Hệ quả: tập train của M3T mới **khác** tập train của M3T gốc (thêm dữ liệu); mục 9 so với M3T gốc vẫn trên tập test
đã thu hẹp, nơi cả hai mô hình đều chưa thấy subject nào.

**Đăng ký trước (26/09/2026):** theo quyết định người dùng để train qua đêm, config chuyển REGISTERED **trước** khi chạy
mục 5–7 (thay vì sau khi Claude đọc output mục 0–7). Vẫn trước mọi kết quả S7/S8. Các nhánh mục 5–7 đã định sẵn;
ảnh cohort trùng pool chặn train; cổng epoch 30 tự dừng train. Notebook chạy bằng `Run all` mỗi phiên.
