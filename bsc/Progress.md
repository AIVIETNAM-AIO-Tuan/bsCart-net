# Progress — nhánh biomarker / ordinal

Ghi theo ngày, chỉ giữ thông tin cần để tiếp tục công việc. Quyết định nghiên cứu lớn nằm ở
`Event.md`, không chép lại ở đây.

---

## 2026-09-25

### Completed
- Triển khai `docs/ke_hoach_m3t_cls.md` bước (1)–(4), xem `Event.md` 2026-09-25 (hai mục):
  `ordinal.bootstrap_delta(groups=)` + `seed_deltas` + `classify_delta`; `bsc/m3t.py`; `bsc/m3t_train.py`;
  `configs/m3t_s9.json` (DRAFT); `io_utils.resolve_path`, `assert_drive_first` kiểm chuỗi POSIX, `load_nii` nhận
  NIfTI 4D có chiều cuối 1; notebook `biomarker_s9_m3t` (mục 0–12) và `biomarker_s10_oaizib_holdout` (mục 0–2).
- Kiểm port cục bộ trên trọng số thật: softmax ghi trong notebook gốc tái lập đúng; `minmax01` bit-đối-bit torchio.
- Hiệu chỉnh ngưỡng dấu vân tay trên npz thật → 0,90.
- Chạy thử S9 và S10 end-to-end bằng harness cục bộ trên dữ liệu tổng hợp (script trong scratchpad, không commit).

### Critical Changes
- `bootstrap_delta`: tham số mới chỉ có từ khoá `groups=`; đầu vào NaN / -1 / sai shape giờ **báo lỗi** thay vì ép
  kiểu im lặng. Đường 1D không groups byte-identical (test so với bản sao nguyên văn).
- `m3t.load_m3t` / `extract` **chặn** trọng số rò rỉ trừ khi `allow_leaky=True`.
- `m3t_train.fit` từ chối chạy tiếp khi cfg/pool/torchio khác, và **không train lại từ đầu** khi có checkpoint hỏng.

### Findings
- Zip M3T có 56 gối với **hai** lần chụp → npz không chỉ V00; ghép theo subject+bên phải cờ mơ hồ.
- Dấu vân tay thô tách được cùng gối (0,96) khỏi khác subject (≤ 0,85) nhưng **không** tách được mọi cặp khác lần chụp.
- `torch.__version__` là lớp con của `str` → làm hỏng `torch.load(weights_only=True)` nếu lưu thẳng vào checkpoint.

### Failures / Risks
- Chưa chạy Colab: mọi cổng S9 (G0, KL4, chuyển đổi, trùng ảnh, epoch 30, sanity) và audit S10 chưa có số thật.
- Mô hình segmentation sinh mask 09_09 chưa xác nhận (giả định d20 fold 0).
- Việc cũ còn treo: sửa Hình 1 (`docs/make_figs.py`, `docs/report_data.py` chưa commit); ghi Event.md về quantile
  cutpoints thua trên dữ liệu thật.

### Decisions
- Mục tiêu mới + train lại không rò rỉ + ngoại lệ `/content/input_cache`: `Event.md` 2026-09-25.

### Current State
- **Verified working (cục bộ):** code + test; harness tổng hợp S9/S10.
- **Unverified:** mọi thứ trên dữ liệu thật; thời gian train RTX 6000.

### Next
1. Chạy S9 mục 0–7 và S10 mục 0–2 trên Colab, gửi output.
2. Commit đăng ký trước (config REGISTERED + Event.md), rồi train S9 mục 8.
3. Sau S9 mục 11: sửa S8 → S7 với feature set `m3t_` và phép so chính.

---

## 2026-09-13

### Completed
- **S6 chạy xong toàn cohort.** 1229/1229 ca, 0 lỗi, ~15,7 s/ca. Bảng
  `s6_fcl/biomarker_table_v2.csv`, 1229 × 88. Cột legacy khớp bảng S3 tới 1,16e-10.
- **S7 chạy xong** với đủ 6 feature set và đối chứng A2. 540 tổ hợp, 41,3 phút.
- **S8 chạy xong** với 10 model × 5 feature set, holdout một lần n_test=246, 3,6 phút.
- Thêm `s6_surface_only` vào S8 cho khớp bộ feature set của S7 (`61317e9`).
- Sổ tay giải thích S6 + S7: https://claude.ai/code/artifact/ad58bf9c-96ef-42b5-bd90-86eba3cfd980
- **Tầng quyết định** viết xong trong `bsc/ordinal.py` và nối vào S8 (`d629de8`, notebook kế
  tiếp). Test 24 → 36, đậu hết. **Chưa chạy trên Colab.** Chi tiết ở `Event.md` mục 13/9 thứ hai.

### Critical Changes
- **`fit_cutpoints` thêm `objective` / `min_recall` / `qwk_slack`; đường mặc định giữ nguyên
  BYTE-IDENTICAL** (test golden khoá lại `[1.13266468, 2.22593294, 3.30818994, 4.15042206]`).
  Mọi con số S7/S8 đã báo cáo đều sinh từ đường đó nên nó là hợp đồng.
- **`m_xgb_reg` đổi từ `cross_val_predict` sang `ORD.inner_oof`.** Cùng fold (`GroupKFold(4)`
  không shuffle) nên C tái lập bit-đối-bit; S8 mục 4 **assert** điều đó so với lần chạy 13/9.
  Gỡ luôn phụ thuộc vào sklearn ≥ 1.4 cho `cross_val_predict(params=)`.
- **`decode_count` và `apply_cutpoints` giờ chốt hình dạng.** `decode_count` với `thr` shape
  `(N,)` khi `N == K-1` trước đây broadcast SAI trục mà numpy không kêu; `np.digitize` nhận bins
  giảm dần cũng không kêu. Hai assert mới chặn cả hai.
- **Per-class P/R/F1 gộp về một định nghĩa** trong `ordinal.py`. Trước đó S7 và S8 mỗi cái giữ một
  bản gọi sklearn inline.
- S8 ghi vào **`s8_holdout_v2/`**, không đè `s8_holdout/`; mục 2 assert phép chia trùng khít bản
  đã ghim.
- `bsc/ordinal.py` thêm ba thành phần dùng chung: `asymmetric_loss`, `TwoBranchTrunk`
  (α + β = 1 bảo đảm theo cấu trúc), và thành phần loss `exp` (expected-count) — đường
  gradient **duy nhất** tới `tau`. Ai bật `learn_thresholds` đều phải bật kèm `exp`.
- `FrankHall.__getstate__` bỏ `make_clf` để pickle được. Đối tượng nạp lại **dự đoán được
  nhưng không fit lại được**.
- `stratified_group_split` có nhánh lùi về `GroupShuffleSplit` (kèm cảnh báo in ra) khi một
  lớp có dưới 2 subject. Gặp trên tập con nhỏ, không gặp trên cohort đầy đủ.

### Findings
- **FCL neo bề mặt mạnh gấp ~4 lần cột trơ cũ.** ρ với KL: `fcl_fem_pct` 0,423 so với
  `denuded_ratio_tibial` 0,105. Cột mạnh nhất là `fcl_fem_maxdef_mm2` (0,433).
- **Bề mặt và bảng cũ bổ sung cho nhau**, không trùng: D đạt 0,666 trên 55 cột bề mặt,
  0,640 trên 15 cột cũ, nhưng **0,717** khi ghép.
- **Loss ordinal trả lãi ổn định +0,02 tới +0,05** qua mọi feature set; lợi thế của mạng so
  với cây thì giảm đơn điệu và **đảo dấu** ở bộ giàu nhất (+0,068 → −0,011).
- **Ngưỡng cuối là mắt xích yếu duy nhất, và là vấn đề hiệu chỉnh chứ không phải phân biệt.**
  AUC của `P(KL>3)` là 0,952, **cao nhất** trong bốn ngưỡng, nhưng trung bình p của ca KL4
  thật chỉ 0,46–0,57.
- **ASL (G) hỏng đồng loạt** — tệ nhất trong 10 model trên cả 5 feature set. Nguyên nhân là
  cấu hình: một `gamma_neg` áp cho bốn ngưỡng có chiều lệch **ngược nhau**. Chi tiết ở
  `Event.md`, mục 2026-09-13.
- **Fusion (I) là biến thể duy nhất có mặt tích cực**: acc 60,6% và MAE 0,480 — cao nhất và
  thấp nhất trong toàn bộ 50 ô của S8. Nhưng chỉ trên 1 trong 2 feature set áp dụng được.
- `tab_*` **tăng +16%** theo KL thay vì sụp. Phép kiểm an toàn đạt, nhưng là bằng chứng gián
  tiếp của nhiễm gai xương.
- **Trade-off recall của C nằm ở mục tiêu đặt điểm cắt, không phải ở bộ hồi quy.** QWK quadratic
  là hệ số tương hợp Lin, mà điểm hồi quy co về trung bình, nên cách rẻ nhất để tăng QWK là nới
  rộng hai bin ngoài cùng — đúng cái đang làm KL3 tụt còn KL4 phình.
- **Điểm cắt phân vị (0 tham số) thắng điểm cắt tối ưu QWK trên held-out tổng hợp**: min-recall
  không bao giờ tệ hơn ở 6/6 seed, QWK tốt hơn ở 5/6. Bỏ bước tìm kiếm lại được cả hai mặt.
- **Tối ưu thẳng macro-recall thì bất ổn**: 2/4 seed tệ hơn cả hai mặt trên held-out. Với ~80 ca
  KL4 để đặt điểm cắt, tìm kiếm tham lam chỉ đuổi theo nhiễu.

### Failures / Risks
- **Độ tin cậy của FCL tính từ mask AI CHƯA ĐO.** Đây là điều kiện cần trước khi công bố bất
  kỳ con số FCL nào. Dữ liệu đã có sẵn (103 ca test OAI-ZIB, 36 ca iMorphics), **không cần
  chạy thêm inference**.
- **`alpha` của I chưa chắc được nhận dạng** — khởi tạo 0,5 và chỉ dịch ~0,10. Phải chạy lại
  với gate khởi tạo −2 / +2 trước khi diễn giải.
- **I trùng khít E trên 3 feature set không có radiomics.** Ba dòng `hiệu = 0.000` ở mục 7
  của S8 nghĩa là *không chạy*, đang bị trình bày như *không tác dụng*.
- **`tau` của H không bao giờ được in ra** → toàn bộ giá trị chẩn đoán của H bị mất.
- **H đổi hai thứ, không phải một** (tau + thành phần loss `exp`), trái với mô tả trong
  markdown của S8.
- **Phép kiểm sensitivity của S7 trộn chất lượng nhãn với cỡ mẫu.** Tập tin cậy cao chỉ bằng
  36% cohort nên train tụt từ ~980 xuống ~358; riêng việc đó đủ giải thích mức giảm 0,04.
- Bảng `s5_fixed_KE_THUA` **lạc quan có hệ thống** (danh sách đặc trưng kế thừa đã nhìn thấy
  nhãn test). Chỉ dùng đối chiếu, không bao giờ dùng kết luận.
- Notebook S1–S5 trong repo **đã lỗi thời**, không dựng lại được cohort 1229 ca. Quy trình
  dựng cohort thật nằm ngoài repo.

### Decisions
- Không đưa G (ASL) vào báo cáo như kiến trúc ứng viên; giữ làm kết quả âm tính có giải thích.
  Xem `Event.md` 2026-09-13.
- Không chạy lại S6 chỉ để lấy `fcl_*_defarea_mm2`. Gộp vào lần chạy S6 tiếp theo vì lý do
  khác. `ndef` + `maxdef` đã phủ phần lớn thông tin đó.
- Dùng S7 để kết luận, dùng S8 khi cần một mô hình cố định và một tập test cố định.

### Current State
- **Verified working:** S6 toàn cohort; S7 15 fold, 6 feature set; S8 holdout 10 model.
  `holdout_split.csv` đã ghim, mọi thí nghiệm sau phải đọc file đó thay vì chia lại.
  Tầng quyết định: 36 test đậu cục bộ, và toàn bộ cell mới của S8 đã chạy thử end-to-end trên
  dữ liệu tổng hợp (mọi assert đậu).
- **Likely working but unverified:** tầng quyết định **trên dữ liệu thật** — S8 v2 chưa chạy
  Colab; `alpha` của nhánh fusion (chưa kiểm nhận dạng); `tau` của H (giờ đã in ra, chưa đọc).
- **Known limitation:** FCL là **cận dưới** — phép đóng chỉ bắt được ổ bị sụn còn lại bao
  quanh, mất sụn ở rìa mảng không được đếm. `fem_med` / `fem_lat` không phải cMF chuẩn. Không
  đo được sụn bánh chè vì mask không có xương bánh chè.

### Next
1. **Chạy S8 v2 trên Colab** (~8–9 phút), đọc bảng frontier ở mục 4b và các cổng ở mục 4c.
   Quy tắc nào đạt trên cả hai feature set có radiomics thì đem sang S7.
2. **Đo độ tin cậy FCL từ mask AI** (ICC + Bland-Altman). Chặn mọi việc công bố. Dữ liệu sẵn.
3. **Chạy quy tắc thắng dưới CV 15 fold của S7** (`s7_ordinal_v2`): Δmacro-recall ≥ +0,03 với
   Δqwk ≥ −0,015 trên OOF gộp 3 seed, và đếm số fold khả thi cho quy tắc có `min_recall`.
4. **Kiểm nhận dạng `alpha`** của I: chạy lại với gate khởi tạo −2 và +2. Nếu cả hai hội tụ về
   ~0,40 thì `alpha` có nghĩa; nếu bám gần giá trị khởi tạo thì bỏ mọi diễn giải.
5. **Xác nhận nhiễm gai xương** bằng điểm gai xương trong `KXR_SQ_BU00.txt`; khung đã dựng
   sẵn ở S6 mục 7, chỉ cần đặt `KL_FILE`.
6. Ensemble điểm liên tục C + Σp(B) + Σp(D) rồi một bộ điểm cắt — dùng lại chính `inner_oof`.

### Ghi chú công cụ
`graphify update .` **không chạy được** trong checkout này: không có `graphify-out/` và CLI không
nằm trong PATH. Chưa chặn việc gì; nếu cần thì phải dựng graph lần đầu trước.
