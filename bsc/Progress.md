# Progress — nhánh biomarker / ordinal

Ghi theo ngày, chỉ giữ thông tin cần để tiếp tục công việc. Quyết định nghiên cứu lớn nằm ở
`Event.md`, không chép lại ở đây.

---

## 2026-09-13

### Completed
- **S6 chạy xong toàn cohort.** 1229/1229 ca, 0 lỗi, ~15,7 s/ca. Bảng
  `s6_fcl/biomarker_table_v2.csv`, 1229 × 88. Cột legacy khớp bảng S3 tới 1,16e-10.
- **S7 chạy xong** với đủ 6 feature set và đối chứng A2. 540 tổ hợp, 41,3 phút.
- **S8 chạy xong** với 10 model × 5 feature set, holdout một lần n_test=246, 3,6 phút.
- Thêm `s6_surface_only` vào S8 cho khớp bộ feature set của S7 (`61317e9`).
- Sổ tay giải thích S6 + S7: https://claude.ai/code/artifact/ad58bf9c-96ef-42b5-bd90-86eba3cfd980

### Critical Changes
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
- **Likely working but unverified:** `alpha` của nhánh fusion (chưa kiểm nhận dạng);
  `tau` của H (chưa in ra bao giờ).
- **Known limitation:** FCL là **cận dưới** — phép đóng chỉ bắt được ổ bị sụn còn lại bao
  quanh, mất sụn ở rìa mảng không được đếm. `fem_med` / `fem_lat` không phải cMF chuẩn. Không
  đo được sụn bánh chè vì mask không có xương bánh chè.

### Next
1. **Đo độ tin cậy FCL từ mask AI** (ICC + Bland-Altman). Chặn mọi việc công bố. Dữ liệu sẵn.
2. **Dò ngưỡng hậu kiểm trên validation** thay vì cố định 0,5 — AUC 0,952 nói vấn đề nằm ở
   vạch cắt, không ở khả năng phân biệt. Rẻ nhất trong các hướng còn lại.
3. **Chạy I dưới cross-validation của S7**, cộng phép kiểm nhận dạng `alpha` (gate −2 / +2).
4. **Xác nhận nhiễm gai xương** bằng điểm gai xương trong `KXR_SQ_BU00.txt`; khung đã dựng
   sẵn ở S6 mục 7, chỉ cần đặt `KL_FILE`.
5. Sửa ba lỗi trình bày của S8 (dòng `0.000` của I, in `tau`, con số 0,26–0,38 đã cũ).
