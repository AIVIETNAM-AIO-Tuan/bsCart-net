# Event.md — việc còn treo của nhánh biomarker FCL

Ghi ngày 2026-09-13. Đây là những gì đã phát hiện nhưng **chưa xử lý**, để quay lại sau.
Phần đã làm xong nằm ở `biomarkers.py`, `ordinal.py`, hai notebook `biomarker_s6_fcl` /
`biomarker_s7_ordinal`, và sổ tay https://claude.ai/code/artifact/40ae338f-d0b2-434e-b5f5-647dffad16fb

Bối cảnh gốc: bảng S3 có `denuded_ratio` và `thickness` nhưng cả hai không đo được mất sụn.
Đã thay bằng họ biomarker neo vào bề mặt xương. Ca chạy thử `9000099_V00_R` đạt, độ dày rơi
đúng dải y văn, nhưng lộ ra rằng phần lớn FCL thô là **đốm ở rìa mảng sụn**, không phải ổ thật.

---

## A. Khoảng cách so với chuẩn y văn (ưu tiên cao nhất)

### A1. Đo ở mức VÙNG CON, không phải cả mảng
Y văn OAI chia mỗi mảng sụn thành vùng trung tâm / trong / ngoài (16 vùng con femorotibial),
và mất sụn dồn vào **vùng trung tâm chịu lực**. Hiện tại `fcl_mt_*` tính trên cả mâm chày
trong, nên một ổ ở vùng trung tâm bị pha loãng vào mẫu số lớn hơn nhiều lần.

- Eckstein 2008, mất sụn tập trung ở central + external medial tibia, central medial femur.
  https://doi.org/10.1002/art.24208
- Cotofana 2013 dùng ngưỡng "vùng con trơ ở mức vừa" = **> 10% diện tích vùng con**.
  https://doi.org/10.1016/j.joca.2013.04.001

**Việc cần làm:** chia footprint thành vùng con. Cách rẻ: chia theo phân vị tọa độ trên
footprint đã có (trục trước-sau và trong-ngoài). Cách đúng: atlas.

### A2. Không phân biệt được vùng trơ do MẤT SỤN với vùng trơ do GAI XƯƠNG TRONG SỤN
Cotofana 2013 (633 gối OAI) tách vùng trơ thành hai kiểu hình: `cartilage-loss-type` và
`intra-chondral-osteophyte-type`. Phép đo hiện tại gộp cả hai.
https://doi.org/10.1016/j.joca.2013.04.001

### A3. Footprint bằng closing chỉ bắt được ổ ĐƯỢC BAO QUANH → FCL là cận dưới
Mất sụn ở rìa mảng, hoặc cả một khoang trơ trụi, sẽ không được đếm. Đã kiểm bằng y văn rằng
tiền đề "mất sụn theo vùng" đúng kể cả ở KL4, nên phương pháp dùng được, nhưng vẫn hụt ở rìa.

**Nâng cấp:** footprint từ **atlas gối lành** thay vì closing. `bsc/atlas.py` đã có khung
`ArticularAtlas` + `build_articular_atlas` dựng riêng trong từng fold, dùng lại được.

---

## B. Phép kiểm chưa chạy

### B1. `tab` KHÔNG được sụp ở KL4 — phép kiểm an toàn quan trọng nhất
Nếu `tab_*_mm2` tụt mạnh ở nhóm KL4 thì không phải bệnh nặng hơn mà là **phép đóng mất chỗ
bám**, và mọi con số FCL ở nhóm đó vô giá trị. Chạy ngay khi có bảng v2 đầy đủ.

Cơ sở để kỳ vọng nó KHÔNG sụp: Eckstein 2011, 109 gối KL4 trong 831 gối OAI vẫn mỏng đi
nhanh nhất (tới 3.9%/năm), tức vẫn còn nhiều sụn. https://doi.org/10.1002/acr.20370

### B2. Độ tin cậy của FCL tính từ mask AI — CHƯA ĐO
Việc cần làm: so FCL tính từ nhãn thật với FCL tính từ mask dự đoán trên **103 ca test
OAI-ZIB** và 36 ca test iMorphics. Báo ICC và Bland-Altman. **Dữ liệu đã có sẵn, không cần
chạy thêm inference nào.** Đây là điều kiện cần trước khi công bố bất kỳ số FCL nào.

### B3. Nhiễm gai xương — [ĐÃ CÓ BẰNG CHỨNG GIÁN TIẾP 13/9, cần xác nhận]
**`tab` KHÔNG sụp ở KL4 mà TĂNG ĐƠN ĐIỆU theo KL.** Trung vị trên 1229 ca:

| KL | tab_fem | tab_mt | tab_lt |
|---|---|---|---|
| 0 | 5936 | 1269 | 1143 |
| 4 | 6882 | 1364 | 1227 |

Femur tăng **+16%** từ KL0 lên KL4. Phép kiểm an toàn thì ĐẠT (closing không mất chỗ bám), nhưng
bản thân xu hướng này là dấu hiệu của đúng thứ mục này lo: **KL3/KL4 theo định nghĩa đòi hỏi gai
xương lớn**, và nếu mũ sụn gai xương bị gán nhãn sụn thì footprint nở ra đúng theo mức KL.

Hai hệ quả:
- **FCL bị kéo XUỐNG ở KL cao** vì mẫu số nở ra. Tức `fcl_*_pct` là ước lượng BẢO THỦ, tin được
  theo chiều dương. Đây là chiều tốt.
- **Nếu đưa `tab_*` vào làm feature cho classifier thì đang dùng tín hiệu gai xương.** Với bài toán
  dự đoán KL thì điều đó hợp lệ, vì KL vốn tính cả gai xương. Nhưng khi ấy **không được gọi nó là
  biomarker sụn thuần** trong báo cáo.

Cách xác nhận vẫn như cũ: dùng điểm gai xương X-quang trong `KXR_SQ_BU00.txt`.

### B3b. Nhiễm gai xương — cách kiểm gốc
Mask không có nhãn gai xương. Gai xương trưởng thành nằm trong nhãn xương; mũ sụn gai xương
non dễ vào nhãn sụn, kéo footprint ra rìa đúng ở gối nặng nhất.

**Cách kiểm rẻ nhất:** file `KXR_SQ_BU00.txt` mà S1 đang đọc để lấy KL cũng chứa điểm gai
xương theo khoang (các cột dạng `OSFM` / `OSFL` / `OSTM` / `OSTL`; in `df_kl_raw.columns`
để xác nhận tên). Trong **cùng một mức KL**, nếu `tab_mt_mm2` hay `cab_fem_mm2` tương quan
dương với điểm gai xương ⇒ biomarker đang bị nhiễm. Notebook S6 mục 7 đã có khung sẵn,
chỉ cần đặt `KL_FILE`.

### B4. Độ nhạy bán kính closing trên dữ liệu THẬT
Trên phantom đã đo: bán kính hành xử như một **ngưỡng**, không phải núm chỉnh liên tục.
Lỗ bán kính trắc địa 4.2 mm → closing 2 mm cho 8.7 mm², còn 4 / 8 / 12 mm đều cho 49.8 mm².
Rủi ro nằm ở phía đặt quá nhỏ. Nhưng **trên giải phẫu thật** đặt quá lớn lại có rủi ro khác:
bắc cầu qua khe có thật. Notebook S6 mục 6 có sẵn, đặt `SENS_N = 20` để chạy.

---

## C. Vấn đề còn lại của chính phép đo

### C1. `fem_med` / `fem_lat` không phải cMF chuẩn
Chia đôi bằng một mặt phẳng suy từ trọng tâm hai sụn chày, nên mỗi nửa mang theo một phần
ròng rọc. Không so trực tiếp được với số công bố theo vùng con của Chondrometrics.

### C2. Không đo được sụn bánh chè
Mask không có xương bánh chè nên không có bề mặt để neo. Nhãn 6/7/8 hoàn toàn không tham gia
vào họ biomarker bề mặt.

### C3. Diện tích bề mặt của cột LEGACY vẫn sai theo hướng
`contact_and_surface_area` gán mỗi voxel biên một diện tích cố định bằng tích hai cạnh nhỏ
nhất, bất kể hướng mặt. Giữ nguyên **có chủ ý** để bảng v2 đối chiếu 1:1 với S3. Nếu về sau
muốn sửa thì phải tạo cột mới, đừng sửa tại chỗ.

---

## D. Chất lượng dữ liệu

### D0. [ĐÃ GIẢI QUYẾT 13/9] Tách external validation — manifest có ghi, S7 đã lọc
`cohort_manifest.csv` giữ **1325 dòng**, gồm 1229 ca cho S3–S7 cộng 96 ca OAIZIB-CM dành
riêng làm external validation. Phép tách được ghi lại bằng **hai cách độc lập**:
`source_dataset = "reserved_external_validation_oaizib_test"` và cột `oaizib_split`.
Bốn nguồn còn lại khớp chính xác báo cáo 13/9: 766 + 375 + 72 + 16 = 1229.

- **S6 tính cả 1325 ca là ĐÚNG**, không cần sửa. Trích biomarker là phép toán per-case,
  không fit xuyên qua các ca nên không rò rỉ; và external validation sau này cũng cần
  biomarker của đúng 96 ca đó.
- **S7 đã được vá** để tách 96 ca trước khi train. Nhãn split **luôn lấy lại từ manifest**,
  không tin cột trong bảng đặc trưng: `biomarker_table.csv` sửa lúc 1:31 chiều còn manifest
  lúc 1:58, nên bảng S3 sinh ra TRƯỚC khi phép tách tồn tại và `source_dataset` trong đó là
  bản cũ. Danh sách 96 case_id ghi ra `s7_ordinal/external_validation_cases.csv`.

### D0b. [MỞ] 5 ca đổi nguồn giữa 28/8 và 13/9, chưa ai giải thích
`gt_real_imorphics` 67 → 72 và `ai_full_8class` 21 → 16, trong khi cột "chuyển sang external
validation" của cả hai đều để trống. Manifest xác nhận đúng con số mới. Đây là phân loại lại
provenance chứ không phải loại bỏ, và nó **đổi định nghĩa tập tin cậy cao** mà S4/S5/S7 dùng
cho sensitivity analysis. Hỏi corntoun1505.

### D0c. [MỞ] Notebook S1–S5 trong repo đã lỗi thời, không dựng lại được cohort
Không notebook nào nhắc tới `new_oai1000_balanced_no_gt` (766 ca, 62% cohort) hay việc tách
external validation. S1 chỉ biết iMorphics + OAIZIB-CM, tối đa 683 ca. Cohort thật đang được
dựng bằng quy trình nằm ngoài repo. Cũng có nhánh CNN (`cnn_roi_cache_joint`,
`model_kl_cnn_gradcam` trên Drive) không xuất hiện trong bất kỳ notebook nào.

### D0d. [MỞ] Trùng tên S7
Báo cáo 13/9 đã mô tả một "Ordinal Classification S7" với kết quả riêng
(`s7_ordinal_test_predictions.csv`, Frank-Hall κ=0,762). Notebook `biomarker_s7_ordinal.ipynb`
là một thứ khác cùng số hiệu. Đổi tên trước khi có người nhầm.

### D1. Nhãn 6/7/8 trên ca OAI-ZIB là AI thuần
OAI-ZIB chỉ có GT cho nhãn 1-5. Meniscus và sụn bánh chè do model d20 sinh ra, **chưa từng
được đo Dice** vì không có GT. Ảnh hưởng tới `vol_med_meniscus_mm3`, `vol_lat_meniscus_mm3`,
`vol_patellar_cart_mm3` và hai cột `extrusion_*`. Không ảnh hưởng FCL / tAB / ThC.

### D2. Ca 9000099_V00_R có sụn chêm trong thấp bất thường
Sụn chêm trong 1.32 cm³ so với ngoài 2.53 cm³. Bên trong thấp hơn thường thấy. Có thể thật,
có thể phân đoạn thiếu. Kiểm khi làm D1.

### D3. Bản S6 đang chạy (2026-09-13) THIẾU cột `fcl_*_defarea_mm2`
Cột này thêm sau khi vòng lặp đã bắt đầu. Bảng lần này vẫn dùng được qua `ndef` + `maxdef`,
chỉ thiếu tổng diện tích khi một khoang có nhiều ổ. **Lần chạy sau nhớ dùng bản mới** và
xóa checkpoint cũ (tên checkpoint đã đổi sang `worker<N>/partial.csv`).

---

## E. BML — nhánh riêng, KHÔNG suy ra được từ mask

Mask chỉ là nhãn giải phẫu, không mang cường độ tín hiệu. Và chuỗi DESS vốn không hợp để đọc
BML: so trực tiếp trên đúng protocol OAI ở 3T, trong 200 BML thấy trên chuỗi IW fat-suppressed
thì DESS **bỏ sót 93 tổn thương**, và DESS cũng cho kích thước nhỏ hơn có ý nghĩa thống kê.
Hayashi 2011, https://doi.org/10.1186/1471-2474-12-198

**Nếu muốn làm thật:** tải chuỗi `SAG_IW_TSE_FS` của OAI cho cùng gối cùng lần khám, đăng ký
về không gian DESS, dùng mask xương làm ROI. Cần nhãn BML để train — hiện chưa dataset nào có.

**Đừng làm:** gọi first-order radiomics của lớp xương dưới sụn trên DESS là "biomarker BML".

---

## F. Phần ordinal (S7)

### F-1. [MỞ] Phép kiểm sensitivity đang TRỘN chất lượng nhãn với cỡ mẫu
Mục 6 của S7 so QWK trên toàn cohort (1229) với chỉ case tin cậy cao (447). Kết quả 13/9 trên
`s6_all_plus_radiomics`: A 0.758→0.710, B 0.773→0.731, E 0.777→0.746. Lệch đều khoảng −0.04,
dưới ngưỡng cảnh báo 0.1 nên không báo động.

**Nhưng phép so này không công bằng.** Tập con chỉ bằng 36% cohort, nên tập huấn luyện tụt từ
~980 xuống ~358 ca. Riêng việc mất 63% dữ liệu đã đủ giải thích mức giảm đó. Không tách được
"nhãn AI làm hỏng kết quả" khỏi "ít dữ liệu hơn".

**Cách sửa:** thêm một nhánh đối chứng lấy NGẪU NHIÊN 447 ca từ toàn cohort, lặp vài lần, rồi
so ba nhóm. Nếu tập tin cậy cao ngang với tập ngẫu nhiên cùng cỡ thì nhãn AI vô hại.

### F0. [XONG 13/9] Chạy lại S7 để có A2 dưới cross-validation
Đã chạy, 540 tổ hợp, 41.3 phút, 6 feature set. Kết quả **xác nhận và làm sắc nét** phát hiện từ
S8, đóng góp của kiến trúc sụp đơn điệu khi đặc trưng tốt lên còn đóng góp của loss thì không:

| Feature set | cột | A→A2 (đổi sang mạng) | A2→D (thêm loss ordinal) |
|---|---|---|---|
| legacy_s3 | 15 | **+0.068** | +0.018 |
| s6_surface_only | 55 | +0.060 | +0.042 |
| s6_all | 70 | +0.033 | +0.048 |
| s5_radiomics | 871 | +0.005 | +0.027 |
| s6_all_plus_radiomics | 926 | **−0.011** | +0.035 |

**Kết luận:** loss ordinal đóng góp ổn định +0.02 tới +0.05 bất kể chất lượng đặc trưng, còn lợi
thế của mạng so với cây CHỈ tồn tại khi đặc trưng nghèo, và đảo dấu ở bộ giàu nhất.

### F0b. [cũ, giữ để đối chiếu] Chạy lại S7 để có A2 dưới cross-validation
S8 (holdout, 2026-09-13) cho phép tách đóng góp của **kiến trúc** khỏi đóng góp của **loss ordinal**,
nhờ đối chứng A2 (MLP + softmax, cùng thân với D/E). Kết quả rất đáng chú ý và **đảo chiều theo chất
lượng đặc trưng**:

| | A → A2 (đổi sang mạng) | A2 → D (thêm loss ordinal) |
|---|---|---|
| 15 biến hình học | **+0.062** | +0.031 |
| có radiomics | +0.004 | **+0.035** |

Trên đặc trưng nghèo, phần lớn mức tăng đến từ **mạng thay cây**, không phải từ loss. Trên đặc trưng
giàu thì ngược lại hoàn toàn. Nhưng đây là **một lần chia, n=246, dao động ±0.055** — chưa đủ để kết
luận. S7 (15 fold) đã có A2 trong code từ commit `d269c16` nhưng lần chạy S7 gần nhất diễn ra TRƯỚC
đó nên bảng kết quả chỉ có A–E. **Chạy lại S7 là việc rẻ nhất cho kết luận chắc nhất.**

### F1. Chưa chạy trên bảng v2
S7 có fallback tự lùi về `biomarker_table.csv` của S3 nếu chưa có v2. Chạy được ngay để gỡ
lỗi pipeline trên feature cũ; khi S6 xong thì chạy lại, nó tự bắt bảng v2.

### F2. Ba vấn đề của loss trong slide "Biomarker-guided ordinal multi-task"
- **Chưa định đầu ra chính thức.** Có 3 head cùng dự đoán KL (softmax 5 lớp, ordinal, OA
  status) và lúc suy luận chúng có thể mâu thuẫn. Phải chọn một và ghi rõ. S7 mục 5 đo tỉ lệ
  bất đồng giữa head softmax và head ngưỡng.
- **`L_mono` chỉ là phạt mềm**, không bảo đảm `p` đơn điệu. Muốn đơn điệu theo cấu trúc thì
  dùng CORAL (bias có thứ tự, chung trọng số) hoặc CORN (xác suất có điều kiện).
  `ordinal.monotonic_violation_rate` đo phần còn vi phạm.
- **`L_OA` trùng với một ngưỡng của `L_ord`.** `KL >= 2` chính là sự kiện `KL > 1`, tức `p_1`.
  Nên nó chỉ là tăng trọng số cho một BCE đã có, không phải nhiệm vụ thứ ba. Slide 8 lại ghi
  `KL >= 1`, mâu thuẫn với slide 5 và 11.

### F3. Quy tắc giải mã phải ghi rõ khi báo cáo
Đang dùng `y = #{k : p_k > 0.5}`. Với ca KL3, `p_1` và `p_2` cao không gây sai; sai chỉ đến từ
`p_3 > 0.5` (lên KL4) hoặc `p_2 <= 0.5` (xuống KL2). Ngưỡng cuối `P(KL>3)` hiệu chỉnh kém nhất
vì KL4 hiếm. Cân nhắc tinh chỉnh ngưỡng từng `k` trên validation theo QWK thay vì cố định 0.5.

---

## G. Nhắc lại ràng buộc cũ vẫn còn hiệu lực

- **Rò rỉ V00/V01:** 51/70 đầu gối iMorphics có hai timepoint ở hai fold khác nhau ⇒ số CV bị
  thổi phồng. Báo cáo số **test**, không báo cáo số CV. S7 dùng `StratifiedGroupKFold` theo
  `subject` nên phần ordinal không dính, nhưng mọi so sánh với baseline cũ thì phải cẩn thận.
- **Non-destructive:** thí nghiệm mới = thư mục mới. Bảng S3 và S4/S5 giữ nguyên.
- **Drive-first:** mọi artifact dưới Drive, không bao giờ ghi vào `/content/`.

---
---

# Nhật ký sự kiện theo thời gian

Từ đây trở xuống dùng đúng format của `CLAUDE.md` mục 8. Phần A–G ở trên là bản ghi theo
chủ đề, giữ nguyên, không viết lại.

## 2026-09-13 — S8: ASL thất bại đồng loạt, fusion bio/radiomics có tín hiệu nhưng n=2

### Context
S7 (15 fold) cho thấy ngưỡng cuối `P(KL>3)` là mắt xích yếu duy nhất của cái thang ordinal:
ca KL4 thật chỉ đạt trung bình 0.46 (B) tới 0.57 (D) ở ngưỡng đó, trong khi AUC của chính
ngưỡng đó lại **cao nhất** trong bốn ngưỡng (0.952). Tức mô hình phân biệt được KL4 nhưng
không vượt nổi vạch cố định 0.5. Ba kiến trúc được thêm vào S8 để tấn công đúng chỗ đó:
G dùng Asymmetric Loss, H học ngưỡng `tau`, I tách hai nhánh bio/radiomics gộp bằng `alpha`.

### Change
Thêm G/H/I vào `bsc/ordinal.py` và `notebooks/biomarker_s8_holdout.ipynb`. Mỗi cái khác E
đúng một thành phần (xem Risks về ngoại lệ của H). Chạy holdout một lần
`GroupShuffleSplit(test_size=0.2, random_state=42)`, n_test=246, 10 model × 5 feature set,
3.6 phút.

### Evidence / Result
**G (ASL) thất bại đồng loạt và rõ ràng.** G là model **tệ nhất trong 10 model, trên cả 5
feature set, theo cả QWK lẫn off-by≥2**. Không một ô nào G không đội sổ.

| feature set | QWK của E | QWK của G | hiệu | off-by≥2 của G | off-by≥2 tốt nhất của set |
|---|---|---|---|---|---|
| legacy_s3 | 0.611 | **0.442** | −0.169 | **27.6%** | 13.8% (D) |
| s6_surface_only | 0.651 | **0.473** | −0.178 | **24.0%** | 12.2% (E) |
| s6_all | 0.698 | **0.555** | −0.143 | **19.1%** | 10.6% (B, D) |
| s5_radiomics | 0.770 | **0.727** | −0.042 | **10.2%** | 6.1% (B) |
| s6_all_plus_radiomics | 0.796 | **0.745** | −0.051 | **11.4%** | 5.3% (D) |

Trên `legacy_s3`, off-by≥2 của G là 27.6% — **tệ hơn cả baseline nominal A (22.0%)**. Tức
ASL làm loại lỗi lâm sàng nặng nhất **phổ biến hơn** so với không làm gì cả.

**Nguyên nhân đã định lượng được, không phải đoán.** `asymmetric_loss` áp **một** `gamma_neg`
cho **cả bốn** cột ngưỡng, nhưng chiều lệch của bốn ngưỡng **đảo dấu**:

| ngưỡng | dương / 1229 | tỉ lệ dương | thiểu số là |
|---|---|---|---|
| t₀ = KL>0 | 945 | 76.9% | **ÂM** |
| t₁ = KL>1 | 712 | 57.9% | ÂM |
| t₂ = KL>2 | 417 | 33.9% | dương |
| t₃ = KL>3 | 106 | 8.6% | **DƯƠNG** |

ASL với `gamma_neg > gamma_pos` hạ trọng số **âm dễ**, giả định âm là đa số. Đúng cho t₃,
**sai ngược** cho t₀ và t₁. Mức hạ rất mạnh: với `gamma_neg=4, clip=0.05`, một mẫu âm ở
p=0.5 chỉ còn trọng số 0.041, tức **hạ 24 lần** so với BCE; ở p=0.3 là hạ 256 lần.

Hệ quả đo được đúng như dự đoán — mọi thứ bị đẩy LÊN (recall %, `s6_all_plus_radiomics`):

| | KL0 | KL1 | KL2 | KL3 | KL4 |
|---|---|---|---|---|---|
| E | 58.2 | 40.0 | 52.6 | 79.3 | 50.0 |
| G | **43.6** | 34.0 | **43.9** | 77.6 | **69.2** |

G **đạt đúng mục tiêu thiết kế** (KL4 recall 50.0 → 69.2) nhưng trả giá bằng KL0 và KL2, và
precision KL4 tụt 86.7 → 72.0, tức nó gọi bừa KL4.

**I (fusion bio/radiomics) có tín hiệu, nhưng chỉ n=2.**

| feature set | QWK E | QWK I | hiệu | acc I | MAE I |
|---|---|---|---|---|---|
| s5_radiomics | 0.770 | **0.794** | +0.025 | **60.6%** | **0.480** |
| s6_all_plus_radiomics | 0.796 | 0.788 | −0.009 | 56.1% | 0.516 |

Trên `s5_radiomics`, I cho **accuracy cao nhất (60.6%) và MAE thấp nhất (0.480) trong toàn
bộ 50 ô của S8**, QWK đứng nhì (chỉ sau C trên set khác, 0.803). Nhưng trên set còn lại nó
hơi thua E. Cả hai hiệu số đều **nằm trong dao động ±0.055 của một lần chia**.

`alpha` (trọng số nhánh biomarker) = **0.395** và **0.403** trên hai feature set — lệch nhau
chỉ 0.008, dù số cột bio sau khi chọn rất khác nhau (5/95 so với 18/107).

**F (head MLP) và H (tau học được) đều là kết quả rỗng.** F: −0.006 tới +0.019. H: −0.028
tới −0.007, âm ở cả 5 set nhưng đều dưới ngưỡng nhiễu.

**Đối chiếu với báo cáo 13/9 — khớp.** A trên `legacy_s3`: acc 41.1%, QWK 0.532,
off-by≥2 22.0%. Báo cáo ghi đúng 22.0% cho S4.

### Significance
- **ASL không bị loại vì ý tưởng sai, mà vì cấu hình sai.** Phân rã ngưỡng sinh ra bốn bài
  toán nhị phân có chiều lệch ngược nhau; áp một `gamma_neg` chung cho cả bốn là sai theo
  cấu trúc. Bản sửa được là `gamma_neg` **theo từng ngưỡng**, hoặc chỉ bật ASL ở t₂/t₃.
- **Mức hỏng tỉ lệ nghịch với chất lượng đặc trưng** (−0.17 trên hình học thuần, −0.04 tới
  −0.05 khi có radiomics). Đặc trưng mạnh che bớt được loss hỏng; đừng dùng điều đó để kết
  luận ASL "chỉ hơi kém".
- **I là biến thể duy nhất có mặt tích cực**, và `alpha` ổn định qua hai feature set là dấu
  hiệu đáng theo đuổi, vì nó cho một con số **đọc được** thay vì suy từ chênh lệch hai model.

### Risks / Open Questions
- **`alpha` chưa chắc đã được nhận dạng.** Nó khởi tạo ở 0.5 và chỉ dịch ~0.10. Chưa loại
  trừ được khả năng gate gần như đứng yên. **Phép kiểm rẻ:** chạy lại I với gate khởi tạo ở
  −2 và +2 (α ≈ 0.12 và 0.88). Nếu cả hai hội tụ về ~0.40 thì α có nghĩa; nếu bám gần giá
  trị khởi tạo thì α vô nghĩa và mọi diễn giải phải bỏ.
- **`alpha` là trọng số trên BIỂU DIỄN ĐÃ HỌC, không phải tỉ lệ thông tin.** Một nhánh có
  thể cho vector biên độ lớn hơn rồi bù bằng α nhỏ. Không được đọc 0.40 thành "40% thông tin".
- **I chỉ có n=2.** Trên ba feature set không có radiomics, guard trong `OrdinalMLP` khiến I
  **trùng khít E**, nên ba dòng `hiệu = 0.000` trong bảng mục 7 nghĩa là *không chạy*, không
  phải *không tác dụng*. Bảng đang trình bày gây hiểu nhầm.
- **H đổi HAI thứ, không phải một.** Bật `learn_thresholds` buộc phải bật luôn thành phần
  loss `exp` (trọng số 1.0), vì đó là đường gradient duy nhất tới `tau`. Câu "mỗi biến thể
  khác E đúng một chỗ" trong markdown không đúng với H.
- **`tau` không bao giờ được in ra.** Notebook có lưu `r["tau"]` nhưng mục 7 chỉ hiện `alpha`.
  Toàn bộ giá trị chẩn đoán của H bị mất — không biết `tau` có dịch khỏi 0 hay không.
- **Con số cũ trong markdown S8 đã lỗi thời.** Ô mô tả G ghi "KL4 chỉ đạt 0.26 tới 0.38 ở
  ngưỡng cuối"; lần chạy S7 mới nhất cho 0.46 tới 0.57.
- **Chưa phân tích `y_alt` của G.** G vẫn giữ head softmax được huấn luyện bình thường
  (`cls=1.0`), nên so `y_count` với `y_softmax` của chính G sẽ tách được "ASL phá head ngưỡng"
  khỏi "ASL phá cả mô hình". Dữ liệu nằm trong `FITTED`, chỉ cần thêm vài dòng.

### Decision
Người dùng đọc kết quả và nhận định ASL không phù hợp, fusion bio/radiomics cho kết quả tốt.
Kiểm chứng xác nhận cả hai theo đúng chiều, với điều chỉnh: ASL hỏng **do cấu hình**, và
fusion **chưa vượt nhiễu**.

### Consequence
1. **Không đưa G vào báo cáo như một kiến trúc ứng viên.** Giữ lại làm kết quả âm tính có
   giải thích, đúng tinh thần CLAUDE.md mục 7.
2. Nếu còn muốn theo ASL: sửa `asymmetric_loss` nhận `gamma_neg` dạng **vector theo ngưỡng**,
   rồi chạy lại. Trước khi làm, cân nhắc rằng dò ngưỡng hậu kiểm trên validation rẻ hơn nhiều
   và nhắm đúng cùng một vấn đề.
3. **Chạy I dưới cross-validation của S7** trước khi kết luận bất cứ điều gì về fusion. Một
   lần chia với n=2 feature set không đủ.
4. Chạy phép kiểm nhận dạng `alpha` (gate khởi tạo −2 / +2).
5. Sửa ba lỗi trình bày: đánh dấu ba dòng `0.000` của I là "không áp dụng", in `tau` ở mục 7,
   cập nhật con số 0.26–0.38 thành 0.46–0.57.

## 2026-09-13 — Tầng quyết định: trade-off recall của C nằm ở MỤC TIÊU đặt điểm cắt, không phải ở bộ hồi quy

### Context
Người dùng nhận xét model C (XGB hồi quy + điểm cắt) đang hoạt động tốt và hỏi có cách nào giảm
đánh đổi để giữ recall cho mọi lớp. Số đo trên `s6_all_plus_radiomics`, n_test = 246: C có **QWK cao
nhất bảng 0.803** và off-by≥2 thấp 6.1%, nhưng **recall KL3 48.3% — thấp nhất trong 10 model** và
recall KL4 80.8% — cao nhất, với precision KL4 chỉ 72.4%. S7 15 fold cho cùng hình: recall KL2 32.9,
KL3 42.1, KL4 77.4, macro-F1 49.0 thấp nhất trong 6 model.

### Change
Không sửa bộ hồi quy. Thêm một **tầng quyết định** vào `bsc/ordinal.py`, dùng chung cho S7 và S8:

- `confusion` / `per_class_prf` / `per_class_recall` / `macro_recall` / `macro_f1` / `OBJECTIVES` /
  `bootstrap_delta`. Trước đây S7 và S8 mỗi cái giữ một bản gọi sklearn inline; gộp về MỘT định
  nghĩa. Bản numpy nhanh ~200 lần nên dùng được **trong** vòng tìm điểm cắt.
- `_coordinate_descent` tách ra dùng chung; `fit_cutpoints` thêm `objective` / `min_recall` /
  `qwk_slack`. Đường mặc định **byte-identical** (test golden khoá lại con số
  `[1.13266468, 2.22593294, 3.30818994, 4.15042206]`).
- `quantile_cutpoints` — cắt sao cho tỉ lệ dự đoán bằng tỉ lệ thật, **không có tham số nào để fit**.
- `fit_threshold_cuts` — `thr[K-1]` thay vạch 0.5 cố định cho B/D/E. **Không ép đơn điệu**, vì tối ưu
  đo được là `[0.65, 0.35, 0.55, 0.20]` và ép sẽ chặn đúng cái cần sửa.
- `inner_oof` — điểm honest cho scorer bất kỳ (FrankHall, MLP torch), `GroupKFold(4)` không shuffle
  nên trùng khít `cross_val_predict` cũ và C tái lập bit-đối-bit.
- `class_balanced_weights` + chốt hình dạng ở `decode_count` và `apply_cutpoints`.

S8 thêm mục 4b (tầng quyết định), 4c (cổng thăng hạng), 4d (chẩn đoán S6), model C2 (hồi quy có
trọng số lớp), và ghi vào `s8_holdout_v2/`. Test 24 → 36.

### Evidence / Result
**Cơ chế đã chứng minh bằng đại số, không phải phỏng đoán.** QWK quadratic bằng
`2·Cov(y,ŷ) / (Var y + Var ŷ + (μy−μŷ)²)`, tức hệ số tương hợp Lin. Điểm hồi quy luôn **co về trung
bình**, `Var(score) < Var(y)`. Cách rẻ nhất để đẩy phân số đó lên là **bơm Var(ŷ)**, và cách bơm là
nới rộng hai bin ngoài cùng — tức cố ý đoán thừa KL0/KL4 và bóp KL2/KL3. Thuật toán không hỏng; nó
đang tối ưu đúng cái ta bảo nó tối ưu.

**Đo trên điểm co tổng hợp** (`s = 0.6·(y−2)+2 + N(0,0.45)`, fit 400 / held-out 200):

| seed | QWK: cắt-QWK → phân vị | min recall: cắt-QWK → phân vị |
|---|---|---|
| 0 | 0.813 → **0.834** | 0.135 → **0.486** |
| 1 | 0.866 → 0.868 | 0.535 → 0.535 |
| 2 | 0.866 → **0.871** | 0.389 → **0.522** |
| 3 | 0.858 → **0.867** | 0.286 → **0.381** |
| 4 | 0.837 → 0.822 | 0.459 → **0.468** |
| 5 | 0.885 → **0.887** | 0.324 → **0.491** |

Điểm cắt phân vị **không bao giờ làm recall thấp nhất tệ đi** (6/6) và thắng QWK ở 5/6 seed. Nghĩa
là bỏ hẳn bước tìm kiếm lại được **cả hai mặt** — vì tìm QWK trên ~400 hàng đang overfit vị trí cắt.

**Ngược lại, tối ưu thẳng `macro_recall` BẤT ỔN:** 2/4 seed tệ hơn trên *cả hai* mặt ở held-out
(seed 0: marginal dự đoán `[92,116,35,108,49]` so với thật `[94,78,94,98,36]` — nó bóp hẳn lớp giữa).
Với ~80 ca KL4 để đặt điểm cắt, tìm kiếm tham lam trên hàm bậc thang chỉ đang đuổi theo nhiễu.

**`min_recall` bất khả thi là chuyện thật**, 1/4 seed ở sàn 0.45 với 400 hàng. Nhánh lùi in cảnh báo
và trả về đúng kết quả không ràng buộc.

**`fit_threshold_cuts` trên p có ngưỡng cuối thiếu tự tin** (`p[:,3] *= 0.6`), 5 seed: recall lớp
cuối 0.08–0.39 → 0.78–1.00, QWK cũng tăng ở cả 5. Ngưỡng tìm được **không đơn điệu** ở 5/5.

**Chạy thử toàn bộ cell mới của S8 trên dữ liệu tổng hợp:** mọi assert đậu, và trên feature set có
radiomics, `quantile` cho QWK 0.887 → 0.914, macro-recall 0.638 → 0.697, min-recall 0.176 → 0.471,
off-by≥2 0.017 → 0.000, optimism 0.139 → 0.060. Đúng chiều đã dự đoán.

**Chưa có số trên dữ liệu thật** — S8 v2 chưa chạy trên Colab.

### Significance
- **Đây là cải tiến rẻ nhất còn lại.** Không train lại gì: scorer fit một lần, giữ điểm honest
  inner-OOF, rồi áp nhiều quy tắc lên cùng bộ điểm. Tám quy tắc chỉ tốn vài giây mỗi cái.
- **Nó cũng nhắm đúng vấn đề của B/D/E**, vốn là cùng một bệnh soi gương: ca KL4 thật chỉ đạt
  P(KL>3) trung bình 0.46–0.57 so với vạch 0.5, trong khi AUC của chính ngưỡng đó là 0.952 — cao
  nhất trong bốn. Phân biệt được nhưng không hiệu chỉnh được, và cái thứ hai rẻ hơn nhiều để sửa.
- **Thay thế hướng ASL.** Tinh chỉnh ngưỡng hậu kiểm nhắm cùng mục tiêu với chi phí huấn luyện bằng 0,
  nên không cần sửa `asymmetric_loss` thành `gamma_neg` theo ngưỡng nữa.

### Risks / Open Questions
- **`macro_recall` không ràng buộc overfit vị trí cắt** — đã đo. Giảm nhẹ: `quantile` là ứng viên
  chính (0 tham số), `qwk_slack` / `min_recall` là bản có điều tiết, cột `optimism` phơi bày trực
  tiếp, và S7 15 fold là trọng tài cuối.
- **`min_recall` có thể bất khả thi trên fold nhỏ** → nhánh lùi làm nó im lặng bằng tham chiếu. Khi
  đưa sang S7 phải **đếm số fold khả thi** và báo cáo.
- **Ngưỡng không đơn điệu** khiến phép đếm nhận mẫu kiểu `[1,0,1,0]`. Đo bằng
  `inconsistent_pattern` và **báo cáo**, không chặn — chặn sẽ hủy đúng phần sửa được.
- **`macro_recall` bỏ qua precision**, có thể mua recall KL4 bằng recall KL3. `macro_f1` đã có sẵn
  trong `OBJECTIVES` nếu bảng frontier cho thấy vậy.
- **C2 là scorer khác, không phải quy tắc quyết định.** Nó có điểm cắt của riêng nó; đọc tách.
- **`graphify update .` không chạy được** — không có `graphify-out/` trong checkout và CLI không nằm
  trong PATH. Chưa chặn việc gì.

### Decision
Theo lựa chọn của người dùng: **giữ QWK gần tối đa, để recall lớp giữa hồi phục "miễn phí"**. Cổng
thăng hạng đặt ở macro-recall ≥ ref + 0.05, QWK ≥ ref − 0.02, off-by≥2 ≤ ref + 0.02, optimism ≤ 0.05,
và CI bootstrap của hiệu macro-recall không chứa 0 — phải đạt trên **cả hai** feature set có
radiomics. `macro_recall` và `minrec0.40` chỉ để hiện frontier, **không thăng hạng từ S8 một mình**.

Về S6: **giữ nguyên cột, chưa đầu tư vùng con / atlas** cho tới khi đo xong độ tin cậy FCL từ mask AI
(mục B2). Chỉ thêm cell chẩn đoán 4d để biết cột S6 nào sống sót qua LASSO và đóng góp gain bao nhiêu.

### Consequence
1. Chạy S8 v2 trên Colab (~8–9 phút), qua các cổng ở mục 4c.
2. Quy tắc nào đạt thì chạy dưới CV 15 fold của S7 (`s7_ordinal_v2`), xác nhận bằng
   Δmacro-recall ≥ +0.03 với Δqwk ≥ −0.015 trên OOF gộp 3 seed.
3. **Không** sửa `asymmetric_loss` thành gamma theo ngưỡng nữa — mục đích đó đã được tầng quyết định
   phục vụ với chi phí thấp hơn. G vẫn là kết quả âm tính có giải thích.
4. Việc kế tiếp, theo thứ tự giá/lợi: đo độ tin cậy FCL từ mask AI (B2, chặn mọi công bố từ S6);
   kiểm nhận dạng `alpha` của I (gate khởi tạo −2/+2); ensemble điểm liên tục
   C + Σp(B) + Σp(D) rồi một bộ điểm cắt, dùng lại chính `inner_oof`; CORAL/CORN nếu tỉ lệ vi phạm
   đơn điệu của E (16.6%) hóa ra quan trọng.

## 2026-09-13 — S8 v2: dò ngưỡng hậu kiểm làm ĐÚNG việc nó sinh ra, nhưng mức lợi tổng không tái lập

### Context
Chạy S8 v2 (`s8_holdout_v2`, 3.8 phút, 11 model × 5 feature set, 95 dòng quyết định) để đo tầng
quyết định trên dữ liệu thật. Ba chốt an toàn đều qua: phép chia trùng khít bản đã ghim, A/B/C tái
lập **bit-đối-bit** (|Δqwk| = 0.0000), và quy tắc mặc định trùng khít dòng gốc. Dòng MLP lệch
0.0008–0.0077 do khác bản dựng torch, đúng như dự kiến nên chỉ in.

### Change
Không sửa gì thêm. Đây là bản ghi kết quả của thay đổi đã mô tả ở mục 13/9 trước đó.

### Evidence / Result

**1. C2 (trọng số theo lớp) là kết quả rỗng, nghiêng âm trên đặc trưng giàu.** Δqwk so với C:

| feature set | C | C2 | hiệu |
|---|---|---|---|
| legacy_s3 | 0.604 | 0.626 | **+0.022** |
| s6_surface_only | 0.663 | 0.622 | **−0.041** |
| s6_all | 0.702 | 0.711 | +0.009 |
| s5_radiomics | 0.793 | 0.786 | −0.007 |
| s6_all_plus_radiomics | 0.803 | 0.785 | **−0.018** |

Trung bình −0.007, mọi hiệu số dưới ngưỡng nhiễu ±0.055. **Nguyên nhân là dư thừa, không phải sai.**
Bước đặt điểm cắt vốn đã thích nghi với phân bố điểm: đổi trọng số làm điểm dịch đi, rồi
`fit_cutpoints` fit lại trên chính điểm đã dịch và bù trừ phần lớn. Cái còn lại là **giá phải trả về
phương sai**: `w_i = N/(K·n_i)` khiến ca KL4 nặng gấp **3.16 lần** ca KL3, và cỡ mẫu hiệu dụng tụt
**983 → 820, mất 16.6%**. Nên trên đặc trưng giàu, nơi bộ hồi quy vốn đã tốt và phương sai mới là
thứ chi phối, C2 thua.

**2. Dò ngưỡng nâng recall KL4 ở 12/12 phép so — hoàn toàn tái lập.**

| | recall KL4 | precision KL4 |
|---|---|---|
| `fixed0.5`, trung bình 6 cặp model × feature set | 0.487 | 0.872 |
| `tuned_qwk`, cùng 6 cặp | **0.744** | 0.704 |

Không một cặp nào đi ngược chiều. Cơ chế hoạt động **chính xác như thiết kế**: nó đổi precision KL4
lấy recall KL4. Đây là câu trả lời trực tiếp cho triệu chứng đã đo ở S7 (ca KL4 thật chỉ đạt
P(KL>3) 0.46–0.57 so với vạch 0.5 dù AUC ngưỡng đó là 0.952).

**3. Nhưng mức lợi TỔNG không tái lập — và dấu hiệu nhiễu thì không thể rõ hơn.**
`E@tuned_mr_slack0.02` trên `s6_all_plus_radiomics` cho **QWK 0.814 và macro-recall 0.614, cả hai đều
cao nhất trong toàn bộ bảng S8** (vượt C mặc định 0.803). Nó qua **cả 5 cổng**. Nhưng trên
`s5_radiomics` cùng quy tắc đó cho 0.783 / 0.524, tức macro-recall **thấp hơn** cả `fixed0.5` (0.545).

Và `B@tuned_mr_slack0.02` là ảnh soi gương: qua đủ 5 cổng trên `s5_radiomics`, trượt trên
`s6_all_plus_radiomics`. **Hai model, hai feature set, mỗi cái đạt đúng một, và ở hai phía ngược
nhau.** Không quy tắc nào đạt trên cả hai, nên theo luật đã đăng ký trước, **không quy tắc nào được
đem sang S7 từ S8 một mình**.

**4. Giải thích được vì sao H từng là kết quả rỗng.** Đặt `tau` học được cạnh ngưỡng mà tìm kiếm hậu
kiểm ra, trên cùng bài toán:

| feature set | `tau` của H quy ra ngưỡng | ngưỡng tìm kiếm ra trên E |
|---|---|---|
| s6_all_plus_radiomics | [0.523, 0.514, 0.509, …] | **[0.40, 0.95, 0.15, 0.15]** |
| s5_radiomics | [0.516, 0.506, 0.505, …] | [0.40, 0.40, 0.65, 0.10] |
| s6_all | [0.525, 0.518, 0.509, …] | [0.70, 0.80, 0.55, 0.15] |

`tau` **gần như không nhúc nhích khỏi 0.5** ở mọi feature set, trong khi tìm kiếm hậu kiểm đẩy ngưỡng
cuối xuống tận 0.10–0.15. Thành phần loss `exp` tạo áp lực gradient quá yếu so với ba thành phần còn
lại. Đây là bằng chứng trực tiếp rằng **tìm kiếm hậu kiểm làm được thứ mà gradient descent không làm
được**, với chi phí huấn luyện bằng 0.

**5. `minrec0.40` phần lớn là lệnh rỗng.** In `CANH BAO` bảy lần trên mười lần gọi, tức bất khả thi
và lùi về kết quả không ràng buộc. Ví dụ C2 trên `s5_radiomics`: dòng `minrec0.4` trùng khít dòng
`qwk` tới từng chữ số.

**6. `optimism` KHÔNG lọc được gì — lỗi thiết kế của chính cổng đó.** Gần như mọi giá trị đều **âm**
(−0.001 tới −0.064), nghĩa là test tốt hơn inner. Lý do có hệ thống: model inner-OOF chỉ được huấn
luyện trên 3/4 tập train (737 so với 983 ca) nên yếu hơn model cuối, khiến ước lượng inner **bi quan
một cách hệ thống**. Cổng `optimism ≤ 0.05` vì thế đạt gần 100% và không phân biệt được gì. Mặt tích
cực: nó cho biết **overfit vị trí cắt KHÔNG phải là chế độ hỏng ở đây** — nhiễu lấy mẫu mới là.

**7. Chẩn đoán S6 — cột bề mặt THẬT SỰ được dùng.** Trên `s6_all_plus_radiomics`:

| họ | LASSO bỏ | giữ, có gain | tỉ trọng gain | gain / cột |
|---|---|---|---|---|
| S6_surface | 42 | **13** | **15.5%** | 1.19% |
| legacy | 10 | 5 | 6.2% | 1.24% |
| radiomics | 767 | 89 | 78.4% | 0.88% |

**Không cột S6 nào rơi vào trạng thái "giữ nhưng gain 0".** Mười ba cột sống sót thì cả mười ba đều
được XGB tách trên chúng, và **cột S6 mạnh nhất đứng hạng 2 trên tổng 926 cột**. Đó là
`fcl_fem_maxdef_mm2`, kế đến là `fcl_mt_ndef` — đúng hai cột đã được chọn thay cho `fcl_*_mm2` thô vì
có ngưỡng 5 mm². Tính theo từng cột, S6 hữu ích hơn radiomics (1.19% so với 0.88%).

### Significance
- **Câu hỏi "S6 có cần không" giờ có câu trả lời tốt hơn nhiều so với hiệu số QWK.** QWK gần như
  không nhúc nhích vì radiomics đã bão hoà, nhưng model **thực sự dùng** cột FCL và xếp một cột FCL
  ở hạng 2 toàn bảng. Không phải "dư thừa rồi bị loại" mà là "được giữ và được dùng".
- **Dò ngưỡng là công cụ đúng cho bài toán KL4**, cơ chế đã tái lập 12/12. Chỉ có mức lợi trên metric
  tổng là chưa chứng minh được bằng một lần chia.
- **Trọng số theo lớp là ngõ cụt cho họ model có bước đặt điểm cắt**, vì hai thứ dư thừa với nhau.
- **Học `tau` trong lúc train thua tìm kiếm hậu kiểm** một cách rõ ràng và giải thích được.

### Risks / Open Questions
- **Cổng `optimism` cần sửa hoặc bỏ.** Nó đang so hai thứ khác cỡ mẫu huấn luyện. Cách sửa: so
  `optimism` của quy tắc với `optimism` của quy tắc tham chiếu (hiệu tương đối), hoặc bỏ hẳn và dựa
  vào S7 làm trọng tài.
- **`minrec0.40` quá chặt cho cohort này** — bảy trên mười lần bất khả thi. Nếu giữ thì hạ sàn xuống
  0.30, hoặc bỏ vì `mr_slack` đã phục vụ cùng mục đích mà luôn khả thi.
- **Chưa loại trừ được khả năng mức lợi của E là thật nhưng phụ thuộc feature set.** Một lần chia
  n_test = 246 không tách được điều đó khỏi nhiễu. Chỉ 15 fold của S7 mới trả lời được.
- `alpha` của I vẫn chưa kiểm nhận dạng: 0.395 và 0.403, dịch −0.105 và −0.097 khỏi khởi tạo 0.5.

### Decision
**Theo đúng luật đã đăng ký trước khi chạy: không quy tắc nào được thăng hạng từ S8.** Không bẻ luật
để lấy `E@tuned_mr_slack0.02` dù nó đang giữ con số cao nhất bảng — đúng vì đó là con số cao nhất
bảng trên **một** feature set và thất bại trên feature set còn lại.

**C2 bị loại**, ghi lại làm kết quả âm tính có giải thích. **Giữ cột S6 trong classifier** và từ nay
trích dẫn con số chẩn đoán (15.5% gain, hạng 2 toàn bảng) thay vì hiệu số QWK, vì nó trả lời đúng
câu hỏi hơn.

### Consequence
1. **Chạy dò ngưỡng dưới CV 15 fold của S7** cho B, D, E. Đây là phép kiểm duy nhất tách được
   "mức lợi thật" khỏi "nhiễu một lần chia". Không cần train lại scorer, chỉ thêm `inner_oof`
   trong từng fold.
2. Ở S7, **báo cáo recall và precision KL4 tách riêng**, vì đó mới là chỗ hiệu ứng tái lập được,
   chứ không chỉ QWK gộp.
3. Sửa cổng `optimism` thành hiệu tương đối so với quy tắc tham chiếu, hoặc bỏ.
4. Bỏ `minrec0.40` khỏi bộ quy tắc, giữ `mr_slack0.02`.
5. **Không** theo tiếp hướng trọng số lớp cho C, và **không** theo tiếp hướng học `tau` cho H.
