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
