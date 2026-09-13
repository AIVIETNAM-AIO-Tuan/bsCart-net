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

### F0. [MỞ, ƯU TIÊN] Chạy lại S7 để có A2 dưới cross-validation
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
