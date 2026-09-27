# Báo cáo 27/9/2026

*Mô hình dự đoán độ nặng thoái hóa khớp gối (Kellgren - Lawrence) từ MRI DESS, biomarker sụn/xương
và đặc trưng radiomics*

**Nguồn dữ liệu:** OAI DESS knee MRI cohort

**Người lập báo cáo:** Hoàng Xuân Bách, Ngô Đức Tuấn

**Ngày lập báo cáo:** 27/09/2026

## 0. Thay đổi chính so với báo cáo 13/9/2026

Tuần này chưa có kết quả mới. Báo cáo chỉ gồm các phần đã làm xong nhưng chưa trình bày ở báo cáo
13/9, cùng hai thí nghiệm đang chạy. Cohort giữ nguyên: 1.229 ca của 1.215 bệnh nhân.

- **Hậu kiểm biomarker neo bề mặt xương** (báo cáo 13/9 hẹn trình bày lần sau): chạy đủ 1.229 ca
  không lỗi, qua phép kiểm an toàn, kèm một nghi vấn về gai xương và một đính chính nhỏ.
- **Đánh giá chéo 15 fold theo bệnh nhân** thay cho một lần chia train/test, đúng đề xuất của báo cáo
  13/9. Mỗi ca được dự đoán bởi mô hình chưa từng thấy nó, nên số liệu tính trên cả 1.229 ca thay vì 246.
- **Lợi thế của mạng nơ-ron so với cây quyết định biến mất khi đặc trưng tốt lên**, còn lợi thế của
  hàm loss thứ tự (ordinal) thì giữ nguyên.
- **Dò ngưỡng quyết định cho KL4:** recall KL4 tăng ở cả 6/6 phép so, nhưng mức lợi trên kappa tổng
  không lặp lại được, nên chưa đưa vào kết luận.
- **Biomarker bề mặt xương được mô hình dùng thật**, và tính theo từng biến còn hữu ích hơn radiomics.
- **Đang chạy, chưa có kết quả:** mô hình ảnh M3T làm mô hình nền, và tập holdout OAI-ZIB 93 ca.

## 1. Biomarker neo bề mặt xương — hậu kiểm

### 1.1. Vì sao hai chỉ số cũ không đo được mất sụn

Trong 15 biomarker cũ có hai chỉ số mang tên đo tổn thương sụn nhưng thực chất không đo được nó:

- **Độ dày sụn** được tính bằng thể tích chia diện tích tiếp xúc, tức là độ dày trung bình của phần
  sụn *còn lại*. Chỗ đã mất sụn không kéo con số này xuống.
- **Tỷ lệ trơ trụi** lấy mẫu số là toàn bộ biên xương trong ảnh, nên chủ yếu phản ánh trường nhìn của
  lần chụp chứ không phải bệnh.

Phép đo mới đặt trên bề mặt xương dưới sụn, nên chỗ không còn sụn được tính là độ dày 0 thay vì bị
loại khỏi phép tính.

### 1.2. Kết quả chạy và đính chính

Chạy đủ 1.229/1.229 ca, không lỗi, khoảng 15,7 giây mỗi ca, ra 88 biến cho mỗi ca. 15 biomarker cũ
được tính lại khớp bảng cũ tới sai số 10⁻¹⁰, nên mọi kết quả cũ chạy lại đều ra đúng số cũ.

**Đính chính Hình 2 của báo cáo 13/9.** Hình đó vẽ 8 trong 11 biến đã đo tương quan với KL. Ba biến bị
bỏ ra đều là biến mới — độ dày trung bình trên vùng nền sụn — và chúng yếu: Spearman ρ chỉ từ −0,07 đến
−0,19 (dấu âm là đúng chiều, cái yếu là độ lớn). Vì vậy ranh giới thật không phải "biến mới mạnh, biến
cũ yếu", mà là **đo diện tích mất sụn thì bắt được bệnh, đo độ dày trung bình thì không**. Tương quan
với KL mới được tính cho 11 trên 88 biến.

### 1.3. Đọc thêm đường mất sụn theo KL (Hình 2b của báo cáo 13/9)

- **KL0 và KL1 gần như phẳng** (sụn đùi 1,11% → 1,07%), đúng với định nghĩa lâm sàng: hai mức này
  chưa có hẹp khe khớp rõ. Mức 0,7–1,1% ở KL0 là sàn nhiễu của phép đo ở rìa mảng sụn, không phải bệnh.
- **Mức tăng dồn vào chặng KL3 → KL4**, chiếm 56–78% toàn bộ mức tăng của cả thang, tùy khoang.

### 1.4. Phép kiểm an toàn

Vùng nền sụn là vùng xương đáng lẽ phải có sụn phủ. Nếu vùng này bị co lại ở nhóm KL4 thì tỷ lệ mất
sụn ở nhóm đó không còn tin được. Kết quả ngược lại: vùng nền **nở ra** theo KL.

| Khoang | KL=0 (mm²) | KL=4 (mm²) | Thay đổi |
|---|---|---|---|
| Sụn đùi | 5.936 | 6.882 | **+16%** |
| Mâm chày trong | 1.269 | 1.364 | +7,5% |
| Mâm chày ngoài | 1.143 | 1.227 | +7,3% |

*Bảng 1. Trung vị diện tích vùng nền sụn theo KL. Phép kiểm an toàn đạt.*

> **Nghi vấn:** KL3 và KL4 theo định nghĩa có gai xương lớn, mà mask phân đoạn không có nhãn gai xương.
> Nếu phần sụn phủ trên gai xương bị gán là sụn thì vùng nền sẽ nở theo KL đúng như quan sát. Hệ quả:
> tỷ lệ mất sụn ở KL cao là ước lượng *bảo thủ* (mẫu số bị phóng to); còn diện tích vùng nền, nếu dùng
> làm đặc trưng, đang mang cả tín hiệu gai xương — hợp lệ để dự đoán KL, nhưng không được gọi là
> biomarker sụn thuần.

## 2. Đánh giá chéo 15 fold theo bệnh nhân

### 2.1. Thiết kế

Dữ liệu được chia 5 fold, phân tầng theo KL, mọi ca của cùng một bệnh nhân nằm trong cùng một fold;
lặp lại với 3 cách chia khác nhau thành 15 fold. Mọi mô hình và mọi bộ đặc trưng dùng chung các fold
này. Khi có hơn 120 biến, bước chọn biến (LASSO) được làm lại bên trong từng fold để tránh rò rỉ.

Sáu mô hình khác nhau đúng một chỗ: cách xử lý thứ tự của thang KL.

| Mô hình | Loại | Xử lý thứ tự |
|---|---|---|
| XGBoost softmax (gốc) | Cây | Không — mô hình của báo cáo 13/9 |
| MLP softmax (đối chứng) | Mạng nơ-ron | Không — cùng mạng với hai MLP ordinal, chỉ khác hàm loss |
| Frank & Hall | Cây | 4 mô hình nhị phân "KL > k", ép đơn điệu |
| Hồi quy + điểm cắt | Cây | Dự đoán điểm liên tục, rồi tối ưu 4 điểm cắt theo kappa |
| MLP ordinal | Mạng nơ-ron | Hàm loss trên 4 ngưỡng "KL > k", phạt vi phạm đơn điệu |
| MLP ordinal đa nhiệm | Mạng nơ-ron | MLP ordinal, thêm nhánh phân loại 5 lớp và nhánh có/không thoái hóa |

*Bảng 2. Sáu mô hình được so sánh.*

### 2.2. Kết quả chính

![Hình 1](figs/fig2_s7_qwk.png)

*Hình 1. QW-Kappa của 6 mô hình trên 3 bộ đặc trưng. Thêm đặc trưng luôn tăng kappa; chênh lệch giữa
các mô hình co lại khi đặc trưng giàu hơn (0,090 → 0,037).*

| Bộ đặc trưng | Số biến | XGBoost softmax (gốc) | Tốt nhất | Mô hình tốt nhất | MAE thấp nhất |
|---|---|---|---|---|---|
| 15 biomarker cũ | 15 | 0,554 | **0,644** | MLP ordinal đa nhiệm | 0,716 |
| Bề mặt xương | 55 | 0,564 | **0,666** | MLP ordinal | 0,708 |
| Cũ + bề mặt xương | 70 | 0,636 | **0,717** | MLP ordinal | 0,633 |
| Cũ + radiomics | 871 | 0,738 | **0,775** | Hồi quy + điểm cắt | 0,549 |
| Tất cả | 926 | 0,750 | **0,776** | Frank & Hall | 0,533 |

*Bảng 3. QW-Kappa đánh giá chéo, n = 1.229. MAE là sai số tuyệt đối trung bình tính theo bậc KL, lấy
giá trị thấp nhất trong 6 mô hình. Số đánh giá chéo không so trực tiếp với số trên một lần chia của báo
cáo 13/9 (mô hình gốc: 0,554 so với 0,532).*

- **Bề mặt xương một mình mạnh ngang 15 biomarker cũ, và hai nhóm bổ sung cho nhau:** 0,666 và 0,644,
  ghép lại lên 0,717.
- **Khi đã có radiomics, bề mặt xương gần như không tăng kappa nhưng giảm sai số:** với Frank & Hall,
  kappa 0,763 → 0,776 và MAE 0,555 → 0,533. Giá trị của nhóm này là tập biến gọn và đọc được: "2,3% diện
  tích mâm chày trong bị trơ" có nghĩa lâm sàng, một hệ số wavelet thì không.

### 2.3. Kiến trúc hay hàm loss?

| Bộ đặc trưng | Đổi cây sang mạng | Thêm loss ordinal |
|---|---|---|
| 15 biomarker cũ | **+0,068** | +0,018 |
| Bề mặt xương | +0,060 | +0,042 |
| Cũ + bề mặt xương | +0,033 | +0,048 |
| Cũ + radiomics | +0,005 | +0,027 |
| Tất cả | **−0,011** | +0,035 |

*Bảng 4. Mức thay đổi QW-Kappa. Cột giữa: XGBoost softmax → MLP softmax. Cột phải: MLP softmax → MLP
ordinal (cùng mạng, cùng số epoch, chỉ đổi hàm loss).*

Lợi thế của mạng so với cây giảm dần rồi đảo dấu khi đặc trưng giàu lên, còn lợi thế của loss ordinal
giữ ở mức +0,02 đến +0,05 trên mọi bộ đặc trưng. **Phần đáng đầu tư là hàm loss thứ tự, không phải chọn
cây hay mạng.**

### 2.4. Ma trận nhầm lẫn

![Hình 2](figs/fig3_s7_confusion.png)

*Hình 2. Ma trận nhầm lẫn của hai mô hình tốt nhất trên bộ đặc trưng đầy đủ, n = 1.229. Không ca KL0
nào bị đoán thành KL4 và ngược lại. Frank & Hall tốt hơn ở KL1 (115/233 so với 89/233), MLP ordinal
tốt hơn ở KL4 (63/106 so với 52/106).*

Sai số tập trung ở các ô sát đường chéo, gần như không còn ca lệch từ 3 bậc trở lên.

### 2.5. Theo từng lớp KL

![Hình 3](figs/fig4_s7_f1.png)

*Hình 3. F1 theo từng lớp KL. KL1 là lớp yếu nhất ở mọi mô hình.*

| Độ KL | XGBoost softmax (gốc) | Frank & Hall | MLP ordinal | MLP ordinal đa nhiệm |
|---|---|---|---|---|
| KL=0 | 64,1 | 60,6 | 60,4 | 61,3 |
| KL=1 | **34,1** | **42,7** | 36,7 | 37,4 |
| KL=2 | 45,1 | 47,2 | 48,2 | 48,5 |
| KL=3 | 61,9 | 63,5 | 64,4 | 64,8 |
| KL=4 | 54,3 | 58,4 | **61,8** | 58,9 |
| Trung bình (macro) | 51,9 | **54,5** | 54,3 | 54,2 |

*Bảng 5. F1 theo lớp (%), bộ đặc trưng đầy đủ, một lần lặp của đánh giá chéo, n = 1.229.*

KL1 chỉ đạt F1 34–43%, khớp thực tế lâm sàng: KL1 là "nghi ngờ hẹp khe khớp", mức mà chính bác sĩ đọc
phim cũng đồng thuận với nhau kém nhất. Lỗi đi cả hai chiều: với Frank & Hall, trong 233 ca KL1 có 47
ca bị đoán xuống KL0 và 67 ca bị đoán lên KL2.

### 2.6. Điểm yếu còn lại: ngưỡng cuối cùng

Xác suất trung bình mà Frank & Hall gán cho "KL > k", theo lớp thật:

| KL thật | P(KL > 0) | P(KL > 1) | P(KL > 2) | P(KL > 3) |
|---|---|---|---|---|
| KL=0 | 0,49 | 0,19 | 0,03 | 0,00 |
| KL=1 | 0,73 | 0,36 | 0,06 | 0,00 |
| KL=2 | 0,90 | 0,66 | 0,22 | 0,01 |
| KL=3 | 0,97 | 0,91 | 0,67 | 0,08 |
| KL=4 | 1,00 | 0,99 | 0,94 | **0,46** |

*Bảng 6. Lý tưởng là một bậc thang: ô nào có KL thật lớn hơn k thì trên 0,5, còn lại dưới 0,5. Ô in
đậm là ô duy nhất nằm sai phía.*

Ca KL4 thật chỉ đạt trung bình 0,46 ở ngưỡng cuối, tức trung bình một ca KL4 không vượt được mốc 0,5 để
được gọi là KL4. Tuy vậy, AUC của bốn ngưỡng lần lượt là 0,872 / 0,902 / 0,934 / 0,952 — ngưỡng cuối lại
phân biệt **tốt nhất**. Mô hình xếp hạng đúng ca KL4, chỉ là xác suất bị kéo thấp vì KL4 chỉ chiếm
8,6% số ca. Đây là vấn đề hiệu chỉnh xác suất, rẻ hơn nhiều để sửa, và là động cơ của mục 3.

## 3. Tập test cố định và cách đặt ngưỡng quyết định

Đánh giá chéo cho số liệu tin cậy nhất nhưng mỗi ca được dự đoán bởi một mô hình khác nhau. Mục này
dùng đúng cách chia train/test của báo cáo 13/9 (246 ca test) để có một mô hình cố định; kết quả ở đây
dùng để loại phương án, còn kết luận dựa vào đánh giá chéo. Lưu ý KL4 chỉ có 26 ca trong tập test, nên
mỗi ca đúng hay sai làm recall KL4 đổi 3,8 điểm phần trăm.

### 3.1. Kết quả với ngưỡng mặc định

| Mô hình | Accuracy | QW-Kappa | MAE | Off-by ≥2 (%) |
|---|---|---|---|---|
| XGBoost softmax (gốc) | 51,2% | 0,750 | 0,593 | 9,8 |
| Frank & Hall | 53,3% | 0,783 | 0,528 | 6,1 |
| Hồi quy + điểm cắt | 54,5% | **0,803** | 0,520 | 6,1 |
| MLP ordinal | 54,5% | 0,796 | 0,516 | **5,3** |
| MLP ordinal đa nhiệm | **57,7%** | 0,797 | **0,500** | 7,3 |
| MLP hai nhánh | 56,1% | 0,788 | 0,516 | 6,9 |

*Bảng 7. Tập test 246 ca, bộ đặc trưng đầy đủ (926 biến, khác bộ 93 biến của báo cáo 13/9). MLP hai
nhánh: một nhánh cho biomarker, một nhánh cho radiomics, trộn bằng trọng số học được. Báo cáo 13/9 có
off-by ≥2 là 7,3% (Frank & Hall) và 6,5% (Regression); ở đây MLP ordinal hạ xuống 5,3%.*

### 3.2. Đổi cách đặt điểm cắt, không huấn luyện lại

Mô hình hồi quy + điểm cắt có kappa cao nhất nhưng recall KL3 thấp nhất (48,3%) và recall KL4 cao nhất
(80,8%): nó đoán thừa hai lớp ngoài cùng và ép hai lớp giữa. Nguyên nhân nằm ở mục tiêu đặt điểm cắt.
QW-Kappa tương đương hệ số tương hợp Lin, mà dự đoán của mô hình hồi quy luôn bị co về giá trị trung
bình, nên cách rẻ nhất để tăng kappa là nới rộng hai lớp ngoài cùng.

![Hình 4](figs/fig5_s8_confusion.png)

*Hình 4. Cùng một mô hình hồi quy đã huấn luyện, chỉ khác cách đặt 4 điểm cắt. Điểm cắt phân vị (tỷ lệ
dự đoán mỗi lớp bằng tỷ lệ thật, không có tham số nào để fit) đưa KL3 lên 32 ca đúng, đổi lại KL4 còn 18.*

### 3.3. Dò ngưỡng cho KL4: hiệu ứng lặp lại được

![Hình 5](figs/fig6_s8_threshold.png)

*Hình 5. Dò ngưỡng quyết định thay cho mốc 0,5 cố định, trên 6 cặp mô hình × bộ đặc trưng. Recall KL4
tăng ở cả 6 cặp, precision KL4 giảm ở cả 6 cặp.*

| | Recall KL4 | Precision KL4 |
|---|---|---|
| Ngưỡng 0,5 cố định | 0,487 | 0,872 |
| Ngưỡng dò được | **0,744** | 0,704 |

*Bảng 8. Trung bình 6 cặp. Dò ngưỡng đổi precision KL4 lấy recall KL4, đúng như thiết kế.*

### 3.4. Nhưng mức lợi tổng không lặp lại

Trên bộ đặc trưng đầy đủ, MLP ordinal đa nhiệm với ngưỡng dò được đạt QW-Kappa 0,814 và macro-recall
0,614 — cao nhất toàn bộ thí nghiệm. Nhưng trên bộ "cũ + radiomics", cùng quy tắc đó chỉ đạt 0,783 và
macro-recall 0,524, thấp hơn cả ngưỡng 0,5 cố định. Frank & Hall với ngưỡng dò được thì ngược lại: đạt
trên bộ "cũ + radiomics", trượt trên bộ đầy đủ. Hai mô hình, hai bộ đặc trưng, mỗi cái chỉ đạt ở một
phía — đây là dấu hiệu của nhiễu. Theo quy tắc đặt ra trước khi chạy, **chưa quy tắc dò ngưỡng nào được
đưa vào kết luận**.

## 4. Biomarker bề mặt xương có được mô hình dùng không

Khi đã có radiomics, thêm biomarker bề mặt xương chỉ tăng kappa +0,004 đến +0,026, nằm trong mức nhiễu.
Nhưng con số đó chưa cho biết các biến này bị loại vì trùng lặp, hay được giữ mà không được dùng.

![Hình 6](figs/fig7_s6_usage.png)

*Hình 6. (a) Tỉ trọng đóng góp vào mô hình theo nhóm biến. (b) Bảy biến bề mặt xương đóng góp nhiều nhất.*

| Nhóm biến | Số biến đầu vào | LASSO loại | Được giữ và được dùng | Tỉ trọng đóng góp | Đóng góp mỗi biến |
|---|---|---|---|---|---|
| Bề mặt xương | 55 | 42 | **13** | **15,5%** | 1,19% |
| 15 biomarker cũ | 15 | 10 | 5 | 6,2% | 1,24% |
| Radiomics | 856 | 767 | 89 | 78,4% | 0,88% |

*Bảng 9. Bộ đặc trưng đầy đủ, mô hình XGBoost. Cả 13 biến bề mặt xương còn lại sau LASSO đều được mô
hình dùng.*

- Biến bề mặt xương mạnh nhất — **diện tích ổ mất sụn lớn nhất ở sụn đùi** — đứng hạng 2 trên toàn bộ
  926 biến; tiếp theo là số ổ mất sụn ở mâm chày trong. Cả hai chỉ đếm ổ từ 5 mm² trở lên, vì rìa mảng
  sụn mỏng dần về 0 nên phép đo sinh ra các đốm giả nhỏ ở rìa: trên một ca thật, tổng diện tích mất sụn
  là 51,3 mm² nhưng ổ lớn nhất chỉ 5,8 mm², tức khoảng 89% con số thô là đốm rìa.
- Tính theo từng biến, **biomarker bề mặt xương hữu ích hơn radiomics** (1,19% so với 0,88%).

## 5. Các hướng đã thử và bị loại

| Hướng | Kết quả | Nguyên nhân |
|---|---|---|
| Asymmetric Loss | Tệ nhất trong 10 mô hình, trên cả 5 bộ đặc trưng | Dùng một mức phạt chung cho cả 4 ngưỡng, trong khi chiều mất cân bằng của chúng ngược nhau: "KL > 0" có 76,9% ca dương, "KL > 3" chỉ có 8,6% |
| Nhánh dự đoán sâu hơn | Thay đổi −0,011 đến +0,026, trong mức nhiễu | Với dữ liệu dạng bảng, mạng sâu hơn không giúp gì |
| Học ngưỡng trong lúc huấn luyện | Ngưỡng học được chỉ dịch tới 0,51–0,53 | Tín hiệu học cho ngưỡng quá yếu. Dò ngưỡng sau huấn luyện đẩy ngưỡng cuối xuống 0,10–0,15 mà không tốn thêm lần huấn luyện nào |
| Trọng số theo lớp | Trung bình −0,007, âm trên cả hai bộ có radiomics | Trùng vai trò với bước đặt điểm cắt vốn đã thích nghi với phân bố; phần còn lại chỉ là giảm cỡ mẫu hiệu dụng (983 → 820 ca, −16,6%) |

*Bảng 10. Bốn hướng đã loại. Ghi lại vì thất bại có giải thích cũng là kết quả nghiên cứu.*

## 6. Đang thử nghiệm, chưa có kết quả

Hai thí nghiệm dưới đây đã xong phần chuẩn bị nhưng chưa có con số để kết luận. Mục này chỉ nêu câu
hỏi, thiết kế và tiến độ. Hiệu năng của M3T trên cohort cố ý chưa được báo cáo, vì phép so chính đã
được chốt trước và chưa chạy.

### 6.1. Mô hình ảnh M3T làm nền: biomarker có cải thiện được không

Đây là bước triển khai đề xuất cuối của báo cáo 13/9: dùng mô hình phân loại ảnh M3T (CNN 3D + CNN 2D
+ transformer, CLS token 128 chiều). Mục tiêu được điều chỉnh ngày 25/9: không còn là "thay radiomics
bằng CLS token", mà là **dùng M3T làm mô hình nền và đo xem biomarker từ phân đoạn có cải thiện được dự
đoán của nó không**.

**Vì sao phải huấn luyện lại M3T.** Bộ trọng số sẵn có đã được huấn luyện trên chính bệnh nhân của
cohort: 853/1.325 ca có bệnh nhân nằm trong tập train của M3T, và M3T đạt kappa 0,851 trên các bệnh
nhân này so với 0,779 trên bệnh nhân chưa thấy. Dùng bộ trọng số đó thì mọi kết quả "M3T + biomarker"
sẽ lạc quan giả. Vì vậy M3T được huấn luyện lại từ đầu, loại mọi bệnh nhân của cohort (cả hai gối, mọi
lần khám, kể cả tập holdout).

**Đã xong (phần chuẩn bị):**

- Tái lập đúng kết quả gốc của M3T trên 1.636 gối test.
- Dữ liệu huấn luyện mới: 6.842 ảnh của 3.468 bệnh nhân, không bệnh nhân nào thuộc cohort; KL4 còn 111
  gối trong tập train.
- Huấn luyện xong (khoảng 9 phút mỗi epoch, dừng sớm ở epoch 99, chọn epoch 94), qua bước kiểm tra so
  với bản gốc.
- Trích CLS token cho 1.317/1.325 ca: 1.221/1.229 ca của cohort và 96/96 ca holdout.

**Phép so chính (chốt ngày 26/9, trước khi có bất kỳ kết quả nào):** mô hình Frank & Hall trên "M3T +
toàn bộ biomarker" (198 biến) so với "chỉ M3T" (128 biến), cùng đánh giá chéo 15 fold theo bệnh nhân.
Mức lợi tối thiểu đáng theo đuổi là +0,02 kappa. Kết luận theo 4 mức: cải thiện vượt ngưỡng, cải thiện
dưới ngưỡng, làm tệ hơn, hoặc chưa đủ bằng chứng.

**Rủi ro đã biết:**

- Ảnh MRI của cohort không chuyển được sang dạng đầu vào của M3T (ảnh của M3T là vùng cắt sát quanh
  khớp), nên chỉ dùng được ca có sẵn ảnh M3T. Thiếu 8 ca, nên mọi bộ đặc trưng sẽ chạy trên cùng 1.221 ca.
- Độ chính xác thống kê còn thô: chạy thử trên một cặp bộ đặc trưng cũ cho khoảng tin cậy rộng khoảng
  ±0,03, lớn hơn mức +0,02. Nếu mức lợi thật nhỏ, kết quả dễ rơi vào "chưa đủ bằng chứng".
- Chỉ thử ghép biomarker dạng bảng với đặc trưng M3T đã đóng băng. Kết quả âm sẽ không loại trừ các
  cách dùng phân đoạn khác (mask làm kênh ảnh, cắt vùng khớp) vì chưa thử.

### 6.2. Tập holdout OAI-ZIB

Câu hỏi: mô hình cuối có giữ được chiều của kết quả trên những ca chưa từng thấy không.

**Đính chính:** báo cáo 13/9 gọi 96 ca tách ra là "external validation". Chính xác hơn đây là **holdout
cùng nguồn OAI-ZIB**, không phải dữ liệu ngoài độc lập, nên kết quả trên đó chỉ là bằng chứng hỗ trợ.

**Đã xong (kiểm tra rò rỉ):**

- 3/96 ca có bệnh nhân xuất hiện trong cohort (gối bên kia hoặc lần khám khác), nên bị loại. Còn
  **93 ca**, KL0–KL4 lần lượt 21/11/20/26/15.
- Mô hình phân đoạn dùng để sinh mask (nnU-Net ResEnc-L, ensemble 5 fold) được huấn luyện trên 544 ca;
  không ca nào trong 93 ca trùng bệnh nhân với 544 ca đó. Không đối chiếu được nội dung ảnh vì ảnh
  huấn luyện không còn được lưu, nên kết luận ở mức sạch theo mã bệnh nhân.
- Phát hiện phụ: 4/103 bệnh nhân của tập test OAI-ZIB có ca iMorphics trong dữ liệu huấn luyện mô hình
  phân đoạn. Dice đã báo cáo trên 103 ca test vì vậy có tối đa 4 ca không hoàn toàn sạch; chưa đo lại,
  ảnh hưởng lên trung bình dự kiến nhỏ.

**Còn lại:** phân đoạn 93 ca → tính biomarker → ghép CLS token → huấn luyện mô hình cuối trên 1.229 ca
→ đánh giá holdout một lần duy nhất. Không dùng holdout để chọn mô hình hay ngưỡng.

## 7. Kết luận và khuyến nghị

### 7.1. Kết quả chính

- Biomarker neo bề mặt xương qua hậu kiểm: chạy đủ 1.229 ca, phép kiểm an toàn đạt. Đo diện tích mất
  sụn thì bắt được bệnh, đo độ dày trung bình thì không.
- Đánh giá chéo 15 fold (n = 1.229): cao nhất QW-Kappa 0,776 (Frank & Hall, bộ đặc trưng đầy đủ). Hàm
  loss ordinal cho mức lợi ổn định +0,02 đến +0,05; chọn cây hay mạng không còn quan trọng khi đặc trưng
  đã giàu.
- KL1 vẫn là lớp khó nhất (F1 34–43%). KL4 bị đoán thiếu do xác suất lệch thấp, không phải do mô hình
  không phân biệt được.
- Dò ngưỡng tăng recall KL4 ổn định, nhưng chưa chứng minh được lợi trên chỉ số tổng.

### 7.2. Hạn chế

- **Độ tin cậy của phép đo mất sụn trên mask AI chưa được đo** — đây là điều kiện bắt buộc trước khi
  công bố bất kỳ con số mất sụn nào.
- Phép đo mất sụn là cận dưới: không đếm được mất sụn ở rìa mảng hay cả khoang trơ trụi. Phép đo tính
  trên cả mảng sụn, chưa chia vùng con như y văn OAI, và chưa đo được sụn bánh chè.
- Nghi vấn gai xương làm nở vùng nền sụn (mục 1.4) chưa được kiểm.
- Phép so trên 447 ca có nhãn tin cậy cao chưa tách được ảnh hưởng của chất lượng nhãn khỏi việc giảm
  cỡ mẫu; cần thêm nhánh đối chứng 447 ca chọn ngẫu nhiên.
- Trọng số trộn của MLP hai nhánh chỉ dịch từ 0,5 tới khoảng 0,40, chưa loại trừ được khả năng nó gần
  như không học.
- Số đánh giá chéo cũ trên iMorphics bị thổi phồng (51/70 gối có hai lần khám nằm ở hai fold khác
  nhau); đánh giá chéo mới chia theo bệnh nhân nên không bị ảnh hưởng.

### 7.3. Việc tiếp theo

1. Chạy phép so chính của M3T và báo kết quả theo 4 mức đã chốt.
2. Tập holdout: xác định đúng bộ trọng số của mô hình phân đoạn đã sinh mask cho cohort, rồi phân đoạn
   93 ca, tính biomarker và đánh giá một lần.
3. Đo độ tin cậy của phép đo mất sụn: so mask AI với nhãn thật (ICC, Bland-Altman) trên 103 ca test
   OAI-ZIB và 36 ca test iMorphics — dữ liệu đã sẵn.
4. Kiểm dò ngưỡng dưới đánh giá chéo 15 fold; không cần huấn luyện lại.
5. Kiểm nghi vấn gai xương bằng điểm gai xương theo khoang trên X-quang của OAI.
6. Tính tương quan với KL cho đủ 88 biến bề mặt xương.
7. Chạy lại MLP hai nhánh với các giá trị khởi tạo khác của trọng số trộn.
8. Chia vùng con cho phép đo mất sụn, sau khi xong việc 3.

# Phụ lục A. Tham số của phép đo bề mặt xương

| Tham số | Giá trị | Vai trò |
|---|---|---|
| Bán kính phép đóng | 8,0 mm | Lấp các lỗ mất sụn được sụn còn lại bao quanh để dựng vùng nền. Hoạt động như một ngưỡng: với lỗ bán kính 4,2 mm, đặt 2 mm cho 8,7 mm², còn 4 / 8 / 12 mm đều cho 49,8 mm². Rủi ro nằm ở phía đặt quá nhỏ. |
| Bước lấy mẫu dọc tia | 0,1 mm | Độ phân giải khi đo độ dày theo pháp tuyến |
| Chiều dài tia tối đa | 6,0 mm | Giới hạn tìm sụn phía trên mặt xương |
| Khoảng hở tối đa | 1,0 mm | Gặp sụn xa hơn mức này thì coi như không có sụn |
| Độ làm mượt | 0,5 mm | Đổi sang voxel riêng cho từng trục (voxel 0,3646 × 0,3646 × 0,70 mm) |
| Mảnh sụn nhỏ nhất được giữ | 100 mm² | Bỏ mảnh sụn rời rạc nhỏ hơn |
| Ổ mất sụn nhỏ nhất | 5 mm² | Ngưỡng để một ổ được tính là tổn thương |

# Phụ lục B. Các chỉ số của phép đo bề mặt xương

Mỗi chỉ số được đo cho 5 vùng: toàn bộ sụn đùi, nửa trong và nửa ngoài của sụn đùi, mâm chày trong,
mâm chày ngoài. Thuật ngữ theo Wirth và Eckstein (2008): tAB là vùng nền sụn, dAB là phần vùng nền
không còn sụn.

| Chỉ số | Ý nghĩa | Ghi chú |
|---|---|---|
| Diện tích vùng nền sụn (tAB) | Vùng xương đáng lẽ phải có sụn phủ | Mẫu số của tỷ lệ mất sụn; nghi mang cả tín hiệu gai xương |
| Tỷ lệ mất sụn toàn bề dày | dAB / tAB, tính theo % | Chỉ số chính để báo cáo |
| Số ổ mất sụn | Số ổ từ 5 mm² trở lên | Tin cậy hơn diện tích thô |
| Diện tích ổ mất sụn lớn nhất | Ổ lớn nhất trong vùng | Biến bề mặt xương mạnh nhất, hạng 2 trên 926 biến |
| Độ dày trung bình trên vùng nền | Chỗ mất sụn tính là 0 mm | Thay thế đúng cho "độ dày sụn" cũ |
| Độ dày trung bình trên phần còn sụn | Chỉ tính chỗ còn sụn | Mù với mất sụn như chỉ số cũ; giữ để đối chiếu |
| Độ dày phân vị 5% | Độ dày tại phân vị 5% của phần còn sụn | Nhạy với chỗ sụn mỏng nhất |
| Tỷ lệ sụn mỏng | % diện tích mỏng dưới 0,5 mm và dưới 1,0 mm | Giai đoạn trước khi mất hẳn |
| Kiểm tra chất lượng | Tỷ lệ pháp tuyến bề mặt bị lật | Phải gần 0 hoặc gần 1; gần 0,5 là phép đo hỏng |
