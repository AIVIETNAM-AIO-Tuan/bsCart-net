# Báo cáo 14/9/2026

*Biomarker neo bề mặt xương và phân loại độ Kellgren–Lawrence theo thang thứ tự,
trên MRI DESS khớp gối*

**Nguồn dữ liệu:** OAI DESS knee MRI cohort
**Ngày lập báo cáo:** 14/09/2026
**Notebook nguồn:** `biomarker_s6_fcl.ipynb`, `biomarker_s7_ordinal.ipynb`, `biomarker_s8_holdout.ipynb`
**Số liệu gốc:** `docs/report_data.py` — sửa ở đó rồi chạy `python docs/make_figs.py` để sinh lại hình.

---

## 0. Thay đổi chính so với báo cáo 13/9/2026

- **Bổ sung họ biomarker mới, neo vào bề mặt xương (S6).** Hai cột cũ mang tên đo tổn thương sụn
  thực ra không đo được: `thickness` là thể tích chia diện tích tiếp xúc nên mù với mất sụn, còn
  `denuded_ratio` lấy mẫu số là toàn bộ biên xương trong ảnh nên chủ yếu phản ánh trường nhìn.
  Cột mới tương quan với KL mạnh **gấp khoảng bốn lần** cột cũ.
- **Đánh giá lại toàn bộ bằng cross-validation 15 fold (S7)**, thay vì một lần chia. Mỗi ca được dự
  đoán out-of-fold nên con số tính trên cả **1.229 ca** chứ không phải 246.
- **Tách được đóng góp của *kiến trúc* khỏi đóng góp của *hàm loss*** nhờ đối chứng A2. Kết quả đảo
  chiều theo chất lượng đặc trưng và không đọc được nếu thiếu đối chứng này.
- **Thêm tầng quyết định (S8):** cùng một mô hình đã huấn luyện, đổi quy tắc đặt ngưỡng thì recall
  của lớp KL4 tăng ở **12/12** phép so. Nhưng mức lợi trên chỉ số tổng **không tái lập** giữa hai bộ
  đặc trưng, nên chưa quy tắc nào được đưa vào kết luận.
- **Bốn hướng bị loại, có giải thích:** Asymmetric Loss, đầu dự đoán sâu hơn, ngưỡng học được trong
  lúc huấn luyện, và trọng số theo lớp. Chi tiết ở mục 6.
- **Baseline khớp tuyệt đối báo cáo 13/9:** S4 gốc trên 15 biến hình học cho accuracy 41,1%,
  κ = 0,532, off-by≥2 = 22,0% — trùng từng chữ số.

---

## 1. Cohort

Không đổi so với báo cáo 13/9: **1.229 ca / 1.215 subject**, sau khi tách 96 ca OAIZIB-CM làm
external validation. Nhãn tách luôn lấy lại từ `cohort_manifest.csv`, không tin cột trong bảng đặc
trưng, vì bảng S3 được sinh trước khi phép tách tồn tại.

| Độ KL | Số ca | Tỷ trọng | Trong test set (n=246) |
|---|---|---|---|
| KL=0 | 284 | 23,1% | 55 |
| KL=1 | 233 | 19,0% | 50 |
| KL=2 | 295 | 24,0% | 57 |
| KL=3 | 311 | 25,3% | 58 |
| KL=4 | 106 | 8,6% | 26 |

*Bảng 1. Phân bố độ KL. Test set dùng `GroupShuffleSplit(test_size=0.2, random_state=42)` theo
subject — đúng cấu hình báo cáo 13/9.*

> **Lớp KL4 chỉ có 26 ca trong test set**, nên mỗi ca đoán đúng hay sai làm recall đổi 3,8 điểm phần
> trăm. Mọi nhận định về KL4 phải dựa vào số cross-validation ở mục 3, không dựa vào mục 4.

---

## 2. Biomarker neo bề mặt xương (S6)

### 2.1. Ý tưởng

Sụn là thứ biến mất, nên không thể lấy sụn làm hệ quy chiếu để đo chính sự biến mất đó. Mọi phép đo
của S6 đặt trên **bề mặt xương dưới sụn**: tại từng điểm trên mặt xương, hỏi phía trên điểm đó có
bao nhiêu milimet sụn. Không có gì thì câu trả lời là **0**, chứ không phải là không có câu hỏi.

Thuật ngữ theo Eckstein và Wirth: `tAB` là vùng đáng lẽ phải có sụn (footprint), `cAB` là phần còn
sụn phủ, `dAB = tAB − cAB` là vùng trơ — chính là **mất sụn toàn bề dày (FCL)**.

### 2.2. Kết quả chạy

1.229/1.229 ca, **không lỗi nào**, khoảng 15,7 giây mỗi ca. Bảng `biomarker_table_v2.csv` có
1.229 × 88 cột. Đối chiếu 15 cột cũ: sai lệch lớn nhất **1,16 × 10⁻¹⁰**, tức bảng mới là **tập cha
thực sự** của bảng S3, mọi kết quả cũ chạy lại đều ra đúng con số cũ.

![Hình 1](figs/fig1_s6_fcl.png)

*Hình 1. **Bảng a** tách thành hai cụm rời hẳn nhau. Năm cột FCL nằm sát nhau trong dải 0,371–0,433,
rồi có một khoảng trống 0,209 trước khi tới cột kế tiếp; hai cột cũ nằm ép sát vạch 0, một trong đó
còn mang dấu âm, tức đo ngược chiều bệnh. Đáng chú ý là ba cột dẫn đầu chỉ lệch nhau 0,010 dù đo ba
thứ khác hẳn nhau — diện tích ổ lớn nhất, số ổ, và phần trăm diện tích — nghĩa là tín hiệu nằm ở bản
thân hiện tượng mất sụn chứ không phải ở cách tổng hợp con số.
**Bảng b** cho ba đường cùng một hình dạng: đi ngang, thậm chí chúc nhẹ xuống ở chặng KL0 sang KL1,
uốn lên từ KL2, rồi dốc đứng ở chặng cuối. Riêng chặng KL3 sang KL4 chiếm 56–78% toàn bộ mức tăng của
cả thang. Thứ tự ba khoang không đổi ở bất kỳ mức nào, và khoảng cách giữa sụn đùi với mâm chày ngoài
giãn ra gấp 3,5 lần từ KL0 tới KL4 — mất sụn không chỉ nhiều hơn mà còn lệch về một khoang rõ hơn khi
bệnh nặng lên.*

Hai điều đáng chú ý ở bảng bên phải. **KL0 và KL1 phẳng, thậm chí hơi nghịch chiều** — đúng với định
nghĩa lâm sàng, vì hai mức này chưa có hẹp khe khớp rõ nên chưa nên có mất sụn toàn bề dày; con số
0,8–1,1% đo được ở KL0 là *sàn đốm rìa* của phép đo chứ không phải bệnh. Và **mức tăng dồn vào chặng
KL3 sang KL4**, gần như gấp đôi.

### 2.3. Phép kiểm an toàn quan trọng nhất

Nếu diện tích footprint **sụp** ở nhóm KL4 thì không phải bệnh nặng hơn mà là phép đóng mất chỗ bám,
và mọi số FCL ở nhóm đó vô giá trị. Nó không sụp — nó **nở**:

| | KL=0 | KL=4 | Thay đổi |
|---|---|---|---|
| `tab_fem` (mm²) | 5.936 | 6.882 | **+16%** |
| `tab_mt` (mm²) | 1.269 | 1.364 | +7,5% |
| `tab_lt` (mm²) | 1.143 | 1.227 | +7,3% |

*Bảng 2. Trung vị diện tích footprint theo KL. Phép kiểm an toàn đạt.*

> **Nhưng chiều tăng này mở ra một nghi vấn.** KL3 và KL4 *theo định nghĩa* đòi hỏi gai xương lớn, mà
> mask không có nhãn gai xương. Nếu mũ sụn của gai xương bị gán nhãn sụn thì footprint sẽ nở đúng
> theo mức KL, y như quan sát được. Hai hệ quả: (a) `fcl_*_pct` bị kéo xuống ở KL cao vì mẫu số nở
> ra, tức nó là ước lượng **bảo thủ**, tin được theo chiều dương; (b) nếu đưa `tab_*` vào làm đặc
> trưng thì đang dùng tín hiệu gai xương — hợp lệ cho bài toán dự đoán KL, nhưng **không được gọi nó
> là biomarker sụn thuần** trong báo cáo.

---

## 3. Phân loại độ KL theo thang thứ tự (S7)

### 3.1. Thiết kế đánh giá

`StratifiedGroupKFold` 5 fold **theo subject**, lặp 3 seed thành 15 fold; cùng bộ fold cho mọi mô
hình và mọi bộ đặc trưng. Impute trung vị fit chỉ trên train. Trên 120 cột thì **chọn đặc trưng lại
trong từng fold** để chống rò rỉ. Tổng 540 tổ hợp, 41,3 phút.

Sáu mô hình khác nhau **đúng một chỗ**: cách xử lý thứ tự.

| | Mô hình | Họ | Xử lý thứ tự bằng |
|---|---|---|---|
| A | XGB softmax | cây | Không. Đây là baseline S4. |
| A2 | MLP softmax | mạng | Không. **Đối chứng.** |
| B | XGB Frank & Hall | cây | 4 bộ phân loại nhị phân, ép đơn điệu, giải mã đếm |
| C | XGB hồi quy + điểm cắt | cây | Điểm liên tục, rồi tối ưu 4 điểm cắt theo QWK |
| D | MLP loss ngưỡng | mạng | BCE trên 4 ngưỡng + phạt đơn điệu |
| E | MLP đủ loss của slide | mạng | D + head softmax + head OA |

### 3.2. Kết quả chính

![Hình 2](figs/fig2_s7_qwk.png)

*Hình 2. Hai điều đọc được theo hai hướng khác nhau. Theo chiều ngang trong mỗi nhóm, thứ tự ba cột
màu không đổi ở cả sáu mô hình: thêm đặc trưng luôn tăng, mức tăng từ bảng cũ lên bộ đầy đủ là +0,117
tới +0,196. Theo chiều dọc giữa các nhóm, **khoảng biến thiên giữa các mô hình co lại khi đặc trưng
tốt lên**: rộng 0,090 trên bảng cũ, 0,081 khi thêm S6, chỉ còn 0,037 trên bộ đầy đủ. Nói cách khác,
chọn mô hình nào là chuyện quan trọng khi đặc trưng nghèo và gần như hết quan trọng khi đặc trưng
giàu. Vùng tô xám bên trái là hai mô hình không dùng thứ tự; chúng bị bỏ xa ở cột xanh dương nhưng
bắt kịp ở cột xanh lục.*

| Bộ đặc trưng | Số cột | A (nền) | Tốt nhất | Mô hình | MAE tốt nhất |
|---|---|---|---|---|---|
| Bảng cũ | 15 | 0,554 | **0,644** | E | 0,716 |
| Bề mặt S6 một mình | 55 | 0,564 | **0,666** | D | 0,708 |
| Cũ + S6 | 70 | 0,636 | **0,717** | D | 0,633 |
| Cũ + radiomics | 871 | 0,738 | **0,775** | C | 0,549 |
| Tất cả | 926 | 0,750 | **0,776** | B | 0,533 |

*Bảng 3. QWK out-of-fold, n = 1.229. Cột "A (nền)" là baseline nominal để so mức tăng.*

**Ba nhận định đọc được từ bảng này.**

**Một — họ bề mặt S6 một mình mạnh ngang cả bảng cũ, và hai bên bổ sung cho nhau.** 55 cột bề mặt cho
0,666, 15 cột cũ cho 0,644, nhưng ghép lại thì lên **0,717**, cao hơn hẳn từng bên. Hai họ không đo
cùng một thứ.

**Hai — khi đã có radiomics, S6 gần như không thêm QWK nhưng có thêm MAE.** Với Frank-Hall:
QWK 0,763 → 0,776, MAE 0,555 → **0,533**. Cách đọc: radiomics trên ảnh đã chứa phần lớn thông tin mà
hình học nói ra; giá trị của họ bề mặt nằm ở chỗ nó là **tập con gọn và đọc được** — bác sĩ hiểu
"2,3% diện tích mâm chày trong bị trơ", không ai đọc được một hệ số wavelet.

**Ba — lợi thế của kiến trúc bốc hơi khi đặc trưng tốt lên, lợi thế của loss thì không.** Đây là phát
hiện chỉ nhìn thấy được nhờ đối chứng A2:

| Bộ đặc trưng | Số cột | A → A2 (đổi cây sang mạng) | A2 → D (thêm loss ordinal) |
|---|---|---|---|
| Bảng cũ | 15 | **+0,068** | +0,018 |
| Bề mặt S6 | 55 | +0,060 | +0,042 |
| Cũ + S6 | 70 | +0,033 | +0,048 |
| Cũ + radiomics | 871 | +0,005 | +0,027 |
| Tất cả | 926 | **−0,011** | +0,035 |

*Bảng 4. Tách đóng góp của kiến trúc khỏi đóng góp của hàm loss. A2 dùng cùng thân mạng, cùng số
epoch, cùng seed với D — chỉ đổi hàm loss.*

Cột trái **giảm đơn điệu và đảo dấu**: mạng chỉ hơn cây khi đặc trưng nghèo, và ở bộ giàu nhất thì
cây thắng ngược. Cột phải **đứng nguyên ở mức +0,02 tới +0,05 ở mọi nơi**. Kết luận thực dụng: đừng
tốn thời gian tranh luận cây hay mạng; **loss ordinal mới là phần trả lãi ổn định**.

### 3.3. Ma trận nhầm lẫn

![Hình 3](figs/fig3_s7_confusion.png)

*Hình 3. Ô tô đậm theo tỷ lệ trong hàng, số là đếm ca, viền xanh là đoán đúng. Cả hai ma trận có khối
đậm chạy dọc đường chéo và lan sang đúng một ô hai bên, còn **hai góc đối diện thì trống hoàn toàn**:
không một ca KL0 nào bị gọi thành KL4 và ngược lại, ở cả hai mô hình. Điểm khác nhau giữa hai bảng gọn
trong một phép đánh đổi. B đọc đúng 115 trên 233 ca KL1 còn D chỉ 89, nhưng ở hàng cuối thì ngược lại:
D bắt được 63 trên 106 ca KL4 còn B chỉ 52. Nhìn hàng KL4 của B thấy rõ vì sao — 52 ca bị đẩy sang ô
KL3 ngay bên cạnh, **đúng bằng** số ca nó gọi đúng.*

Sai số của cả hai tập trung ở cận chéo, đúng tinh thần của thang thứ tự, và gần như không còn ca lệch
từ 3 bậc trở lên.

### 3.4. Recall và F1 theo từng lớp

![Hình 4](figs/fig4_s7_f1.png)

*Hình 4. Cả ba mô hình vẽ ra cùng một đường gấp khúc hình chữ M: cao ở KL0, sụt sâu ở KL1, hồi một
phần ở KL2, đạt đỉnh ở KL3 rồi lại tụt ở KL4. Hố KL1 sâu 21–28 điểm so với đỉnh KL3 ngay cạnh nó,
và nó là hố duy nhất không mô hình nào lấp được. Ở bốn lớp còn lại ba cột gần bằng nhau, chênh lệch
giữa các mô hình dồn hết vào hai lớp: tại KL1 thì B dẫn trước A tới 9 điểm, còn tại KL4 thì D dẫn
trước A 8 điểm. Baseline A chỉ thắng ở đúng một chỗ là KL0, và thua ở mọi chỗ khác.*

| Độ KL | A (S4 gốc) | B (Frank-Hall) | D (MLP ordinal) | E (MLP slide) |
|---|---|---|---|---|
| KL=0 | 64,1 | 60,6 | 60,4 | 61,3 |
| KL=1 | **34,1** | **42,7** | 36,7 | 37,4 |
| KL=2 | 45,1 | 47,2 | 48,2 | 48,5 |
| KL=3 | 61,9 | 63,5 | 64,4 | 64,8 |
| KL=4 | 54,3 | 58,4 | **61,8** | 58,9 |
| **macro** | 51,9 | **54,5** | 54,3 | 54,2 |

*Bảng 5. F1 theo lớp (%), seed 0, out-of-fold n = 1.229.*

**KL1 là lớp yếu ở mọi mô hình**, F1 chỉ 34–43%. Điều này khớp thực tế lâm sàng: KL1 nghĩa là "nghi
ngờ hẹp khe khớp", và chính các bác sĩ đọc phim cũng đồng thuận với nhau kém nhất ở mức này. Lỗi của
KL1 đi cả hai chiều: trong 233 ca thì 47 rơi xuống KL0 và 67 bị đẩy lên KL2.

### 3.5. Mắt xích yếu duy nhất: ngưỡng cuối

Trung bình xác suất `P(KL > k)` theo lớp thật, Frank-Hall. Lý tưởng là một bậc thang:

| KL thật | P(KL>0) | P(KL>1) | P(KL>2) | P(KL>3) |
|---|---|---|---|---|
| KL=0 | 0,49 | 0,19 | 0,03 | 0,00 |
| KL=1 | 0,73 | 0,36 | 0,06 | 0,00 |
| KL=2 | 0,90 | 0,66 | 0,22 | 0,01 |
| KL=3 | 0,97 | 0,91 | 0,67 | 0,08 |
| KL=4 | 1,00 | 0,99 | 0,94 | **0,46** |

*Bảng 6. Bậc thang xác suất. Ô in đậm là ô duy nhất nằm sai phía vạch quyết định 0,5.*

Bậc thang gần như đúng, trừ **đúng một ô**: ca KL4 thật chỉ đạt trung bình 0,46 ở ngưỡng cuối, tức
trung bình một ca KL4 thật **không vượt nổi vạch để được gọi là KL4**. Mọi ô khác trong vùng "đáng lẽ
phải bật" đều từ 0,66 trở lên.

> **Nhưng đây là vấn đề hiệu chỉnh, không phải vấn đề phân biệt.** AUC từng ngưỡng của cùng mô hình
> đó là **0,872 / 0,902 / 0,934 / 0,952** — ngưỡng cuối có AUC *cao nhất* trong bốn ngưỡng. Mô hình
> biết rõ ca nào là KL4, nó xếp hạng chuẩn hơn cả ba ngưỡng kia; nó chỉ từ chối nói ra ở vạch cố định
> 0,5, vì nhãn `KL>3` lệch 1 chọi 11. Phân biệt được nhưng không hiệu chỉnh được là hai chuyện khác
> nhau, và chuyện thứ hai **rẻ hơn nhiều để sửa**. Đó là động cơ của mục 4.

---

## 4. Mô hình cố định và tầng quyết định (S8)

S7 cho con số trung thực nhất nhưng **mỗi ca được dự đoán bởi một mô hình khác nhau**, và không tồn
tại "mô hình cuối cùng" nào để đem đi làm việc khác. S8 dùng đúng cách chia của báo cáo 13/9 để có
*một* mô hình và *một* tập test cố định. **Dùng S8 để loại, dùng S7 để kết luận.**

### 4.1. Kết quả mặc định, định dạng như Bảng 5 của báo cáo 13/9

| Mô hình | Accuracy | QW-Kappa | MAE | Off-by≥2 (%) |
|---|---|---|---|---|
| A — XGB softmax | 51,2% | 0,750 | 0,593 | 9,8 |
| B — Frank & Hall | 53,3% | 0,783 | 0,528 | **6,1** |
| C — hồi quy + điểm cắt | 54,5% | **0,803** | 0,520 | **6,1** |
| D — MLP loss ngưỡng | 54,5% | 0,796 | 0,516 | **5,3** |
| E — MLP đủ loss slide | **57,7%** | 0,797 | **0,500** | 7,3 |
| I — hai nhánh bio/radiomics | 56,1% | 0,788 | 0,516 | 6,9 |

*Bảng 7. Tập test cố định n = 246, bộ đặc trưng đầy đủ. Off-by≥2 là tỷ lệ ca lệch từ 2 bậc KL trở
lên — lỗi lâm sàng nghiêm trọng nhất. Báo cáo 13/9 đo được 22,0% cho S4 gốc, 7,3% cho Frank-Hall và
6,5% cho Regression; ở đây D hạ xuống **5,3%**.*

### 4.2. Tầng quyết định: đổi quy tắc đặt ngưỡng, không huấn luyện lại gì

Mô hình C có QWK cao nhất bảng nhưng recall KL3 thấp nhất trong cả mười mô hình (48,3%) và recall KL4
cao nhất (80,8%) với precision KL4 chỉ 72,4%. Nó **đoán thừa hai lớp biên và bóp hai lớp giữa**.

**Đó không phải lỗi của bộ hồi quy mà là lỗi của mục tiêu đặt điểm cắt.** QWK quadratic bằng
`2·Cov(y, ŷ) / (Var y + Var ŷ + (μy − μŷ)²)`, tức hệ số tương hợp Lin. Điểm hồi quy luôn bị **co về
trung bình**, `Var(score) < Var(y)`. Cách rẻ nhất để đẩy phân số đó lên là **bơm `Var(ŷ)`**, và cách
bơm là nới rộng hai bin ngoài cùng. Thuật toán không hỏng — nó đang tối ưu đúng cái ta bảo nó tối ưu.

![Hình 5](figs/fig5_s8_confusion.png)

*Hình 5. Hai ma trận đến từ **cùng một bộ hồi quy đã huấn luyện**, khác nhau duy nhất ở chỗ đặt bốn
điểm cắt. So hai hàng cuối thấy ngay cái giá của quy tắc tối ưu QWK: bảng trái đẩy được 21 ca KL4 vào
đúng ô, nhưng hàng KL3 ngay trên đó chỉ còn 28 ca đúng và để 8 ca trôi sang KL4. Bảng phải làm ngược
lại, KL3 lên 32 và KL4 xuống 18. Cột KL4 của bảng trái dài hơn hẳn cột tương ứng bên phải — đó chính
là hình ảnh của việc nới rộng bin ngoài cùng. Điểm cắt phân vị đặt cắt sao cho tỷ lệ dự đoán bằng tỷ
lệ thật và **không có tham số nào để fit**.*

### 4.3. Hiệu ứng tái lập được: ngưỡng dò đổi precision KL4 lấy recall KL4

![Hình 6](figs/fig6_s8_threshold.png)

*Hình 6. Sáu cặp là 3 mô hình (B, D, E) nhân 2 bộ đặc trưng có radiomics. **Không một đoạn nối nào
bắt chéo hướng chung**: sáu đoạn bên trái đều đi lên, sáu đoạn bên phải đều đi xuống. Độ dốc thì rất
khác nhau — cặp thấp nhất ở bảng a nhảy từ 0,39 lên 0,89 trong khi cặp cao nhất chỉ đi từ 0,54 lên
0,65, tức **mô hình nào đang bỏ sót KL4 nhiều nhất thì được lợi nhiều nhất**. Ở bảng b, năm trong sáu
điểm xuất phát chụm trong dải 0,83–0,88 rồi tỏa ra thành 0,63–0,83, cho thấy cái giá phải trả không
đồng đều giữa các mô hình.*

| | Recall KL4 | Precision KL4 |
|---|---|---|
| Vạch 0,5 cố định | 0,487 | 0,872 |
| Ngưỡng dò được | **0,744** | 0,704 |

*Bảng 8. Trung bình 6 cặp. Cơ chế hoạt động chính xác như thiết kế.*

### 4.4. Nhưng mức lợi tổng **không** tái lập

Quy tắc tốt nhất trên bộ đầy đủ là `E @ ngưỡng dò`, cho **QWK 0,814 và macro-recall 0,614 — cả hai
đều cao nhất trong toàn bộ bảng S8**, vượt C mặc định 0,803. Nó qua cả năm cổng thăng hạng.

Nhưng trên bộ `cũ + radiomics`, đúng quy tắc đó cho 0,783 và macro-recall 0,524, tức **thấp hơn cả
vạch 0,5 cố định**. Và `B @ ngưỡng dò` là ảnh soi gương: đạt trên bộ `cũ + radiomics`, trượt trên bộ
đầy đủ.

> **Hai mô hình, hai bộ đặc trưng, mỗi cái đạt đúng một, và ở hai phía ngược nhau.** Đó là chữ ký của
> nhiễu chứ không phải tín hiệu. Theo luật đã đăng ký *trước khi chạy*, **không quy tắc nào được đưa
> vào kết luận**, dù `E @ ngưỡng dò` đang giữ con số đẹp nhất bảng.

---

## 5. Cột S6 có thực sự được mô hình dùng không

Hiệu số QWK không trả lời được câu này: có radiomics thì thêm S6 chỉ nhích +0,004 tới +0,026, nằm
trong nhiễu. Nhưng đó có thể là "bị loại vì dư thừa" hoặc "được giữ mà vô dụng" — hai chuyện rất
khác nhau. Tách ba trạng thái thì thấy rõ.

![Hình 7](figs/fig7_s6_usage.png)

*Hình 7. **Bảng a** nhìn thoáng thì radiomics áp đảo, nhưng phải đọc kèm dòng chữ nhỏ dưới mỗi cột:
78,4% gain đó trải trên 89 cột, còn 15,5% của S6 chỉ trải trên 13 cột. Quy về từng cột thì thứ tự đảo
ngược — bảng cũ 1,24%, S6 1,19%, radiomics 0,88%. Chiều dài cột cam nói lên số lượng chứ không nói
lên chất lượng.
**Bảng b** cho thấy ba cột dẫn đầu của S6 đều thuộc họ FCL, và hai cột đứng đầu chính là hai cột đã
qua ngưỡng 5 mm². Gain giảm đều từ 2,47% xuống 1,00% mà không có bậc hụt nào, tức không phải một cột
duy nhất gánh toàn bộ đóng góp của S6.*

| Họ đặc trưng | Cột vào | LASSO bỏ | Giữ và có gain | Tỷ trọng gain | Gain / cột |
|---|---|---|---|---|---|
| Bề mặt S6 | 55 | 42 | **13** | **15,5%** | 1,19% |
| Bảng cũ | 15 | 10 | 5 | 6,2% | 1,24% |
| Radiomics | 856 | 767 | 89 | 78,4% | 0,88% |

*Bảng 9. Không cột S6 nào rơi vào trạng thái "sống sót LASSO nhưng gain 0" — cả 13 cột sống sót đều
được cây tách trên chúng.*

**Cột S6 mạnh nhất đứng hạng 2 trên tổng 926 cột**: `fcl_fem_maxdef_mm2`, kế đến là `fcl_mt_ndef`.
Đúng hai cột đã được chọn thay cho `fcl_*_mm2` thô nhờ ngưỡng 5 mm² — mép mảng sụn thuôn dần về 0 nên
phép đóng luôn bắc qua một vành mỏng và sinh đốm giả ở rìa; đo trên ca thật, tổng FCL 51,3 mm² nhưng
ổ lớn nhất chỉ 5,8 mm², tức khoảng 89% con số thô là đốm rìa.

Tính theo từng cột, **S6 hữu ích hơn radiomics** (1,19% so với 0,88% gain mỗi cột).

---

## 6. Kết quả âm tính — giữ lại, không xoá

| Hướng | Kết quả | Nguyên nhân |
|---|---|---|
| **Asymmetric Loss** | Tệ nhất trong 10 mô hình, trên cả 5 bộ đặc trưng, theo cả QWK lẫn off-by≥2 | Áp **một** `gamma_neg` cho **cả bốn** ngưỡng, trong khi chiều lệch của chúng **đảo dấu**: `KL>0` có 76,9% dương (thiểu số là *âm*), `KL>3` chỉ 8,6% dương (thiểu số là *dương*) |
| **Đầu dự đoán sâu hơn** | −0,011 tới +0,026, dưới ngưỡng nhiễu | Trên bảng biomarker, làm đầu dự đoán sâu hơn không giúp gì |
| **Ngưỡng `tau` học được** | `tau` quy ra ngưỡng 0,51–0,53, gần như không rời vạch 0,5 | Thành phần loss sinh gradient cho `tau` quá yếu so với ba thành phần còn lại. Tìm kiếm hậu kiểm đẩy ngưỡng cuối xuống tận **0,10–0,15** — làm được thứ gradient descent không làm được, với chi phí huấn luyện bằng 0 |
| **Trọng số theo lớp** | Trung bình −0,007, âm trên cả hai bộ có radiomics | **Dư thừa**, không phải sai: bước đặt điểm cắt vốn đã thích nghi với phân bố điểm, nên đổi trọng số bị bù trừ lại. Cái còn lại là giá về phương sai — cỡ mẫu hiệu dụng tụt 983 → 820, mất 16,6% |

*Bảng 10. Bốn hướng đã thử và bị loại. Ghi lại vì thất bại có giải thích là thông tin nghiên cứu.*

---

## 7. Hạn chế và việc tiếp theo

### 7.1. Điều kiện chặn trước khi công bố bất kỳ con số FCL nào

**Độ tin cậy của FCL tính từ mask AI chưa được đo.** Việc cần làm là so FCL tính từ nhãn thật với FCL
tính từ mask dự đoán trên 103 ca test OAI-ZIB và 36 ca test iMorphics, báo cáo ICC và Bland-Altman.
**Dữ liệu đã có sẵn, không cần chạy thêm inference nào.**

### 7.2. Hạn chế đã biết của phép đo

- **FCL là cận dưới.** Phép đóng chỉ lấp được lỗ *bị sụn còn lại bao quanh*; mất sụn ở rìa mảng hoặc
  cả một khoang trơ trụi sẽ không được đếm. Đây là chiều an toàn nhưng phải nói ra.
- **Đo ở mức cả mảng, không phải vùng con.** Y văn OAI chia mỗi mảng thành vùng trung tâm / trong /
  ngoài, và mất sụn dồn vào vùng trung tâm chịu lực. Hiện một ổ ở vùng trung tâm bị pha loãng vào mẫu
  số lớn hơn nhiều lần. Đây nhiều khả năng là lý do `thc_tab` chỉ đạt ρ khoảng −0,07 tới −0,19 dù về
  nguyên tắc nó phải là cột nhạy nhất.
- **`fem_med` / `fem_lat` không phải cMF chuẩn**, nên không so trực tiếp được với số công bố theo
  vùng con của Chondrometrics.
- **Không đo được sụn bánh chè** vì mask không có xương bánh chè để neo.

### 7.3. Hạn chế của phần đánh giá

- **Phép kiểm độ nhạy trộn chất lượng nhãn với cỡ mẫu.** So QWK trên toàn cohort với chỉ 447 ca tin
  cậy cao cho mức giảm khoảng 0,04, nhưng tập con chỉ bằng 36% cohort nên tập huấn luyện tụt từ ~980
  xuống ~358 ca — riêng việc đó đã đủ giải thích. Cần thêm nhánh đối chứng lấy ngẫu nhiên 447 ca.
- **Trọng số `alpha` của nhánh gộp chưa kiểm nhận dạng.** Nó khởi tạo ở 0,5 và chỉ dịch tới
  0,395–0,403, chưa loại trừ được khả năng gate gần như đứng yên. Phép kiểm: chạy lại với gate khởi
  tạo −2 và +2.
- **So sánh với baseline cũ phải cẩn thận.** 51/70 đầu gối iMorphics có hai timepoint ở hai fold khác
  nhau nên số CV cũ bị thổi phồng. S7 chia theo subject nên phần ordinal không dính.

### 7.4. Việc tiếp theo, xếp theo giá trên lợi

1. **Đo độ tin cậy FCL từ mask AI** — chặn mọi việc công bố, dữ liệu đã sẵn.
2. **Chạy dò ngưỡng dưới cross-validation 15 fold của S7** cho B, D, E. Đây là phép kiểm duy nhất
   tách được mức lợi thật khỏi nhiễu một lần chia, và không phải huấn luyện lại mô hình nào.
3. **Xác nhận nhiễm gai xương** bằng điểm gai xương theo khoang trong `KXR_SQ_BU00.txt`.
4. **Kiểm nhận dạng `alpha`** của nhánh gộp bio/radiomics.
5. **Chia vùng con cho S6** — nâng cấp có giá trị cao nhất còn lại cho phép đo, nhưng chỉ sau khi
   mục 7.1 xong.

---

## Phụ lục A. Tham số của S6

| Tham số | Giá trị | Vai trò |
|---|---|---|
| `close_mm` | 8,0 | Bán kính đóng trắc địa. Hành xử như **ngưỡng**, không phải núm chỉnh liên tục: với lỗ bán kính 4,2 mm, đặt 2 mm cho 8,7 mm² còn 4 / 8 / 12 mm đều cho đúng 49,8 mm². Rủi ro nằm ở phía đặt quá nhỏ. |
| `step_mm` | 0,1 | Bước lấy mẫu dọc tia |
| `max_mm` | 6,0 | Chiều dài tia tối đa |
| `gap_mm` | 1,0 | Chạm xa hơn mức này thì coi như không có sụn |
| `smooth_mm` | 0,5 | Sigma làm mượt, đổi sang voxel **riêng từng trục** (voxel 0,3646 × 0,3646 × 0,70 mm) |
| `min_island_mm2` | 100 | Bỏ mảnh sụn rời rạc |
| `min_defect_mm2` | 5 | Ngưỡng để một ổ được tính là tổn thương |

## Phụ lục B. Đọc tên cột của S6

Quy tắc: `<metric>_<khoang>_<đơn vị>`. Năm khoang: `fem` cả sụn đùi, `fem_med` / `fem_lat` hai nửa,
`mt` mâm chày trong, `lt` mâm chày ngoài.

| Cột | Nghĩa | Dùng khi nào |
|---|---|---|
| `tab_*_mm2` | Diện tích footprint, vùng đáng lẽ có sụn | Mẫu số. Cũng là cột nghi nhiễm gai xương. |
| `fcl_*_pct` | Trơ tính theo phần trăm footprint | Cột chính để báo cáo tỷ lệ |
| `fcl_*_ndef` | Số ổ trơ từ 5 mm² trở lên | **Đáng tin hơn cột thô** |
| `fcl_*_maxdef_mm2` | Diện tích ổ trơ lớn nhất | **Cột mạnh nhất trong bảng**, hạng 2 trên 926 cột |
| `thc_tab_*_mm` | Độ dày trung bình trên cả footprint, **tính lỗ là 0** | Cột thay thế đúng cho `thickness` cũ |
| `thc_cab_*_mm` | Độ dày trung bình chỉ trên phần còn sụn | Hành xử giống cột cũ, tức mù với mất sụn. Giữ để đối chiếu. |
| `thin_le05_*_pct`, `thin_le10_*_pct` | Phần trăm diện tích mỏng dưới 0,5 và 1,0 mm | Giai đoạn trước khi mất hẳn |
| `qc_flipfrac_*` | Tỷ lệ pháp tuyến bị lật | Phải gần 0 hoặc gần 1. Gần 0,5 là hỏng. |
