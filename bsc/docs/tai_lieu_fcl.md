# Tài liệu tham chiếu cho FCL / dAB — Eckstein và Wirth

Tra cứu trên **PubMed** ngày 14/09/2026. Mọi bài dưới đây đều có Eckstein F hoặc Wirth W đứng tên.
Xếp theo việc chúng trả lời cho dự án, không theo năm.

Ghi chú: `tAB` = total area of subchondral bone (footprint), `cAB` = cartilage-covered,
`dAB` = denuded area = **FCL**, `ThC.tAB` = độ dày trung bình trên cả tAB (tính lỗ là 0).

---

## 0. Bài PHƯƠNG PHÁP gốc — chính là ý tưởng ở mục 2.1 của báo cáo

**Wirth W, Eckstein F (2008).** *A technique for regional analysis of femorotibial cartilage thickness
based on quantitative magnetic resonance imaging.* IEEE Trans Med Imaging 27(6):737–44.
[DOI](https://doi.org/10.1109/TMI.2007.907323) · PMID 18541481

> **Đây là bài phải trích cho ý tưởng "neo vào bề mặt xương", không phải bài thuật ngữ 2006.**

Phần tóm tắt của họ nói thẳng: mâm chày được chia thành vùng trung tâm của **tổng diện tích xương dưới
sụn (tAB)** cùng các vùng trước, sau, trong, ngoài bao quanh, và **khoảng cách Euclid giữa tAB với bề
mặt sụn được dùng để xác định độ dày sụn**. Đó đúng là hệ quy chiếu mà S6 đang dùng: lấy mặt xương làm
gốc rồi đo lên, chứ không lấy mảng sụn làm gốc.

Bài cũng cho **sai số độ lặp lại** — thứ ta đang thiếu mốc để so:

| | Giá trị của họ |
|---|---|
| Sai số độ lặp lại, độ dày sụn trung bình theo vùng | 19 µm (1,5%) tới 84 µm (4,7%) |
| Độ ổn định diện tích vùng con khi đặt lại tư thế | độ lệch chuẩn 0,0–0,3% |
| Cỡ mẫu | 12 người test-retest (6 lành, 6 thoái hóa) |

> **Một khác biệt phải nói rõ khi trích.** Họ đo bằng **khoảng cách Euclid** từ tAB tới mặt sụn; S6 của
> ta **bắn tia dọc pháp tuyến** rồi lấy đoạn liên tục đầu tiên. Hai cách trùng nhau khi mảng sụn gần
> song song với mặt xương, nhưng lệch ở vùng cong. Nên câu đúng để viết là "theo tinh thần của
> Wirth & Eckstein 2008", **không phải** "cài đặt lại phương pháp của Wirth & Eckstein 2008".

---

## 1. Bài định nghĩa thuật ngữ — nền của toàn bộ họ cột S6

**Eckstein F, Ateshian G, Burgkart R, và cs. (2006).** *Proposal for a nomenclature for magnetic
resonance imaging based measures of articular cartilage in osteoarthritis.* Osteoarthritis Cartilage
14(10):974–83. [DOI](https://doi.org/10.1016/j.joca.2006.03.005) · PMID 16730462

Đây là bài **đặt ra chính các ký hiệu ta đang dùng**: tên biến ghép từ một đại lượng đo cộng một
nhãn mô, ví dụ `ThC.tAB`. Nếu muốn tên cột của mình so được với số công bố thì đây là tài liệu phải
trích. Nhóm tác giả quốc tế, thống nhất tại hội nghị Chicago 12/2004.

> **Dùng cho:** Phụ lục B của báo cáo 14/9, phần đặt tên cột. Nên trích ở mục 2.1.

---

## 2. Bài trả lời thẳng câu hỏi "dAB xuất hiện thế nào theo độ KL"

**Frobell RB, Wirth W, Nevitt M, và cs. (2010).** *Presence, location, type and size of denuded areas
of subchondral bone in the knee as a function of radiographic stage of OA — data from the OA
initiative.* Osteoarthritis Cartilage 18(5):668–76.
[DOI](https://doi.org/10.1016/j.joca.2009.12.011) · PMID 20175972 · **toàn văn miễn phí** (PMC3066411)

633 gối OAI, 3T, phân đoạn thủ công FLASH coronal. Con số của họ:

| | Kết quả của họ |
|---|---|
| Tỉ lệ có dAB | 39% toàn bộ (48% nam, 33% nữ) |
| KL0 | **1 trên 47** gối có dAB |
| KL1 | 29% đã có dAB |
| KL4 | **29 trên 32** gối có dAB |
| Bản chất dAB | **61% là gai xương trong sụn**, không phải mất sụn |

**Đây là bài quan trọng nhất cho ta**, vì hai lý do.

Thứ nhất, **61% dAB là gai xương chứ không phải mất sụn** — xác nhận đúng mối lo ở `Event.md` mục
A2 và B3. Họ còn chỉ ra hướng: gai xương trong sụn hay gặp ở **bên ngoài** (mâm chày sau, lồi cầu
trong), còn mất sụn thật hay gặp ở **bên trong** (mâm chày ngoài và sụn đùi). Đó là một phép kiểm
rẻ ta chưa làm: nếu `fcl_lt_*` của ta hành xử khác `fcl_mt_*` theo đúng chiều này thì là dấu hiệu
tốt.

Thứ hai, họ thấy **chỉ 1/47 gối KL0 có dAB**, trong khi S6 của ta cho **100% ca có FCL > 0 ở mọi
mức KL**. Chênh lệch đó chính là *sàn đốm rìa* đã ghi trong báo cáo, và bài này là bằng chứng ngoài
rằng con số thô `fcl_*_mm2` không dùng trực tiếp được — phải qua ngưỡng 5 mm², tức dùng
`ndef` / `maxdef` như ta đang làm.

---

## 3. Bài xác nhận ta chọn đúng ba họ cột

**Buck RJ, Wyman BT, Hellio Le Graverand MP, Wirth W, Eckstein F (2010).** *An efficient subset of
morphological measures for articular cartilage in the healthy and diseased human knee.* Magn Reson
Med 63(3):680–90. [DOI](https://doi.org/10.1002/mrm.22207) · PMID 20187178

152 phụ nữ, 3T, theo dõi 24 tháng. Kết luận: **ba đại lượng** — độ dày trung bình trên tAB, diện tích
tAB, và **phần trăm dAB** — giải thích **trên 90%** biến thiên của toàn bộ các phép đo hình thái sụn
khác, cả cắt ngang lẫn theo dõi dọc.

> **Dùng cho:** biện minh cho việc S6 sinh đúng ba họ `thc_tab_*`, `tab_*`, `fcl_*_pct` thay vì hàng
> chục cột. Trích được ở mục 2.1 của báo cáo.

---

## 4. Bài về kiểu hình dAB và đau — hai loại dAB khác nhau

**Cotofana S, Wyman BT, Benichou O, và cs. (2013).** *Relationship between knee pain and the presence,
location, size and phenotype of femorotibial denuded areas of subchondral bone as visualized by MRI.*
Osteoarthritis Cartilage 21(9):1214–22. [DOI](https://doi.org/10.1016/j.joca.2013.04.001) · PMID 23973133

Cùng 633 gối. Tách dAB thành hai kiểu hình: **kiểu mất sụn** và **kiểu gai xương trong sụn**. Người có
dAB kiểu mất sụn có tỉ lệ đau thường xuyên cao hơn chút (PR 1,13). Quan trọng hơn cho ta: họ định nghĩa
**"vùng con trơ ở mức vừa" là trên 10% diện tích vùng con** — một ngưỡng có thể dùng lại trực tiếp nếu
ta làm phần chia vùng con ở mục 7.2 của báo cáo.

---

## 5. Bài mới nhất, và trùng đúng bài toán kỹ thuật của ta

**Wirth W, Eckstein F (2025).** *A fully-automated technique for cartilage morphometry in knees with
severe radiographic osteoarthritis — Method development and validation.* Osteoarthritis Cartilage Open
7(3):100645. [DOI](https://doi.org/10.1016/j.ocarto.2025.100645) · PMID 40697622 ·
**toàn văn miễn phí** (PMC12281388)

> **Đây là bài đáng đọc trước nhất.** Nó nói đúng vấn đề ta đang có, do chính hai tác giả bạn hỏi,
> và mới ra tháng 7 năm nay.

Bài đặt vấn đề y hệt ta: **vùng trơ dAB làm hỏng phân đoạn tự động bằng CNN ở gối KL4**, vì mạng
không biết đâu là chỗ sụn *đáng lẽ* phải có. Cách họ giải: **hậu xử lý bằng đăng ký đa atlas có chọn
lọc để dựng lại tAB**. Đó chính xác là hướng "footprint từ atlas" ta ghi trong `Event.md` mục A3, và
là phương án thay cho phép đóng trắc địa hiện tại.

Vài kết quả của họ đáng để ta so:

- CNN huấn luyện trên **KL2–4** cho kết quả tốt hơn huấn luyện **chỉ KL4**: Dice 0,80–0,89, sai lệch
  hệ thống của độ dày sụn 1,2–8,4% và của diện tích tAB chỉ −0,4 tới 4,3%.
- Có bước hậu xử lý bằng đăng ký thì **tốt hơn hẳn** không có.
- Độ nhạy với thay đổi một năm: phân đoạn thủ công trên DESS cao nhất (SRM ≥ −0,69), tự động đạt
  ≥ −0,56.

**Việc nên làm với bài này:** đọc phần phương pháp để biết họ dựng atlas thế nào, rồi so với khung
`ArticularAtlas` đã có sẵn trong `bsc/atlas.py`. Và con số sai lệch tAB −0,4 tới 4,3% của họ là
**mốc để chấm điểm** cho phép kiểm độ tin cậy FCL từ mask AI ở mục 7.1 của báo cáo.

---

## 6. Bài cho câu hỏi "KL4 có còn sụn để mất không"

**Eckstein F, Nevitt M, Gimona A, và cs. (2011).** *Rates of change and sensitivity to change in
cartilage morphology in healthy knees and in knees with mild, moderate, and end-stage radiographic
osteoarthritis: results from 831 participants from the Osteoarthritis Initiative.* Arthritis Care Res
63(3):311–9. [DOI](https://doi.org/10.1002/acr.20370) · PMID 20957657 ·
**toàn văn miễn phí** (PMC3106126)

831 gối, trong đó **109 gối KL4**. Kết quả: gối KL4 có **tốc độ mất sụn lớn nhất trong mọi mức KL**,
tới **−3,9% một năm**, SRM tới −0,51. Gối lành gần như không đổi (±0,7%).

> **Dùng cho:** cơ sở để tin rằng gối KL4 **vẫn còn nhiều sụn**, tức phép kiểm an toàn "tab không được
> sụp ở KL4" của ta là kỳ vọng đúng chứ không phải mong muốn. Đã trích trong `Event.md` mục B1.

---

## 7. Bài về dAB nền dự báo tiến triển

**Eckstein F, Wirth W, Hudelmaier MI, và cs. (2009).** *Relationship of compartment-specific structural
knee status at baseline with change in cartilage morphology: a prospective observational study using
data from the osteoarthritis initiative.* Arthritis Res Ther 11(3):R90.
[DOI](https://doi.org/10.1186/ar2732) · PMID 19534783 · **toàn văn miễn phí** (PMC2714146)

Gối **có vùng trơ dưới sụn ở thời điểm nền** mất sụn nhanh hơn đáng kể về sau: SRM tới −0,64 so với
−0,33 của cả nhóm. Đáng chú ý: **tình trạng gai xương ở thời điểm nền KHÔNG liên quan** tới mất sụn
trong.

> **Dùng cho:** lập luận rằng FCL có giá trị tiên lượng, không chỉ mô tả. Hữu ích nếu sau này muốn
> dùng cột FCL cho bài toán dự báo tiến triển thay vì chỉ phân loại KL.

---

## 8. Bài về phân bố mất sụn theo trục chi

**Eckstein F, Wirth W, Hudelmaier M, và cs. (2008).** *Patterns of femorotibial cartilage loss in knees
with neutral, varus, and valgus alignment.* Arthritis Rheum 59(11):1563–70.
[DOI](https://doi.org/10.1002/art.24208) · PMID 18975356

174 gối, theo dõi 26,6 tháng. Tỉ lệ mất sụn trong so với ngoài: 1,4:1 ở gối trục thẳng, **3,7:1 ở gối
vẹo trong**, 1:6,0 ở gối vẹo ngoài. Câu quan trọng cho ta: **khi mất sụn nhẹ thì chủ yếu là mỏng đi,
còn khi mất sụn nhanh thì chủ yếu là vùng trơ nở ra**.

Vùng con bị nặng nhất: mâm chày trong phần trung tâm và ngoài, sụn đùi trong phần trung tâm.

> **Dùng cho:** chọn vùng con khi làm mục 7.2. Đã trích trong `Event.md` mục A1.

---

## 9. Bài ứng dụng — dAB làm tiêu chí đánh giá điều trị

**Jansen MP, Maschek S, van Heerwaarden RJ, và cs. (2021).** *Changes in Cartilage Thickness and
Denuded Bone Area after Knee Joint Distraction and High Tibial Osteotomy.* J Clin Med 10(2):368.
[DOI](https://doi.org/10.3390/jcm10020368) · PMID 33478012 · **toàn văn miễn phí** (PMC7835945)

Dùng **phần trăm dAB (`dABp`)** làm tiêu chí chính bên cạnh độ dày sụn, trong hai thử nghiệm ngẫu nhiên.
Cho thấy dAB đủ chín để làm tiêu chí đánh giá lâm sàng, không chỉ là chỉ số mô tả.

---

## Nên trích bài nào cho mục nào của báo cáo 14/9

| Câu trong báo cáo | Trích bài |
|---|---|
| **2.1** "neo vào bề mặt xương, không neo vào sụn" | **Wirth & Eckstein 2008** (mục 0) — bài phương pháp |
| **2.1** tên `tAB` / `cAB` / `dAB` / `ThC.tAB` | Eckstein 2006 (mục 1) — bài thuật ngữ |
| **2.1** "tính lỗ là 0, đó là chỗ cột cũ mù" | Eckstein 2006 + Buck 2010 (mục 3) |
| **2.1** vì sao chỉ cần ba họ cột | Buck 2010 (mục 3) |
| **2.2** mốc để chấm độ tin cậy | Wirth & Eckstein 2008 (1,5–4,7%) và 2025 (tAB −0,4 tới 4,3%) |
| **2.3** kỳ vọng footprint không sụp ở KL4 | Eckstein 2011 (mục 6) |
| **2.3** nghi vấn nhiễm gai xương | Frobell 2010 (mục 2) — 61% dAB là gai xương |
| **7.2** chia vùng con | Eckstein 2008 (mục 8) + ngưỡng 10% của Cotofana 2013 (mục 4) |
| **7.4** hướng atlas | Wirth & Eckstein 2025 (mục 5) |

**Nếu chỉ được trích một bài cho mục 2:** lấy **Wirth & Eckstein 2008**. Nó là bài phương pháp, đúng
hai tác giả, và mô tả đúng hệ quy chiếu mà S6 dùng. Bài 2006 chỉ đặt tên, không đặt ra cách đo.

## Thứ tự nên đọc

1. **Wirth & Eckstein 2008** (mục 0) — nguồn của ý tưởng mục 2.1.
2. **Wirth & Eckstein 2025** (mục 5) — trùng bài toán kỹ thuật của ta nhất, và miễn phí toàn văn.
3. **Frobell 2010** (mục 2) — con số dAB theo KL để đối chiếu với S6, và cảnh báo 61% là gai xương.
4. **Eckstein 2006** (mục 1) — chốt lại cách đặt tên cột.
5. **Buck 2010** (mục 3) — biện minh cho ba họ cột.

Bốn bài còn lại đọc khi cần lập luận cụ thể.

---

## Ba việc trong dự án mà bộ tài liệu này thay đổi

1. **Có mốc để chấm điểm phép kiểm ở mục 7.1.** Wirth & Eckstein 2025 báo sai lệch hệ thống diện tích
   tAB **−0,4 tới 4,3%** và Dice 0,80–0,89 cho phân đoạn tự động trên gối KL4. Khi ta đo độ tin cậy
   FCL từ mask AI thì so với mốc này.
2. **Nghi vấn gai xương có số liệu ngoài.** Frobell 2010: 61% dAB là gai xương trong sụn, và nó **thiên
   về bên ngoài** còn mất sụn thật thiên về bên trong. Đây là phép kiểm rẻ hơn nhiều so với đọc điểm
   gai xương X-quang: chỉ cần so hành vi của `fcl_lt_*` với `fcl_mt_*`.
3. **Hướng atlas không còn là suy đoán.** Chính hai tác giả này đã làm và công bố, nên `bsc/atlas.py`
   có tài liệu tham chiếu để bám theo thay vì tự nghĩ ra.

---

*Nguồn tra cứu: PubMed. Mọi thông tin tóm tắt ở trên lấy từ phần tóm tắt của chính các bài báo được
liệt kê; xin xem bản gốc qua các liên kết DOI trước khi trích dẫn trong báo cáo.*
