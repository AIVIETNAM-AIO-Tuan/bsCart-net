# STAGE 1 (MVP) — KẾT LUẬN: KẾT QUẢ ÂM TÍNH CÓ KIỂM CHỨNG

**Ngày:** 2026-07-20 · **cập nhật 2026-08-31** (bổ sung `med_tib_cart` — Stage 1 ĐÓNG)
**Lớp:** `femoral_cart` (lớp dẫn, plan §3.1) **và** `med_tib_cart` (lớp stress-test) —
**cả hai đều trượt §3.7, 0/10 ca, CI âm.** Xem §5.
**Quyết định:** ⛔ **ĐÓNG Stage 1. Công bố kết quả âm tính.** Không PROCEED Stage 2 với
kiến trúc MVP hiện tại.
**Cơ sở:** plan §3.7 — *"Nếu mô hình dùng bề mặt xương ground-truth không vượt ResEnc,
dừng hoặc sửa giả thuyết trước khi xây mô hình graph."* Ta đã dùng **xương GT (oracle)**
và đã loại trừ có hệ thống mọi nguyên nhân kỹ thuật (§4).

---

## 1. Phát hiện trung tâm

> **Hệ tọa độ bề mặt xương mang thông tin thật và tốt hơn hẳn các hệ quy chiếu đối chứng —
> nhưng mô hình 1D đọc profile dọc tia KHÔNG khai thác được thông tin đó từ MRI để cạnh
> tranh với ResEnc ở vùng sụn mỏng.**

Đây là **nghịch lý biểu diễn–học được**, và cả hai vế đều có bằng chứng định lượng:

| Vế | Bằng chứng | Số |
|---|---|---|
| Biểu diễn **TỐT** | Oracle reconstruction (occupancy GT hoàn hảo qua splat) | **0.0256mm** — chính xác hơn ResEnc **~21×** |
| Hệ tọa độ **CÓ GIÁ TRỊ** | M6: pháp tuyến vs axial/tangent/random, paired bootstrap | thắng cả 3, **CI dương** cả 3 |
| Nhưng model **KHÔNG học được** | §3.7 vùng mỏng, checkpoint-locked (femoral) | **2.188mm** vs B0 **0.552mm**, **0/10 ca** |
| Lặp lại ở lớp thứ hai | §3.7 vùng mỏng, checkpoint-locked (med_tib) | **0.990mm** vs B0 **0.398mm**, **0/10 ca**, CI [−0.729, −0.458] |

Khoảng cách giữa trần lý thuyết (0.026mm) và hiệu năng thực (1.75–2.19mm) là **~70–85 lần**.
Nó không do hình học, không do phép biến đổi tọa độ, không do bug, không do loss, không do
thiếu dữ liệu — tất cả đã được kiểm và loại trừ (§4).

---

## 2. Thiết lập thí nghiệm

| | |
|---|---|
| Dữ liệu | OAI-ZIB, `Dataset001_KneeOA`, spacing `(0.3646, 0.3646, 0.70)`mm |
| Split | `engineering_split_v1_fixed`: 40 train (fold 1–4) / 10 val (fold 0), từ `splits_zib_v1_fixed.json` (đã vá rò rỉ V00/V01) |
| Test set | **103 ca KHÔNG đụng tới** — chỉ dùng một lần ở M0, giữ kín cho Gate 4 |
| Baseline (B0) | ResEnc-L 150ep, prediction **out-of-fold** (model không train trên ca đó) |
| Bề mặt xương | **S0 = ground-truth** (điều kiện oracle — thuận lợi nhất cho giả thuyết) |
| Miền khớp | D1 = atlas quần thể xây **chỉ từ ca train của fold**, `assert_no_leak` |
| Metric quyết định | **lỗi biên vùng mỏng** (bin absent + ≤0.5 + ≤1.0mm), **KHÔNG** dùng ASSD tổng |

**Vì sao cấm ASSD tổng** (M0 §6): mục tiêu §3.7 quy về ASSD tổng chỉ **0.0058mm** (femoral),
trong khi sai khác giữa hai implementation metric (scipy vs SimpleITK) là **0.0033mm**. Hiệu
ứng chìm dưới độ rung định nghĩa ⇒ không đo được. Mọi báo cáo ở mẫu số vùng mỏng.

---

## 3. Nền tảng đã kiểm chứng (mọi cổng ĐẠT)

### 3.1. QC hình học trên xương thật (plan §6 bước 3/5/6)

| lớp | M3 normals | M4 single-interval | M2 Dice | M2 ASSD | khung suy biến |
|---|---|---|---|---|---|
| femoral_cart | **99.8%** | **99.8%** | 0.995 | **0.009mm** | 0% |
| med_tib_cart | **99.6%** | **99.7%** | 0.993 | **0.008mm** | 0% |

Cổng: M3 >99.5% · M4 >95% · M2 ASSD <0.1mm. **Đạt biên độ lớn** (M2 ASSD thấp hơn cổng ~11×).
⇒ Trích bề mặt, trường pháp tuyến, phóng tia, dựng lại — **hoạt động đúng**.

### 3.2. Atlas D1 — không rò rỉ, tổng quát hóa được

- Xây **chỉ từ 40 ca train**; `assert_no_leak` xác nhận không ca val nào lọt vào (§2.2).
- Trên ca val **chưa từng thấy**: recall **98.2–99.8%**, precision 87–92%, chọn ~41% node
  (oracle 37%) ⇒ có lọc thật, thiên recall đúng thiết kế (atlas là bộ sinh ứng viên).
- Phân bố lưỡng cực rõ: 32% ô ≥0.8 · 61.6% ô ≤0.2 (không "chọn bừa").
- Kiểm chéo femoral↔med_tib: atlas **đúng lớp** recall ~99%, atlas **sai lớp** recall **0%**
  ⇒ mỗi atlas mã hóa đúng giải phẫu riêng.

*Ghi chú phương pháp: đăng ký PCA bị bác bỏ (xương gần đối xứng ⇒ trục/dấu không xác định,
axis_sep chỉ ~1.05); thay bằng chuẩn hóa trục-gốc (tịnh tiến + tỉ lệ), = mục 1 của §4.4 Step 2.*

### 3.3. Trần biểu diễn (phép thử quyết định)

Cho occupancy GT **hoàn hảo** đi qua đúng đường splat, đo lỗi vùng mỏng:

| | lỗi biên vùng mỏng |
|---|---|
| **Oracle (occupancy GT qua splat)** | **0.0256mm** |
| ResEnc B0 | 0.5517mm |
| ray model đã train | 1.75–2.19mm |

⇒ **Phép biến đổi tọa độ + splat KHÔNG phá sụn mỏng.** Trần chính xác hơn ResEnc ~21×.
Toàn bộ lỗi của ray model là **lỗi dự đoán**, không phải giới hạn biểu diễn.

*(Nhất quán với M0: trên toàn bề mặt headroom nhỏ (prize 0.058mm), nhưng riêng vùng mỏng
khoảng cách ResEnc→trần là 0.52mm — đúng thứ rescope M0 nhắm tới.)*

### 3.4. M6 — negative control: hệ tọa độ CÓ giá trị (plan §3.5, §3.7 điều 5)

**Bài học phương pháp — metric sai làm đảo ngược kết luận.** Đo bằng occ-Dice không gian
tia cho kết quả SAI:

| hướng | tỉ lệ dương | occ-Dice (SAI) | lỗi biên vùng mỏng (ĐÚNG) |
|---|---|---|---|
| **normal** | 21.2% | 0.825 *(hạng 3)* | **1.99mm** ← tốt nhất |
| random | 24.5% | 0.797 *(hạng 4)* | 2.32mm |
| tangent | 35.7% | 0.837 *(hạng 2)* | 2.97mm |
| axial | 33.8% | **0.856** *(hạng 1)* | **3.68mm** ← tệ nhất |

Tia pháp tuyến cắt **vuông góc** lớp sụn ⇒ đoạn giao ngắn nhất ⇒ tỉ lệ dương thấp nhất ⇒
occ-Dice khó nhất **về mặt số học**. Đó chính là tính chất giúp định vị biên chính xác.
Plan §3.5 M6 yêu cầu *"boundary and thickness metrics"* — đo đúng thì **thứ hạng lật ngược
hoàn toàn**.

Thống kê ghép cặp (§2.4, n=10, bootstrap 10 000):

| đối chứng | lỗi TB | normal tốt hơn | boot 95% CI | số ca |
|---|---|---|---|---|
| axial | 3.676mm | **+1.689mm** | [+1.385, +1.927] | 10/10 |
| tangent | 2.968mm | **+0.982mm** | [+0.677, +1.291] | 10/10 |
| random | 2.316mm | **+0.330mm** | [+0.215, +0.463] | 9/10 |

**Cả ba CI dương ⇒ §3.7 điều 5 ĐẠT.** Lợi ích đến từ **hệ tọa độ**, không phải "thêm một
mạng nữa". Đây là bằng chứng dương tính mạnh nhất của Stage 1.

---

## 4. Loại trừ có hệ thống — vì sao kết luận âm tính đáng tin

Mỗi giả thuyết "sửa được" đều có một phép thử riêng, và **tất cả đều bị bác bỏ bằng dữ liệu**:

| # | Giả thuyết | Phép thử | Kết quả | Kết luận |
|---|---|---|---|---|
| 1 | Hình học sai | QC M2/M3/M4 xương thật | 99.8% / 99.8% / 0.009mm | ❌ không phải |
| 2 | Atlas D1 hỏng/rò rỉ | recall trên ca held-out + cross-check | 98–99% / 0% chéo | ❌ không phải |
| 3 | Biểu diễn/splat là trần | oracle reconstruction | **0.026mm** | ❌ không phải |
| 4 | Bug trong cấu hình | B1 micro-overfit S0-D1-H1 | Dice 0.879, **interior err 2.69%**, presence-acc **0.988** | ❌ không phải |
| 5 | Loss/sampling bias | B2 `stratify`, `absent_fp_weight` | stratify **+0.052 (hại)**, absentFP −0.239 (≈nhiễu) | ❌ **bác bỏ** |
| 6 | Hard presence gate | B2 `presence_gate="soft"` | −0.528 (có tác dụng nhưng không đủ) | ⚠️ một phần |
| 7 | **Thiếu dữ liệu** | **B3: 140 ca (3.5×)** | **1.751mm** — không cải thiện | ❌ **bác bỏ** |
| 8 | **Thiếu ngữ cảnh bề mặt** | **bsc_03: 4 phép đo** | tín hiệu CÓ (lift 2.67×) nhưng FP thành mảng (38.1%), làm mượt chỉ +0.067mm | ❌ **bác bỏ** |
| 9 | **Thiếu ngữ cảnh THÔ / slab 3D CNN** | **bsc_03 §5: k-NN phi tham số** | k-NN nhớ 80k tia **không vượt nổi majority** (−0.001); láng giềng thô thêm **0** | ❌ **bác bỏ — giới hạn THÔNG TIN** |

### 4.1. B1 — loại trừ bug (micro-overfit đúng cấu hình đang hỏng)

Train S0-D1-H1 trên **1 ca**, 1200 tia phân tầng, 150 epoch:

| | |
|---|---|
| loss | 0.460 → 0.106 |
| occ-Dice (train) | 0.879 @thr 0.70 |
| **lỗi trong lòng sụn** | **2.69%** |
| lỗi ở ô biên | 33.03% (39% tổng khối lượng lỗi) |
| **absent presence-acc** | **0.988** |
| thin-ray recall | 0.827 |

occ-Dice 0.879 **không phải underfit** — nó chạm **trần lượng tử hóa biên**: M5 cũ đỉnh 0.897,
M2 phantom đo sụn 0.4mm chỉ đạt Dice 0.848 *dù occupancy hoàn hảo*. Lỗi dồn ở ô biên (39%),
trong lòng chỉ 2.69% ⇒ model **memorize được**. Presence-acc 0.988 trên train ⇒ cơ chế không hỏng.

*Cải chính: cổng 0.97 mình đặt ban đầu nằm TRÊN trần vật lý của occ-Dice ray-space ở cấu trúc
mỏng — đó là lỗi đặt cổng, đã ghi nhận.*

### 4.2. B2 — bác bỏ giả thuyết loss/sampling bias

Sweep 4 cấu hình, cùng split/seed, mỗi lần đổi một lever (n=10 val):

| cấu hình | thin err | vs P2 (2.188) | vs B0 (0.552) | tốt/n | presence F1 |
|---|---|---|---|---|---|
| baseline (P2) | 1.913mm | −0.275 | +1.361 | **0/10** | 0.880 |
| **stratify** | **2.241mm** | **+0.052** | +1.689 | **0/10** | 0.900 |
| absentFP=3 | 1.949mm | −0.239 | +1.398 | **0/10** | 0.888 |
| soft-gate | **1.660mm** | −0.528 | +1.108 | **0/10** | 0.886 |

**Toàn bộ nằm trong dải 1.66–2.24mm**, biên độ 0.58mm — trong khi **run-to-run variation của
chính P2 đã đo được là ~0.25mm** (hai instance cho 1.94 và 2.19mm). Phần lớn chênh lệch
không phân biệt được với nhiễu.

Điểm quan trọng: **`stratify` làm TỆ hơn** và **`absent_fp_weight` gần như vô hiệu** — hai
lever nhắm thẳng vào giả thuyết "sụn dày chi phối gradient" (review §8). Dữ liệu **không
ủng hộ** giả thuyết đó.

### 4.3. B3 — bác bỏ giả thuyết thiếu dữ liệu (phép thử quyết định cuối)

Giữ lever duy nhất có tác dụng (`soft-gate`), tăng dữ liệu **3.5×**:

| | ca train | thin err | vs B0 | tốt/n |
|---|---|---|---|---|
| P2 khóa | 40 | 2.188mm | +1.636 | 0/10 |
| soft-gate | 40 | 1.660mm | +1.108 | 0/10 |
| **B3** | **140** | **1.751mm** | **+1.199** | **0/10** |

**Đường cong phẳng hoàn toàn: 40 → 140 ca không kéo được lỗi xuống.** 1.751 thậm chí nhỉnh
hơn cấu hình 40 ca (1.660), chênh lệch trong nhiễu. Vẫn cách B0 **~3.2×**, vẫn **0/10**.

**Cổng đã ghi TRƯỚC khi chạy** (để không dời cột gôn sau): *≤1.0mm ⇒ scale tiếp; ~1.6–1.9mm
⇒ dừng Phase B, viết kết quả âm tính.* Kết quả 1.751mm rơi đúng nhánh dừng.

### 4.4. Cơ chế thất bại — ảo giác ở vùng absent

Phân rã khối lượng lỗi theo bin độ dày (P2, exploratory n=5):

| bin độ dày | ray err | B0 err | ray n | B0 n |
|---|---|---|---|---|
| **absent** | **3.380mm** | 1.057mm | **23 897** | 13 155 |
| ≤0.5mm | 0.708 | 0.431 | 2 430 | 2 787 |
| ≤1.0mm | 0.702 | 0.308 | 22 954 | 27 266 |

Bin `absent` (voxel bề mặt PRED ở vùng GT không có sụn = **dương tính giả**) chiếm **~82%
khối lượng lỗi vùng mỏng**. Ray tạo **1.8×** số voxel sụn ảo so với B0, và chúng nằm **xa
hơn 3.2×**.

Recall theo độ dày (n=5):

| độ dày GT | voxel sụn GT | ray recall |
|---|---|---|
| >2.0mm (thân dày) | 622 916 | **80.9%** |
| ≤1.0mm (rìa mỏng) | 16 291 | **37.0%** |
| ≤0.5mm (cực mỏng) | 1 350 | **30.7%** |

**Mô hình bắt tốt phần thân dày (81%) nhưng hỏng ở rìa mỏng (37%)** — **ngược hẳn** điều giả
thuyết hứa hẹn (giúp nhiều nhất ở vùng mỏng/mất).

### 4.4b. Ngữ cảnh bề mặt (intra-surface context) — bác bỏ, và lộ ra cơ chế mới

Giả thuyết: tia 1D vứt bỏ ngữ cảnh **tiếp tuyến**; cho mỗi node thấy láng giềng trên bề mặt
(plan §4.4 Step 5) sẽ biết *"cả vùng này không có sụn"*. Đo **trước khi xây** (notebook
`bsc_03`, n=10 val, k=8 láng giềng, dùng lại checkpoint — không train gì mới):

| Phép đo | Kết quả | Cổng | |
|---|---|---|---|
| **[1] Coherence** — vắng sụn có liên tục không? | node absent có **97.4%** láng giềng cùng absent (nền 36.7%) ⇒ lift **2.67×** | >1.5 | ✅ |
| **[2] Neighbor-oracle** — biết presence GT của láng giềng thì đoán được gì? | F1 **0.991** · absent-recall **0.983** | — | ✅ trần rất cao |
| **[3] FP isolation** — FP rời rạc hay thành mảng? | **38.1%** láng giềng quanh FP được đoán đúng | >0.60 | ❌ **thành MẢNG** |
| **[4] Smoothing sweep** — làm mượt presence rồi đo lại | 1.752 → **1.684mm** (α=0.7), gain **0.067mm** | >0.10mm | ❌ |

Sweep đầy đủ:

| α | thin err | vs B0 | presence F1 | absent-recall |
|---|---|---|---|---|
| 0.00 | 1.752mm | +1.200 | 0.895 | 0.902 |
| 0.30 | 1.713 | +1.162 | 0.902 | 0.908 |
| 0.50 | 1.696 | +1.144 | 0.905 | 0.912 |
| **0.70** | **1.684** | **+1.133** | 0.908 | 0.915 |
| 1.00 | 1.686 | +1.134 | 0.910 | 0.917 |

**Nghịch lý thứ hai, song song với §1:** ngữ cảnh bề mặt **chứa gần đủ thông tin** (oracle
F1 0.991) nhưng **không dùng được**, vì lỗi của model **có cấu trúc không gian** — láng giềng
của một FP cũng đang sai như nhau, nên tổng hợp láng giềng chỉ **củng cố** cái sai. Đúng cảnh
báo MM3 của plan (*"graph propagation có thể lấp luôn focal defect"*).

**Định lượng khoảng cách:** làm mượt lấp được 0.067mm trên khoảng cách 1.133mm tới B0 =
**5.9%**. Một graph transformer đầy đủ phải tốt gấp **~17×** làm mượt ngây thơ mới chạm B0 —
không phải kịch bản hợp lý.

**Chi tiết đáng chú ý:** presence F1 *có* cải thiện (0.895 → 0.910) và absent-recall tăng
(0.902 → 0.917), nhưng lỗi biên gần như không đổi ⇒ làm mượt dọn được vài FP dễ, còn **những
FP thực sự gây lỗi biên thì nằm trong mảng**, không đụng tới được.

**Chẩn đoán mới rút ra:** model **không** sai ngẫu nhiên (thứ có thể trung bình hóa đi) mà
sai **nhất quán theo vùng** ⇒ dấu hiệu vấn đề nằm ở **tín hiệu ảnh trong vùng đó**, không
phải ở thiếu ngữ cảnh.

*Xác nhận độ tin cậy: đo trên checkpoint `b889b49a297a787c` (train lại độc lập sau khi mất
dữ liệu Drive), α=0 cho **1.752mm** — B3 trước đó cho **1.751mm**. Trùng tới 0.001mm trên hai
lần train độc lập ⇒ ~1.75mm là hiệu năng thật của cấu hình, không phải một checkpoint xui.*

### 4.4c. Giới hạn THÔNG TIN — đặc trưng tia không tách được absent vs cực mỏng

§4.4b bác bỏ việc tổng hợp **dự đoán** của láng giềng. Nhưng nó không trả lời: *nếu cho model
nhìn **đặc trưng MRI thô** của tia lân cận (tức 3D CNN trên slab neo pháp tuyến, plan §5.2
Extension B) thì sao?* Đo bằng **k-NN phi tham số** — baseline mạnh nhất có thể, nó *thuộc
lòng* toàn bộ tập train, không train gì.

Chỉ xét **tia khó** (độ dày ≤1mm — nơi §3.7 thất bại): 80k tia train, 40k tia val.
Trong đó chỉ **10.1%** thực sự có sụn ⇒ "luôn đoán absent" đạt **acc 0.899**.

**Đo trên ĐỦ 4 bộ kênh** (dựng hình học một lần với `(mri, grad, sdf)` rồi cắt kênh — lần đo
đầu chỉ dùng `(mri, sdf)`, **bỏ sót `grad`** là kênh M8 đo được đóng góp nhiều nhất):

| bộ kênh | acc | lift vs majority | **recall** | **precision** | +láng giềng | gain |
|---|---|---|---|---|---|---|
| mri | 0.898 | −0.001 | 0.046 | 0.455 | 0.897 | −0.001 |
| mri+grad | 0.902 | +0.002 | 0.098 | 0.572 | 0.900 | −0.002 |
| mri+sdf | 0.898 | −0.001 | 0.110 | 0.481 | 0.897 | −0.001 |
| **mri+grad+sdf** | 0.902 | **+0.003** | **0.141** | 0.555 | 0.902 | −0.001 |

**Không bộ kênh nào vượt nổi việc đoán bừa "absent"** (lift tối đa +0.003, cổng >+0.05).
Ghép đặc trưng thô của 4 láng giềng thêm **đúng 0** (−0.001 tới −0.002, **nhất quán ở cả 4 bộ**).

**Sắc thái quan trọng — tín hiệu KHÔNG bằng 0.** Cột recall tăng **đơn điệu** theo lượng thông
tin đầu vào: 0.046 (mri) → 0.098 (+grad) → 0.141 (cả ba), với precision giữ ~0.55 tức **gấp
5.5 lần** tỉ lệ nền 10.1%. Nhưng bộ tốt nhất chỉ bắt được **14%** số tia sụn mỏng — **86% vô
hình** trong đặc trưng tia. Accuracy không nhúc nhích vì ở tỉ lệ 10/90, vài TP bắt thêm bị bù
trừ bởi FP.

*Gradient thật sự có ích (recall gấp đôi, 0.046 → 0.098) — xác nhận lo ngại về kênh bị bỏ sót
là đúng. Nhưng nó KHÔNG lật kết luận, chỉ làm kết luận vững hơn vì giờ đã bao gồm kênh mạnh nhất.*

*Xu hướng đơn điệu này là **tia hy vọng duy nhất còn lại**: thêm thông tin THẬT (echo thứ hai
của DESS — xem §9) có thể đẩy tiếp. Nhưng hai kênh hình học chỉ mua được +9 điểm phần trăm
recall, còn rất xa mức dùng được (~70–80%).*

⇒ **Đặc trưng dọc tia không phân biệt được "không có sụn" với "sụn 0.3mm".** Đây là **giới hạn
thông tin**, không phải giới hạn kiến trúc — và nó **bác bỏ trực tiếp** hướng slab/3D CNN: nếu
đặc trưng thô của láng giềng không thêm gì, một CNN tổng hợp chính những đặc trưng đó không có
gì để khai thác.

**Đối chiếu quyết định với §4.4b:** biết **nhãn GT** của láng giềng ⇒ đoán gần hoàn hảo
(neighbor-oracle F1 **0.991**). Biết **đặc trưng ảnh** của láng giềng ⇒ vô dụng (gain −0.001).
Khoảng cách giữa hai con số đó chính là thứ **không tồn tại trong ảnh**.

*Có tín hiệu yếu, không phải bằng 0: F1 0.179 ứng với precision ~49% (so với nền 10.1%) nhưng
recall chỉ ~11% — bắt được một phần mười số tia sụn mỏng. Quá ít để hữu ích.*

**Caveat:** k-NN dùng khoảng cách Euclid trên đặc trưng thô 128–640 chiều; ở số chiều cao đây
là thước đo tương tự yếu (lời nguyền chiều), nên về nguyên tắc một **biểu diễn học được** có
thể tách tốt hơn. Nhưng mô hình đã train *chính là* một biểu diễn học được, và nó cũng thất
bại đúng ở vùng này. Hai cách tiếp cận rất khác nhau — phi tham số nhớ hết, và CNN học đặc
trưng — cùng đụng một bức tường. Đó là bằng chứng **hội tụ**, dù không phải chứng minh tuyệt đối.

### 4.5. Vì sao chỉnh ngưỡng presence không cứu được

| presence_thr | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|
| thin err | 2.00 | 1.84 | 2.26 | 3.99 | **8.70** |

Nâng ngưỡng đáng lẽ giảm FP, lại làm lỗi **tăng vọt**. Vì presence là **một giá trị mỗi tia**
hard-gate cả tia: ở vùng mỏng tín hiệu "có sụn" yếu, gần như không tách được với absent. Nâng
ngưỡng vừa chặn FP absent vừa chặn luôn **sụn mỏng thật** (biến FP thành FN).

⇒ Đây chính là **§9.3 của plan** ("giảm nhầm lẫn absent vs cực mỏng") — và MVP thất bại đúng
tại đó.

---

## 5. Kết quả Go/No-Go theo plan §3.7

| # | Điều kiện §3.7 | Kết quả | |
|---|---|---|---|
| 1 | Cải thiện outer-boundary vùng mỏng ≥10% tương đối | **−226%** (1.751 vs 0.552) | ❌ |
| 2 | Surface Dice @0.5mm cải thiện rõ | không đo được (điều 1 đã trượt nặng) | ❌ |
| 3 | Dice tổng không giảm đáng kể | Dice ray ~0.81 vs B0 ~0.88 | ⚠️ |
| 4 | Presence head phát hiện absent tốt hơn ngưỡng hóa ResEnc | **FP absent 1.8× B0** | ❌ |
| 5 | Tia pháp tuyến vượt các hướng đối chứng | **3/3 CI dương** | ✅ |
| 6 | Model bề mặt dự đoán giữ phần lớn lợi ích | không áp dụng (P2 chưa vượt B0 ⇒ "retained gain" vô nghĩa) | — |
| 7 | Lặp lại được qua fold/seed | chưa chạy đa fold/seed | — |

**1 đạt / 4 trượt / 2 không áp dụng.** Điều kiện 1 — điều kiện cốt lõi — trượt với biên độ lớn.

### 5.1. `med_tib_cart` — lớp stress-test (plan §3.1, §6 bước 11)

Chạy **2026-08-31** để đóng hồ sơ Stage 1. Cấu hình **y hệt canonical P2 femoral**
(40 train / 10 val, 20k tia, 30 epoch, lr 3e-4, seed 1) ⇒ chỉ đổi **đúng một yếu tố**
là lớp (QĐ4, §3.2).

**Vì sao vẫn phải chạy dù femoral đã âm tính:** M0 cho med_tib prize **0.0885mm** — cao hơn
femoral (0.0582mm) **52%** — và khối lượng vùng mỏng **33.5%** vs 21.4%. Giả thuyết dự đoán
lớp nhiều headroom hơn thì có cơ hội hơn. Kết quả **không** hiển nhiên trước khi đo.

| lớp | B0 (mm) | ray (mm) | tỉ lệ | tốt hơn | CI95 (B0 − ray) | presence F1 | absent-recall |
|---|---|---|---|---|---|---|---|
| `femoral_cart` | 0.5517 | 2.1881 | **4.0×** | 0/10 | — | 0.899 | 0.879 |
| `med_tib_cart` | 0.3981 | 0.9899 | **2.5×** | 0/10 | **[−0.729, −0.458]** | 0.865 | 0.935 |

⇒ **Trượt §3.7 trên CẢ HAI lớp.** CI hoàn toàn âm, không ca nào tốt hơn B0.
Kết luận âm tính **không** phải hiện tượng riêng của lớp dẫn.

**Ba quan sát đáng ghi, không đảo kết luận:**

1. **Khoảng cách thu hẹp 1.6×** (4.0× → 2.5×). Giả thuyết "lớp nhiều headroom hơn thì gần
   hơn" **đúng về hướng** nhưng sai về biên độ: cần 2.5× nữa mới chạm B0, trong khi prize
   chỉ hơn 52%. Headroom **không** phải ràng buộc quyết định — khớp với chẩn đoán
   giới hạn thông tin (§4.4c).

2. **Tỉ lệ khoảng cách bám theo absent-recall, không bám theo occ-Dice.** med_tib có
   occ-Dice *thấp hơn* (0.756 vs 0.816) và presence F1 *thấp hơn* (0.865 vs 0.899) nhưng
   absent-recall *cao hơn* (0.935 vs 0.879) — và đó là lớp gần B0 hơn. Đây là **xác nhận
   độc lập cho cơ chế §4.4**: lỗi trội là **ảo giác ở bin absent** (82% khối lượng lỗi vùng
   mỏng ở femoral). Bớt ảo giác ⇒ bớt lỗi theo tỉ lệ.
   *Cảnh báo: hai điểm dữ liệu không đủ để ngoại suy tuyến tính. Đây là bằng chứng nhất quán
   với cơ chế, không phải phép đo quan hệ liều–đáp ứng.*

3. **nnUNet mạnh hơn ở med_tib vùng mỏng theo số tuyệt đối** (B0 0.398 vs 0.552) dù Dice
   tổng thấp hơn (0.852 vs 0.891). Dice tổng và lỗi biên vùng mỏng xếp hạng **ngược nhau**
   giữa hai lớp — thêm một dẫn chứng cho §2 (*"cấm dùng ASSD/Dice tổng làm metric quyết định"*).

**Checkpoint khóa:** `MVP_MTC_S0_D1_I2_H1_v1_Fold0_Seed1` / `2fe1a3e6a69bb9e2`,
git `3b99692c`, atlas `a08332dffe909ad9`, split `47179e4dd81345c7` (**trùng split femoral** —
cùng ca train/val, chỉ khác lớp), config `4eeff38f19430691`. Atlas med_tib phủ 39.1% lưới,
P(khớp) TB 0.124, `assert_no_leak` ĐẠT, sanity-check khác atlas femoral ĐẠT.

---

## 6. Reproducibility — mọi số đều truy vết được

Sau khi phát hiện bug bookkeeping (trộn instance model giữa các phiên, eval dùng kết quả cũ,
không khóa commit), toàn bộ đã được sửa và **canonical rerun**:

| | |
|---|---|
| experiment_id | `MVP_FC_S0_D1_I2_H1_v1_Fold0_Seed1` |
| checkpoint_sha256 | `4e133ba5fa0a74c6` |
| git commit | `dc6f5849` |
| atlas_hash | `e4efc01b335a907a` |
| split_hash | `47179e4dd81345c7` |
| config_hash | `b208b4eaf85b39fc` |
| §3.7 (từ model **reload từ đĩa**) | **2.1881mm** · 0/10 · presence F1 0.899 |

Hạ tầng: `save_run`/`load_run_model` (checkpoint + hash), đường dẫn kết quả theo
`runs/<experiment_id>/<checkpoint_hash>/`, test round-trip chứng minh **reload cho dự đoán
y hệt**. **109 test** chạy trên phantom, không cần dữ liệu thật.

### 6.1. Mapping sensitivity audit

Phép gán `voxel → node bề mặt gần nhất` gán nhầm 2 398 voxel sụn GT vào bin `absent`. Audit
so Mapping A (nearest node) vs Mapping B (nearest sampled-ray proxy), n=10:

| | Mapping A | Mapping B |
|---|---|---|
| B0 thin err | 0.5517mm | 0.6033mm (**+9.3%**) |
| GT voxel gán nhầm `absent` | 2 398 | **848** (−65%) |
| QC Mapping B | — | unassigned 0.3% · dist median 0.17mm · p95 0.34mm |

9.3% ở vùng xám (5–10%). **Kết luận không lung lay:** kể cả dưới Mapping B, B0 (0.603) vẫn
hơn P2 (2.188) **~3.6×**. Mapping B đúng hơn thật ⇒ dùng cho báo cáo cuối.

---

## 7. Giới hạn — điều nghiên cứu này KHÔNG khẳng định

Trung thực về phạm vi:

1. ~~Chỉ femoral_cart~~ — **đã đóng 2026-08-31**: `med_tib_cart` cũng trượt (§5.1),
   0/10 ca, CI [−0.729, −0.458]. Hai lớp, **không phải hiện tượng riêng của lớp dẫn**.
   *Còn lại:* `lat_tib_cart` và `patellar_cart` chưa kiểm — nhưng chúng không nằm trong
   plan §3.1 (chỉ định 2 lớp) nên không phải lỗ hổng của Stage 1.
2. **n=10 val, 1 fold (fold 0), 1 seed** cho mỗi lớp. Chưa 5-fold, chưa nhiều seed
   (§3.7 điều 7). Hai lớp dùng **cùng split** (`47179e4dd81345c7`) nên là hai phép đo
   trên cùng bệnh nhân — độc lập về lớp, **không** độc lập về mẫu.
3. **Chưa thử toàn bộ ~320 ca train.** Đường cong 40→140 phẳng nên xác suất lật thấp, nhưng
   không phải bằng chứng tuyệt đối.
4. **P3/M8-D (coarse ResEnc probability) chưa chạy** — cần OOF softmax. Đây là kênh có thể
   giúp model "biết" chỗ nào ResEnc đã tin là sụn.
5. **Chưa thử presence theo ĐỘ SÂU** (thay vì một giá trị mỗi tia), boundary-endpoint loss,
   structured interval output, hay cross-surface context (Stage 2).
6. **Oracle 0.026mm dùng occupancy GT + presence GT** — là trần lý thuyết, không khẳng định
   model *có thể* đạt tới.
7. **Convenience split `[:40]/[:10]`**, không phân tầng theo KL/độ dày/domain. Hợp lệ cho
   engineering, không phải final unbiased estimate.
8. Bảng phân rã bin (§4.4) dùng **n=5** (exploratory). Các số Gate (§3.7, oracle, M6 paired)
   dùng đủ 10.

---

## 8. Giá trị khoa học của kết quả âm tính

Kết quả này **không phải thất bại kỹ thuật** mà là một phát hiện có cấu trúc:

1. **Biểu diễn tốt ≠ hiệu năng tốt.** Hệ tọa độ giữ được sụn mỏng tới 0.026mm (chính xác hơn
   ResEnc 21×) và thắng mọi hướng đối chứng — nhưng model 1D không khai thác được. Khoảng
   cách giữa "thông tin có trong biểu diễn" và "học được từ ảnh" là ~70–85×.

2. **Sụn mỏng khó vì lý do khác localization — và giờ đã ĐO ĐƯỢC.** Ghép với kết quả âm tính
   **ROI cascade** (crop bbox GT hoàn hảo chỉ đổi Dice ±0.003), ta có bộ đôi: *localization
   không phải nút thắt* + *hệ quy chiếu tốt không tự động thành hiệu năng*. §4.4c định lượng
   trực tiếp: k-NN nhớ 80k tia **không vượt nổi việc đoán bừa "absent"** ⇒ nút thắt nằm ở
   **tín hiệu ảnh tại vùng sụn cực mỏng**, không ở cách tổ chức không gian. Đây là chuyển từ
   *suy đoán* sang *đo đạc*.

3. **Nhầm lẫn absent ↔ cực mỏng là failure mode chi phối** (82% khối lượng lỗi). Presence
   ở mức một-giá-trị-mỗi-tia không tách được hai lớp này — và sweep ngưỡng chứng minh không
   có điểm cắt nào cứu được.

4. **Lỗi của model có cấu trúc không gian, không phải nhiễu ngẫu nhiên** (§4.4b). Ngữ cảnh
   bề mặt về nguyên tắc gần như đủ để xác định presence (oracle F1 0.991, coherence 2.67×),
   nhưng model sai nhất quán theo từng vùng nên tổng hợp láng giềng chỉ củng cố cái sai. Đây
   là **nghịch lý thứ hai song song với §1**: thông tin có sẵn trong cấu trúc, nhưng model
   không ở vị trí khai thác được nó.

4. **Bài học phương pháp lặp lại nhiều lần trong Stage 1:**
   - Chọn sai metric **đảo ngược kết luận** (M6: occ-Dice ray-space xếp axial nhất, boundary
     metric xếp axial bét).
   - Cổng phải đặt **dưới trần vật lý** đã đo (cổng 0.97 cho occ-Dice là bất khả thi).
   - Phải **chạy phép thử quyết định** trước khi kết luận từ hình ảnh (dự đoán oracle sẽ cao
     ⇒ STOP; thực tế 0.026mm ⇒ REVISE).
   - Bookkeeping lỏng lẻo làm **mọi số chi tiết thành provisional** — checkpoint hash là bắt buộc.

---

## 9. ĐÓNG Stage 1

**Câu hỏi Stage 1** (plan §3): *"Khi bề mặt xương và pháp tuyến được biết chính xác, mô hình
đọc MRI dọc pháp tuyến có xác định sụn chính xác hơn ResEnc **ở vùng sụn mỏng** không?"*

**Trả lời: KHÔNG.** 2.19mm vs 0.55mm, 0/10 ca, ở **điều kiện oracle** (xương GT — thuận lợi
nhất có thể cho giả thuyết). Và cơ chế đã đo được: giới hạn nằm ở **thông tin trong ảnh**
(§4.4c), không ở kiến trúc.

Plan §3.7 đã quy định sẵn: *"Nếu mô hình dùng bề mặt xương ground-truth không vượt ResEnc,
**dừng** hoặc sửa giả thuyết trước khi xây mô hình graph."* ⇒ **Đóng Stage 1 là làm theo plan.**

Không sweep hyperparameter thêm — B2/B3 đã bão hòa, mọi lever rẻ đã cạn.

### 9.1. Stage 1 — ĐÃ ĐÓNG (2026-08-31)

| # | Việc | Trạng thái |
|---|---|---|
| 1 | Chạy `med_tib_cart` — lớp stress-test (plan §3.1, §6 bước 11) | ✅ **XONG 2026-08-31.** Trượt §3.7, 0/10, CI âm (§5.1). Đóng lỗ hổng *"sao chỉ thử lớp dễ nhất?"* |
| 2 | Viết bài báo kết quả âm tính | ⬜ còn lại — bộ bằng chứng đã đủ và mạch lạc |
| 3 | Tải `runs/` + `atlas/` về máy local | ⬜ Drive **đã mất dữ liệu một lần**; giờ có 2 checkpoint canonical cần giữ |

**Câu hỏi Stage 1 đã có câu trả lời trên cả hai lớp được chỉ định. Không xây Stage 2
trên kiến trúc này.**

### 9.2. Đã loại bằng ĐO ĐẠC — không mở lại

| Hướng | Bằng chứng loại |
|---|---|
| ~~Graph / intra-surface transformer~~ | §4.4b: làm mượt láng giềng chỉ lấp 5.9% khoảng cách; FP thành mảng nên bỏ phiếu **củng cố** cái sai |
| ~~Slab + 3D CNN (Extension B)~~ | §4.4c: k-NN không vượt majority ở **cả 4 bộ kênh**; láng giềng thô thêm **0** |
| ~~Sửa loss / sampling~~ | §4.2: stratify **hại**, absentFP vô hiệu |
| ~~Thêm dữ liệu~~ | §4.3: 3.5× ca, đường cong **phẳng** |

### 9.3. NGOÀI phạm vi Stage 1 — là hướng nghiên cứu MỚI, không phải cứu Stage 1

Stage 1 hỏi: *"hệ tọa độ bề mặt xương có giúp model đọc MRI tốt hơn ở vùng sụn mỏng không?"*
Câu trả lời là **không**, đã đo. Hai hướng dưới đây **không trả lời câu hỏi đó** — chúng đặt
câu hỏi khác, và áp dụng cho **mọi** phương pháp (kể cả nnUNet), không riêng hệ tọa độ:

| Hướng | Câu hỏi thật sự của nó |
|---|---|
| **Echo thứ hai của DESS** | *Ảnh hiện tại có đủ thông tin không?* `imagesTr` chỉ có `_0000` = **một kênh**, trong khi DESS thu **hai echo** với tương phản sụn/dịch khác nhau. §4.4c cho thấy recall tăng **đơn điệu** theo lượng thông tin (0.046 → 0.141) ⇒ đây là chỗ duy nhất giới hạn **có thể** dịch chuyển. Là bài toán **dữ liệu/thu nhận**, không phải kiến trúc. |
| **Đo giới hạn chú thích trên OAI-ZIB** | *Bài toán có well-posed ở vùng mỏng không?* Con số 0.85–0.90 hiện **trích từ văn liệu, chưa tự đo**. Nếu người cũng không phân biệt được thì nó giải thích vì sao **mọi** phương pháp dừng ở đó. Là bài toán **đặc tả dataset**. |

*Đã loại khỏi danh sách: "bỏ quyết định nhị phân, chuyển sang độ dày có bất định" (Stage 3
Extension D) — đó là **định nghĩa lại thành công**, không phải trả lời câu hỏi đã đặt.*

**Không khuyến nghị** xây Stage 2 (graph transformer, cross-surface attention) trên nền
kiến trúc này: nếu mô hình 1D không khai thác được biểu diễn ở điều kiện **oracle** (xương
GT), thêm độ phức tạp lên trên khó có khả năng cứu — đây đúng là bài học mà ROI cascade đã dạy.

---

## 10. Tái lập

| Artifact | Đường dẫn |
|---|---|
| Canonical P2 (checkpoint + manifest + gate37 + mapping audit) | `BSC_ROOT/runs/MVP_FC_S0_D1_I2_H1_v1_Fold0_Seed1/4e133ba5fa0a74c6/` |
| QC hình học per-case | `BSC_ROOT/runs/mvp_geom_qc.jsonl` |
| Atlas femoral / med_tib | `BSC_ROOT/atlas/atlas_{cls}_fold0.npz` |
| Phase B sweep (B2) | `BSC_ROOT/runs/phaseB_sweep_femoral_cart.jsonl` |
| Phase B quyết định (B3) | `BSC_ROOT/runs/phaseB_B3_femoral_cart.json` |
| M0 headroom + ε_metric | `BSC_ROOT/runs/M0_percase.jsonl`, `M0_eps_metric.jsonl` |
| Code | `bsc/{core,metrics,headroom,model,atlas,experiment,mvp,io_utils}.py` |
| Notebook | `bsc/notebooks/bsc_02_mvp.ipynb` |
| Test | `bsc/tests/` — **109 test**, chạy trên phantom |

**Tài liệu liên quan:**
`M0_gate1_results_and_decision.md` (Gate 1, headroom, ε_metric) ·
`mvp_stage1_femoral_findings_and_decision.md` (chi tiết chẩn đoán + review) ·
`p3_m8d_standardization_decision_vi.md` (ma trận S/D/I/H) ·
`mapping_split_canonical_p2_decisions_vi.md` (mapping audit, canonical protocol) ·
`bsc_02_mvp_experiment_method_review_vi.md` (phản biện phương pháp)

Bootstrap: 10 000 lần, seed 0, ghép cặp per-case. Spacing lấy **theo từng ca**, không dùng
mặc định module (`core.SPACING` giả định trục 0.70mm ở đầu, dữ liệu thật có nó ở cuối).

---

## 11. Câu kết luận cho bài báo

> A bone-surface coordinate system was implemented and validated for knee cartilage
> segmentation on OAI-ZIB. The representation itself is sound: ground-truth occupancy passed
> through the full normal-ray projection and splatting pipeline reconstructs thin-region
> boundaries to within 0.026 mm — approximately 21× more accurate than the ResEnc baseline
> (0.552 mm). Negative-direction controls confirm the coordinate frame carries genuine
> anatomical information: surface-normal rays outperform axial, tangent, and random
> directions on thin-region boundary error, with bootstrap confidence intervals excluding
> zero in all three comparisons.
>
> However, a 1D ray encoder trained on MRI intensity, gradient, and bone signed-distance
> along these rays did not translate that representational advantage into predictive
> performance. Under oracle conditions (ground-truth bone surfaces), thin-region boundary
> error reached 2.19 mm versus 0.55 mm for ResEnc on femoral cartilage, with zero of ten
> validation cases improving. The result replicates on medial tibial cartilage — the
> stress-test class, selected because it has 52% greater measured headroom — at 0.99 mm
> versus 0.40 mm, again zero of ten, with a paired bootstrap confidence interval of
> [−0.729, −0.458] mm. Notably, the gap ratio tracks absent-region recall (0.879 → 4.0×;
> 0.935 → 2.5×) rather than ray-space occupancy Dice, independently corroborating the
> failure mechanism below. The dominant failure mode is hallucinated cartilage in truly absent regions,
> accounting for approximately 82% of thin-region error mass, with recall falling from 81%
> in thick cartilage to 37% at thin margins. This failure survived systematic elimination of
> alternative explanations: geometric quality control, atlas construction, micro-overfitting
> capacity, stratified sampling, absent-region loss weighting, presence-gating strategy, and
> a 3.5× increase in training data all failed to close the gap.
>
> We therefore report a negative result. Together with our earlier negative finding that ROI
> cascading does not improve thin cartilage segmentation, this suggests that the bottleneck
> lies in the image signal available at extremely thin cartilage rather than in spatial
> localization or coordinate parameterisation. Representational adequacy does not imply
> learnability.
