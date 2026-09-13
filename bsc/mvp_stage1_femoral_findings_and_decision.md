# MVP Stage 1 (femoral cartilage) — Kết quả, chẩn đoán và quyết định

**Ngày:** 2026-07-20
**Lớp:** `femoral_cart` (lớp dẫn debug pipeline, plan §3.1). `med_tib_cart` chưa train.
**Quyết định:** 🔄 **REVISE** trong phạm vi MVP (không STOP, không PROCEED Stage 2). Xem §9.
**Dữ liệu:** OAI-ZIB, 40 ca train (fold 1–4) + 10 ca val (fold 0), `splits_zib_v1_fixed.json`.
Baseline = prediction **out-of-fold** 150ep (B0/ResEnc). 103 ca test **không đụng tới**.
**Bề mặt xương:** ground-truth (S0, điều kiện oracle). Spacing `(0.3646, 0.3646, 0.70)`.

---

## 1. Câu hỏi Stage 1 và cách trả lời

Giả thuyết trung tâm (plan §1): *neo hệ tọa độ vào bề mặt xương giúp định vị biên sụn
chính xác hơn ResEnc, **đặc biệt ở vùng cực mỏng / mất sụn**.*

Cổng quyết định §3.7 điều 1: *"mô hình xương-GT cải thiện outer-boundary error **trong
vùng sụn mỏng** ~10% tương đối trở lên."* Ràng buộc báo cáo (M0 §6): **không dùng ASSD
tổng** (mục tiêu quy về ASSD tổng chỉ 0.006mm, nhỏ ngang sai khác giữa hai implementation
metric). Báo cáo ở **mẫu số vùng mỏng** + presence F1.

---

## 2. QC hình học trên xương thật (plan §6 bước 3/5/6) — ✅ ĐẠT

Chạy M2/M3/M4 trên xương thật trước khi train. Đây là cổng: hỏng ⇒ dừng.

| lớp | M3 normals | M4 single-interval | M2 Dice | M2 ASSD | khung suy biến |
|---|---|---|---|---|---|
| femoral_cart | 99.8% | 99.8% | 0.995 | 0.009mm | 0% |
| med_tib_cart | 99.6% | 99.7% | 0.993 | 0.008mm | 0% |

Cả ba cổng đạt với biên độ lớn (M3 >99.5%, M4 >95%, M2 ASSD <0.1mm). Không ca nào có
khung tọa độ suy biến → atlas D1 dùng được cho mọi ca. **Hình học không phải nút thắt.**

---

## 3. Atlas D1 (plan §2.2, §3.4 Step 4 MVP-B) — ✅ hoạt động

Xây **chỉ từ 40 ca train** của fold, `assert_no_leak` xác nhận không ca val nào lọt vào.

- Phân bố P(khớp) lưỡng cực rõ: 32% ô ≥0.8, 61.6% ô ≤0.2 (không "chọn bừa").
- Trên ca val (chưa từng thấy): **recall 98.2–99.8%**, precision 87–92%, chọn ~41% node
  (oracle 37%) → thật sự có lọc, thiên recall đúng thiết kế.
- Kiểm chéo femoral↔med_tib: atlas đúng lớp recall ~99%, atlas sai lớp recall **0%** →
  mỗi atlas mã hóa đúng giải phẫu riêng, không lẫn.

*Ghi chú phương pháp: cách đăng ký PCA ban đầu bị bác bỏ (xương đối xứng → trục/dấu không
xác định), thay bằng chuẩn hóa trục-gốc (tịnh tiến + tỉ lệ), khớp §4.4 Step 2 mục 1.*

---

## 4. Ma trận P0–P3 + ablation đầu vào M8 (plan §7/§8) — occ-Dice không gian tia (val)

| run | cấu hình | kênh | occ-Dice | presence F1 | absent-recall |
|---|---|---|---|---|---|
| P0 | S0-D0-I0-H0 | mri | 0.833 | — (H0) | — |
| P1 | S0-D0-I1-H0 | mri+grad | 0.849 | — (H0) | — |
| P2 | S0-D1-I2-H1 | mri+sdf | 0.825 | 0.884 | 0.916 |
| P3 | S0-D1-I3-H1 | mri+prob | *bỏ qua* (chưa có OOF softmax) | | |
| M8-A | S0-D1-I0-H1 | mri | 0.821 | 0.880 | 0.907 |
| M8-B | S0-D1-I1-H1 | mri+grad | 0.839 | 0.890 | 0.927 |
| M8-C (=P2) | S0-D1-I2-H1 | mri+sdf | 0.825 | 0.884 | 0.916 |

**Nhận xét đóng góp kênh:** gradient (I1, +0.018 so với I0) đóng góp gấp ~4 lần SDF
(I2, +0.004). Đáng chú ý vì P2 (SDF) đang là cấu hình chuẩn, nhưng dữ liệu nói I1 nhỉnh hơn.
P3/M8-D (coarse ResEnc prob) chưa đo được — cần OOF softmax (hoãn theo quyết định trước).

*Cảnh báo: occ-Dice không gian tia KHÔNG phải metric quyết định (xem §5, §6). Nó chỉ đo
mô hình có học được profile 1D không.*

---

## 5. M5 — tiny-set overfitting (plan §3.5 bước 8) — ✅ đạt, nhưng lộ UNDERFIT

Train 4 ca, ~4000 tia, 80 epoch:

- loss 0.457 → 0.105 (giảm 77%)
- occ-Dice train **0.897** tại ngưỡng tối ưu 0.60 (>0.90 sau khi chọn ngưỡng đúng; ngưỡng
  0.5 cho 0.883 vì loss là focal-BCE, tối ưu hiệu chỉnh chứ không tối ưu Dice).
- Phân rã lỗi: **27% lỗi ở ô biên**, 3.7% trong lòng → phần lớn là lượng tử hóa biên.
- **loss vẫn giảm 12% ở 10 epoch cuối** → mô hình CHƯA hội tụ, còn underfit.

Dấu hiệu underfit này là manh mối quan trọng cho §9.

---

## 6. M6 — negative control hướng tia (plan §3.5 bước 9) — ✅ ĐẠT (sau khi sửa metric)

**Bài học phương pháp — metric sai làm đảo kết luận:**

Đo lần đầu bằng occ-Dice không gian tia cho kết quả **sai**:

| hướng | tỉ lệ dương | occ-Dice (SAI) |
|---|---|---|
| normal | 21.2% | 0.825 |
| axial | 33.8% | 0.856 |
| tangent | 35.7% | 0.837 |
| random | 24.5% | 0.797 |

Tia pháp tuyến cắt **vuông góc** lớp sụn → đoạn giao ngắn nhất → tỉ lệ dương thấp nhất →
occ-Dice khó nhất. Axial cắt xiên → đoạn giao dài → Dice dễ hơn. So occ-Dice giữa các
hướng là so **hai bài toán khác nhau**. Plan §3.5 M6 viết rõ *"boundary and thickness
metrics"* — phải đo ở **không gian voxel sau dựng lại**.

Đo đúng (lỗi biên vùng mỏng, thống kê ghép cặp §2.4, n=10):

| đối chứng | lỗi TB | normal tốt hơn | boot 95% CI | số ca |
|---|---|---|---|---|
| axial | 3.68mm | **+1.69mm** | [+1.39, +1.93] | 10/10 |
| tangent | 2.97mm | **+0.98mm** | [+0.68, +1.29] | 10/10 |
| random | 2.32mm | **+0.33mm** | [+0.21, +0.46] | 9/10 |
| **normal** | **1.99mm** | | | |

**Pháp tuyến thắng cả ba đối chứng, CI đều dương.** → Lợi ích đến từ **hệ tọa độ**, không
phải "thêm một mạng". Đây là điều kiện §3.7 điều 5, và nó ĐẠT — luận điểm hình học có cơ sở.

---

## 7. M7 jitter + P5 xương dự đoán (plan §3.5 bước 10/12)

**M7 (nhiễu bề mặt lúc test):** occ-Dice suy giảm **đều đặn**, không sụp đột ngột:
0.825 (0mm) → 0.735 (0.25mm) → 0.623 (0.5mm) → 0.492 (1.0mm).

⚠️ *Caveat (review §4.3): đây là **sensitivity test**, KHÔNG phải robustness test — model train
KHÔNG jitter, chỉ thêm jitter lúc test; và displacement + rotation bị **ghép chung** (không tách
được lỗi vị trí vs lỗi normal); và dùng occ-Dice không gian tia, không phải metric quyết định.
Chưa được kết luận "hệ tọa độ không giòn".*

**P5 (S2 — xương DỰ ĐOÁN thay GT):** occ-Dice 0.793 vs P2 0.825.

⚠️ *Caveat (review §4.4): con số "giữ 96%" tính từ occ-Dice không gian tia, mà S0/S2 có target
occupancy + reconstruction ceiling khác nhau nên không cùng bài toán. Quan trọng hơn: "retained
gain" `(E_B0−E_P5)/(E_B0−E_P2)` **vô nghĩa khi P2 còn tệ hơn B0** (mẫu số âm). P5 phải đo lại
bằng metric vùng mỏng SAU khi P2 vượt B0. Chưa được kết luận điều 6 §3.7 đạt.*

---

## 8. Cổng §3.7 sơ cấp — ❌ TRƯỢT NẶNG, và chẩn đoán cơ chế

### 8.1. Kết quả cổng (n=10 val, mẫu số vùng mỏng)

| | lỗi biên vùng mỏng |
|---|---|
| ResEnc B0 | **0.5517mm** |
| ray model (P2) | **~2.00mm** |
| cải thiện tương đối | **−291%** (tệ hơn 4×) |
| số ca cải thiện | **0/10** |

Điều 1 (cải thiện vùng mỏng) và điều 4 (không tăng FP ở absent) đều trượt. Đây là tín hiệu
STOP/REVISE theo chính cổng của plan. **Nhưng phải chẩn đoán cơ chế trước khi kết luận.**

### 8.2. Tái tạo KHÔNG hỏng (voxel Dice)

| ca | Dice ray | Dice B0 | vol ray/GT | recall thể tích |
|---|---|---|---|---|
| 003 | 0.858 | 0.905 | 1.03 | 87% |
| 010 | 0.839 | 0.888 | 1.12 | 89% |
| 022 | 0.677 | 0.862 | 0.67 | 56% (ngoại lệ) |
| 028 | 0.837 | 0.880 | 0.99 | 84% |
| 035 | 0.824 | 0.861 | 1.03 | 84% |

Dice ray ~0.81 vs B0 ~0.88 — chỉ kém 0.07 ở **tổng thể**. Loại bỏ giả thuyết "tái tạo vỡ".

### 8.3. Lỗi vùng mỏng = ẢO GIÁC ở bin `absent` (phân rã theo độ dày)

| bin độ dày | ray err | B0 err | ray n | B0 n |
|---|---|---|---|---|
| **absent** | **3.380mm** | 1.057mm | **23897** | 13155 |
| ≤0.5mm | 0.708 | 0.431 | 2430 | 2787 |
| ≤1.0mm | 0.702 | 0.308 | 22954 | 27266 |

Bin `absent` (voxel PRED ở vùng GT không sụn = **dương tính giả**) chiếm **82% khối lượng
lỗi vùng mỏng** của ray. Ray tạo 23897 voxel sụn ảo (gấp 1.8× B0), nằm xa hơn (3.38 vs
1.06mm). Đây đúng là §3.7 điều 4 và ngược §9.3 ("giảm nhầm lẫn absent vs cực mỏng").

### 8.4. Recall sụt theo độ mỏng (bức tranh "mỏng vs dày")

| độ dày GT | voxel sụn GT | ray recall | FP ray |
|---|---|---|---|
| >2.0mm (dày, trung tâm) | 622916 | **80.9%** | 49864 |
| ≤2.0mm | 158038 | 78.1% | 45800 |
| ≤1.0mm (rìa mỏng) | 16291 | **37.0%** | 6515 |
| ≤0.5mm (cực mỏng) | 1350 | **30.7%** | 1160 |
| absent | — | — | **29023** |

**Mô hình bắt tốt phần thân dày (81%), hỏng ở rìa mỏng (37%)** — bỏ sót ~70% sụn mỏng,
đồng thời ảo giác ở vùng absent. Đây **ngược hẳn** điều giả thuyết hứa (giúp nhất ở vùng mỏng).

### 8.5. Hard scalar presence gating KHÔNG đủ **dưới setup hiện tại**

| presence_thr | ray thin err |
|---|---|
| 0.5 | 2.00mm |
| 0.7 | 1.84mm (tốt hơn tí) |
| 0.8 | 2.26mm |
| 0.9 | 3.99mm |
| 0.95 | 8.70mm |

Nâng ngưỡng đáng lẽ giảm FP, lại làm lỗi **tăng**. Vì presence-**một-giá-trị-mỗi-tia** hard-gate
cả tia: ở vùng mỏng tín hiệu "có sụn" yếu, gần như không tách được với absent. Nâng ngưỡng vừa
chặn FP absent vừa chặn luôn sụn mỏng thật (biến FP thành FN).

**Cải chính (review §9):** kết quả này chứng minh *"hard scalar presence gating với threshold
hiện tại, dưới loss/sampling hiện tại, không tách được thin/absent"* — **KHÔNG** chứng minh
*"mọi ray-level presence đều bất khả thi"*. Chưa thử: balanced-ray sampling, bỏ hard-gate,
presence suy từ occupancy, OOF ResEnc prior, structured interval output. Xem §14.

---

## 9. Phép thử QUYẾT ĐỊNH — oracle reconstruction

Câu hỏi: 2.0mm là do **model đoán kém** (sửa được) hay **trần biến đổi tọa độ + splat**
(fundamental)? Cho occupancy GT hoàn hảo đi qua đúng đường splat, đo lỗi vùng mỏng:

| | lỗi biên vùng mỏng |
|---|---|
| **Oracle (occupancy GT hoàn hảo qua splat)** | **0.0256mm** |
| ResEnc B0 | 0.5517mm |
| ray model đã train | ~2.00mm |

**Đây là kết quả lật ngược:**

- Trần của phép biểu diễn là **0.026mm** — chính xác hơn ResEnc **~20 lần**. Splat + tọa độ
  pháp tuyến **KHÔNG** phá sụn mỏng.
- Toàn bộ 2.0mm lỗi của ray là **lỗi dự đoán của MODEL**, không phải trần fundamental.
- **Có headroom thật:** khoảng cách ResEnc→trần ở vùng mỏng là 0.52mm.

*(Nhất quán với M0: trên toàn bề mặt headroom nhỏ (prize 0.058mm), nhưng riêng vùng mỏng
khoảng cách tới trần là 0.52mm — đúng thứ rescope M0 §6 nhắm tới. Và nhất quán với M2: biên
đo bằng khoảng cách mm thì nhỏ tí (0.026mm) dù Dice trên cấu trúc ~1 voxel trông tệ.)*

---

## 10. Quyết định: REVISE (không STOP)

Không phải "giả thuyết sai" mà "MODEL hiện tại quá yếu để hiện thực hóa phần biểu diễn vốn
đã tốt". Bằng chứng hội tụ:

1. Phép biểu diễn: trần 0.026mm (§9) — không phải nút thắt.
2. Hình học: QC đạt biên độ lớn (§2) — không phải nút thắt.
3. Hệ tọa độ: M6 chứng minh pháp tuyến thắng đối chứng (§6) — luận điểm có cơ sở.
4. Model: underfit (M5 loss còn giảm 12% ở cuối, §5); train trên **40/320 ca, 30 epoch**;
   presence-mỗi-tia không tách được thin/absent (§8.5).

### Lever sửa (cập nhật theo review §8 — nguyên nhân gốc là LOSS/SAMPLING, không chỉ underfit)

Review §8 chỉ ra cơ chế sâu hơn "thiếu ca": occupancy loss **trung bình trên mọi ô độ sâu của
mọi tia** → tia sụn dày có nhiều ô dương, tia mỏng rất ít, tia absent toàn âm; uniform sampling
không đảm bảo đủ ví dụ thin/absent. Nên **gradient dương bị sụn dày chi phối** → khớp đúng
recall 81% (dày) vs 37% (mỏng). **Scale training đơn thuần KHÔNG sửa bias cấu trúc này.**

Thứ tự sửa:

1. **Sửa objective + sampling** (gốc rễ, §8): stratified sampling theo bin độ dày + hard
   negatives cạnh atlas margin; per-ray normalization; boundary-endpoint loss; phạt FP-absent
   mạnh hơn; thin-ray weighting. **Bỏ hard scalar gate** ở revision đầu.
2. **Micro-overfit đúng cấu hình** S0-D1-H1 (không phải P1/D0/H0 như M5 cũ, review §3.5), 1 ca,
   tia phân tầng, mục tiêu train Dice 0.97–0.99, kiểm thin/absent RIÊNG. Không đạt thì chưa scale.
3. **Scale** (nhiều ca + epoch + ≥3 seed) — chỉ sau khi (1)(2) đạt.
4. **Presence theo độ sâu / OOF ResEnc prior (I3)** nếu FP absent vẫn trơ.

### Cổng ghi TRƯỚC (tránh hợp lý hóa ngược)

Sau khi scale, đo lại §3.7 vùng mỏng trên val:

- `ray thin err < 0.5517mm` (dưới B0) **và** CI bootstrap dương → **PROCEED** Stage 2.
- Chững lại xa trên 0.55mm dù đã scale hết + presence theo độ sâu → **STOP**, công bố âm tính.

---

## 11. Giới hạn — điều nghiên cứu này CHƯA khẳng định

- Chỉ **femoral_cart**; med_tib_cart (lớp stress-test §3.1) chưa train — atlas đã dựng sẵn.
- n=10 val, 1 fold (fold 0). Chưa 5-fold, chưa nhiều seed.
- P3/M8-D (coarse ResEnc prob) chưa chạy — cần OOF softmax.
- Oracle test dùng occupancy GT **và** presence GT (mọi thứ hoàn hảo) → là trần lý thuyết;
  không khẳng định model CÓ THỂ đạt được, chỉ khẳng định phép biểu diễn không chặn.
- Điều kiện S0 (xương GT). P5 (xương dự đoán) mới sơ bộ trên occ-Dice, chưa đo §3.7.

---

## 12. Tái lập

| Artifact | Đường dẫn |
|---|---|
| QC hình học per-case | `BSC_ROOT/runs/mvp_geom_qc.jsonl` |
| Atlas femoral / med_tib | `BSC_ROOT/atlas/atlas_{cls}_fold0.npz` |
| Đánh giá §3.7 per-case | `BSC_ROOT/runs/mvp_eval_femoral_cart_P2.jsonl` |
| Registry + gate | `BSC_ROOT/runs/MVP_stage1_femoral_cart.json` |
| Cache hình học | `BSC_ROOT/geom_cache/` |
| Code | `bsc/mvp.py`, `bsc/model.py`, `bsc/atlas.py`, `bsc/experiment.py`, `bsc/core.py` |
| Notebook | `bsc/notebooks/bsc_02_mvp.ipynb` |
| Test | `bsc/tests/test_mvp.py` (+ test_model/atlas/experiment), 94 test pass |

Cấu hình P2: `MVP_FC_S0_D1_I2_H1_v1_Fold0_Seed1`. Bootstrap 10000 lần, seed 0, ghép cặp
per-case. **Lưu ý kỹ thuật:** notebook chưa lưu checkpoint model ra đĩa → tắt session mất
weight, phải train lại (đang được sửa).

---

## 13. Thay đổi so với kỳ vọng ban đầu — ghi để minh bạch (§7 plan)

| Kỳ vọng ban đầu | Thực tế | Bằng chứng |
|---|---|---|
| Coordinate-transform có thể là trần fundamental ở vùng mỏng | **Sai** — trần 0.026mm | §9 oracle test |
| Giả thuyết "giúp nhất ở vùng mỏng" | **Ngược** ở MVP hiện tại — tệ nhất ở vùng mỏng | §8.4 recall 37% |
| Chỉnh ngưỡng presence sẽ giảm FP | **Sai** — nâng ngưỡng làm tệ hơn | §8.5 sweep |
| occ-Dice đo được M6 | **Sai metric** — đảo kết luận | §6 |

Người dự đoán oracle sẽ cao (~2mm, STOP); dữ liệu cho 0.026mm (REVISE). Ghi lại để không
lặp lại thiên kiến bi quan: **chạy phép thử quyết định trước khi kết luận từ hình ảnh.**

---

## 14. Phản biện phương pháp (review `bsc_02_mvp_experiment_method_review_vi.md`) — điều chỉnh

Kết luận LỚN không đổi (**REVISE, giữ hướng hình học**), nhưng review chỉ ra bug bookkeeping +
overclaim làm một số kết luận PHỤ chưa chắc. Ghi lại đầy đủ, không giấu.

### 14.1. Bug bookkeeping — PHẢI sửa TRƯỚC khi chạy lại (nếu không kết quả bị nhiễu)

| # | Bug | Hệ quả |
|---|---|---|
| a | `git pull` mỗi lần chạy, không khóa commit | train và eval có thể ở hai revision code khác nhau, cùng experiment_id |
| b | **Không lưu checkpoint model** | tắt session mất weight → phải train lại; các cell trộn `net` và `RUNS[BEST].net` → nguy cơ hai instance P2 khác nhau trong cùng báo cáo |
| c | Eval jsonl chỉ khóa theo `case_id` | train lại P2 nhưng ca cũ bị skip → **dùng kết quả model cũ**. Đường dẫn phải là `eval/<experiment_id>/<checkpoint_hash>/` |
| d | Gate JSON dùng `m6["normal"]["occ_dice"]` (metric SAI cũ) | registry ghi M6 fail dù phân tích đúng (boundary) cho pass. Phải dùng `all(ci_low>0)` của `m6_stats` |
| e | Gate M5 dùng `m5_dice` @thr0.5, config P1/D0/H0, 3 ca (summary ghi 4) | M5 test SAI cấu hình đang hỏng; phải test S0-D1-H1, mục tiêu 0.97–0.99 |
| f | Cache key **thiếu** direction_seed, jitter_seed, atlas_hash, code_commit | `random`/`tangent`/jitter có thể tái dùng hình học SAI seed; atlas mới dùng domain cũ |

### 14.2. Claim BỊ DOWNGRADE (đã sửa inline ở §7, §8.5)

- **"presence-per-ray fundamental"** → chỉ là *"hard scalar gating không đủ dưới setup hiện tại"* (§8.5).
- **P5 "giữ 96%"** → số ray-space, retained-gain vô nghĩa khi P2<B0 (§7).
- **M7 "không giòn"** → mới là sensitivity test, chưa robustness; displacement+rotation ghép chung (§7).
- **Bảng phân rã bin (§8.3) và recall-độ-dày (§8.4) dùng n=5**, không phải 10 — là **exploratory subset**.
  (Các số Gate: §3.7, oracle §9, M6 paired §6 dùng đủ 10 — không ảnh hưởng.)

### 14.3. Red flag mapping — bin `absent` có 1123 voxel sụn GT (review §6)

Voxel sụn GT về khái niệm **không thể** nằm ở bin "không có sụn". Nó lộ ra phép gán
`voxel sụn → node xương gần nhất (Euclidean)` gán nhầm vài voxel rìa vào node thickness 0.
⇒ **Không diễn giải "absent recall 30.1%" như hiện tượng giải phẫu.** Trước khi dùng bảng binwise
cho kết luận chính, phải: bin theo **ray/node sở hữu voxel** (provenance) thay vì nearest-Euclidean,
và báo cáo tỉ lệ mapping inconsistency.

### 14.4. Phép thử phương pháp CÒN THIẾU (làm trong revise)

- **M6 tách representation vs learnability:** mỗi hướng có oracle ceiling KHÁC nhau. Phải báo cáo
  `excess = E_model(d) − E_oracle(d)` cho từng hướng. Hiện chỉ khẳng định được *"normal tốt hơn
  controls trong setup này"*, chưa phải *"model đọc MRI tốt hơn nhờ hướng normal"*.
- **Tách metric vùng mỏng thành 3** (review §5): GT→Pred (bỏ sót/lệch biên sụn thật) · Pred→GT
  absent (hallucination) · ray-endpoint boundary MAE. Giữ số gộp làm composite an toàn.
- **M7 grid tách** displacement và rotation; đo bằng metric vùng mỏng; rồi train CÓ jitter.
- **Chọn mẫu:** `[:40]/[:10]` là convenience subset. Gate chính phải stratify theo KL/độ dày/
  absent area/domain/baseline-thin-err, ghi manifest, không đổi sau khi xem kết quả.
- **`BEST` = P2 hard-code** dù M8-B cao hơn ray-Dice → đổi tên `PRIMARY_RUN`; nếu chọn input theo
  val phải chọn bằng metric thin-region rồi đánh giá trên split KHÁC (tránh selection bias).

### 14.5. Điều review XÁC NHẬN vẫn tin được ngay

Geometry QC đạt · atlas không leak trực tiếp · **P2 hiện tại thất bại ở vùng mỏng** · oracle
self-consistency tốt (0.026mm) · hallucination absent là failure mode lớn. Và quyết định
**REVISE, giữ hướng hình học** — không đổi.

---

## 15. Phase A — canonical P2 (checkpoint-locked, 2026-07-20)

Sau khi sửa bookkeeping (Step 0), chạy lại P2 **config cũ** sạch, khóa checkpoint-hash. Đây
là engineering baseline coherent (thay các số `provisional` từ nhiều instance RAM ở §14.2).

| | |
|---|---|
| experiment_id | `MVP_FC_S0_D1_I2_H1_v1_Fold0_Seed1` |
| checkpoint_sha256 | `4e133ba5fa0a74c6` · git `dc6f5849` |
| §3.7 vùng mỏng (từ model RELOAD) | B0 **0.5517mm** → ray **2.1881mm** · rel −326% · **0/10** |
| val occ-Dice | 0.816 · presence F1 0.899 · absent-recall 0.879 |

**Số của record giờ là 2.19mm** (không phải 1.94–2.0 ở §8 — đó là instance RAM khác phiên,
đúng bug bookkeeping đã sửa). Dao động 1.94–2.19 giữa instance = run-to-run variation, nhỏ
không đáng kể so với gap tới B0. Kết luận P2 thất bại: **xác nhận sạch, hash-locked.**

### 15.1. Mapping audit A vs B (§2.6 của `mapping_split...`) — vùng xám, kết luận vững

| | Mapping A (node) | Mapping B (nearest-ray) |
|---|---|---|
| B0 thin err | 0.5517mm | 0.6033mm (**+9.3%**) |
| GT voxel gán nhầm `absent` | 2398 | **848** (−65%) |
| QC Mapping B | — | unassigned 0.3% · dist median 0.17mm · p95 0.34mm |

9.3% ở vùng xám (5–10%), sát ngưỡng. **Nhưng kết luận không lung lay:** kể cả dưới Mapping B,
B0 (0.60) vẫn hơn P2 (2.19) ~3.6×. Mapping B **đúng hơn thật** (giảm mis-binning 2398→848, QC
sạch). Caveat: audit này đo trên B0, chưa đo P2 dưới B (P2 có thể giảm chút nhưng vẫn >> 0.60).

**Chốt:** Mapping A cho Phase B engineering (không lật so sánh); **Mapping B cho final Gate**
(tốt hơn, rẻ, thay cho full provenance). Ghi rõ khi báo cáo.

### 15.2. Phase B — kế hoạch đã khóa

So mọi revision với **P2 khóa (2.19mm)** và **B0 (0.55mm)** trên cùng engineering split 40/10:

1. Micro-overfit S0-D1-H1 (1 ca, tia phân tầng, mục tiêu 0.97–0.99) — không đạt thì chưa scale.
2. Stratified ray sampling theo bin độ dày + hard negatives cạnh atlas margin.
3. Bỏ hard scalar presence gate; loss per-ray-norm + boundary-endpoint + phạt FP-absent.
4. Cổng Phase B (§5.2 của decision): thin err giảm mạnh từ 2.19mm, absent FP giảm, thin recall
   tăng — chưa cần vượt B0 ngay khi còn debug. Vượt B0 (paired CI dương) là cổng Stage 2.

### 15.3. B1 — micro-overfit (2026-07-20): bug bị LOẠI TRỪ, lỗi là generalization

Micro-overfit S0-D1-H1 trên 1 ca (1200 tia stratified, 150ep): occ-Dice train **0.879**,
thin-recall 0.827, **absent presence-acc 0.988**. Phân rã lỗi: **trong lòng 2.69%** (memorize
tốt, hơn M5 3.70%), biên 33% (trần lượng tử hóa), 39% lỗi ở biên.

**Cải chính cổng:** mốc 0.97 mình đặt nằm TRÊN trần lượng tử hóa ray-space (~0.90, xác nhận
qua M5 đỉnh 0.897 + M2). occ-Dice ray-space là metric TỆ cho micro-overfit ở đây; Phase B đo
boundary + §3.7 thin, không đo ray occ-Dice.

**Kết luận:** cấu hình biểu diễn được ca train (interior 2.69%, presence 0.988) ⇒ **bug loại
trừ**. Presence memorize train (0.988) nhưng ảo giác FP trên val ⇒ lỗi §3.7 là **generalization
gap của presence**, KHÔNG phải giới hạn biểu diễn (sắc hơn "presence-per-ray fundamental" ở
§8.5). ⇒ Phase B step 2 nhắm **generalization**: nhiều ca + stratified + regularization/phạt
FP-absent, KHÔNG phải truy bug.
