# M0 Headroom — Kết quả, phương pháp và quyết định Gate 1

**Ngày:** 2026-07-19
**Giai đoạn:** Stage 1 (MVP) — trước khi xây bất cứ mô hình nào
**Quyết định:** ✅ **PROCEED** xây MVP, kèm cổng mới và ràng buộc báo cáo (xem §6, §7)
**Dữ liệu:** OAI-ZIB test, 103 ca (`Dataset001_KneeOA/labelsTs`, nhãn `[0..5]`)
**Baseline:** B0 = `Dataset020_KneeUnion` ResEnc-L 250ep fold_0

---

## 1. Câu hỏi M0 đặt ra

M0 **không có trong plan doc**. Nó hỏi đúng câu mà thí nghiệm ROI cascade đã quên hỏi:

> Phần thưởng lớn cỡ nào — **trước** khi xây bất cứ thứ gì?

ROI cascade đã đốt một chu kỳ vì bắt đầu xây (crop, train ResEnc-L 250ep) rồi mới đo xem
localization có giúp không, và câu trả lời là KHÔNG (Dice ±0.003 dù bbox hoàn hảo).

Giả thuyết của dự án hứa cải thiện ở **biên và độ dày**, đặc biệt nơi sụn **mỏng/mất**.
Nên câu hỏi cụ thể: **lỗi của ResEnc có thực sự nằm nhiều ở vùng mỏng/mất không?**

---

## 2. Phương pháp — cách từng con số được tạo ra

### 2.1. Trường độ dày GT

`bsc/core.py` + `headroom.gt_thickness_per_node()`:
SDF xương (mm) → marching cubes lấy node bề mặt → pháp tuyến từ ∇φ → phóng tia dọc pháp
tuyến → lấy mẫu mask sụn GT dọc tia → độ dày mỗi node.

Bin độ dày theo plan §2.3: `absent` (0mm), `≤0.5`, `≤1.0`, `≤2.0`, `>2.0`.

### 2.2. Error mass theo bin (`error_mass_by_thickness`)

Hai chiều, đối xứng như ASSD:
- voxel bề mặt **GT** → khoảng cách tới bề mặt PRED, bin theo độ dày của chính nó
  (khối lượng bỏ sót / false-negative);
- voxel bề mặt **PRED** → khoảng cách tới bề mặt GT, bin theo độ dày node GT **gần nhất**
  (khối lượng thừa / false-positive — đây là cách bin `absent` có dữ liệu, vì nó không có
  bề mặt GT riêng; chính chiều này làm chẩn đoán hoạt động ở OA nặng).

`error_mass = Σ khoảng cách` trong bin. Đây là **khối lượng lỗi**, không phải diện tích.

### 2.3. Prize counterfactual (`prize_counterfactual`)

Bỏ các voxel bề mặt thuộc bin `{absent, ≤0.5, ≤1.0}` khỏi phép tính ASSD (coi như **khớp
hoàn hảo**, khoảng cách 0), rồi tính lại ASSD:

```
assd_base  = mean(mọi khoảng cách)
assd_prize = Σ(khoảng cách còn lại) / N_tổng      # voxel bị bỏ tính là 0
prize      = assd_base − assd_prize
```

**⚠️ Prize là TRẦN, không phải cải thiện kỳ vọng.** Nó giả định hiệu năng **hoàn hảo tuyệt
đối** ở vùng mỏng. Hiểu nhầm điểm này là gốc rễ của lỗi ngưỡng ở §5.

Tính per-case → paired bootstrap CI (10 000 lần, seed 0).

### 2.4. Metric

`bsc/metrics.py` — **một định nghĩa duy nhất** cho toàn Stage 1:
`skimage.find_boundaries(mode="inner")` + `scipy.distance_transform_edt`, spacing mm theo
thứ tự trục (z, y, x). Cố ý **không** dùng SimpleITK để có hai cài đặt độc lập đối chiếu.

---

## 3. Gate 0 — kiểm chứng metric (điều kiện tiên quyết)

Không thể đo hiệu ứng 0.1mm bằng metric chưa kiểm chứng.

### 3.1. Đối chiếu bảng đã công bố (404 ca CV, bản 150ep)

| lớp | Dice | ASSD (scipy) | Dice đã công bố |
|---|---|---|---|
| femoral_cart | 0.889 | 0.271 | 0.891 |
| med_tib_cart | 0.844 | 0.280 | 0.852 |
| lat_tib_cart | 0.861 | 0.291 | 0.868 |

Dice khớp trong ±0.02. **Lưu ý phương pháp:** đây là 150ep-CV so với 250ep-test — **khác
model, khác tập** — nên chỉ là phép kiểm tỉnh táo, **không** kết luận gì về ASSD.

### 3.2. Đặc trưng hóa độ lệch scipy vs SimpleITK (40 ca, per-case)

δ = ASSD_scipy − ASSD_sitk:

| lớp | mean | median | SD | IQR | %dương | boot 95% CI |
|---|---|---|---|---|---|---|
| femoral_cart | **+0.0644** | +0.0643 | 0.0088 | [+0.058, +0.071] | 100% | [+0.062, +0.067] |
| med_tib_cart | +0.0041 | +0.0082 | 0.0210 | [−0.002, +0.020] | 72% | [−0.003, +0.010] |
| lat_tib_cart | −0.0053 | −0.0047 | 0.0155 | [−0.011, +0.004] | 40% | [−0.010, −0.001] |

Δdice = 0.0000 cả ba lớp → không có bug đọc file/nhãn.

**Kết luận:** femoral lệch **+0.064mm nhưng là hằng số cứng** (SD 0.009, 100% dương) —
khác **định nghĩa bề mặt** (`edt`-tới-tâm-voxel vs `SignedMaurer`-tới-mặt-phân-giới), phụ
thuộc hình học (femoral có tỷ lệ mặt cong/xiên cao + spacing z bất đối xứng 0.70 vs 0.36mm).
Tibial ~0.

**Quy tắc rút ra:** không bao giờ đặt ASSD scipy cạnh số đã công bố 0.21mm (tính bằng
SimpleITK). Chỉ báo cáo scipy, nhất quán cho mọi baseline/model.

---

## 4. Gate 1 — kết quả M0 (103 ca test)

### 4.1. Bảng chính

| | ΔASSD_prize | boot 95% CI | thin+absent error mass | diện tích | **tập trung** | bậc thang z |
|---|---|---|---|---|---|---|
| femoral_cart | 0.0582mm | [0.0529, 0.0643] | 21.4% | 9.6% | **2.23×** | 0.871 |
| med_tib_cart | 0.0885mm | [0.0761, 0.1039] | 33.5% | 20.3% | **1.65×** | 0.709 |

### 4.2. Phân rã error mass theo bin

**femoral_cart**

| bin | N | lỗi TB (mm) | % error mass |
|---|---|---|---|
| absent | 2 079 | **0.991** | 9.2% |
| ≤0.5mm | 749 | 0.596 | 1.7% |
| ≤1.0mm | 6 186 | 0.416 | 10.5% |
| ≤2.0mm | 25 889 | 0.248 | 26.1% |
| >2.0mm | 59 177 | 0.224 | 52.5% |

**med_tib_cart**

| bin | N | lỗi TB (mm) | % error mass |
|---|---|---|---|
| absent | 544 | **0.830** | 10.8% |
| ≤0.5mm | 371 | 0.518 | 3.8% |
| ≤1.0mm | 2 958 | 0.324 | 18.8% |
| ≤2.0mm | 9 404 | 0.200 | 40.1% |
| >2.0mm | 5 826 | 0.209 | 26.4% |

### 4.3. Nhận xét

- **Bin `absent` sai gấp ~4 lần mọi bin khác** (0.99 và 0.83 vs ~0.21–0.25mm). Đây là điểm
  yếu cụ thể, có thật của ResEnc — và đúng là thứ presence head nhắm tới.
- **Lỗi tập trung bất tương xứng ở vùng mỏng:** femoral 2.23×, med_tib 1.65×.
- **Bậc thang z KHÔNG chi phối** (0.871 và 0.709, đều <1) → nhiễu lượng tử hóa theo z không
  phải thành phần áp đảo. Nhánh STOP nhanh của Gate 1 không kích hoạt.
- Femoral phần lớn vẫn dày: >2.0mm chiếm 52.5% error mass. Phương pháp **không giúp được** ở
  phần đó.

### 4.4. Độ bền với định nghĩa metric (`ε_metric`, 103 ca)

Chạy **cùng counterfactual** bằng hai định nghĩa khoảng cách:

| lớp | Prize_scipy | Prize_sitk | ε | \|ε\|/prize | boot 95% CI(ε) |
|---|---|---|---|---|---|
| femoral_cart | 0.0582 | 0.0549 | +0.0033 | **6%** | [+0.0029, +0.0037] |
| med_tib_cart | 0.0885 | 0.0808 | +0.0077 | **9%** | [+0.0068, +0.0086] |

**Kết luận: kết luận Gate 1 KHÔNG phụ thuộc định nghĩa metric.** Không quyết định nào bị lật
khi đổi quy ước (femoral RESCOPE cả hai; med_tib ≥0.08 cả hai).

**Bài học phương pháp luận:** gap tuyệt đối δ **không nói gì** về độ tin của prize. Femoral
δ = +0.064 nhưng ε chỉ +0.0033 (triệt tiêu 96%); ngược lại med_tib δ ≈ +0.004 nhưng ε **lớn
hơn** (+0.0077). Phải đo ε trực tiếp, không suy từ δ.

*(Đã kiểm: bảng Gate 1 và bảng ε dùng cùng 103 ca, cùng pipeline — join per-case cho
`max|A−B| ≈ 1e-16`, tức chỉ là sai số dấu phẩy động.)*

---

## 5. Phát hiện quan trọng nhất — cổng M0 cũ bị hỏng

Cổng M0 ban đầu (do nhóm tự đặt, **không có trong plan**):

```
PROCEED : ΔASSD_prize ≥ 0.08mm  VÀ  thin+absent ≥ 35% error mass
RESCOPE : ΔASSD_prize ∈ [0.04, 0.08)
STOP    : ΔASSD_prize < 0.04mm  HOẶC  bậc thang z chi phối
```

### 5.1. Lỗi phạm trù

`prize` là **TRẦN** (giả định hoàn hảo tuyệt đối), còn `0.08` là mức **cải thiện mong muốn
đạt được**. So trần với mức-đạt-được là **so sai loại đại lượng**.

### 5.2. Hệ quả số học — bất khả thi

Quy ngưỡng 0.08 về mẫu số vùng mỏng:

| lớp | trần tối đa | ngưỡng 0.08 đòi | khả thi? |
|---|---|---|---|
| femoral_cart | 0.0582mm | **137%** lỗi vùng mỏng | ❌ **KHÔNG THỂ ĐẠT** |
| med_tib_cart | 0.0885mm | 90% lỗi vùng mỏng | ⚠️ đòi gần như hoàn hảo |

**Femoral không bao giờ qua được Gate 1 — kể cả phương pháp hoàn hảo tuyệt đối**, vì ngưỡng
đòi nhiều hơn tổng lượng lỗi tồn tại ở vùng mỏng. Kết luận này **độc lập với mọi số đo**:
nó đúng ngay cả trước khi chạy bất cứ thứ gì.

### 5.3. Sai mẫu số so với plan

Plan **§3.7 điều 1** viết rõ: *"cải thiện outer-boundary error **trong vùng sụn mỏng** ~10%
tương đối trở lên"* — mẫu số là **vùng mỏng**, không phải toàn bề mặt.

| lớp | lỗi TB vùng mỏng | mục tiêu §3.7 (10%) | quy về ASSD tổng | = % trần |
|---|---|---|---|---|
| femoral_cart | 0.5950mm (CI [0.556, 0.636]) | 0.0595mm | 0.0058mm | **10%** |
| med_tib_cart | 0.4338mm (CI [0.390, 0.485]) | 0.0434mm | 0.0089mm | **10%** |

Plan đòi bắt **10% trần**. Cổng M0 cũ đòi **90–137% trần**. Lệch **9–14 lần**.

### 5.4. Ngưỡng 35% cũng đo sai thứ nó tuyên bố đo

Docstring `headroom.py` nêu lý do: *"vùng mỏng gom lỗi **bất tương xứng so với diện tích**"*.
Nhưng code kiểm **35% tuyệt đối**. Hai thứ khác nhau, và cho **thứ hạng ngược nhau**:

| | % diện tích | % error mass | tập trung | theo luật 35% | theo lý do đã nêu |
|---|---|---|---|---|---|
| femoral | 9.6% | 21.4% | **2.23×** | ❌ trượt | ✅ **mạnh nhất** |
| med_tib | 20.3% | 33.5% | 1.65× | ⚠️ suýt | ✅ đạt |

Số tuyệt đối bị chi phối bởi **lớp đó có bao nhiêu sụn mỏng**, không phải vùng mỏng có khó
bất tương xứng hay không.

---

## 6. Ràng buộc báo cáo (bắt buộc)

Mục tiêu §3.7 quy về ASSD tổng là **0.0058mm** (femoral) / **0.0089mm** (med_tib).
`ε_metric` đo được là **0.0033mm** / **0.0077mm**.

> **Mục tiêu nhỏ ngang ngửa sai khác giữa hai implementation metric.**

⇒ **ASSD tổng KHÔNG dùng để báo cáo hiệu ứng này được** — hiệu ứng chìm dưới độ rung định
nghĩa. Mọi báo cáo **bắt buộc** ở mẫu số **vùng mỏng** (0.0595 trên nền 0.595 = 10% sạch sẽ)
hoặc **presence F1 / thickness MAE**. Đây là ràng buộc kỹ thuật, không phải lựa chọn phong cách.

Điều này **khớp plan §9**, vốn đã ghi *"kiến trúc mới không nhất thiết phải tăng Dice tổng
nhiều"* và liệt kê tiêu chí thành công gồm §9.3 (giảm nhầm lẫn absent vs cực mỏng) và §9.4
(cải thiện ước lượng độ dày).

---

## 7. Cổng Gate 1 mới + quyết định

### 7.1. Bỏ cổng cũ

Ngưỡng `0.04 / 0.08` trên ASSD tổng bị **thay**, không phải hạ — lý do: so sai loại đại lượng
(§5.1) và bất khả thi về số học (§5.2). Ngưỡng 35% tuyệt đối thay bằng tỷ số tập trung (§5.4).

### 7.2. Cổng mới, suy trực tiếp từ §3.7

| # | Tiêu chí | femoral | med_tib |
|---|---|---|---|
| 1 | Trần ≥ mục tiêu §3.7 / tỷ lệ bắt 25% <br>(cần ≥0.023 / ≥0.036mm) | 0.058 ✅ | 0.089 ✅ |
| 2 | Tỷ số tập trung ≥ 1.5× | 2.23 ✅ | 1.65 ✅ |
| 3 | Bậc thang z không chi phối (<1) | 0.871 ✅ | 0.709 ✅ |
| 4 | Báo cáo ở mẫu số vùng mỏng / presence / thickness | bắt buộc | bắt buộc |

**⇒ QUYẾT ĐỊNH: PROCEED xây MVP.**

### 7.3. Giả định hiện rõ (thay cho giả định ẩn)

Cổng cũ **ngầm** giả định model bắt được ~29% trần mà không nói ra. Cổng mới giả định **25%**
và **nói rõ**. Đây là giả định **chưa kiểm chứng** — nó chỉ đo được **sau khi xây MVP**.

---

## 8. Giới hạn — điều M0 KHÔNG trả lời được

- **Trả lời được:** lỗi có tập trung bất tương xứng ở vùng mỏng không? → **Có**, chắc chắn
  (CI không chạm 0, metric-robust, tập trung 1.65–2.23×).
- **KHÔNG trả lời được:** model thật bắt được bao nhiêu % trần. Bất định thật sự đã **dời chỗ**
  từ *"có headroom không"* sang *"bắt được bao nhiêu"* — chỉ trả lời được sau khi xây.

> **PROCEED ở đây nghĩa là "có đủ căn cứ để xây MVP", KHÔNG phải "đã chứng minh phương pháp
> hiệu quả".**

Rủi ro còn lại đã biết:
- Femoral có >52% error mass ở sụn dày (>2mm) — phương pháp **không chạm tới** phần đó.
- Tỷ lệ bắt 25% là giả định, chưa kiểm chứng.
- Phantom (`tests/test_core.py`) cho thấy round-trip tọa độ **tự nó** mất 15% Dice ở sụn
  0.4mm → cổng M2 phải đặt trên **ASSD**, không phải Dice.

---

## 9. Lớp dẫn — giữ nguyên plan §3.1

**Không đảo** thứ tự plan. §3.1 chọn femoral làm lớp chính vì lý do **kỹ thuật** (bề mặt lớn,
phân đoạn xương ổn định, dễ debug pipeline) — không phải vì cỡ hiệu ứng.

- **femoral** — dựng và debug pipeline (M2/M3/M4/M5). Tập trung lỗi cao nhất (2.23×).
- **med_tib** — stress test + hiệu ứng chính báo cáo (trần 0.089, thin mass 33.5%).

---

## 10. Tái lập

| Artifact | Đường dẫn |
|---|---|
| Per-case Gate 0 (ASSD scipy, 404 ca) | `BSC_ROOT/runs/gate0_percase.csv` |
| Per-case ASSD SimpleITK (40 ca) | `BSC_ROOT/runs/gate0b_sitk_assd.csv` |
| Per-case M0 (prize/mass/stair, 103 ca) | `BSC_ROOT/runs/M0_percase.jsonl` |
| Per-case ε_metric (103 ca) | `BSC_ROOT/runs/M0_eps_metric.jsonl` |
| Tổng hợp Gate 1 | `BSC_ROOT/runs/M0_headroom_B0_zibTs.json` |
| Notebook | `bsc/notebooks/bsc_00_bootstrap.ipynb`, `bsc_01_headroom.ipynb` |
| Code | `bsc/core.py`, `bsc/metrics.py`, `bsc/headroom.py` |

Bootstrap: 10 000 lần, `seed=0`, paired per-case. Seed chỉ ảnh hưởng CI, không ảnh hưởng
điểm ước lượng.

---

## 11. Thay đổi so với tài liệu trước — ghi để minh bạch (§7 plan)

| Thay đổi | Lý do | Trước → Sau |
|---|---|---|
| Bỏ ngưỡng 0.08mm ASSD tổng | So sai loại đại lượng; đòi 137% lỗi tồn tại (femoral) ⇒ bất khả thi | 0.08mm → tiêu chí §7.2 |
| Bỏ ngưỡng 35% tuyệt đối | Đo sai thứ docstring tuyên bố đo; cho thứ hạng ngược | 35% tuyệt đối → tập trung ≥1.5× |
| Cấm báo cáo ASSD tổng | Mục tiêu (0.006mm) < sai khác implementation (0.003–0.008mm) | tự do → chỉ vùng mỏng/presence/thickness |
| Giả định tỷ lệ bắt | Cổng cũ giả định ~29% **ẩn** | ẩn 29% → **hiện rõ 25%** |

**Không thay đổi:** femoral vẫn là lớp dẫn kỹ thuật (§3.1); prize vẫn tính bằng `bsc.metrics`
scipy; bin độ dày vẫn theo plan §2.3.
