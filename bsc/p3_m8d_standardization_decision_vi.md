# QUYẾT ĐỊNH CHUẨN HÓA P3 VÀ M8-D TRONG KẾ HOẠCH MVP

## 1. Bối cảnh

Sau khi chạy lại phân tích metric robustness trên toàn bộ 103 ca, kết quả thu được:

| Class | N | SciPy prize | SimpleITK prize | \(\epsilon\) | \(|\epsilon|/\text{prize}\) | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---|
| Femoral cartilage | 103 | 0.0582 mm | 0.0549 mm | +0.0033 mm | 6% | [0.0029, 0.0037] |
| Medial tibial cartilage | 103 | 0.0885 mm | 0.0808 mm | +0.0077 mm | 9% | [0.0068, 0.0086] |

Hai implementation cho cùng dấu prize, và độ chênh chỉ chiếm 6–9% giá trị prize.

Kết luận:

> Gate 1 không phụ thuộc đáng kể vào định nghĩa ASSD. Có thể khóa SciPy làm canonical metric contract và tiếp tục theo quyết định: **rescope scientific claim, proceed with focused MVP**.

Khi bắt đầu dựng MVP, xuất hiện một điểm không nhất quán trong plan:

- `M8-D`: MRI + coarse cartilage probability.
- `P3`: MRI + coarse logits, Atlas domain, occupancy + presence.

Hai cấu hình này trông gần như giống nhau, dẫn đến câu hỏi liệu chúng có phải hai thí nghiệm khác nhau hay không.

---

## 2. Vấn đề cốt lõi

### 2.1. M8-D và P3 đang thuộc hai cấp mô tả khác nhau

#### M8-D là một câu hỏi ablation

M8-D thuộc nhóm **input ablation**.

Câu hỏi nó muốn trả lời là:

> Khi giữ cố định surface source, articular domain, output heads và training protocol, việc thêm coarse cartilage prior từ ResEnc có cải thiện mô hình so với MRI-only hay không?

Do đó, M8-D mô tả **một biến đầu vào cần kiểm thử**, không phải một cấu hình pipeline đầy đủ.

#### P3 là một cấu hình thí nghiệm hoàn chỉnh

P3 xác định đầy đủ:

- Nguồn bone surface.
- Articular domain.
- Input channels.
- Output heads.
- Training setup.

P3 là một run cụ thể có thể được train, lưu checkpoint và báo cáo metric.

Do đó:

\[
\boxed{
\text{M8-D là câu hỏi ablation; P3 là run dùng để trả lời câu hỏi đó}
}
\]

Hai tên không nên được hiểu là hai experiment độc lập.

---

## 3. Vì sao giữ cả hai như hai run riêng là không hợp lý?

### 3.1. Trùng lặp thí nghiệm

Nếu M8-D và P3 cùng sử dụng:

- GT bone surface.
- Fold-specific atlas.
- MRI.
- Coarse cartilage prior.
- Occupancy + presence heads.
- Cùng training protocol.

thì chúng là cùng một cấu hình.

Train cả hai sẽ:

- Tốn compute không cần thiết.
- Tạo hai experiment ID cho cùng một giả thuyết.
- Gây khó khăn khi tổng hợp bảng kết quả.
- Có nguy cơ xuất hiện hai kết quả hơi khác nhau do seed, dù bản chất cấu hình giống nhau.
- Làm audit trail khó hiểu.

### 3.2. Làm mờ ý nghĩa của ablation

Ablation chỉ hợp lệ khi mỗi cặp so sánh thay đổi đúng một yếu tố.

Ví dụ, để đo đóng góp của coarse ResEnc prior:

\[
S0 + D1 + I0 + H1
\]

so với:

\[
S0 + D1 + I3 + H1
\]

Trong đó chỉ input thay đổi từ MRI-only sang MRI + coarse probability.

Nếu M8-D và P3 đồng thời khác nhau ở:

- Articular domain.
- Presence head.
- Input representation.
- Surface source.

thì không thể kết luận improvement đến từ coarse prior.

### 3.3. Dễ dẫn đến báo cáo sai

Nếu M8-D được mô tả như một input ablation nhưng P3 được báo cáo như một model riêng, người đọc có thể hiểu rằng:

- M8-D chỉ là thêm một channel.
- P3 là một kiến trúc khác.

Trong khi thực tế P3 chỉ là implementation cụ thể của M8-D trong scaffold đã chọn.

---

## 4. Coarse probability và coarse logits không phải cùng một thứ

Đây là điểm cần chuẩn hóa rõ ràng.

### 4.1. Target-class probability

\[
p_c(x)=\operatorname{softmax}(z(x))_c
\]

Đặc điểm:

- Nằm trong khoảng \([0,1]\).
- Dễ diễn giải.
- Dễ kiểm tra calibration.
- Dễ ensemble giữa nhiều folds.
- Ít nhạy với scale nội bộ của network.
- Có thể bị saturation ở gần 0 hoặc 1.

### 4.2. Raw target-class logit

\[
z_c(x)
\]

Đặc điểm:

- Không bị giới hạn.
- Giữ thông tin confidence thô.
- Scale có thể khác giữa fold hoặc checkpoint.
- Không phù hợp để average trực tiếp giữa các model nếu chưa calibration.
- Có thể khiến ray model học scale đặc thù của ResEnc thay vì học anatomy.

### 4.3. Logit margin hoặc log-odds

Một lựa chọn giàu thông tin hơn raw probability:

\[
m_c(x)
=
z_c(x)
-
\log\sum_{j\neq c}e^{z_j(x)}
\]

Nó biểu diễn bằng chứng của target cartilage so với các class cạnh tranh.

Tuy nhiên, đây nên là extension hoặc ablation sau MVP, không phải input mặc định ban đầu.

---

## 5. Quyết định input chuẩn cho MVP

### 5.1. Chọn ensemble mean target-cartilage probability

Cấu hình canonical:

\[
p_c^{ens}(x)
=
\frac{1}{K}
\sum_{k=1}^{K}p_c^{(k)}(x)
\]

Trong đó \(K\) là số fold hoặc số model trong ensemble.

#### Lý do

1. Backbone ResEnc đang frozen.
2. Five-fold probabilities có thể average một cách có ý nghĩa.
3. Probability ổn định hơn raw logits giữa các folds.
4. Dễ giải thích và visualize.
5. Ít đưa thêm biến calibration vào proof of concept.
6. MVP nên kiểm tra giá trị của surface coordinate, không nên đồng thời kiểm tra logit scaling.
7. Nếu probability-based MVP thất bại, việc đổi sang raw logits khó có khả năng cứu được giả thuyết hình học một cách thuyết phục.

### 5.2. Không dùng từ “logits” thay thế cho “probability”

Từ giờ, plan và code phải ghi chính xác:

- `target_cartilage_probability`
- hoặc `target_cartilage_logit_margin`

Không dùng chung chung `coarse logits` nếu input thực tế là probability.

---

## 6. Cách chuẩn hóa experiment design

Thay vì để P-series và M8-series tồn tại như hai danh sách run song song, nên tách thành các trục cấu hình độc lập.

### 6.1. Surface source

| Code | Surface source |
|---|---|
| S0 | Ground-truth bone surface |
| S1 | Ground-truth surface + controlled jitter |
| S2 | Predicted bone surface |

### 6.2. Articular domain

| Code | Domain |
|---|---|
| D0 | Oracle cartilage-derived articular domain |
| D1 | Fold-specific population atlas |

### 6.3. Input channels

| Code | Input |
|---|---|
| I0 | MRI only |
| I1 | MRI + gradient |
| I2 | MRI + bone signed-distance field |
| I3 | MRI + ensemble target-cartilage probability |
| I4 | MRI + high-resolution ResEnc features |
| I5 | MRI + probability + ResEnc features |

### 6.4. Output heads

| Code | Output |
|---|---|
| H0 | Occupancy only |
| H1 | Occupancy + presence |

---

## 7. Experiment matrix được chuẩn hóa

| Experiment | Surface | Domain | Input | Head | Mục đích |
|---|---|---|---|---|---|
| P0 | S0 | D0 | I0 | H0 | Oracle-coordinate baseline |
| P1 | S0 | D0 | I1 | H0 | Kiểm tra giá trị của gradient |
| P2 | S0 | D1 | I2 | H1 | Atlas + geometry-only prior |
| P3 | S0 | D1 | I3 | H1 | Kiểm tra giá trị của coarse ResEnc probability |
| P4 | S1 | D1 | Best input | H1 | Surface-jitter robustness |
| P5 | S2 | D1 | Best input | H1 | Predicted-surface pilot |

Trong framework này:

\[
\boxed{
\text{P3 là run chính thức tạo ra kết quả M8-D}
}
\]

M8-D không được train như một run riêng.

---

## 8. Cách định nghĩa M8 sau khi chuẩn hóa

M8 không còn là danh sách experiment độc lập.

M8 là tên của một **analysis group**:

> Input ablation dưới một scaffold cố định.

Ví dụ scaffold:

\[
S0 + D1 + H1
\]

Các cấu hình:

| Input ablation | Cấu hình |
|---|---|
| M8-A | S0-D1-I0-H1 |
| M8-B | S0-D1-I1-H1 |
| M8-C | S0-D1-I2-H1 |
| M8-D | S0-D1-I3-H1 |
| M8-E | S0-D1-I4-H1 |

Do đó:

- P3 là tên run trong experiment registry.
- M8-D là vai trò của P3 trong bảng ablation.

Không tạo checkpoint `M8-D` riêng nếu checkpoint `P3` đã tồn tại.

---

## 9. Cấu hình P3 được chốt

### P3 — Coarse-prior MVP configuration

#### Surface source

Ground-truth bone surface.

\[
S0
\]

#### Articular domain

Fold-specific population atlas.

\[
D1
\]

#### Ray input

- MRI intensity.
- Bone signed-distance field nếu đây là channel nền đã được giữ cố định trong scaffold.
- Ensemble target-cartilage probability từ frozen ResEnc.

\[
I3
\]

#### Outputs

- Cartilage occupancy profile.
- Cartilage-presence probability.

\[
H1
\]

#### Mục đích

> Định lượng giá trị bổ sung của coarse ResEnc cartilage prior khi ray model đã được đặt trong cùng một surface-coordinate scaffold.

#### So sánh hợp lệ

P3 phải được so với một cấu hình giống hệt, chỉ bỏ coarse probability:

\[
S0+D1+I0+H1
\]

hoặc, nếu bone SDF là input nền bắt buộc:

\[
S0+D1+I2+H1
\]

so với:

\[
S0+D1+(I2+\text{probability})+H1
\]

Cần ghi rõ input nền để tránh việc I3 vô tình vừa thêm probability vừa bỏ SDF.

---

## 10. Những ablation cần giữ tách biệt

### 10.1. Giá trị của coarse prior

Giữ cố định:

- Surface.
- Atlas.
- Presence head.
- Loss.
- Training cases.
- Seed policy.
- Reconstruction.

Chỉ thay:

- Không coarse prior.
- Có coarse probability.

### 10.2. Giá trị của presence head

Giữ input cố định:

\[
S0+D1+I3
\]

So sánh:

\[
H0
\]

với:

\[
H1
\]

### 10.3. Giá trị của atlas domain

Giữ input và heads cố định.

So sánh:

\[
D0
\]

với:

\[
D1
\]

### 10.4. Giá trị của predicted surface

Giữ input, domain và heads cố định.

So sánh:

\[
S0
\]

với:

\[
S2
\]

---

## 11. Quy tắc đặt tên experiment

Tên run nên thể hiện cấu hình thực tế.

Ví dụ:

```text
MVP_MTC_S0_D1_I3_H1_Fold0_Seed1
```

Diễn giải:

- `MVP`: giai đoạn.
- `MTC`: medial tibial cartilage.
- `S0`: GT bone surface.
- `D1`: fold-specific atlas.
- `I3`: MRI + target-cartilage probability.
- `H1`: occupancy + presence.
- `Fold0`.
- `Seed1`.

Có thể thêm version:

```text
MVP_MTC_S0_D1_I3_H1_v1_Fold0_Seed1
```

P3 có thể được lưu trong metadata:

```yaml
plan_alias: P3
ablation_roles:
  - M8-D
```

Như vậy một run có thể có:

- Một ID kỹ thuật duy nhất.
- Một alias trong plan.
- Một hoặc nhiều vai trò trong bảng ablation.

---

## 12. Audit trail cần ghi lại

Trong experiment registry, lưu:

- Experiment ID.
- Plan alias.
- Ablation role.
- Exact input channels.
- Probability hay logit.
- Ensemble method.
- Number of folds.
- Calibration method nếu có.
- Surface source.
- Atlas version.
- Output heads.
- Loss terms.
- Code commit.
- Dataset revision.
- Fold.
- Seed.

Không ghi chỉ `coarse prior` vì quá mơ hồ.

Ví dụ:

```yaml
experiment_id: MVP_MTC_S0_D1_I3_H1_Fold0_Seed1
plan_alias: P3
ablation_role: M8-D
surface_source: gt_bone
articular_domain: fold_specific_atlas
inputs:
  - mri_intensity
  - bone_sdf
  - resenc_ensemble_target_cartilage_probability
resenc_prior:
  representation: probability
  ensemble: arithmetic_mean
  n_models: 5
outputs:
  - occupancy
  - presence
```

---

## 13. Quyết định cuối cùng

### Quyết định 1

\[
\boxed{
\text{P3 và M8-D không phải hai experiment độc lập}
}
\]

P3 là run cụ thể; M8-D là câu hỏi input ablation mà P3 trả lời.

### Quyết định 2

\[
\boxed{
\text{Chỉ train và lưu một cấu hình canonical}
}
\]

Không tạo checkpoint riêng cho M8-D nếu P3 đã đại diện cho cùng cấu hình.

### Quyết định 3

\[
\boxed{
\text{Dùng ensemble target-cartilage probability trong MVP}
}
\]

Không dùng raw logits làm mặc định.

### Quyết định 4

\[
\boxed{
\text{Mỗi ablation chỉ thay đúng một yếu tố}
}
\]

Không thay đồng thời input, domain và output heads.

### Quyết định 5

\[
\boxed{
\text{P-series là run registry; M-series là analysis/test grouping}
}
\]

Điều này giúp:

- Không duplicate compute.
- Không duplicate checkpoint.
- Không nhầm mục tiêu khoa học.
- Dễ audit.
- Dễ viết bảng ablation.
- Dễ tái lập thí nghiệm.

---

## 14. Việc cần làm tiếp theo

1. Sửa plan để P3 ghi `ensemble target-cartilage probability`, không ghi `coarse logits`.
2. Thêm bảng mapping giữa P-series và M-series.
3. Xác định input nền của mọi M8 run:
   - MRI only.
   - MRI + SDF.
   - MRI + probability.
   - MRI + SDF + probability.
4. Đảm bảo M8-A và M8-D chỉ khác đúng coarse prior.
5. Cập nhật experiment registry.
6. Không train duplicate M8-D.
7. Chạy P3 như một phần của input-ablation matrix.
8. Sau MVP, mới thử logit margin như một extension nhỏ.

---

## 15. Câu kết luận ngắn cho tài liệu dự án

> P3 and M8-D represent the same underlying configuration at two different levels of the experimental plan. M8-D defines the input-ablation question, whereas P3 is the concrete experiment used to answer it. To avoid duplicate training and confounded comparisons, the project will retain one canonical P3 run using the ensemble target-cartilage probability from the frozen ResEnc model. The M8-D result will be derived from this run within the fixed input-ablation scaffold.
