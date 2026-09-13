# QUYẾT ĐỊNH GATE 1 VÀ KẾ HOẠCH KIỂM CHỨNG TIẾP THEO

## 1. Kết luận điều hành

### Quyết định hiện tại

**RESCOPE THE CLAIM, PROCEED WITH THE MVP.**

Diễn giải bằng tiếng Việt:

> Không dừng hướng nghiên cứu sử dụng hệ tọa độ neo vào bề mặt xương. Tuy nhiên, cần thu hẹp tuyên bố khoa học và chỉ tiếp tục bằng một MVP được kiểm soát chặt chẽ, thay vì triển khai ngay Main Model hoàn chỉnh.

Kết quả Gate 1 hiện tại cho thấy:

1. Baseline ResEnc còn **headroom dương và ổn định** tại các vùng sụn mỏng hoặc mất hoàn toàn.
2. Lỗi của ResEnc có xu hướng **tập trung bất tương xứng** tại các vùng mục tiêu của kiến trúc surface-coordinate.
3. Kết luận headroom **không phụ thuộc đáng kể vào implementation ASSD**.
4. Tuy nhiên, counterfactual prize chỉ là **trần trên lý tưởng**, chưa chứng minh mô hình thực tế có thể thu hồi được bao nhiêu phần trăm headroom đó.
5. Vì vậy, kết quả đủ mạnh để bác bỏ quyết định `STOP`, nhưng chưa đủ để cam kết triển khai toàn bộ Main Model.

---

# 2. Kết quả Gate 1 hiện tại

## 2.1. Femoral cartilage

\[
\Delta ASSD_{\text{prize}} = 0.0582\text{ mm}
\]

Bootstrap 95% CI:

\[
[0.0529,\ 0.0643]\text{ mm}
\]

Thin + absent error mass:

\[
21.4\%
\]

Phân bố lỗi:

| Thickness bin | N | Mean distance (mm) | Error mass |
|---|---:|---:|---:|
| Absent | 2,079 | 0.991 | 9.2% |
| \(\leq 0.5\) mm | 749 | 0.596 | 1.7% |
| \(\leq 1.0\) mm | 6,186 | 0.416 | 10.5% |
| \(\leq 2.0\) mm | 25,889 | 0.248 | 26.1% |
| \(>2.0\) mm | 59,177 | 0.224 | 52.5% |

M0c z/in-plane ratio:

\[
0.871
\]

### Diễn giải

- Headroom khoảng 0.058 mm là **không nhỏ và CI hoàn toàn dương**.
- Error mass tuyệt đối ở thin + absent chỉ là 21.4%, nên không qua ngưỡng hard-coded 35%.
- Tuy nhiên, khi chuẩn hóa theo diện tích bề mặt của vùng thin + absent, error enrichment được ước tính khoảng:

\[
E_{\text{thin}} \approx 2.23
\]

Nghĩa là vùng thin + absent tạo ra lỗi nhiều hơn khoảng 2.23 lần so với mức kỳ vọng nếu lỗi phân bố tỷ lệ thuận với diện tích.

---

## 2.2. Medial tibial cartilage

\[
\Delta ASSD_{\text{prize}} = 0.0885\text{ mm}
\]

Bootstrap 95% CI:

\[
[0.0761,\ 0.1039]\text{ mm}
\]

Thin + absent error mass:

\[
33.5\%
\]

Phân bố lỗi:

| Thickness bin | N | Mean distance (mm) | Error mass |
|---|---:|---:|---:|
| Absent | 544 | 0.830 | 10.8% |
| \(\leq 0.5\) mm | 371 | 0.518 | 3.8% |
| \(\leq 1.0\) mm | 2,958 | 0.324 | 18.8% |
| \(\leq 2.0\) mm | 9,404 | 0.200 | 40.1% |
| \(>2.0\) mm | 5,826 | 0.209 | 26.4% |

M0c z/in-plane ratio:

\[
0.709
\]

### Diễn giải

- Headroom gần hoặc vượt ngưỡng 0.08 mm tùy cohort và revision code.
- Thin + absent error mass là 33.5%, gần ngưỡng 35%.
- Error enrichment được ước tính khoảng:

\[
E_{\text{thin}} \approx 1.65
\]

Nghĩa là lỗi ở vùng mục tiêu cao hơn khoảng 1.65 lần so với mức kỳ vọng theo diện tích.

---

# 3. Lỗi triển khai trong tiêu chí 35%

## 3.1. Điều mà tài liệu tuyên bố muốn đo

Phần mô tả của `headroom.py` nói rằng vùng sụn mỏng cần tạo ra lượng lỗi:

> bất tương xứng so với diện tích của vùng đó.

Tiêu chí đúng phải so sánh:

\[
\text{Error mass fraction}
\]

với:

\[
\text{Surface-area fraction}
\]

Metric phù hợp là:

\[
E_{\text{thin}}
=
\frac{
\text{thin+absent error-mass fraction}
}{
\text{thin+absent surface-area fraction}
}
\]

## 3.2. Điều code hiện tại thực sự đo

Code hiện tại chỉ tính:

\[
\text{thin+absent error-mass fraction}
\]

sau đó so trực tiếp với:

\[
0.35
\]

Nó không đo mức độ bất tương xứng theo diện tích.

Do đó, ngưỡng 35% không thực hiện đúng đặc tả khoa học của chính nó.

## 3.3. Hệ quả

Theo absolute error-mass fraction:

- Medial tibial được xếp cao hơn femoral.
- Femoral không qua ngưỡng 35%.

Theo enrichment ratio:

- Femoral khoảng 2.23.
- Medial tibial khoảng 1.65.

Thứ hạng bị đảo ngược.

Điều này cho thấy ngưỡng 35% không chỉ là một lựa chọn chưa tối ưu, mà có thể dẫn đến kết luận sai về nơi lỗi thực sự tập trung bất tương xứng.

---

# 4. Cách sửa tiêu chí error concentration

## 4.1. Không thay 35% bằng một con số tuyệt đối khác

Không nên hạ 35% xuống 20% hoặc 30% chỉ để làm kết quả hiện tại thành `PROCEED`.

Đó sẽ là post-hoc threshold tuning.

## 4.2. Thay metric absolute fraction bằng enrichment ratio

Sử dụng:

\[
E_{\text{thin}}
=
\frac{
f_{\text{error, thin}}
}{
f_{\text{area, thin}}
}
\]

Trong đó:

- \(f_{\text{error, thin}}\): tỷ lệ tổng surface-distance error nằm ở thin + absent.
- \(f_{\text{area, thin}}\): tỷ lệ tổng surface points thuộc thin + absent.

Working interpretation:

| Enrichment ratio | Diễn giải |
|---:|---|
| \(E<1\) | Vùng thin không tập trung lỗi |
| \(E\approx1\) | Lỗi tỷ lệ thuận với diện tích |
| \(1<E<1.5\) | Enrichment nhẹ |
| \(1.5\leq E<2\) | Enrichment đáng chú ý |
| \(E\geq2\) | Enrichment mạnh |

Các mức này là working thresholds và cần được kiểm chứng bằng bootstrap ở case level.

## 4.3. Báo cáo cả absolute fraction và enrichment

Không bỏ absolute error-mass fraction.

Nên báo cáo đồng thời:

- Thin + absent surface-area fraction.
- Thin + absent error-mass fraction.
- Enrichment ratio.
- Bootstrap CI của enrichment.

---

# 5. Diễn giải đúng về ngưỡng prize 0.08 mm

## 5.1. Prize là upper bound, không phải expected gain

`prize_counterfactual` giả định rằng toàn bộ lỗi ở các bin:

- `absent`
- `<=0.5 mm`
- `<=1.0 mm`

được sửa hoàn hảo.

Do đó:

\[
H=\Delta ASSD_{\text{counterfactual}}
\]

là headroom tối đa lý tưởng.

Model thật chỉ có thể thu hồi một tỷ lệ:

\[
q\in[0,1]
\]

Expected gain:

\[
\Delta ASSD_{\text{actual}}=qH
\]

## 5.2. Ngưỡng 0.08 mm đang ẩn một giả định

Nếu target practical gain là khoảng:

\[
0.023\text{ mm}
\]

và headroom yêu cầu là:

\[
0.08\text{ mm}
\]

thì tỷ lệ thu hồi giả định là:

\[
q=\frac{0.023}{0.08}\approx29\%
\]

Giả định 29% này chưa được nêu và chưa được kiểm chứng.

## 5.3. Không nên hạ 0.08 mm theo kết quả hiện tại

Giữ lại 0.08 mm trong lịch sử thí nghiệm như một **strong-investment threshold**.

Nhưng cần diễn giải lại:

> 0.08 mm không phải ngưỡng xác nhận hay phủ nhận giả thuyết surface-coordinate. Nó là ngưỡng headroom đủ lớn để biện minh cho việc đầu tư trực tiếp vào một kiến trúc phức tạp.

Headroom thấp hơn 0.08 nhưng CI dương và enrichment rõ vẫn có thể biện minh cho một MVP.

---

# 6. Metric-definition robustness

Kết quả so sánh prize giữa SciPy và SimpleITK:

| Class | N | SciPy prize | SimpleITK prize | \(\epsilon\) | \(|\epsilon|/\text{prize}\) | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---|
| Femoral cartilage | 15 | 0.0550 | 0.0525 | +0.0025 | 5% | [0.0020, 0.0031] |
| Medial tibial cartilage | 15 | 0.0794 | 0.0709 | +0.0085 | 11% | [0.0060, 0.0113] |

Trong đó:

\[
\epsilon_{\text{metric}}
=
Prize_{\text{SciPy}}
-
Prize_{\text{SITK}}
\]

## 6.1. Kết luận

- Hai implementation cho prize cùng dấu.
- \(|\epsilon|/\text{prize}\) chỉ 5% và 11%.
- Metric-definition difference nhỏ hơn nhiều so với headroom.
- Kết luận Gate 1 không bị đảo chiều.

Vì vậy:

> Có thể chọn SciPy làm canonical metric contract và tiếp tục. Kết luận headroom hiện tại không phụ thuộc đáng kể vào implementation ASSD.

## 6.2. Working robustness rule

### Robust

- Hai prize cùng dấu.
- \(|\epsilon|/\text{prize}<20\%\).
- Model ranking không thay đổi.

### Metric-sensitive

- \(|\epsilon|/\text{prize}>50\%\).
- Hai prize khác dấu.
- Model ranking thay đổi.

Vùng 20–50% nên được xem là gray zone và cần kiểm tra thêm.

---

# 7. Chênh lệch giữa hai bộ số liệu cần được làm rõ

Hai bảng hiện đang cho point estimates khác nhau:

| Class | Gate 1 output | Metric-robustness SciPy |
|---|---:|---:|
| Femoral | 0.0582 mm | 0.0550 mm |
| Medial tibial | 0.0885 mm | 0.0794 mm |

Trước khi khóa quyết định, cần xác nhận hai bảng có sử dụng cùng:

- Cohort.
- Số case.
- Inclusion criteria.
- NaN handling.
- Prediction files.
- Thickness assignment.
- Surface extraction.
- Metric revision.
- Label mapping.
- Per-case aggregation.

Nếu khác nhau, phải ghi rõ tên cohort và version code trong báo cáo.

Không nên sử dụng hai con số như thể chúng là cùng một estimate.

---

# 8. Quyết định khoa học hiện tại

## 8.1. Không STOP

Không nên dừng hướng nghiên cứu vì:

1. Counterfactual headroom có CI hoàn toàn dương.
2. Enrichment ratio lớn hơn 1 ở cả hai class.
3. Femoral có enrichment mạnh khoảng 2.23.
4. Medial tibial có headroom tuyệt đối lớn.
5. Metric-definition robustness tốt.
6. M0c ratio dưới 1 và chưa cho thấy z-staircase thống trị toàn bộ lỗi.
7. Tín hiệu quan sát được không giống random noise.

## 8.2. Chưa triển khai Main Model hoàn chỉnh

Chưa có bằng chứng rằng model thực tế có thể thu hồi đủ tỷ lệ headroom.

Các module sau chưa nên triển khai ngay:

- Surface Graph Transformer.
- Cross-surface attention.
- Conditional implicit field.
- Multi-ray cone.
- Learned normal correction.
- End-to-end joint fine-tuning.

## 8.3. Tiếp tục MVP

MVP nên kiểm tra duy nhất câu hỏi:

> Một 1D normal-ray refinement model đơn giản, với GT bone surface, có thu hồi được một phần có ý nghĩa của counterfactual headroom hay không?

---

# 9. Rescope điều gì?

## 9.1. Không rescope core direction

Không chuyển sang một hướng nghiên cứu hoàn toàn khác.

Core direction vẫn là:

> cartilage refinement trong hệ tọa độ neo vào bề mặt xương.

## 9.2. Rescope scientific claim

### Claim quá rộng

> Surface-coordinate segmentation will substantially improve overall cartilage ASSD.

### Claim phù hợp với dữ liệu hiện tại

> Surface-coordinate refinement targets the disproportionate boundary error occurring in locally thin or absent cartilage regions, with particular emphasis on presence detection and submillimetre boundary localization.

## 9.3. Rescope primary outcomes

Primary outcomes của MVP:

- ASSD improvement trong các bin `absent`, `<=0.5 mm`, `<=1.0 mm`.
- Outer-boundary MAE.
- Surface Dice @ 0.5 mm ở thin regions.
- Presence/absence F1.
- False-positive cartilage ở absent regions.
- Thin-cartilage recall.
- Headroom capture ratio.

Secondary outcomes:

- Overall ASSD.
- Overall Dice.
- Overall Surface Dice.
- Volume error.

---

# 10. Những điều cần kiểm chứng trong MVP

## 10.1. Coordinate-system validity

### Round-trip test

1. Project GT cartilage vào normal-ray coordinates.
2. Reconstruct lại volume.
3. So sánh reconstructed GT với original GT.

Kiểm chứng:

- Coordinate transformation có làm mất thông tin đáng kể không?
- Rasterization có tạo bias không?
- Femoral có chịu reconstruction error cao hơn tibial không?

Yêu cầu trước khi train:

- Round-trip Dice cao.
- Boundary error nhỏ hơn expected MVP gain.
- Không có systematic loss tại cartilage margins.

---

## 10.2. Surface-normal advantage

So sánh:

- Surface-normal rays.
- Axial rays.
- Random-direction rays.
- Tangential rays.

Câu hỏi:

> Improvement đến từ anatomical coordinate system hay chỉ từ việc thêm một local network?

Normal-ray model phải tốt hơn negative controls ở thin-region boundary metrics.

---

## 10.3. Oracle-surface performance

Sử dụng GT bone surface.

Câu hỏi:

> Khi loại bỏ bone-segmentation error, ray model có thật sự cải thiện cartilage boundary không?

Nếu GT-surface model không cải thiện, không nên xây Main Model.

---

## 10.4. Predicted-surface degradation

Sau oracle test, dùng predicted bone surfaces.

Đo:

\[
\text{Retained gain}
=
\frac{
\Delta ASSD_{\text{predicted surface}}
}{
\Delta ASSD_{\text{GT surface}}
}
\]

Cần xác định:

- Bone error làm mất bao nhiêu gain?
- Normal error có phải bottleneck chính không?
- Có cần multi-ray hoặc jitter augmentation không?

---

## 10.5. Thin versus absent discrimination

Đánh giá riêng:

- True absence.
- Extremely thin cartilage.
- Thin cartilage.
- Non-articular surface.

Câu hỏi:

> Model có phân biệt được sụn mất hoàn toàn với sụn còn cực mỏng không?

Metric:

- Presence F1.
- AUPRC.
- False-positive rate ở absent regions.
- Recall ở extremely thin regions.
- Calibration.

---

## 10.6. Error localization

So sánh improvement theo thickness bins:

| Bin | Câu hỏi |
|---|---|
| Absent | Có giảm hallucinated cartilage không? |
| \(\leq0.5\) mm | Có giữ được sụn cực mỏng không? |
| \(\leq1.0\) mm | Có giảm boundary error không? |
| \(\leq2.0\) mm | Có benefit phụ không? |
| \(>2.0\) mm | Có làm hỏng vùng ResEnc vốn đã tốt không? |

MVP chỉ được xem là phù hợp nếu gain tập trung đúng ở các bin mục tiêu.

---

## 10.7. Geometry sensitivity

Kiểm tra theo:

- Curvature.
- Surface orientation.
- z-normal component.
- Cartilage class.
- Joint compartment.
- Bone-surface error.

Câu hỏi:

> Femoral enrichment cao là do surface geometry phức tạp, hay do một artefact của thickness projection?

---

# 11. Headroom Capture Ratio — Gate quan trọng tiếp theo

Định nghĩa:

\[
q_{\text{capture}}
=
\frac{
ASSD_{\text{baseline}}
-
ASSD_{\text{MVP}}
}{
\Delta ASSD_{\text{counterfactual}}
}
\]

Đây là tỷ lệ headroom lý tưởng mà MVP thực sự thu hồi được.

Working interpretation:

| Capture ratio | Diễn giải |
|---:|---|
| \(q<10\%\) | Surface-coordinate khó chuyển headroom thành gain thực |
| \(10\%\leq q<25\%\) | Signal yếu đến vừa; cần rescope thêm |
| \(25\%\leq q<50\%\) | Signal tốt; hợp lý để xây Main Model |
| \(q\geq50\%\) | Signal rất mạnh |

Các ngưỡng này là working thresholds, cần báo cáo cùng bootstrap CI.

## Ví dụ medial tibial

Nếu headroom là:

\[
H=0.0794\text{ mm}
\]

thì:

| Capture | Actual gain |
|---:|---:|
| 10% | 0.0079 mm |
| 25% | 0.0199 mm |
| 29% | 0.0230 mm |
| 50% | 0.0397 mm |

MVP là bước trực tiếp kiểm chứng giả định model có thể thu hồi khoảng 29% headroom hay không.

---

# 12. Gate framework được đề xuất lại

## Gate 1A — Target-region enrichment

\[
E_{\text{thin}}
=
\frac{
f_{\text{error, thin}}
}{
f_{\text{area, thin}}
}
\]

Yêu cầu:

- Point estimate \(>1\).
- Bootstrap CI ưu tiên nằm trên 1.
- Báo cáo theo case, class và KL group.

## Gate 1B — Counterfactual headroom

\[
H=\Delta ASSD_{\text{counterfactual}}
\]

Yêu cầu:

- Headroom dương.
- Bootstrap CI không bao gồm 0.
- Không bị chi phối bởi một số ít outlier.

## Gate 1C — Metric-definition robustness

\[
R_{\text{metric}}
=
\frac{
|\epsilon_{\text{metric}}|
}{
|Prize_{\text{canonical}}|
}
\]

Yêu cầu:

- Hai prize cùng dấu.
- \(R_{\text{metric}}<20\%\) là robust working zone.

## Gate 2 — MVP headroom capture

\[
q_{\text{capture}}
=
\frac{
\Delta ASSD_{\text{MVP}}
}{
H
}
\]

Quyết định:

| Điều kiện | Quyết định |
|---|---|
| Không enrichment hoặc headroom gần 0 | STOP |
| Enrichment + headroom dương | PROCEED TO MVP |
| MVP capture thấp | RESCOPE hoặc STOP |
| MVP capture khoảng 25% trở lên | PROCEED TO MAIN MODEL |
| MVP capture mạnh và ổn định | STRONG PROCEED |

---

# 13. Ưu tiên class cho MVP

## Primary MVP: medial tibial cartilage

Lý do:

- Headroom tuyệt đối lớn hơn.
- Thin + absent error mass cao hơn.
- Geometry tương đối đơn giản.
- Phù hợp để chứng minh proof of concept nhanh.

## Secondary validation: femoral cartilage

Lý do:

- Error enrichment mạnh hơn.
- Geometry cong và phức tạp.
- Phù hợp để kiểm tra lợi ích của orientation canonicalization.
- Có thể bộc lộ các failure modes về normal và reconstruction.

Không nên bắt đầu chỉ với femoral vì complexity cao có thể che khuất tín hiệu proof of concept.

---

# 14. Side tests bắt buộc trước khi quyết định Main Model

1. Synthetic shell phantom.
2. GT coordinate round-trip.
3. Normal orientation validation.
4. Single-interval occupancy analysis.
5. Tiny-set overfit.
6. Negative-direction controls.
7. Surface-position jitter.
8. Normal-angle jitter.
9. GT versus predicted bone surfaces.
10. Thin-versus-absent confusion analysis.
11. Per-bin improvement analysis.
12. Capture-ratio bootstrap.
13. Reconcile hai bộ point estimates hiện tại.
14. Full logging theo experiment ID, cohort và code commit.

---

# 15. Quyết định cuối cùng

## Quyết định

\[
\boxed{
\text{PROCEED TO A FOCUSED MVP, RESCOPE THE CLAIM}
}
\]

## Không làm ngay

- Không xây Main Model hoàn chỉnh.
- Không thêm graph, implicit field hoặc cross-surface attention.
- Không sửa ngưỡng 0.08 để phù hợp kết quả.
- Không thay 35% bằng một ngưỡng absolute khác.
- Không dùng overall Dice làm primary success criterion.

## Làm tiếp

1. Sửa metric concentration thành enrichment ratio.
2. Giữ full audit trail của logic Gate 1 cũ và mới.
3. Khóa canonical SciPy ASSD contract.
4. Reconcile các bộ số liệu 0.0582/0.0550 và 0.0885/0.0794.
5. Triển khai MVP đơn giản trên medial tibial cartilage.
6. Validate lại trên femoral cartilage.
7. Tính headroom capture ratio.
8. Chỉ mở Main Model nếu MVP thu hồi đủ headroom và gain đúng ở thin/absent regions.

---

# 16. Câu kết luận đề xuất cho báo cáo

> Baseline ResEnc errors are disproportionately enriched in thin and absent cartilage regions. Counterfactual analysis identifies a positive and metric-robust ASSD headroom for both femoral and medial tibial cartilage. These findings justify a focused surface-normal MVP targeting local cartilage presence and submillimetre boundary refinement. They do not yet justify the complete surface-graph architecture or a claim of large overall ASSD improvement. Progression to the Main Model will depend on the proportion of counterfactual headroom recovered by the MVP.
