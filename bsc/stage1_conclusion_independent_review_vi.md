# REVIEW STAGE 1 CONCLUSION — QUYẾT ĐỊNH SAU PHASE B

## 1. Kết luận điều hành

Sau khi xem toàn bộ `STAGE1_CONCLUSION.md`, quyết định hợp lý nhất là:

\[
\boxed{\text{STOP PHASE B CHO RAY-ONLY 1D MVP HIỆN TẠI}}
\]

Nhưng **chưa nên diễn giải thành**:

\[
\boxed{\text{“surface-coordinate direction đã thất bại”}}
\]

và cũng chưa đủ bằng chứng để khẳng định:

> “nút thắt nằm ở tín hiệu MRI tại sụn cực mỏng.”

Kết quả hiện tại chứng minh chắc hơn một phát biểu hẹp:

> **Một ray encoder 1D cục bộ, dùng MRI/SDF và các biến thể loss/sampling đã thử, không chuyển được lợi thế hình học của normal coordinates thành hiệu năng cạnh tranh với ResEnc ở femoral cartilage.**

Đây là một negative result có giá trị cho **kiến trúc ray-only hiện tại**.

Trước khi đóng toàn bộ hướng surface-coordinate refinement, có một phép thử còn rất quan trọng:

\[
\boxed{\text{P3/M8-D với clean OOF ResEnc probability}}
\]

vì kiến trúc chính ban đầu là refinement/fusion với ResEnc, không phải thay ResEnc bằng một mạng 1D chỉ đọc local MRI.

---

# 2. Những kết luận hiện tại mình đồng ý mạnh

## 2.1. P2 thất bại rõ ràng

Canonical checkpoint:

- P2 thin-region error: **2.188 mm**
- B0: **0.552 mm**
- tốt hơn baseline: **0/10 ca**

Khoảng cách này rất lớn.

Dù metric mapping thay đổi một ít, rất khó có khả năng lật thành P2 tốt hơn B0.

Do đó:

\[
\boxed{\text{P2 hiện tại không đạt Gate §3.7}}
\]

Điểm này đủ vững để dừng việc tối ưu nhỏ quanh P2.

---

## 2.2. Coordinate transform/splat không phải trần chính

Oracle reconstruction:

\[
0.0256\text{ mm}
\]

so với B0:

\[
0.5517\text{ mm}
\]

cho thấy pipeline projection + reconstruction có thể biểu diễn target với boundary error rất nhỏ.

Kết luận hợp lệ:

> **Representation không tự áp đặt một floor khoảng 1–2 mm.**

Không nên diễn giải 0.0256 mm là mức model thực tế có thể đạt từ MRI.

---

## 2.3. Normal coordinates có tín hiệu dương

Normal rays tốt hơn:

- random: +0.330 mm
- tangent: +0.982 mm
- axial: +1.689 mm

và CI paired bootstrap đều dương.

Điều này hỗ trợ kết luận:

> Normal-coordinate pipeline là lựa chọn tốt hơn các direction controls trong setup hiện tại.

Đây là bằng chứng tích cực quan trọng và nên giữ trong paper.

---

## 2.4. Bookkeeping hiện đã đủ tốt để tin canonical P2

Việc khóa:

- checkpoint hash;
- git commit;
- atlas hash;
- split hash;
- config hash;

và evaluate từ model reload từ đĩa giải quyết đúng vấn đề stale evaluation trước đây.

Canonical P2 2.188 mm bây giờ là mốc engineering đáng tin.

---

# 3. Điểm quan trọng nhất cần sửa trong kết luận: B1 chưa loại trừ underfitting như tài liệu đang nói

Tài liệu hiện viết:

> occ-Dice train 0.879 không phải underfit vì M2 phantom ở sụn 0.4 mm chỉ có Dice 0.848 dù occupancy hoàn hảo.

Ở đây có một vấn đề về **không gian metric**.

## 3.1. B1 đang báo cáo ray-space occupancy Dice

Target trực tiếp của network là:

\[
\hat o_{ik}
\quad \text{so với}\quad
o_{ik}
\]

trên ray grid.

Nếu model memorise hoàn hảo training rays:

\[
\hat o_{ik}=o_{ik}
\]

thì ray-space occupancy Dice về lý thuyết có thể đạt:

\[
1.0
\]

## 3.2. M2 phantom Dice là voxel-space round-trip Dice

M2 đo:

```text
GT volume
→ occupancy GT
→ splat
→ reconstructed volume
→ voxel Dice
```

Dice 0.848 ở sụn 0.4 mm xuất phát từ rasterization/splat và độ mỏng dưới voxel.

Nó **không phải ceiling của ray-space occupancy Dice**.

Do đó:

\[
\boxed{\text{0.879 ray-space Dice không thể được giải thích bằng M2 voxel Dice ceiling}}
\]

### Điều B1 thực sự chứng minh

B1 cho thấy:

- model học được interior tốt;
- presence trên training gần hoàn hảo;
- không có dấu hiệu pipeline hoàn toàn hỏng;
- lỗi tập trung nhiều ở transitions/boundaries.

Nhưng nó **chưa chứng minh model đã memorise target gần tối đa**.

Vì vậy nên đổi:

> “Bug/underfit đã bị loại trừ.”

thành:

> “Không thấy dấu hiệu lỗi pipeline nghiêm trọng; tuy nhiên khả năng fit boundary transitions của ray encoder vẫn chưa đạt hoàn hảo.”

Điểm này không lật quyết định STOP P2, nhưng làm câu chuyện khoa học chính xác hơn.

---

# 4. B2 không “bác bỏ loss/sampling bias” nói chung

B2 đã thử:

- một stratified sampling scheme;
- `absent_fp_weight=3`;
- soft presence gate.

Kết quả không đủ để cứu model.

Điều có thể nói chắc:

> **Các loss/sampling modifications cụ thể đã thử không đóng được khoảng cách với B0.**

Không nên viết:

> “Loss/sampling bias đã bị bác bỏ.”

vì vẫn chưa thử:

- boundary endpoint regression;
- interval-structured decoder;
- per-ray normalized loss;
- distance-transform/boundary loss;
- depth-conditioned presence;
- neighborhood/context losses.

Những thứ này không nhất thiết phải tiếp tục làm ở Phase B, nhưng vì chưa thử nên không thể nói toàn bộ họ giả thuyết loss/output formulation đã bị loại trừ.

---

# 5. Soft-gate có tín hiệu lớn hơn tài liệu đang đánh giá

Soft-gate:

\[
2.188 \rightarrow 1.660\text{ mm}
\]

tức cải thiện khoảng:

\[
0.528\text{ mm}
\]

Đây là một thay đổi lớn hơn run-to-run difference khoảng 0.25 mm được quan sát trước đó.

Tuy nhiên, tất cả mới chạy một seed nên chưa thể biết chính xác signal/noise của training randomness.

Cách diễn giải phù hợp:

> Soft gating là lever duy nhất cho tín hiệu cải thiện có kích thước đáng kể, nhưng hiệu năng vẫn còn rất xa B0 và chưa được xác nhận qua nhiều seed.

Không cần tiếp tục sweep soft-gate nếu mục tiêu là resource allocation, nhưng không nên gọi nó “gần như noise”.

---

# 6. B3 không đủ để nói “thiếu dữ liệu đã bị bác bỏ”

B3:

- 40 ca: 1.660 mm
- 140 ca: 1.751 mm

cho thấy:

> **Không có bằng chứng rằng tăng dữ liệu 40 → 140 giúp cấu hình soft-gate hiện tại.**

Đây là kết quả hữu ích.

Nhưng chưa đủ để phát biểu:

> “Thiếu dữ liệu đã bị bác bỏ.”

Lý do:

1. Chỉ có hai điểm trên scaling curve.
2. Một seed.
3. Chưa chạy toàn bộ ~320 ca.
4. Architecture có thể là bottleneck nên data scaling không có tác dụng.
5. Hyperparameters tối ưu ở 40 ca chưa chắc tối ưu ở 140 ca.

Vì vậy nên đổi câu:

> “Missing data ruled out.”

thành:

> **“A 3.5× increase in training cases did not improve the current architecture under the tested training protocol.”**

Đó là phát biểu đủ mạnh và hoàn toàn được dữ liệu hỗ trợ.

---

# 7. Mapping sensitivity audit còn một điểm cần kiểm

Audit báo:

| | Mapping A | Mapping B |
|---|---:|---:|
| B0 | 0.5517 | 0.6033 |

Nhưng tài liệu vẫn so:

\[
B0_{\text{Mapping B}}=0.603
\]

với:

\[
P2=2.188
\]

Nếu 2.188 là P2 dưới **Mapping A**, đây là hai định nghĩa metric khác nhau.

## Cần làm

Tính cả:

\[
P2_{\text{Mapping B}}
\]

trên cùng 10 ca.

Sau đó so:

\[
B0_B \quad \text{vs}\quad P2_B
\]

### Dự đoán

Với khoảng cách hiện tại rất lớn, mình không kỳ vọng kết luận lật.

Nhưng final report nên tuyệt đối apples-to-apples.

---

# 8. Kết luận “nút thắt nằm ở tín hiệu ảnh” hiện quá mạnh

Tài liệu kết luận:

> Bottleneck lies in image signal available at extremely thin cartilage rather than spatial localization or coordinate parameterisation.

Phần:

> “không phải spatial localization”

được hỗ trợ khá tốt bởi ROI cascade + coordinate tests.

Nhưng phần:

> “nằm ở image signal”

chưa được chứng minh.

Một khả năng khác rất quan trọng là:

\[
\boxed{\text{local 1D context không đủ}}
\]

ResEnc B0 nhìn thấy:

- vùng 3D rộng;
- hình thái lân cận;
- nhiều slice;
- quan hệ spatial giữa các structure;
- contextual cues xa hơn một ray.

RayEncoder hiện tại chỉ nhìn profile cục bộ dọc một tia.

Do đó thất bại có thể do:

- local MRI signal yếu;
- **hoặc thiếu cross-ray / 2D / 3D context**;
- hoặc cả hai.

Cách viết an toàn:

> The findings suggest that localization and coordinate parameterisation are not the dominant bottlenecks. The remaining limitation is consistent with insufficient information in a local 1D intensity profile and/or the absence of broader spatial context.

---

# 9. Đây là lý do P3/M8-D trở thành phép thử rất quan trọng

P2 hỏi:

> Một ray model có thể tự segment cartilage từ local MRI + geometry không?

Kết quả hiện tại:

> Không.

Nhưng kiến trúc dự kiến ban đầu còn một câu hỏi khác:

> Một ray model dùng contextual prediction từ ResEnc có thể **refine** baseline trong surface coordinates không?

Đó chính là P3/M8-D:

\[
MRI + p_{\text{ResEnc,OOF}}
\rightarrow
\text{ray refinement}
\]

Đây không phải hyperparameter sweep.

Nó kiểm tra **một giả thuyết khác về vai trò của surface coordinates**:

### P2
Surface model = standalone segmenter.

### P3
Surface model = geometry-aware refiner của một 3D model đã có context.

Nếu P3 vượt B0 trong thin-region metric:

> Surface coordinates vẫn có giá trị như refinement mechanism, dù ray-only segmentation thất bại.

Nếu P3 vẫn không vượt B0:

> Lúc đó bằng chứng để đóng Phase B và không xây Stage 2 mạnh hơn rất nhiều.

---

# 10. Ưu tiên P3 hay medial tibial?

Tài liệu hiện ưu tiên `med_tib_cart`.

Mình sẽ đổi thứ tự.

## Ưu tiên 1 — P3/M8-D trên femoral

Lý do:

- cùng class;
- cùng split;
- cùng atlas;
- trực tiếp giải quyết confound “local ray vs contextual ResEnc”;
- là phần chưa kiểm của chính architecture matrix;
- không cần suy từ class khác.

## Ưu tiên 2 — med_tib_cart

Medial tibial vẫn đáng chạy vì M0 cho:

- headroom 0.0885 mm;
- thin+absent mass 33.5%.

Nó kiểm tra externality across cartilage class.

Nhưng nó không trả lời câu hỏi quan trọng nhất còn thiếu ở femoral:

> geometry-aware refinement có hoạt động khi có ResEnc prior hay không?

---

# 11. Có nên xây Stage 2 ngay không?

Không.

\[
\boxed{\text{KHÔNG PROCEED GRAPH/TRANSFORMER NGAY}}
\]

Điểm này mình đồng ý với tài liệu.

Nhưng lý do nên là:

> Stage 1 chưa cho thấy lợi ích downstream so với B0, nên chưa đủ căn cứ đầu tư vào graph model.

Không nên nói:

> 1D thất bại nên graph khó có khả năng cứu.

Graph/cross-ray context có thể chính là thứ 1D thiếu.

Nếu sau này thử graph, đó phải là **một giả thuyết mới**:

> spatial context across neighboring surface nodes resolves thin-vs-absent ambiguity.

Nó cần gate/headroom riêng, không được xem đơn giản là “thêm complexity để cứu P2”.

---

# 12. “Công bố kết quả âm tính” — nội bộ và paper là hai mức khác nhau

## Đủ để dừng engineering branch

Có.

Các bằng chứng đủ mạnh để:

\[
\boxed{\text{DỪNG TỐI ƯU P2/RAY-ONLY}}
\]

## Chưa đủ mạnh cho một claim âm tính rộng trong paper

Hiện chỉ có:

- 1 fold;
- 1 seed;
- n=10 validation;
- convenience split;
- một class;
- P3 chưa chạy.

Vì vậy một bài báo nên tránh claim:

> surface-coordinate cartilage segmentation does not work.

Claim hợp lệ hơn:

> A local 1D normal-ray segmentation model failed to outperform a 3D ResEnc baseline despite a geometrically accurate surface-coordinate representation.

Nếu muốn negative claim mạnh hơn, tối thiểu nên có một confirmatory package nhỏ:

1. P3/M8-D trên femoral;
2. best configuration qua 2–3 seeds **hoặc** một fold thứ hai;
3. metric Mapping B cho cả B0 và ray model;
4. sau đó med_tib như class replication nếu compute cho phép.

---

# 13. Đánh giá câu kết luận bài báo hiện tại

Phần sau rất tốt:

> Representational adequacy does not imply learnability.

Mình sẽ giữ.

Nhưng thay câu:

> “the bottleneck lies in the image signal available at extremely thin cartilage”

bằng:

> **“the results indicate that spatial localization and coordinate parameterisation are not sufficient to resolve extremely thin cartilage. The remaining limitation is consistent with weak local image evidence and/or insufficient contextual information in a one-dimensional ray representation.”**

Và thay:

> “loss, sampling, and data were ruled out”

bằng:

> **“the tested sampling, loss-weighting, gating, and 3.5× data-scaling interventions did not close the gap.”**

Hai thay đổi này làm paper khó bị reviewer phản bác hơn.

---

# 14. Quyết định cuối cùng của mình

## Chốt ngay

\[
\boxed{
\text{STOP optimization of the current P2 ray-only architecture}
}
\]

## Không làm tiếp

- thêm epoch vô hạn;
- sweep FP weight;
- sweep threshold;
- thêm nhiều small hyperparameter tweaks;
- graph transformer ngay.

## Làm một phép thử quyết định cuối trước khi đóng hướng refinement

\[
\boxed{
\text{Run P3/M8-D with clean OOF ResEnc softmax}
}
\]

Gate:

### Nếu P3 vẫn không vượt B0

Dừng Phase B mạnh mẽ:

\[
\boxed{\text{NEGATIVE RESULT CONFIRMED FOR FEMORAL}}
\]

sau đó med_tib là replication/secondary question.

### Nếu P3 vượt B0

Rescope:

\[
\boxed{
\text{Surface coordinates work as a refinement space, not as a standalone segmentation space}
}
\]

Đây sẽ là một phát hiện rất quan trọng và trực tiếp định hình Main Model.

---

# 15. Tóm tắt mức độ tin cậy

| Kết luận | Mức tin |
|---|---|
| P2 ray-only thất bại | **Rất cao** |
| Geometry/splat không phải bottleneck | **Rất cao** |
| Normal tốt hơn direction controls | **Cao** |
| Hard gate hiện tại không phù hợp | **Cao** |
| Specific stratification/FP weighting không cứu P2 | **Cao** |
| Thiếu dữ liệu không phải bottleneck | **Trung bình/thấp** |
| Mọi loss formulation đều không cứu được | **Không đủ bằng chứng** |
| MRI signal là bottleneck duy nhất | **Không đủ bằng chứng** |
| Surface-coordinate refinement nói chung thất bại | **Chưa kết luận — P3 chưa chạy** |
