# QUYẾT ĐỊNH VỀ LỖI NEAREST-NODE MAPPING VÀ STALE EVALUATION

## 1. Kết luận ngắn

### Nearest-node mapping

\[
\boxed{
\text{Chưa cần dừng MVP để viết lại toàn bộ mapping, nhưng phải audit trước khi khóa Gate cuối}
}
\]

Việc xuất hiện 1,123 voxel sụn GT trong bin `absent` chứng minh phép gán theo nearest surface node không phải là một ánh xạ giải phẫu chính xác tuyệt đối.

Tuy nhiên, lỗi này hiện chưa đủ để bác bỏ kết luận chính rằng P2 thất bại nặng ở vùng mục tiêu, vì:

- bảng recall theo độ dày chỉ là phân tích chẩn đoán phụ;
- lỗi P2 so với B0 rất lớn;
- kết luận chính không phụ thuộc riêng vào dòng `absent recall`;
- oracle reconstruction và các kiểm thử hình học vẫn cho kết quả nhất quán.

Dù vậy, không nên khẳng định rằng main metric hoàn toàn không bị ảnh hưởng, vì `error_mass_by_thickness` cũng dùng nearest-node assignment để gán thickness bin cho các surface points.

Do đó, mức kết luận phù hợp là:

> Nearest-node inconsistency chắc chắn làm bảng recall theo độ dày kém tin cậy; ảnh hưởng của nó lên thin-region composite metric có khả năng nhỏ hơn, nhưng cần một sensitivity test để xác nhận.

### Stale evaluation

\[
\boxed{
\text{Kết luận định tính P2 thất bại vẫn đáng tin; các con số chi tiết hiện chỉ là provisional}
}
\]

Hai lần train độc lập cho lỗi vùng mỏng khoảng 1.94–2.0 mm, trong khi B0 là khoảng 0.55 mm.

Khoảng cách này đủ lớn để kết luận:

> P2 hiện tại chưa vượt baseline và cần được sửa.

Bookkeeping không tốt khó có khả năng biến một model lỗi khoảng 2.0 mm thành một model tốt hơn 0.55 mm.

Tuy nhiên, các con số chi tiết như:

- tỷ lệ 82% error mass từ bin absent;
- số voxel false positive;
- per-bin mean error;
- recall theo thickness;
- chênh lệch nhỏ giữa các ablation;

không nên được xem là kết quả chính thức cho đến khi toàn bộ chúng được tạo từ cùng một checkpoint và cùng một code revision.

---

# 2. Đánh giá lỗi nearest-node mapping

## 2.1. Điều lỗi 1,123 voxel chứng minh

Nếu một voxel thuộc GT cartilage nhưng được gán vào node có thickness bằng 0, điều đó nghĩa là:

\[
\text{nearest Euclidean node}
\neq
\text{anatomical source ray của voxel}
\]

Điều này dễ xảy ra ở:

- rìa mỏng;
- vùng cong;
- nơi hai surface nodes gần nhau trong Euclidean space;
- nơi normal rays hội tụ hoặc phân kỳ;
- vùng atlas margin;
- cấu trúc chỉ dày khoảng một voxel.

Vì vậy, bảng voxel-recall theo bin độ dày không thể được xem là exact anatomical stratification.

---

## 2.2. Vì sao kết luận chính vẫn tương đối vững

Kết luận chính là:

\[
E_{\text{P2}}\approx 2.0\text{ mm}
\quad \text{so với}\quad
E_{\text{B0}}\approx0.5517\text{ mm}
\]

Đây là một chênh lệch rất lớn.

Ngay cả khi nearest-node mapping gây một phần sai phân bin, rất khó để sai số đó giải thích toàn bộ khoảng cách hơn 1.4 mm giữa P2 và B0.

Ngoài ra, các bằng chứng độc lập khác cũng chỉ về cùng hướng:

- voxel Dice của ray model thấp hơn B0;
- predicted volume có nhiều false-positive components;
- thin recall thấp;
- presence-threshold sweep không cứu được model;
- oracle reconstruction tốt hơn rất nhiều so với learned model.

Do đó:

> Lỗi mapping làm suy yếu mức độ tin cậy của chẩn đoán “lỗi nằm chính xác bao nhiêu ở từng bin”, nhưng không làm mất chẩn đoán lớn rằng learned P2 đang dự đoán kém.

---

## 2.3. Nhưng main metric không hoàn toàn độc lập với nearest-node

`error_mass_by_thickness` gán thickness cho:

- GT surface points;
- predicted surface points;

bằng nearest GT surface node.

Do đó, thin-region error cũng có thể bị ảnh hưởng bởi:

- surface point bị gán nhầm từ thin sang absent;
- thick point bị gán vào thin;
- predicted false-positive surface bị gán vào node không đúng về giải phẫu.

Vì vậy không nên viết:

> Mapping error không ảnh hưởng main metric.

Cách viết đúng hơn:

> Mapping error chắc chắn ảnh hưởng bảng voxel-level recall; mức ảnh hưởng lên surface-error Gate chưa được định lượng nhưng chưa có dấu hiệu đủ lớn để lật quyết định REVISE.

---

# 3. Có cần chuyển sang provenance mapping ngay không?

## 3.1. Không cần chặn mọi revision để viết lại mapping

Không cần dừng việc:

- sửa loss;
- bỏ hard presence gate;
- phân tầng sampling;
- micro-overfit;
- thêm gradient hoặc OOF prior;

chỉ để hoàn thiện provenance mapping trước.

Lý do:

- learned model hiện thất bại quá xa baseline;
- nút thắt trước mắt là prediction model;
- full provenance implementation có thể tốn thời gian;
- quyết định `REVISE` không phụ thuộc vào một vài phần trăm thay đổi bin assignment.

## 3.2. Nhưng provenance phải có trước final MVP Gate

Provenance mapping sẽ cần thiết cho:

- endpoint boundary MAE;
- exact thin/absent stratification;
- ray-specific error analysis;
- xác định source ray của reconstructed voxel;
- phân biệt hallucination trong articular domain với lỗi mapping.

Do đó:

\[
\boxed{
\text{Provenance không phải blocker cho micro-overfit, nhưng là blocker cho final Gate claim}
}
\]

---

# 4. Sensitivity test rẻ trước khi viết lại toàn bộ mapping

Trước khi triển khai full provenance, nên chạy một phép thử nhỏ trên đủ 10 validation cases.

## 4.1. Hai cách phân bin

### Mapping A — hiện tại

```text
surface point → nearest GT bone-surface node
```

### Mapping B — provenance hoặc ray ownership gần đúng

Đối với reconstructed prediction:

```text
predicted voxel/surface point → ray đã splat giá trị vào voxel đó
```

Đối với GT:

```text
GT surface point → ray/normal coordinate tạo occupancy tương ứng
```

## 4.2. So sánh

Tính lại:

- thin mean error;
- absent error mass;
- \(\leq0.5\) mm error;
- \(\leq1.0\) mm error;
- số surface points trong từng bin;
- P2–B0 paired difference.

## 4.3. Working decision rule

Nếu:

- thin error thay đổi dưới khoảng 5% relative;
- P2 vẫn tệ hơn B0 ở 10/10 ca;
- failure ranking không đổi;

thì nearest-node mapping có thể tiếp tục dùng cho exploratory analysis với caveat.

Nếu:

- thay đổi trên khoảng 10%;
- absent mass thay đổi mạnh;
- một số ca đổi chiều;
- Gate conclusion thay đổi;

thì phải chuyển sang provenance mapping ngay và tính lại toàn bộ §3.7.

Khoảng 5% và 10% là engineering thresholds, không phải chuẩn phổ quát.

---

# 5. Đánh giá stale evaluation

## 5.1. Điều sự nhất quán 1.94–2.0 mm cho thấy

Hai session độc lập tạo kết quả gần nhau.

Điều này làm tăng độ tin rằng:

- P2 thực sự đang ở vùng hiệu năng khoảng 2 mm;
- failure không phải do một checkpoint đặc biệt xấu;
- lỗi đủ lớn để quyết định REVISE.

Nếu training và data order đều deterministic hoặc gần deterministic, kết quả gần nhau càng dễ hiểu.

Tuy nhiên, không nên suy rằng hai session chắc chắn là cùng model.

---

## 5.2. Điều chưa thể bảo đảm

Không có checkpoint hash nên không biết chắc:

- §3.7 dùng model nào;
- per-bin table dùng model nào;
- voxel Dice dùng model nào;
- presence sweep dùng model nào;
- code revision có giống nhau không;
- atlas/cache có giống nhau không.

Vì vậy, hiện có thể tin:

> Failure mode lớn lặp lại.

Nhưng chưa thể tin tuyệt đối:

> Tất cả các số chi tiết thuộc cùng một coherent experiment.

---

# 6. Mức độ ảnh hưởng đến quyết định

## 6.1. Quyết định định tính

Quyết định:

\[
\text{P2 fails} \Rightarrow \text{REVISE}
\]

vẫn vững.

Để lật quyết định, một rerun sạch phải giảm error từ khoảng 2.0 mm xuống dưới 0.55 mm — một thay đổi quá lớn để chỉ do stale bookkeeping.

## 6.2. Các claim định lượng

Các claim sau nên đánh dấu `provisional`:

- 82% thin-region mass đến từ absent;
- 23,897 false-positive surface voxels;
- absent mean error 3.38 mm;
- thin recall 37%;
- cực mỏng recall 30.7%;
- exact effect của threshold 0.7;
- exact gap giữa input ablations.

Các con số này có thể đúng về xu hướng nhưng cần canonical rerun.

---

# 7. Canonical rerun bắt buộc

Trước khi so sánh revision mới với P2, cần tạo một canonical P2 run.

## 7.1. Lưu ngay sau training

```yaml
experiment_id: ...
checkpoint_path: ...
checkpoint_sha256: ...
git_commit: ...
dataset_revision: ...
split_manifest_hash: ...
atlas_hash: ...
config_hash: ...
seed: ...
```

## 7.2. Mọi evaluation phải load checkpoint từ đĩa

Không dùng trực tiếp biến `net` còn trong RAM.

Pipeline:

```text
train → save checkpoint → reload checkpoint → evaluate all metrics
```

Bước reload chứng minh checkpoint đủ để tái lập kết quả.

## 7.3. Tính lại toàn bộ downstream outputs

Từ cùng checkpoint:

- occupancy metrics;
- voxel reconstruction;
- thin-region error;
- per-bin error;
- thickness recall;
- presence sweep;
- M6/M7/P5 nếu cần.

## 7.4. Không reuse stale per-case file

Output path nên chứa:

```text
experiment_id/checkpoint_hash/code_commit
```

Nếu một thành phần thay đổi, phải tạo thư mục kết quả mới.

---

# 8. Quyết định cuối cùng

## Về nearest-node mapping

\[
\boxed{
\text{Giữ caveat cho kết luận hiện tại, chạy sensitivity audit sớm, provenance trước final Gate}
}
\]

Không cần để provenance rewrite chặn micro-overfit và model revision.

Nhưng không nên dùng bảng recall theo thickness làm bằng chứng chính cho đến khi mapping được audit.

## Về stale evaluation

\[
\boxed{
\text{Giữ quyết định REVISE, nhưng coi các bảng chẩn đoán hiện tại là provisional}
}
\]

Cần một canonical rerun của P2 trước khi:

- báo cáo con số chính thức;
- so P2 với revision;
- lựa chọn input/loss;
- tính capture ratio;
- quyết định Stage 2.

---

# 9. Cách diễn đạt đề xuất trong tài liệu

> The nearest-node assignment produces a small but non-zero inconsistency in voxel-level thickness stratification, including GT cartilage voxels assigned to the absent bin. This directly limits the reliability of the voxel-recall-by-thickness table. The primary thin-region surface-error result is less directly affected, although it also relies on nearest-node thickness assignment and therefore requires a sensitivity analysis before final reporting. The current conclusion that P2 substantially underperforms B0 remains robust because the observed performance gap is large and is supported by several independent diagnostics.

> The primary Gate result and the subsequent diagnostic analyses were generated in separate sessions with independently trained P2 instances. Their close agreement supports the qualitative conclusion that P2 fails, but the absence of checkpoint and code hashes prevents treating all reported numbers as one coherent experiment. The current decision to revise the MVP is retained, while detailed per-bin and recall estimates remain provisional until a canonical checkpoint-locked rerun is completed.
