# REVIEW CÁCH THỬ NGHIỆM TRONG `bsc_02_mvp.ipynb`

## 1. Kết luận tổng quát

Cách thử nghiệm có nền tảng khoa học tốt hơn nhiều so với một quy trình chỉ train model rồi nhìn Dice:

- có cổng QC hình học trước khi train;
- có atlas chống leakage;
- có input ablation;
- có negative controls;
- có jitter test;
- có so sánh GT bone với predicted bone;
- có metric riêng cho vùng sụn mỏng;
- có oracle reconstruction để tách lỗi representation khỏi lỗi model;
- giữ 103 ca test chưa đụng tới.

Quyết định tổng thể vẫn hợp lý:

\[
\boxed{\text{GIỮ HƯỚNG HÌNH HỌC, REVISE MVP, CHƯA PROCEED STAGE 2}}
\]

Tuy nhiên, notebook hiện có một số lỗi về bookkeeping, cache, model identity và metric interpretation. Các lỗi này không làm mất kết luận lớn rằng **P2 hiện tại thất bại ở vùng mỏng**, nhưng chúng làm một số kết luận phụ chưa đủ chắc chắn.

---

# 2. Những kết luận hiện có mức tin cậy cao

## 2.1. Pipeline hình học không phải lỗi chính

QC cho thấy:

- normal đúng khoảng 99.8%;
- single interval khoảng 99.8%;
- round-trip Dice khoảng 0.995;
- round-trip ASSD khoảng 0.009 mm;
- không có khung suy biến.

Đây là bằng chứng mạnh rằng:

> Surface extraction, normal field và project–reconstruct cơ bản đang hoạt động đúng.

## 2.2. Atlas không có leakage trực tiếp

Atlas được xây từ 40 ca train và có `assert_no_leak` đối với validation.

Kết quả recall rất cao và atlas sai lớp gần như không chọn vùng của lớp còn lại.

Kết luận:

> D1 atlas có khả năng xác định vùng khớp và chưa có dấu hiệu là nguyên nhân chính làm P2 thất bại.

## 2.3. P2 hiện tại thực sự thất bại ở mục tiêu chính

Metric quyết định cho thấy:

- B0 thin-region error khoảng 0.5517 mm;
- P2 khoảng 2.0 mm;
- 0/10 ca tốt hơn baseline.

Điều này đủ rõ để kết luận:

> Cấu hình P2 hiện tại chưa hiện thực hóa được lợi ích của surface coordinates.

## 2.4. Oracle reconstruction chứng minh representation không tự phá target

GT occupancy qua cùng đường reconstruction cho thin-region error khoảng 0.0256 mm.

Điều này chứng minh:

> Lỗi 2.0 mm không phải là trần bắt buộc do splatting hoặc coordinate transform.

Nhưng 0.0256 mm chỉ là **self-consistency ceiling**, không phải mức model chắc chắn có thể học từ MRI.

---

# 3. Các lỗi hoặc điểm không nhất quán cần sửa ngay

## 3.1. Notebook không khóa code commit

Cell setup dùng:

```python
git pull
```

mỗi khi chạy.

Như vậy notebook có thể:

- được train bằng một revision code;
- được đánh giá lại bằng revision khác;
- dùng cache được tạo từ code cũ;
- nhưng vẫn ghi cùng experiment ID.

### Quyết định

Mỗi run phải khóa:

```text
git_commit
dataset_revision
config_hash
atlas_hash
checkpoint_hash
```

Không dùng `git pull` tự động trong một notebook dùng để tạo kết quả chính thức.

---

## 3.2. Không lưu checkpoint và đang trộn nhiều model object

Notebook ghi rõ session mới làm mất weights và phải train lại P2.

Sau đó:

- một số cell dùng `net`;
- một số cell dùng `RUNS[BEST].net`;
- file evaluation có thể chứa kết quả từ model trước;
- chẩn đoán sau đó có thể dùng model train lại.

Như vậy cùng một báo cáo có nguy cơ trộn hai instances của P2.

### Quyết định

Mỗi run phải:

1. lưu checkpoint ngay sau training;
2. ghi model hash;
3. load đúng checkpoint cho tất cả evaluation;
4. không dùng biến `net` và `RUNS[BEST].net` song song;
5. mọi output phải chứa `experiment_id` và `checkpoint_hash`.

---

## 3.3. Evaluation checkpoint có nguy cơ dùng kết quả cũ

File:

```text
mvp_eval_femoral_cart_P2.jsonl
```

chỉ kiểm tra `case_id`.

Nó không kiểm tra:

- code commit;
- seed;
- checkpoint;
- atlas version;
- loss;
- epoch;
- config version.

Nếu P2 được train lại hoặc code thay đổi, các ca đã có sẽ bị skip và kết quả cũ vẫn được dùng.

### Quyết định

Đường dẫn kết quả phải dùng full run identity:

```text
eval/<experiment_id>/<checkpoint_hash>/per_case.jsonl
```

Không append vào file cũ nếu run hash thay đổi.

---

## 3.4. Gate JSON đang dùng biến M6 cũ và metric sai

M6 ban đầu dùng ray-space occupancy Dice và cho kết luận sai.

Sau đó notebook sửa M6 bằng voxel-space thin boundary error. Tuy nhiên cell ghi Gate vẫn dùng:

```python
m6["normal"]["occ_dice"] > ctrl
```

tức là dùng lại metric cũ.

Do đó registry có thể ghi M6 là fail dù phần phân tích mới cho thấy pass.

### Quyết định

Gate phải dùng:

```python
all(v["ci_low"] > 0 for v in m6_boundary_stats.values())
```

Không được tham chiếu lại ray-space Dice.

---

## 3.5. M5 Gate cũng dùng biến cũ

M5 ban đầu tính Dice tại threshold 0.5.

Sau đó notebook sweep threshold và đánh giá lại, nhưng Gate cuối vẫn dùng:

```python
m5_dice > 0.90
```

với `m5_dice` từ threshold 0.5.

Ngoài ra:

- code dùng 3 ca, không phải 4;
- summary ghi 4 ca;
- output M5 không được lưu trong notebook;
- M5 dùng P1/D0/H0, không phải P2/D1/H1 đang thất bại.

### Quyết định

Micro-overfit phải chạy trên chính cấu hình cần sửa:

```text
S0-D1-(best input)-H1
```

và nên bắt đầu từ:

- 1 case;
- vài trăm ray đã phân tầng;
- không augmentation;
- không subsampling ngẫu nhiên lại giữa các lần.

Mục tiêu nên gần 0.97–0.99 train occupancy Dice, không chỉ vừa qua 0.90.

---

## 3.6. Cache key chưa đủ an toàn

Geometry cache key hiện có một số config hình học, nhưng thiếu:

- seed của random direction;
- seed của jitter;
- atlas hash/version;
- threshold atlas;
- dataset revision;
- predicted-bone model revision;
- code commit.

Điều này có thể khiến:

- random direction tái dùng hướng từ seed khác;
- jitter tái dùng geometry cũ;
- atlas mới vẫn dùng domain cũ;
- predicted bone mới dùng surface cache cũ.

### Quyết định

Cache key phải bao gồm toàn bộ provenance hoặc cache phải bị xóa khi một thành phần thay đổi.

Đặc biệt cần thêm:

```text
code_commit
case_data_hash
surface_source_hash
atlas_hash
direction
direction_seed
jitter_seed
jitter_parameters
ray_config_hash
```

---

# 4. Review từng phép thử

## 4.1. P0–P3 và M8

### Điểm đúng

M8-A/B/C giữ cùng scaffold S0-D1-H1 và chỉ đổi input, nên ablation MRI, gradient và SDF là hợp lệ.

Kết quả hiện tại gợi ý:

- gradient có ích hơn SDF;
- SDF gần như chỉ cung cấp lại thông tin depth vốn đã gần cố định theo ray.

### Điểm cần sửa

Tên `BEST` đang hard-code P2 dù M8-B có ray-space Dice cao hơn.

Điều này không sai nếu P2 đã được pre-specify, nhưng không nên gọi là `BEST`.

### Quyết định

Đổi tên:

```text
PRIMARY_RUN = P2
```

Nếu chọn input tốt nhất dựa trên validation, phải:

- chọn bằng metric thin-region;
- sau đó đánh giá trên split khác;
- không chọn và báo cáo final Gate trên cùng 10 ca.

Nên thêm input:

```text
MRI + gradient + explicit depth coordinate
```

hoặc:

```text
MRI + gradient + SDF
```

vì gradient mang thông tin biên, còn depth/SDF mang vị trí dọc tia.

---

## 4.2. M6 negative-direction controls

### Điểm tốt

Bạn đã phát hiện ray-space occupancy Dice là metric không công bằng và chuyển sang voxel-space boundary metric. Đây là sửa đổi đúng.

### Hạn chế còn lại

Mỗi direction có:

- target occupancy khác;
- positive fraction khác;
- oracle reconstruction ceiling khác;
- coverage và ray overlap khác.

Do đó, normal tốt hơn có thể đến từ hai nguồn:

1. representation bằng normal phù hợp hơn;
2. model học dễ hơn theo normal.

Notebook hiện chưa tách hai nguồn này.

### Phép thử cần thêm

Với mỗi direction \(d\), tính:

\[
E_{\text{oracle}}(d)
\]

và:

\[
E_{\text{model}}(d)
\]

Sau đó báo cáo:

\[
\text{excess error}(d)
=
E_{\text{model}}(d)-E_{\text{oracle}}(d)
\]

Nếu normal có oracle ceiling tốt hơn, đó là lợi ích representation.

Nếu normal còn có excess error thấp hơn, đó là lợi ích learnability.

### Nhận xét về kết luận hiện tại

Có thể nói:

> Normal-coordinate pipeline tốt hơn các direction controls trong setup hiện tại.

Chưa nên khẳng định tuyệt đối:

> Model đọc MRI tốt hơn chỉ vì hướng normal.

---

## 4.3. M7 jitter

### Vấn đề

Notebook ghép đồng thời:

- displacement 0.25 mm với rotation 5°;
- 0.5 mm với 10°;
- 1.0 mm với 15°.

Do đó không biết degradation đến từ:

- sai vị trí surface;
- sai normal;
- hay tương tác giữa hai loại lỗi.

M7 chỉ thêm jitter lúc test cho model train không jitter. Đây là sensitivity test, chưa phải robustness test.

Ngoài ra, M7 dùng ray-space occupancy Dice, không phải metric quyết định.

### Quyết định

Chạy grid riêng:

| Surface offset | Normal rotation |
|---:|---:|
| 0, 0.25, 0.5, 1.0 mm | 0° |
| 0 mm | 0°, 5°, 10°, 15° |
| selected offsets | selected rotations |

Đo:

- thin-region boundary error;
- absent FP mass;
- thin recall;
- voxel Dice phụ.

Sau đó train một model có jitter augmentation và đo lại.

---

## 4.4. P5 predicted bone

### Vấn đề

P5 “giữ 96%” được tính từ ray-space occupancy Dice:

\[
0.793/0.825
\]

Nhưng S0 và S2 có:

- surface khác;
- ray khác;
- target occupancy khác;
- reconstruction ceiling khác.

Hai Dice ray-space không hoàn toàn cùng bài toán.

### Quyết định

P5 phải được đánh giá bằng:

- voxel-space thin boundary error;
- absent FP;
- thin recall;
- oracle reconstruction ceiling riêng cho S2.

Nên báo cáo retained gain:

\[
\frac{
E_{B0}-E_{P5}
}{
E_{B0}-E_{P2}
}
\]

chỉ khi P2 đã tốt hơn B0.

Khi P2 còn tệ hơn B0, “retained gain” chưa có ý nghĩa.

---

# 5. Metric vùng mỏng đang gộp nhiều loại lỗi

Hàm `thin_region_boundary_error` hiện gộp:

- GT surface → prediction;
- prediction surface → GT;
- absent;
- cực mỏng;
- cả các surface components.

Nó là symmetric thin-region surface error, không hoàn toàn là “outer-boundary error”.

### Vấn đề

Một con số duy nhất có thể bị chi phối bởi absent false positives, như kết quả hiện tại.

Điều đó hữu ích để phát hiện failure, nhưng không đủ để trả lời riêng:

- outer boundary có chính xác hơn không;
- thin cartilage có được giữ lại không;
- hallucination có giảm không.

### Quyết định

Báo cáo tối thiểu ba metric riêng:

1. **GT→Pred thin distance**: đo bỏ sót và lệch biên ở sụn thật.
2. **Pred→GT absent distance/mass**: đo hallucination.
3. **Ray endpoint boundary MAE**: đo inner/outer boundary trực tiếp theo ray.

Metric gộp vẫn giữ làm composite safety metric.

---

# 6. Có một red flag trong phân tầng độ dày

Bảng recall theo thickness có:

```text
GT cartilage voxels trong bin absent = 1123
```

Về khái niệm, voxel GT cartilage không nên thuộc vùng “không có sụn”.

Điều này cho thấy phép gán:

```text
voxel cartilage → nearest bone-surface node
```

có thể gán một số voxel sụn thật vào node thickness bằng 0.

Nguyên nhân có thể là:

- nearest Euclidean node không phải ray sở hữu voxel;
- vùng atlas margin;
- surface curvature;
- node coverage;
- lỗi mapping ở rìa.

### Hệ quả

Không nên diễn giải dòng “absent recall 30.1%” như một hiện tượng giải phẫu thật.

### Quyết định

Với ray reconstruction, nên theo dõi provenance trực tiếp:

```text
predicted voxel → source ray/node
```

và phân bin theo thickness của source ray.

Với GT cartilage, phân bin theo ray ownership hoặc signed surface coordinates, không chỉ nearest Euclidean node.

Cần báo cáo tỷ lệ mapping inconsistency trước khi dùng các bảng binwise cho kết luận chính.

---

# 7. Một số bảng chỉ dùng 5/10 ca

Các cell phân rã error mass và recall thickness dùng:

```python
val_ids[:5]
```

nhưng bản tổng kết dễ khiến người đọc hiểu là toàn bộ 10 ca.

### Quyết định

Mọi bảng phải ghi rõ:

```text
exploratory subset n=5
```

hoặc chạy lại đủ 10 ca.

Các con số dùng để quyết định Gate phải luôn dùng đủ validation set đã khóa.

---

# 8. Failure mode có thể đến trực tiếp từ loss và sampling

Current occupancy loss được average trên mọi depth cell của mọi ray.

Điều này tạo bất lợi cấu trúc:

- ray sụn dày có nhiều positive cells;
- ray sụn cực mỏng có rất ít positive cells;
- absent ray toàn negative;
- uniform ray sampling không bảo đảm đủ thin/absent examples.

Vì vậy thick cartilage đóng góp nhiều positive gradients hơn thin cartilage.

Kết quả:

- recall \(>2\) mm khoảng 81%;
- recall \(\leq1\) mm khoảng 37%;

phù hợp với bias này.

### Quyết định sửa

Batch sampling cần phân tầng theo:

- absent;
- \(\leq0.5\) mm;
- \(\leq1.0\) mm;
- 1–2 mm;
- \(>2\) mm;
- hard negatives gần atlas margin.

Loss cần có:

- per-ray normalization;
- boundary endpoint loss;
- stronger absent false-positive penalty;
- thin-ray weighting;
- optional interval-consistency loss.

Scalar presence không nên hard-gate toàn bộ ray trong revision đầu tiên.

---

# 9. “Presence-per-ray là fundamental failure” là kết luận quá mạnh

Threshold sweep chứng minh:

> Presence head hiện tại và threshold hiện tại không tách được thin khỏi absent.

Nó chưa chứng minh:

> Mọi ray-level presence formulation đều bất khả thi.

Có thể failure đến từ:

- loss;
- imbalance;
- insufficient context;
- calibration;
- hard gating;
- model underfit;
- input thiếu coarse prior.

### Cách viết chính xác hơn

> Hard scalar presence gating is not adequate under the current training setup.

Không nên gọi là giới hạn fundamental trước khi thử:

- balanced rays;
- no hard gate;
- occupancy-derived presence;
- OOF ResEnc prior;
- calibrated structured interval output.

---

# 10. Chọn mẫu train/validation

Notebook lấy:

```python
sorted_ids[:40]
sorted_fold0_ids[:10]
```

Đây là convenience subset, không phải mẫu ngẫu nhiên hoặc phân tầng.

Có thể vô tình lệch theo:

- severity;
- acquisition order;
- cartilage thickness;
- scanner/domain.

### Quyết định

Đối với debug: chấp nhận.

Đối với Gate chính:

- stratify theo KL/severity;
- mean thin thickness;
- absent area;
- dataset/domain;
- baseline thin error.

Ghi rõ split manifest và không thay sau khi xem kết quả.

---

# 11. Thứ tự chạy lại được đề xuất

## Step 0 — Sửa reproducibility

- khóa commit;
- lưu checkpoint;
- hash atlas/cache/config;
- xóa stale cache;
- sửa evaluation registry;
- sửa Gate dùng metric mới.

## Step 1 — Micro-overfit đúng cấu hình

Dùng chính S0-D1-H1 với input revision.

- 1 case;
- stratified rays;
- không augmentation;
- full provenance;
- target train Dice 0.97–0.99;
- kiểm tra thin và absent riêng.

Nếu không đạt, chưa scale.

## Step 2 — Sửa model/loss

Ưu tiên:

1. MRI + gradient + depth/SDF;
2. bỏ hard scalar gate;
3. stratified sampling;
4. per-ray/boundary-aware loss;
5. OOF ResEnc probability khi có.

## Step 3 — 40/10 engineering run

Chạy ít nhất 3 seeds.

Engineering gate:

- thin error giảm mạnh từ 2.0 mm;
- absent FP giảm;
- thin recall tăng;
- không yêu cầu ngay vượt B0 nếu đang debug.

## Step 4 — Rerun controls

- M6 với oracle-normalized direction comparison;
- M7 displacement và rotation tách riêng;
- P5 bằng voxel metrics;
- đầy đủ 10 ca.

## Step 5 — Scale

Chỉ khi engineering gate đạt:

- tăng lên toàn bộ development cohort;
- train lâu hơn;
- giữ 103 test chưa đụng tới;
- sau đó mới chạy med_tib và Stage 2 decision.

---

# 12. Đánh giá cuối cùng

## Điều có thể tin ngay

- Geometry QC đạt.
- Atlas không leak trực tiếp.
- P2 hiện tại thất bại ở thin region.
- Oracle representation có self-consistency tốt.
- Hallucination absent là failure mode lớn.

## Điều có tín hiệu nhưng cần kiểm chứng lại

- Normal direction tốt hơn controls.
- Predicted bone giữ phần lớn hiệu năng.
- Hệ tọa độ không giòn.
- Model chủ yếu underfit.

## Điều chưa nên khẳng định

- Presence-per-ray fundamentally không thể hoạt động.
- P5 giữ chính xác 96% lợi ích.
- M7 chứng minh robustness.
- Bảng binwise n=5 đại diện toàn validation set.
- 0.0256 mm là mức model có thể đạt.

## Quyết định

\[
\boxed{
\text{REVISE MVP SAU KHI SỬA PIPELINE THÍ NGHIỆM VÀ MODEL OBJECTIVE}
}
\]

Không cần bỏ surface-coordinate direction, nhưng cần sửa bookkeeping trước để lần chạy tiếp theo có thể tạo ra một kết luận không bị nhiễu bởi stale cache, stale evaluation hoặc mixed checkpoints.
