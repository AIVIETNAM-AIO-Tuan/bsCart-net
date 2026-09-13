# QUYẾT ĐỊNH VỀ MAPPING AUDIT, SPLIT MANIFEST VÀ CANONICAL P2

## 1. Kết luận điều hành

Ba quyết định được chốt như sau:

1. **Dùng nearest-ray-sample làm Mapping B cho sensitivity audit.**  
   Đây là một ray-ownership proxy, không gọi là full provenance.

2. **Giữ convenience split 40/10 cho canonical engineering P2 và các revision runs.**  
   Chỉ chuyển sang split phân tầng đã khóa khi bước vào final MVP Gate. Khi đó phải xây lại atlas và vô hiệu hóa các cache phụ thuộc atlas/split.

3. **Chạy lại P2 với config cũ sau khi sửa bookkeeping.**  
   Đây là checkpoint-locked engineering baseline để xác nhận mốc khoảng 2.0 mm và làm chuẩn so sánh cho các revision. Sau đó mới thay loss, sampling và model logic.

Quy trình:

```text
Fix bookkeeping
→ rerun old P2 cleanly
→ lock checkpoint and diagnostics
→ revise model/loss/sampling
→ compare revisions on the same engineering split
→ freeze design
→ create stratified final split
→ rebuild atlas/dependent cache
→ run final Gate
```

---

# 2. Mapping B cho sensitivity audit

## 2.1. Xác nhận hạn chế của full provenance

`core.splat_rays` hiện thực hiện trilinear splatting bằng cách:

- rải mỗi ray sample vào tám voxel lân cận;
- cộng toàn bộ weighted contributions bằng `np.bincount`;
- chia tổng giá trị cho tổng trọng số tại mỗi voxel.

Sau phép cộng này, output chỉ còn:

```text
acc[voxel]
wgt[voxel]
```

Nó không lưu:

- ray ID nào đã đóng góp;
- sample depth nào đã đóng góp;
- contribution của từng ray;
- ray nào là nguồn chính của voxel.

Vì vậy full provenance không thể được khôi phục từ output volume hiện tại. Muốn có provenance thật phải sửa splat hoặc chạy lại một phép tích lũy có lưu source IDs.

Kết luận:

\[
\boxed{
\text{Full provenance không phải là một audit rẻ với implementation hiện tại}
}
\]

---

## 2.2. Chấp nhận nearest-ray-sample làm Mapping B

Mapping B được đổi tên thành:

> **Nearest sampled-ray ownership proxy**

Với mỗi query point \(q\):

1. Tạo toàn bộ ray sample points:

\[
P_{ik}=v_i+d_k n_i
\]

2. Xây một KDTree trên các điểm \(P_{ik}\).
3. Tìm sample point gần \(q\) nhất.
4. Lấy ray index \(i\) của sample đó.
5. Gán thickness của ray \(i\) cho \(q\).

Cách này gần tinh thần của ray ownership hơn nearest-node vì nó dùng cả:

- vị trí node;
- hướng ray;
- depth của sample.

Nó vẫn là approximation vì một voxel reconstructed có thể nhận contribution từ nhiều ray samples.

---

## 2.3. Không gọi Mapping B là provenance

Tên báo cáo nên là:

```text
Mapping A: nearest bone-surface node
Mapping B: nearest sampled-ray proxy
Mapping C: full splat provenance, future final-Gate implementation
```

Không dùng từ `provenance` cho Mapping B vì nó không xác định source ray thật đã đóng góp nhiều nhất vào voxel.

---

## 2.4. Cách triển khai audit hợp lý

Không cần query mọi voxel trong volume.

Chỉ query các điểm thực sự tham gia metric:

- GT surface voxel centres;
- predicted surface voxel centres;
- GT cartilage voxels nếu chạy bảng recall theo thickness.

Như vậy KDTree được xây một lần mỗi ca, sau đó dùng nhiều query batches.

Cần làm trong tọa độ mm, không dùng index-space distance.

---

## 2.5. Quality-control cho Mapping B

Ngoài thickness bin, lưu khoảng cách:

\[
d_{\text{sample}}(q)
=
\min_{i,k}\|q-P_{ik}\|
\]

Báo cáo:

- median distance;
- 95th percentile;
- maximum;
- fraction vượt ngưỡng;
- fraction không có sample đủ gần.

Nên đặt một `max_dist_mm` thay vì cưỡng ép mọi query point nhận một ray.

Working threshold có thể dựa trên:

\[
\frac{1}{2}\text{ voxel diagonal}
+
\frac{1}{2}\text{ ray sampling step}
\]

hoặc được chọn từ phân bố khoảng cách trên oracle reconstruction.

Nếu query point quá xa mọi ray sample, gán `unassigned`, không ép vào một thickness bin.

---

## 2.6. Sensitivity decision rule

So sánh Mapping A và Mapping B trên đủ 10 validation cases:

- P2 thin-region error;
- B0 thin-region error;
- paired P2–B0 gap;
- absent error mass;
- counts của các thickness bins;
- number of GT cartilage voxels bị gán `absent`.

### Không cần full provenance ngay nếu

- P2 vẫn tệ hơn B0 trên 10/10 ca;
- mean thin error thay đổi dưới khoảng 5%;
- paired gap không đổi đáng kể;
- absent remains the dominant failure;
- mapping inconsistency giảm rõ.

### Ưu tiên full provenance trước final Gate nếu

- thin error thay đổi trên khoảng 10%;
- case-level ranking đổi;
- P2–B0 gap giảm mạnh;
- absent mass thay đổi về bản chất;
- kết luận Gate nhạy với mapping.

Các mức 5% và 10% là engineering thresholds.

---

# 3. Split manifest và atlas/cache

## 3.1. Giữ convenience split cho engineering runs

Xác nhận:

\[
\boxed{
\text{Giữ split 40 train / 10 validation hiện tại cho canonical P2 và revision engineering}
}
\]

Lý do:

- các diagnostics hiện tại đều dựa trên split này;
- atlas D1 đã được xây từ đúng 40 ca;
- giữ cùng split cho phép so sánh apples-to-apples;
- không cần trả chi phí rebuild atlas/cache trong lúc model còn đang hỏng cơ bản;
- engineering stage nhằm sửa pipeline, không nhằm tạo final unbiased estimate.

Tuy nhiên, không nên gọi đây là final canonical evaluation split.

Tên phù hợp:

```text
engineering_split_v1_fixed
```

---

## 3.2. Final Gate phải dùng split phân tầng mới

Sau khi:

- micro-overfit đạt;
- loss/sampling được chốt;
- input được chốt;
- model architecture không còn thay đổi;
- engineering result đủ tốt;

mới tạo:

```text
final_gate_stratified_split_v1
```

Phân tầng nên cân nhắc:

- KL/severity;
- thin cartilage area;
- absent area;
- baseline thin error;
- acquisition/domain nếu có;
- class-specific thickness distribution.

Split phải được khóa trước khi xem final results.

---

## 3.3. Không được tune trên final split

Convenience validation set đã được dùng nhiều lần để:

- chọn diagnosis;
- kiểm tra thresholds;
- xem input ablations;
- quyết định revision.

Do đó không thể dùng nó như final unbiased Gate set.

Final split chỉ dùng sau khi mọi lựa chọn đã freeze.

---

## 3.4. Phải rebuild gì khi đổi split?

Chắc chắn phải rebuild:

- fold-specific atlas D1;
- atlas-derived domain masks;
- training datasets;
- validation datasets;
- result registry;
- any cache whose key contains train IDs, atlas or domain.

Không nhất thiết phải rebuild pure per-case geometry nếu cache đó chỉ chứa:

- bone surface;
- normals;
- ray points;

và key đã bao gồm:

- case data hash;
- surface source;
- spacing;
- RayConfig;
- code version.

Nếu cache hiện tại không tách pure geometry khỏi atlas-dependent domain, cách an toàn là rebuild toàn bộ.

Về lâu dài, cache nên có dependency hashes để split mới chỉ vô hiệu hóa artifacts thực sự phụ thuộc atlas.

---

# 4. Canonical P2 dùng config cũ hay config mới?

## 4.1. Chốt: canonical engineering P2 dùng config cũ

Thứ tự đúng là:

```text
Step 0: sửa bookkeeping, không thay model semantics
Step 1: chạy lại P2 config cũ
Step 2: save + reload checkpoint
Step 3: chạy toàn bộ diagnostics từ checkpoint đó
Step 4: khóa kết quả thành engineering baseline
Step 5: bắt đầu revision
```

Lý do:

1. Xác nhận mốc khoảng 2.0 mm bằng một coherent run.
2. Tách improvement do revision khỏi variation giữa sessions.
3. Kiểm tra hệ thống checkpoint/evaluation mới thực sự hoạt động.
4. Có baseline rõ để đo đóng góp của từng thay đổi.
5. Tránh so model revision mới với một tập hợp con số cũ được tạo từ nhiều model instances.

---

## 4.2. “Canonical” ở đây chỉ có nghĩa engineering reference

Nên đặt tên:

```text
P2_oldcfg_engref_v1
```

hoặc:

```text
MVP_FC_S0_D1_I2_H1_oldloss_EngSplitV1_Seed1
```

Không gọi nó là final canonical model.

Nó là:

> checkpoint-locked reference implementation của P2 cũ trên engineering split.

---

## 4.3. Step 0 không được vô tình thay đổi kết quả số

Step 0 chỉ nên thêm:

- checkpoint save/load;
- hashes;
- immutable result paths;
- cache validation;
- corrected registry;
- explicit model identity.

Không thay:

- loss;
- sampling;
- epochs;
- seed;
- model architecture;
- input normalization;
- domain;
- threshold;
- data split.

Nếu cần sửa một bug có thể thay đổi numerical output, phải:

- ghi version mới;
- không gọi nó là exact rerun của old P2;
- giữ old behavior có thể tái lập hoặc ghi rõ không còn tái lập được.

---

## 4.4. Canonical rerun tối thiểu cần tạo gì?

Từ một checkpoint duy nhất:

- ray-space occupancy Dice;
- presence metrics;
- reconstructed voxel Dice;
- §3.7 thin-region error;
- per-bin error mass;
- thin/absent recall diagnostics;
- presence-threshold sweep;
- checkpoint hash;
- git commit;
- atlas hash;
- split hash;
- config hash.

Tất cả output phải chỉ tới cùng checkpoint.

---

# 5. So sánh revision thế nào?

## 5.1. Giữ cùng engineering split

Canonical old P2 và mọi revision ban đầu phải dùng cùng:

- train IDs;
- validation IDs;
- atlas;
- surface source;
- number of rays;
- seed policy;
- epoch budget hoặc compute budget;
- evaluation functions.

Mỗi revision chỉ thay một yếu tố khi có thể.

---

## 5.2. Hai tầng so sánh

### Tầng A — so với old P2

Trả lời:

> Revision có sửa được failure mode của model cũ không?

Đo:

- thin error giảm từ khoảng 2.0 mm;
- absent FP mass;
- thin recall;
- micro-overfit;
- calibration.

### Tầng B — so với ResEnc B0

Trả lời:

> Revised ray model đã đủ tốt để tiếp tục hướng nghiên cứu chưa?

Gate chính:

\[
E_{\text{revised}} < E_{B0}
\]

với paired CI dương và không tăng absent FP.

Old P2 chỉ là engineering reference; B0 vẫn là scientific baseline.

---

# 6. Khi nào bỏ qua rerun old P2?

Chỉ nên bỏ qua nếu:

- training old P2 cực kỳ tốn kém;
- code cũ không thể tái lập;
- revision đã thay đổi toàn bộ pipeline đến mức old comparison không còn hữu ích;
- mục tiêu chỉ là exploratory coding.

Trong trường hợp hiện tại, old P2 chỉ dùng 40 ca và 30 epochs, trong khi sự mơ hồ bookkeeping đang ảnh hưởng toàn bộ chẩn đoán.

Do đó lợi ích của một clean rerun lớn hơn chi phí.

Quyết định:

\[
\boxed{
\text{Không bỏ qua canonical old-P2 rerun}
}
\]

---

# 7. Lộ trình được chốt

## Phase A — Engineering baseline

1. Sửa bookkeeping.
2. Khóa commit.
3. Xác nhận cache provenance.
4. Train old P2 trên convenience split.
5. Save checkpoint.
6. Reload checkpoint.
7. Chạy toàn bộ diagnostics.
8. Chạy Mapping A versus nearest-ray-sample Mapping B.
9. Khóa `P2_oldcfg_engref_v1`.

## Phase B — Model revision

1. Micro-overfit.
2. Bỏ hard scalar presence gate.
3. Stratified ray sampling.
4. Boundary-aware/per-ray-normalized loss.
5. MRI + gradient/depth input.
6. Clean OOF ResEnc probability nếu cần.
7. So với old P2 và B0 trên cùng engineering split.

## Phase C — Final Gate

1. Freeze model design.
2. Tạo stratified split manifest.
3. Rebuild fold-specific atlas.
4. Invalidate atlas/split-dependent cache.
5. Train/evaluate theo protocol đã khóa.
6. Chạy full provenance nếu Mapping audit cho thấy cần.
7. Quyết định Stage 2.

---

# 8. Câu kết luận ngắn cho tài liệu

> For the mapping sensitivity audit, nearest sampled-ray assignment will be used as a low-cost ray-ownership proxy. It is not equivalent to true splat provenance, because the current trilinear splatting implementation aggregates contributions with `np.bincount` and discards source-ray identities. Full provenance will be reserved for the final Gate unless the proxy audit shows that the main conclusion is sensitive to the mapping method.

> The existing 40/10 convenience split will remain fixed for the clean rerun of the old P2 configuration and for engineering revisions, preserving comparability with the current diagnostics and avoiding premature atlas reconstruction. A new stratified split, together with a rebuilt fold-specific atlas and invalidated dependent caches, will be introduced only after the model design is frozen for the final MVP Gate.

> The first checkpoint-locked rerun will use the unchanged P2 configuration. Its purpose is to establish a coherent engineering reference for the approximately 2.0 mm failure and to validate the new bookkeeping system. Model revisions will then be compared against this reference and against ResEnc B0 under the same engineering protocol.
