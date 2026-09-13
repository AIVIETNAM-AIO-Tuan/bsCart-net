# LITERATURE CHECK — CÁC HƯỚNG GEOMETRY/SHAPE PRIOR CÓ VƯỢT nnU-Net KHÔNG?

## Kết luận ngắn

Có. Một số nghiên cứu về knee-cartilage segmentation có dùng shape/anatomical priors hoặc surface-aware modeling và báo cáo vượt nnU-Net.

Tuy nhiên, bằng chứng không đồng nhất:

- Có paper báo cáo **gain rất lớn** so với 3D nnU-Net.
- Có paper chỉ tăng Dice khoảng **0.1–0.3 điểm phần trăm**.
- Có paper tăng overall score nhưng **cartilage Dice gần như không hơn nnU-Net**.
- Paper gần nhất với bone-surface implicit modeling (CartiSurface) cho kết quả rất tốt, nhưng bảng kết quả công khai hiện không có dòng nnU-Net dù phần Methods liệt kê nnU-Net là baseline, nên không nên dùng paper đó để tuyên bố direct nnU-Net superiority.

Điểm nhất quán đáng chú ý: các phương pháp thành công thường dùng **global/contextual backbone + geometry/shape prior**, không chỉ đọc profile MRI cục bộ theo một tia.

---

## 1. Multi-Task Consistency Network with Deep Shape Prior — 2026

Dataset: OAI-ZIB, 507 MRI, 2-fold cross-validation.

Phương pháp:
- visual-transformer backbone;
- segmentation branch;
- medial-surface branch;
- contour branch;
- reconstruction consistency;
- learned deep shape prior;
- recursive refinement.

So với 3D nnU-Net:

| Metric | 3D nnU-Net | Proposed |
|---|---:|---:|
| Femoral Dice | 93.06% | **96.72%** |
| Femoral ASD | 0.22 mm | **0.12 mm** |
| Tibial Dice | 86.52% | **92.77%** |
| Tibial ASD | 0.29 mm | **0.20 mm** |

Đây là bằng chứng trực tiếp mạnh rằng shape/geometry information có thể vượt nnU-Net trên OAI-ZIB.

Nhưng kiến trúc này không phải local ray model: nó vẫn có backbone giàu context và dùng geometry như multi-task supervision + prior constraint.

---

## 2. PCAM — Position-prior Clustering-based Self-attention — MICCAI 2022

Dataset: OAI-ZIB, 507 MRI.

So trực tiếp:

| Metric | nnU-Net | nnU-Net + PCAM |
|---|---:|---:|
| Femoral Dice | 89.03% | **89.35%** |
| Femoral ASSD | 0.2551 mm | **0.2389 mm** |
| Tibial Dice | 86.00% | **86.14%** |
| Tibial ASSD | **0.2117 mm** | 0.2165 mm |

Kết quả:
- Dice tăng rất nhỏ: +0.32 pp femoral, +0.14 pp tibial.
- Femoral ASSD tốt hơn.
- Tibial ASSD hơi tệ hơn.

Meta-review của MICCAI cũng lưu ý kết quả nnU-Net và nnU-Net+PCAM rất gần nhau và yêu cầu thảo luận significance. Authors báo significance rõ cho segmentation continuity (Betti number), không phải một superiority lớn trên mọi segmentation metric.

Diễn giải: position/context prior có thể cải thiện nnU-Net, nhưng gain có thể rất nhỏ.

---

## 3. Adversarial prior-shape constraint — Frontiers in Medicine 2022

Dataset: SKI10.

Phương pháp dùng adversarial loss như shape consistency prior, kết hợp nhiều stage.

Overall SKI10 score:
- Cascade nnU-Net: 75.4
- nnU-Net 3D low-res: 75.3
- Proposed: **76.2**

Nhưng cartilage Dice:

| | nnU-Net 3D full | Proposed |
|---|---:|---:|
| Femoral cartilage | 0.89 | 0.89 |
| Tibial cartilage | 0.88 | 0.88 |

Vì vậy paper này vượt nnU-Net về overall challenge score, nhưng **không chứng minh gain rõ trên cartilage Dice**.

Đây là ví dụ quan trọng: anatomical/shape constraints không tự động làm cartilage segmentation tốt hơn nnU-Net.

---

## 4. CartiSurface — npj Digital Medicine 2025

Đây là công trình gần ý tưởng của dự án nhất:

- implicit SDF;
- conditioned on femoral/tibial bone surfaces;
- dùng distances và normals từ bone surfaces;
- geometry-aware losses:
  - spacing;
  - parallelism;
  - smoothness;
- continuous surface/thickness modeling.

Kết quả được báo cáo:

- Dice: **0.91**
- thickness MAE: **0.28 mm**
- Hausdorff: **2.13 mm**
- removing bone-surface conditioning làm MAE tăng **0.28 → 0.37 mm**.

Paper viết rằng nnU-Net được dùng như một voxel baseline và tuyên bố CartiSurface outperform các baselines.

Tuy nhiên, bảng định lượng công khai của paper hiện hiển thị U-Net, SegResNet, SurfaceFlow, TopoShapeNet, ImplicitRecon... nhưng **không có dòng nnU-Net**.

Do đó nên dùng CartiSurface để nói:

> bone-conditioned implicit geometry mang lại lợi ích rõ.

Không nên dùng nó để nói:

> CartiSurface đã chứng minh trực tiếp vượt nnU-Net,

trừ khi supplementary/source khác cung cấp con số nnU-Net cụ thể.

---

# Ý nghĩa đối với MVP hiện tại

Literature không mâu thuẫn với kết quả P2 âm tính.

Các phương pháp geometry-aware thành công thường có cấu trúc:

```text
MRI
  ↓
strong 2D/3D contextual encoder
  ↓
semantic/context features
  +
shape / surface / position prior
  ↓
geometry-aware refinement / constraint
```

P2 hiện tại gần hơn với:

```text
MRI local profile trên một normal ray
  ↓
1D encoder
  ↓
cartilage occupancy
```

P2 đã bỏ phần lớn cross-ray/global context mà nnU-Net đang có.

Vì vậy P2 thất bại không chứng minh geometry prior vô ích. Nó chứng minh:

> geometry alone + local ray evidence không đủ để thay một strong 3D contextual segmenter.

## Phép thử quan trọng nhất còn thiếu

P3/M8-D:

```text
MRI ray
+
clean OOF ResEnc probability
+
surface coordinate
→ refinement
```

sẽ gần tinh thần của literature thành công hơn.

Nếu P3 vượt B0 ở thin-region metric:
- kết luận: surface coordinate hữu ích như **refinement space**.

Nếu P3 vẫn không vượt B0:
- negative result cho hướng femoral sẽ mạnh hơn nhiều.

---

# Điểm cần thận trọng khi so số với project

Không so trực tiếp Dice/ASD giữa paper và project như cùng benchmark vì:

- split khác;
- preprocessing khác;
- nnU-Net version/config khác;
- OAI-ZIB vs OAI-iMorphics/SKI10 khác;
- severity distribution khác;
- metric surface definition có thể khác;
- một số paper resample isotropic trước evaluation;
- project đang nhắm thin/absent-specific error chứ không chỉ global Dice.

Điều có thể so đáng tin hơn là **relative comparison trong cùng paper**: proposed method so với nnU-Net dưới cùng experimental protocol.
