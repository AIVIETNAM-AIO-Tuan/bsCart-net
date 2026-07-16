# bsc — Bone-Surface Coordinate Cartilage Segmentation (Stage 1 / MVP)

Biệt lập hoàn toàn với phần còn lại của repo. Đọc data/model cũ **read-only**.

## Giả thuyết

Neo hệ tọa độ vào **bề mặt xương** (1 trục theo pháp tuyến, 2 trục theo mặt) ⇒ sụn
thành bài toán 1D dọc chiều dày. Kỳ vọng cải thiện ở **biên và độ dày** nơi sụn cực
mỏng / mất hẳn (OA nặng) — **không** phải ở Dice tổng thể.

Bối cảnh: ROI cascade đã bác bỏ "thu hẹp field-of-view giúp sụn" (Dice ±0.003 dù bbox
GT hoàn hảo). Đây là hướng khác — đổi *biểu diễn*, không đổi *vùng nhìn*.

## Điều đã đo bằng phantom (không phải phỏng đoán) — `pytest bsc/tests`

| | Kết quả | Hệ quả |
|---|---|---|
| Round-trip sụn 1.6mm | Dice 0.9995, ASSD 0.001mm | Phép đổi tọa độ gần như vô tổn ở sụn dày |
| Round-trip sụn **0.4mm** | Dice **0.848**, ASSD 0.065mm | **Cổng M2 phải đặt trên ASSD, không phải Dice** |
| Remesh thưa 8× | mất 39% voxel sụn | **KHÔNG remesh khi inference** (bác bỏ Step 2 của doc) |
| Đảo dấu pháp tuyên | occupancy → 0 | Bug im lặng; `check_normals` bắt nó |

## Thứ tự chạy & cổng chặn

| Phase | Notebook / module | Cổng | Chi phí |
|---|---|---|---|
| 0. Bootstrap | `notebooks/bsc_00_bootstrap.ipynb` | **Gate 0**: metric khớp bảng ±0.005 | Colab CPU ~30ph |
| 1. **M0 headroom** | `notebooks/bsc_01_headroom.ipynb` + `headroom.py` | **Gate 1** ⭐ | Colab CPU ~1h |
| 2. Geometry QC | (Phase 2) M3→M2→M4 | **Gate 2** | Colab CPU ~1h |
| 3. Ray DB | (Phase 3) domain + `core.py` | **Gate 3**: recall ≥0.99 | Colab CPU ~30ph |
| 4. Train MVP | (Phase 4) `model.py` + M5 | **Gate 4** (khoa học) | Colab GPU ~2h |
| 5. Controls | (Phase 5) M6/M7/M8 | **Gate 5** (go/no-go) | Colab GPU ~3h |

**Gate 1 là cổng chặn chính.** ~1h CPU có thể kết thúc dự án — đó là mục đích, và là
thứ ROI cascade đã không có:
- ✅ PROCEED: `ΔASSD_prize ≥ 0.08mm` **và** thin+absent ≥ 35% error mass
- ⚠️ RESCOPE: `ΔASSD_prize ∈ [0.04, 0.08)` → viết lại mục tiêu quanh presence F1 + thickness MAE
- ❌ STOP: `ΔASSD_prize < 0.04mm` **hoặc** bậc thang z chiếm ưu thế → công bố kết quả âm tính

## Module

| File | Vai trò | Test |
|---|---|---|
| `core.py` | SDF, bề mặt, pháp tuyến, tia, occupancy, splat, domain | `test_core.py` (16) |
| `metrics.py` | Dice/ASSD/HD95/**Surface Dice**/**boundary MAE** + bootstrap CI | `test_metrics.py` (20) |
| `headroom.py` | **M0** — phân rã error mass, counterfactual, Gate 1 | `test_headroom.py` (4) |
| `track.py` | exp_id, git sha, config, metric per-case (§7) | — |
| `io_utils.py` | I/O Colab (nibabel), giải nén baseline | — (chạy trên Colab) |
| `make_splits.py` | Khôi phục + ghim splits, sửa rò rỉ V00/V01 | — |
| `splits/*.json` | Splits đã ghim (immutable) | — |

Chạy test local: `python -m pytest bsc/tests -v` (không cần data, không cần GPU).

## Drive-first (bắt buộc)

Mọi artifact nằm dưới `BSC_ROOT = /content/drive/MyDrive/bsc/`. **Không bao giờ ghi vào
`/content/`** — đó là lý do 544 prediction đã mất một lần. `track._assert_drive_first`
và `io_utils.assert_drive_first` ném lỗi nếu ai đó vi phạm.

```
BSC_ROOT/
├── splits/       # đã ghim, immutable
├── baselines/    # B0 (250ep f0), B1 (150ep f0-4, 544 pred CV), B2 (ROI cascade)
├── geom/         # verts, normals (float16)
├── raydb/        # X, y, meta (float16) — 60 train / 20 val, KHÔNG cần 404 ca
├── atlas/        # atlas fold-specific
└── runs/<exp_id>/# config.json + metrics.csv per-case + ckpt + qc/
```

## Dùng từ Colab

```python
from google.colab import drive; drive.mount("/content/drive")
!git clone <repo> /content/repo
import sys; sys.path.insert(0, "/content/repo")
from bsc import core, metrics, headroom
```

## Rủi ro đã biết (đọc trước khi báo cáo số)

- **Rò rỉ V00/V01:** 51/70 đầu gối iMorphics có 2 timepoint ở 2 fold khác nhau ⇒ số
  CV bị thổi phồng. Số **test** vẫn sạch. Dùng `splits_zib_v1_fixed.json` cho mọi
  training mới. Stage 1 chỉ dùng OAI-ZIB nên ảnh hưởng gián tiếp.
- **KL grade:** ở HuggingFace `YongchengYAO/OAIZIB-CM`. Bin độ dày là stratifier chính;
  KL là bổ sung nếu map được.
- **σ làm mượt:** "σ=1.0mm làm M3 trượt" **không tái lập** trên phantom. Kiểm lại trên
  xương thật ở Phase 2 trước khi coi là knob.
