# nnUNet-OAI — Knee MRI segmentation (OAI)

## Boi canh nhanh

Muc tieu: cai thien phan doan **sun** khop goi tren MRI DESS. Xuong da gan nhu bao hoa
(Dice 0.986/0.987, ASSD 0.17-0.25mm); sun dung o ~0.85-0.89 va **co the da sat tran
nhieu chu thich** (inter-observer ~0.85-0.90).

## Trang thai da xac lap (dung lam lai)

- **ROI cascade => AM TINH.** `Dataset021_CartROI` (crop GT bbox) truot cong upper-bound:
  Dice sun chi dich +-0.003 du localization HOAN HAO. => **Localization KHONG phai nut that.**
  Khong thu lai ROI/finer-spacing. Target spacing da la native `[0.70, 0.3646, 0.3646]`.
- **SKM-TEA => DROPPED.** Cross-domain collapse (fem 0.405, tib-cart ~0) da xac nhan la
  domain shift THAT (qDESS vs DESS) qua viz 2D+3D, khong phai bug converter.
- **RO RI DU LIEU (da xac nhan):** 51/70 dau goi iMorphics co V00 va V01 o hai fold khac nhau
  (nnUNet KFold thuan theo ten case). => **So CV bi thoi phong.** So *test* van sach
  (iMorphics train/test CO tach theo benh nhan: 70 bn x2 train, 18 bn x2 test).
  Bao cao so test, khong bao cao so CV.

## Dataset

| ID | Ten | Ghi chu |
|---|---|---|
| 001 | `Dataset001_KneeOA` | OAI-ZIB, 404 train + 103 test. **NGUON DUY NHAT CO GT XUONG** (label 1,3) |
| 012/013 | iMorphics | 140 train / 36 test. Co sun+meniscus+patella, **KHONG co xuong** |
| 020 | `Dataset020_KneeUnion` | 544 ca, 8-class. Model chinh (= baseline B0) |
| 021 | `Dataset021_CartROI` | Negative baseline (ROI cascade) |

Label union 8-class: `1 femoral_bone, 2 femoral_cart, 3 tibial_bone, 4 med_tib_cart,
5 lat_tib_cart, 6 med_meniscus, 7 lat_meniscus, 8 patellar_cart`

Baseline B0 (d020 ResEnc-L 250ep fold0, OAI-ZIB test n=103):
fem_cart **0.891** (ASSD 0.21) | med_tib **0.852** (0.27) | lat_tib **0.868** (0.27)

## Huong dang trien khai: Bone-Surface Coordinate

Xem `bone_surface_cartilage_architecture_validation_plan.md`. Gia thuyet: neo he toa do
vao **be mat xuong** (1 truc theo phap tuyen, 2 truc theo mat), bien sun thanh bai toan 1D
doc chieu day. Ky vong cai thien o **bien va do day** noi sun cuc mong / mat han — KHONG
phai o Dice tong the.

**Cong chan chinh = M0 (do headroom) truoc khi xay bat cu thu gi.** Day la cau hoi ROI
cascade da quen hoi. Neu `dASSD_prize < 0.04mm` => DUNG, cong bo ket qua am tinh.

### Da do bang phantom o dung spacing that (`bsc/tests/test_core.py`, 16/16 dau)

**Tran round-trip SUP o vung sun mong** - dung vung ma gia thuyet nham toi:

| do day sun | M2 Dice | M2 ASSD |
|---|---|---|
| 1.6mm | 0.9995 | 0.001mm |
| 0.5mm | 0.925 | 0.030mm |
| 0.4mm | **0.848** | 0.065mm |

Voi sun 0.4mm, phep doi toa do TU NO mat 15% Dice du du doan hoan hao.
=> **Cong M2 cua plan doc (Dice >= 0.97) se TRUOT o sun cuc mong. DAT CONG TREN ASSD**
(0.001-0.065mm, duoi xa muc tieu 0.1mm). Dieu nay CUNG CO quyet dinh cua ke hoach:
**metric BIEN la chinh, khong phai Dice** - Dice tren cau truc day ~1 voxel nhay den
muc tan nhan.

**Remesh dong deu (Step 2 cua doc) PHA reconstruction** => da bac bo:
full MC bo sot 0.0% voxel sun | thua 4x mat 14.2% | thua 8x mat 39.3%.
=> subsample tu do khi TRAIN; FULL marching-cubes density khi INFERENCE.

**KHONG tai lap:** "sigma=1.0mm lam M3 truot". Phantom cau tron cho M3=100% o moi
sigma. Doi mm->voxel VAN dung (co test khoa), nhung sigma chua chung minh la knob
quyet dinh - kiem lai tren xuong that o Phase 2.

## Quy tac lam viec

- **Non-destructive:** thi nghiem moi = dataset ID moi + folder moi + notebook moi.
  Doc data/model cu **read-only**. Khong ghi de ket qua cu.
- **Drive-first:** MOI artifact (data, checkpoint, ket qua) nam duoi
  `/content/drive/MyDrive/bsc/`. **KHONG BAO GIO ghi vao `/content/`** — do la ly do
  544 prediction va toan bo output CV da mat mot lan.
- **Secrets:** token lay tu Colab Secrets (`userdata.get("HF_TOKEN")`), **khong hardcode**.
- **Metric co dung MOT dinh nghia** trong `bsc_metrics.py`. Khong copy-paste sang notebook
  (`surf()` da bi trung o merge_s6 cell 22 va merge_s9 cell 18 roi troi khac nhau).
- Moi thu chay tren **Colab**. May local: torch CPU-only, thieu SimpleITK/nibabel/nnunetv2.

## Layout

```
bsc_core.py      # sdf/surface/normals/rays/splat  (chi scipy+skimage, khong can mesh lib)
bsc_metrics.py   # dice/assd/hd95/surface_dice/boundary_mae/vol_err
bsc_model.py     # RayEncoder1D + train loop
bsc_track.py     # exp_id, git sha, config, per-case metrics
tests/           # phantom asserts, chay local khong can data
notebooks/bsc_*  # mot notebook moi phase, chay top-to-bottom tren Colab
```
