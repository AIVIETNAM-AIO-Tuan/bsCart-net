# DETAILED VALIDATION PLAN FOR A BONE-SURFACE COORDINATE CARTILAGE SEGMENTATION ARCHITECTURE

## 1. Overall Research Objective

Validate the following hypothesis:

> Representing cartilage in a coordinate system anchored to the bone surface, with one axis along the surface normal and two axes along the surface, can localize cartilage boundaries more accurately than a voxel-space ResEnc model, especially in regions with extremely thin or completely absent cartilage in severe osteoarthritis.

The system will be developed in three stages:

| Stage | Primary question | Complexity |
|---|---|---|
| 1. MVP | Does the surface-normal coordinate system actually provide a useful signal? | Low |
| 2. Main model | Can this signal be turned into a stable end-to-end model? | Medium to high |
| 3. Extension | Do continuous fields, multi-ray sampling, and uncertainty modeling provide additional value? | High |

---

# 2. Common Experimental Setup

## 2.1. Lock the Baselines

Use the current ResEnc predictions and checkpoints as fixed baselines:

- ResEnc five-fold models.
- The same preprocessing.
- The same target spacing.
- The same training, validation, and test splits.
- The same post-processing.
- Do not retrain the baseline between ablations unless strictly necessary.

The following baselines should be preserved:

| ID | Baseline |
|---|---|
| B0 | Current ResEnc model |
| B1 | Five-fold ResEnc ensemble |
| B2 | Previously tested ROI cascade |
| B3 | ResEnc with ground-truth bone masks, used only for upper-bound analysis |

The ROI cascade should be retained as a **negative or neutral baseline**, demonstrating that reducing the field of view alone is not sufficient to solve the problem.

---

## 2.2. Lock the Data Splits

Every experiment must use the same splits.

Principles:

- The same patient must never appear in both training and test sets.
- If multiple time points exist, all time points from one patient must belong to the same fold.
- If multiple datasets or domains are available, preserve separate:
  - Internal validation.
  - External validation.
  - KL4 test subset.
- A bone-surface atlas must be built separately inside each training fold.
- Validation and test cartilage masks must never be used to construct the articular atlas.

---

## 2.3. Common Metrics

### Volumetric metrics

- Dice.
- Precision.
- Recall.
- False-positive volume.
- False-negative volume.

### Surface metrics

- Average Symmetric Surface Distance, or ASSD.
- 95th-percentile Hausdorff Distance, or HD95.
- Surface Dice at 0.5 mm.
- Surface Dice at 1.0 mm.
- Inner-boundary mean absolute error.
- Outer-boundary mean absolute error.

### Morphological metrics

- Cartilage thickness mean absolute error.
- Cartilage volume error.
- Full-thickness cartilage-loss area error.
- Presence/absence F1 score on the articular surface.

### Stratified reporting

Every metric should be reported by:

- Cartilage class.
- KL0–2, KL3, and KL4.
- Cartilage volume.
- Local cartilage thickness.
- Dataset or acquisition domain.
- Bone-surface prediction quality.

Proposed thickness groups:

| Group | Ground-truth local thickness |
|---|---:|
| Absent | 0 mm |
| Extremely thin | \(0 < t \leq 0.5\) mm |
| Thin | \(0.5 < t \leq 1.0\) mm |
| Moderate | \(1.0 < t \leq 2.0\) mm |
| Thick | \(t > 2.0\) mm |

These thresholds may be adjusted after examining the real thickness distribution and image spacing.

---

## 2.4. Statistical Analysis

Do not compare only mean Dice.

Required analyses:

- Paired per-case comparisons.
- Bootstrap 95% confidence intervals.
- Wilcoxon signed-rank tests as supplementary analyses.
- Effect sizes.
- The number of cases that improved, remained unchanged, or degraded.

A model should be considered improved only when:

1. The effect is consistent across multiple folds.
2. The result is not driven only by a few easy cases.
3. The benefit is clear in thin-cartilage or KL4 subgroups.
4. False-positive cartilage does not increase substantially in regions with complete cartilage loss.

---

# STAGE 1 — MVP: PROOF OF CONCEPT

## 3. Objective

Answer the simplest possible question:

> When the bone surface and its normal directions are known accurately, can a model reading MRI intensity and features along the surface normal identify cartilage more accurately than ResEnc in thin-cartilage regions?

This stage does not require:

- A surface graph.
- Cross-surface attention.
- An implicit neural field.
- End-to-end bone prediction.
- Multi-ray sampling.
- Joint fine-tuning of the complete ResEnc model.

The MVP should remain as simple as possible so that the source of any improvement is identifiable.

---

## 3.1. Cartilage Scope

Start with one or two classes.

### Primary class

**Femoral cartilage**

Reasons:

- Large surface area.
- Many available sampling points.
- Bone segmentation is generally stable.
- The pipeline is relatively easy to debug.

### Stress-test class

**Medial tibial cartilage**

Reasons:

- It is commonly affected by thinning and severe degeneration.
- It is suitable for KL3–4 stress testing.
- Its geometry is generally simpler than patellar cartilage.

Do not implement all cartilage classes in the first experiment.

---

## 3.2. Input

The MVP uses:

1. MRI intensity.
2. Ground-truth bone masks.
3. Ground-truth cartilage masks for target construction.
4. Coarse cartilage probabilities from ResEnc in a separate ablation.
5. High-resolution ResEnc features in a separate ablation.

The main input for each ray is:

\[
X_i \in \mathbb{R}^{K \times C}
\]

where:

- \(K = 48\) or \(64\) samples along depth.
- The ray may cover, for example, \(-1\) mm to \(+6\) mm relative to the bone surface.
- The initial channels contain only:
  - MRI intensity.
  - MRI gradient.
  - Bone signed-distance value.

Additional channels should be introduced only through controlled ablations.

---

## 3.3. Output

### Primary output

Cartilage occupancy along each ray:

\[
\hat c_i(d_k) \in [0,1]
\]

### Auxiliary output

Cartilage-presence probability:

\[
\hat z_i = P(\text{cartilage is present on the ray})
\]

Direct thickness regression is not required in the first MVP.

Thickness can initially be derived from the occupancy target and prediction.

---

## 3.4. Processing Pipeline

### Step 1: Construct the bone signed-distance field

From the ground-truth bone mask:

\[
\phi_b(x) = \text{signed distance to the bone boundary}
\]

Requirements:

- Distance must be measured in millimetres, not voxels.
- Negative values inside the bone.
- Positive values outside the bone.
- Apply light smoothing before computing gradients.

### Step 2: Extract the bone surface

Possible methods:

- Marching cubes.
- Zero-level samples from the signed-distance field.
- Surface-voxel centres for the earliest prototype.

The surface should then be remeshed or subsampled so that point spacing is reasonably uniform.

### Step 3: Calculate surface normals

\[
n(s) =
\frac{\nabla \phi_b(s)}
{\|\nabla \phi_b(s)\|+\epsilon}
\]

Normals must point outward from the bone.

### Step 4: Define the articular domain

The MVP should test two versions.

#### MVP-A: Oracle articular region

Define the articular region from the attachment region of the ground-truth cartilage on the bone.

Purpose:

- Test only localization along the normal direction.
- Do not yet test cartilage absence or atlas generalization.

#### MVP-B: Fold-specific population atlas

Construct the articular domain from the training cases inside each fold.

Purpose:

- Support testing of complete cartilage absence.
- Avoid using the test case’s cartilage mask to define its region of interest.

### Step 5: Sample rays

For each surface point:

\[
r_i(d_k)=s_i+d_kn_i
\]

Use trilinear interpolation to sample:

- MRI intensity.
- MRI gradient.
- Bone signed-distance values.
- Coarse ResEnc probability, when included.

### Step 6: Create occupancy targets

Sample the ground-truth cartilage mask along the same ray:

\[
c_i(d_k)\in\{0,1\}
\]

Store:

- Cartilage presence.
- First cartilage intersection.
- Last cartilage intersection.
- Number of cartilage intervals.
- Local ground-truth thickness.

### Step 7: Train the 1D Ray Encoder

Proposed MVP architecture:

```text
Input K × C
→ Conv1D block
→ Dilated Conv1D block
→ Dilated Conv1D block
→ Depth-wise attention or pooling
→ Occupancy head: K × 1
→ Presence head: 1 × 1
```

Loss:

\[
L =
L_{\text{occupancy}}
+
\lambda_p L_{\text{presence}}
\]

Possible choices:

- Binary cross-entropy or focal binary cross-entropy for occupancy.
- Balanced binary cross-entropy for presence.
- Avoid introducing many loss terms in the MVP.

### Step 8: Reconstruct the volume

Map ray predictions back to the voxel grid using:

- Weighted splatting.
- Or nearest-surface querying.

The result is a probability volume:

\[
P_{\text{ray}}(x)
\]

Evaluate two modes:

1. Ray-only segmentation.
2. Simple fusion with ResEnc:

\[
P_{\text{fused}}
=
\alpha P_{\text{ray}}
+
(1-\alpha)P_{\text{ResEnc}}
\]

Select \(\alpha\) using the validation set only.

---

## 3.5. Mandatory MVP Side Tests

### Test M1 — Synthetic shell phantom

Create synthetic data containing:

- A sphere or ellipsoid representing bone.
- A cartilage shell with a thickness of 1–5 voxels.
- Regions with zero thickness.
- MRI-like blur and noise.
- Accurate normals and perturbed normals.

Purpose:

- Verify that the pipeline can learn basic shell geometry.
- Test thin-shell detection.
- Test complete-absence detection.

Acceptance criteria:

- Near-perfect performance under low noise.
- Controlled degradation as the shell becomes thinner or normal directions become noisier.

---

### Test M2 — Coordinate round-trip test

Using the ground-truth cartilage:

1. Project the ground-truth mask into normal-ray coordinates.
2. Reconstruct it back into voxel space.
3. Compare the reconstructed target with the original mask.

Purpose:

> Determine the upper bound imposed by the coordinate transformation and rasterization process.

Target acceptance:

- Very high round-trip Dice, ideally above 0.97–0.98 in the central articular region.
- Surface error below a fraction of a voxel.
- If round-trip performance is poor, do not train the model yet.

---

### Test M3 — Normal-orientation test

Verify:

\[
\phi(s-\epsilon n)<0
\]

and:

\[
\phi(s+\epsilon n)>0
\]

Acceptance criteria:

- More than 99.5% of normals point outward correctly.
- Incorrect normals must be visualized and either corrected or removed.

---

### Test M4 — Single-interval analysis

Calculate:

\[
R_{\text{single}}
=
\frac{
\text{number of rays containing at most one cartilage interval}
}{
\text{number of rays containing cartilage}
}
\]

Decision rules:

- If \(R_{\text{single}}>95\%\), a boundary-based representation is feasible.
- If \(R_{\text{single}}<90\%\), retain the occupancy decoder and do not use two-boundary regression yet.
- Pay particular attention to cartilage margins and high-curvature regions.

---

### Test M5 — Tiny-set overfitting

Train on only 2–4 cases.

Purpose:

- Confirm that the model can nearly memorize the training rays.
- Failure to overfit suggests a problem in the input, target, architecture, loss, or reconstruction.

Acceptance criteria:

- Very high training occupancy Dice.
- Clear reduction in training boundary error.
- Visual alignment between prediction and target.

---

### Test M6 — Negative-control directions

Compare:

1. Surface-normal rays.
2. Axial-direction rays.
3. Randomly rotated rays.
4. Surface-tangent rays.

Purpose:

> Demonstrate that the benefit comes from the bone-surface coordinate system, not merely from adding another network.

The surface-normal model should perform clearly better on boundary and thickness metrics.

---

### Test M7 — Surface-jitter test

Perturb the ground-truth surface by:

\[
\Delta s \in
\{0,\ 0.25,\ 0.5,\ 1.0\}\text{ mm}
\]

Rotate normals by:

\[
\Delta\theta \in
\{0^\circ,\ 5^\circ,\ 10^\circ,\ 15^\circ\}
\]

Evaluate:

- Occupancy Dice.
- Boundary mean absolute error.
- Presence F1 score.

Purpose:

- Measure sensitivity before switching to predicted bone surfaces.

---

### Test M8 — Input ablation

| ID | Input |
|---|---|
| M8-A | MRI only |
| M8-B | MRI + gradient |
| M8-C | MRI + bone signed-distance field |
| M8-D | MRI + coarse cartilage probability |
| M8-E | MRI + high-resolution ResEnc features |

Objectives:

- Determine whether the ray model can interpret MRI independently or mainly acts as a ResEnc correction model.
- Avoid retaining unnecessary input channels.

---

## 3.6. MVP Experiment Matrix

| ID | Bone surface | Articular region | Input | Output |
|---|---|---|---|---|
| P0 | Ground truth | Oracle | MRI | Occupancy |
| P1 | Ground truth | Oracle | MRI + gradient | Occupancy |
| P2 | Ground truth | Atlas | MRI + signed-distance field | Occupancy + presence |
| P3 | Ground truth | Atlas | MRI + coarse logits | Occupancy + presence |
| P4 | Ground truth + jitter | Atlas | Best P-model input | Occupancy + presence |
| P5 | Predicted | Atlas | Best P-model input | Occupancy + presence |

---

## 3.7. MVP Go/No-Go Gate

Proceed to the Main Model only if most of the following conditions are satisfied:

1. The ground-truth-surface ray model improves outer-boundary error in thin-cartilage regions by approximately 10% or more relative.
2. Surface Dice at 0.5 mm improves clearly.
3. Overall Dice does not decrease materially.
4. The presence head identifies cartilage absence better than thresholding the ResEnc output directly.
5. Surface-normal rays outperform negative-control directions.
6. The predicted-surface model preserves a meaningful portion of the gain achieved with ground-truth surfaces.
7. The result is repeatable across multiple folds or random seeds.

Suggested working targets:

- Boundary mean absolute error decreases by at least 0.1 mm or 10%.
- Surface Dice at 0.5 mm increases by at least 2 percentage points.
- Overall Dice decreases by no more than 0.5 percentage points.
- Thin-cartilage Dice or recall improves consistently.

If the ground-truth-surface model does not outperform ResEnc, stop or revise the hypothesis before building the graph-based model.

---

# STAGE 2 — MAIN MODEL

## 4. Objective

Build a model that operates with predicted bone surfaces and has sufficient context to handle:

- Extremely thin cartilage.
- Full-thickness cartilage loss.
- Focal defects.
- Meniscus–cartilage confusion.
- Surface-normal errors.
- Relationships between opposing bone surfaces.

The Main Model can still use discrete occupancy or boundary bins. A continuous implicit field is reserved for the Extension stage.

---

## 4.1. Input

### Volumetric input from ResEnc

- MRI.
- Bone probability maps.
- Coarse cartilage probability maps.
- Meniscus probability maps.
- High-resolution feature maps.
- Intermediate-resolution semantic feature maps.

### Surface input

At each surface node:

- Surface coordinate.
- Surface normal.
- Tangent frame.
- Curvature.
- Bone identity.
- Compartment identity.
- Atlas coordinate.
- Articular probability.

### Ray input

Along each normal ray:

- MRI intensity.
- MRI gradient.
- Bone probability.
- Source-cartilage probability.
- Opposing-cartilage probability.
- Meniscus probability.
- Source-bone signed-distance field.
- Opposing-bone signed-distance field.
- Sampled ResEnc features.

### Joint-geometry input

- Distance to the opposing bone.
- Angle between the source normal and the direction to the opposing surface.
- Opposing-bone curvature.
- Joint-space width.
- Meniscus probability within the space between the bones.

---

## 4.2. Output

The Main Model produces:

1. Cartilage occupancy profile.
2. Cartilage-presence probability.
3. Inner-boundary distribution.
4. Outer-boundary distribution.
5. Local cartilage thickness.
6. Prediction uncertainty.
7. Reconstructed cartilage volume.
8. Gated fusion output with ResEnc.

Not all heads should be activated in the first run. Add them through controlled ablations.

---

## 4.3. Architecture

```text
ResEnc backbone
      │
      ├── Bone probabilities
      ├── Coarse cartilage probabilities
      └── Feature pyramid
               │
               ▼
Predicted bone signed-distance fields and surfaces
               │
               ▼
Surface-normal ray sampling
               │
               ▼
1D Ray Encoder
               │
               ▼
Intra-surface Graph Transformer
               │
               ▼
Cross-surface geometry and attention
               │
        ┌──────┼────────┐
        ▼      ▼        ▼
 Presence  Occupancy  Boundary
   head      head       heads
        └──────┬────────┘
               ▼
Volumetric reconstruction
               │
               ▼
Surface–voxel gated fusion
               │
               ▼
Final prediction
```

---

## 4.4. Detailed Processing

### Step 1: Predicted bone-surface generation

Create bone surfaces from the probability maps or thresholded predictions.

Preserve two variants:

- Hard binary surface.
- Soft signed-distance or probability representation.

Compare them to determine which is more stable.

### Step 2: Articular-atlas mapping

Map the population atlas to each patient-specific bone surface.

Progress from simple to complex:

1. Landmark-based rigid or affine registration.
2. Non-rigid surface registration.
3. Projection of atlas probabilities using nearest-surface mapping.
4. Learnable atlas-refinement head.

Begin with fixed registration before adding learnable deformation.

### Step 3: Ray sampling

Generate rays from predicted surfaces.

Use variable ray length:

\[
d_{\max,i}
=
\min(
d_{\text{fixed max}},
D_{\text{opposing bone},i}-\epsilon
)
\]

Do not allow rays to extend deeply into the opposing bone.

### Step 4: Ray-feature encoding

The 1D encoder produces:

- Depth-wise feature tokens.
- A ray-level embedding.

### Step 5: Intra-surface message passing

Connect nodes using:

- Mesh adjacency.
- Geodesic nearest neighbours.
- Local surface patches.

Avoid global full-mesh attention initially because it is expensive and may oversmooth focal defects.

### Step 6: Cross-surface context

Each node attends to a limited set of opposing nodes selected by:

- Physical distance.
- A normal-direction cone.
- Anatomical compartment.

Cross-surface context should remain soft information rather than a hard one-to-one correspondence.

### Step 7: Structured prediction

The presence head predicts:

\[
P(Z_i=1)
\]

When cartilage is present, the occupancy or boundary head predicts its location.

Do not force an outer boundary to exist when the presence target is zero.

### Step 8: Reconstruction

Reconstruct each cartilage class into a voxel probability volume.

### Step 9: Fusion

A gating network combines:

- ResEnc logits.
- Surface-model logits.
- Distance to the bone surface.
- Prediction uncertainty.

Outside the articular band, prioritize ResEnc.

Inside thin-cartilage regions, prioritize the surface branch when its uncertainty is low.

---

## 4.5. Training Strategy

### Phase 2A: Frozen-backbone training

- Freeze ResEnc.
- Precompute feature maps or sample them online.
- Train the ray encoder and surface model.

Objectives:

- Stabilize the new branch.
- Reduce memory use.
- Simplify debugging.

### Phase 2B: Mixed ground-truth and predicted surfaces

Sampling strategy:

- Some batches use ground-truth bone surfaces.
- Some use predicted surfaces.
- Some use predicted surfaces with jitter.

A possible curriculum:

| Training stage | Ground-truth surface | Predicted surface | Jittered surface |
|---|---:|---:|---:|
| Early | 70% | 30% | 0% |
| Middle | 30% | 50% | 20% |
| Late | 0–10% | 60% | 30–40% |

### Phase 2C: Partial joint fine-tuning

Unfreeze:

- High-resolution ResEnc blocks.
- Coarse cartilage decoder blocks.
- Do not necessarily unfreeze the complete backbone.

Use a lower learning rate for ResEnc than for the new surface branch.

### Phase 2D: Fusion calibration

Train or calibrate the gating head separately.

Do not tune fusion thresholds on the test set.

---

## 4.6. Loss

\[
\begin{aligned}
L={}&
\lambda_1L_{\text{occupancy}}
+\lambda_2L_{\text{presence}}
+\lambda_3L_{\text{boundary}}\\
&+\lambda_4L_{\text{volume}}
+\lambda_5L_{\text{penetration}}
+\lambda_6L_{\text{surface consistency}}\\
&+\lambda_7L_{\text{uncertainty}}
+\lambda_8L_{\text{fusion}}
\end{aligned}
\]

Do not activate every term simultaneously.

Recommended sequence:

1. Occupancy + presence.
2. Add volumetric reconstruction.
3. Add boundary supervision.
4. Add geometric constraints.
5. Add uncertainty modeling.
6. Add fusion loss.

---

## 4.7. Main-Model Side Tests

### Test MM1 — Oracle-gap analysis

Compare:

| Condition | Bone surface | Articular domain |
|---|---|---|
| Oracle | Ground truth | Ground truth or atlas |
| Bone-error only | Predicted | Ground truth or atlas |
| Atlas-error only | Ground truth | Predicted or registered |
| End-to-end | Predicted | Predicted or registered |

Purpose:

- Quantify performance loss caused by bone segmentation.
- Quantify performance loss caused by atlas mapping.

---

### Test MM2 — Bone-quality stratification

Group cases by bone-surface ASSD:

- Low error.
- Medium error.
- High error.

Plot:

\[
\text{cartilage improvement}
\quad\text{versus}\quad
\text{bone-surface error}
\]

If the cartilage model works only with nearly perfect bone surfaces, improve robustness before adding further complexity.

---

### Test MM3 — Graph-oversmoothing test

Monitor:

- Node-feature variance across graph layers.
- Focal-defect recall.
- Thickness contrast across defect boundaries.

Ablate:

- No graph.
- One graph layer.
- Two graph layers.
- Four graph layers.
- With and without defect-aware edge weights.

Objective:

- Ensure that graph propagation does not fill genuine focal defects.

---

### Test MM4 — Cross-surface ablation

| ID | Opposing geometry | Cross-attention |
|---|---:|---:|
| C0 | None | No |
| C1 | Distance only | No |
| C2 | Distance + angle | No |
| C3 | Geometry features | Yes |
| C4 | Geometry + meniscus context | Yes |

The opposing-surface module should demonstrate measurable added value.

---

### Test MM5 — Hard-negative test

Create separate negative groups:

- Meniscus close to cartilage.
- Opposing cartilage.
- Joint fluid.
- Osteophyte regions.
- Bone edges.
- Non-articular bone surfaces.

Report false-positive rates for each group.

---

### Test MM6 — Presence/absence calibration

Evaluate:

- AUROC.
- AUPRC.
- F1 score.
- Expected calibration error.
- Reliability diagrams.

Select the presence threshold using validation data.

Evaluate separately:

- True full-thickness loss.
- Extremely thin cartilage.
- Non-articular surface.

These are the three categories most likely to be confused.

---

### Test MM7 — Fusion-safety test

Compare:

1. ResEnc only.
2. Surface model only.
3. Fixed-weight fusion.
4. Learned gated fusion.
5. Uncertainty-gated fusion.

Requirements:

- Fusion must not reduce performance in thick-cartilage regions.
- The surface branch should intervene strongly only where it adds value.

---

### Test MM8 — Computational profiling

Record:

- Peak GPU memory.
- Training throughput.
- Inference time per case.
- Number of surface nodes.
- Number of rays.
- Reconstruction time.
- Model parameter count.

Test multiple mesh densities to establish the accuracy–compute trade-off.

---

### Test MM9 — Reproducibility

Each major configuration should include:

- Multiple random seeds on a pilot fold.
- Full five-fold validation for the selected configuration.
- Complete configuration and Git commit logging.
- Deterministic preprocessing artefacts where possible.

---

## 4.8. Main-Model Ablation Matrix

| ID | Ray | Graph | Opposing geometry | Presence | Fusion |
|---|---:|---:|---:|---:|---:|
| A0 | No | No | No | No | No |
| A1 | Yes | No | No | No | No |
| A2 | Yes | No | No | Yes | No |
| A3 | Yes | Yes | No | Yes | No |
| A4 | Yes | Yes | Yes | Yes | No |
| A5 | Yes | Yes | Yes | Yes | Fixed |
| A6 | Yes | Yes | Yes | Yes | Learned |
| A7 | Yes | Yes | Yes | Yes | Uncertainty-gated |

---

## 4.9. Main-Model Go/No-Go Gate

The Main Model should be considered successful only if:

1. It improves at least two target metrics:
   - Outer-boundary mean absolute error.
   - Surface Dice at 0.5 mm.
   - Thin-cartilage Dice or recall.
2. The gain remains when predicted bone surfaces are used.
3. Improvement is concentrated in KL3–4 or thin-cartilage groups.
4. False-positive cartilage does not increase substantially in absence regions.
5. Focal defects are not removed by graph smoothing.
6. Gated fusion outperforms both ResEnc-only and surface-only predictions.
7. Bootstrap confidence intervals support a genuine effect.
8. Results are repeatable across folds.

Suggested working targets:

- Preserve at least 60–70% of the oracle-surface gain when switching to predicted bone surfaces.
- Reduce boundary error in thin cartilage by approximately 10% or more.
- Improve presence/absence F1 meaningfully.
- Reduce Dice in thick cartilage by no more than 0.5 percentage points.

---

# STAGE 3 — EXTENSION

## 5. Objective

Extensions must not be added merely to increase architectural complexity. Each extension should target a specific remaining failure mode of the Main Model:

- Boundary quantization caused by discrete depth bins.
- Surface normals that do not intersect cartilage optimally.
- Inadequate uncertainty representation.
- Need for wider local context around focal defects.
- Limited domain generalization.
- Discrete occupancy that produces insufficiently smooth surfaces.

Each extension must be introduced independently and validated through ablation.

---

## 5.1. Extension A — Conditional Implicit Cartilage Field

### Input

- Surface token from the Main Model.
- Continuous normal distance \(d\).
- Positional encoding of \(d\).
- Local ray features.
- Opposing-surface geometry.

### Output

\[
f_\theta(z_i,d)
\rightarrow
p_{\text{cartilage}}(s_i,d)
\]

Optional outputs:

- Signed-distance value.
- Boundary probability.
- Aleatoric uncertainty.

### Processing

1. Sample continuous query points in the candidate band.
2. Oversample near the ground-truth boundaries.
3. Query the implicit decoder.
4. Reconstruct at arbitrary resolution.
5. Fuse with the volumetric prediction.

### Side tests

#### E-A1 — Query-resolution sweep

Reconstruct at:

- Native resolution.
- Two-times resolution.
- Four-times resolution.

Check whether estimated boundaries converge.

#### E-A2 — Subvoxel phantom test

Shift cartilage boundaries by:

- 0.1 voxel.
- 0.25 voxel.
- 0.5 voxel.

Evaluate whether the implicit decoder recovers the correct ordering and offset.

#### E-A3 — Smoothing-versus-defect test

Verify that the implicit field:

- Smooths noise.
- Does not fill true full-thickness defects.

#### E-A4 — Discrete-versus-continuous ablation

Keep the surface encoder fixed and change only the decoder:

- Discrete occupancy.
- Boundary distributions.
- Implicit occupancy.
- Implicit signed-distance field.

---

## 5.2. Extension B — Multi-Ray Cone or Normal-Aligned Slab

### Objective

Reduce sensitivity to inaccurate normal directions and complex local geometry.

### Input

Instead of one ray, sample:

- The primary normal.
- Rays rotated by approximately ±5°–15° along tangent directions.
- Or a small normal-aligned slab.

### Output

- Fused ray embedding.
- Learned ray weights.
- Cartilage occupancy or boundary predictions.

### Processing

Encode each ray separately:

\[
z_i^{(m)}=E_{\text{ray}}(X_i^{(m)})
\]

Fuse by attention:

\[
z_i=
\sum_m\alpha_i^{(m)}z_i^{(m)}
\]

### Side tests

#### E-B1 — Normal-perturbation stress test

Compare:

- Single ray.
- Five-ray cone.
- Nine-ray cone.
- Two-dimensional ribbon.
- Three-dimensional slab.

Use the same levels of surface and normal jitter.

#### E-B2 — Edge-region test

Evaluate separately at:

- Cartilage margins.
- High-curvature regions.
- Patellar cartilage.
- Osteophyte regions.

#### E-B3 — Compute-normalized comparison

Keep parameter count or floating-point operations approximately comparable where possible, preventing gains from being attributed only to a larger model.

---

## 5.3. Extension C — Learned Normal Correction

### Input

- Initial surface normal.
- Tangent frame.
- Ray or slab features.
- Surface curvature.
- Neighbouring-node context.

### Output

Two correction coefficients:

\[
a_i,b_i
\]

Corrected normal:

\[
\hat n_i=
\operatorname{normalize}
(n_i+a_it_{1i}+b_it_{2i})
\]

Restrict the maximum correction angle.

### Side tests

- Distribution of correction angles.
- Correlation between correction magnitude and bone-surface error.
- Performance when the initial normal is already accurate.
- Performance under perturbed normals.
- Check whether the model exploits correction to move toward opposing cartilage.

If large corrections occur at most surface points, the initial coordinate system may be inappropriate.

---

## 5.4. Extension D — Zero-Inflated Probabilistic Thickness

### Input

Surface tokens and implicit or ray features.

### Output

1. Probability of zero thickness:

\[
\pi_i=P(t_i=0)
\]

2. Conditional positive-thickness distribution:

\[
p_+(t_i\mid t_i>0)
\]

3. Predictive uncertainty.

### Processing

\[
p(t_i)
=
\pi_i\delta_0
+
(1-\pi_i)p_+(t_i)
\]

This explicitly distinguishes:

- Cartilage absence.
- Extremely thin cartilage.
- Normal or thick cartilage.

### Side tests

- Confusion matrix between absent and extremely thin cartilage.
- Thickness calibration.
- Negative log likelihood.
- Prediction-interval coverage.
- Error by KL grade.
- Full-thickness-loss area error.

---

## 5.5. Extension E — Uncertainty-Driven Selective Refinement

### Input

- ResEnc uncertainty.
- Surface-model uncertainty.
- Presence entropy.
- Boundary-distribution entropy.
- Bone-surface uncertainty.

### Output

- Confidence map.
- Refinement-priority map.
- Optional second-pass local slab prediction.

### Processing

Run expensive local refinement only in regions with:

- High uncertainty.
- Thin predicted cartilage.
- Disagreement between ResEnc and the surface model.
- Predicted defect boundaries.

### Side tests

#### E-E1 — Error–uncertainty correlation

Prediction uncertainty should be higher in regions with larger errors.

#### E-E2 — Selective-prediction curve

Progressively remove the most uncertain predictions and measure performance on the retained subset.

#### E-E3 — Compute–accuracy curve

Compare:

- Refining 10% of nodes.
- Refining 25%.
- Refining 50%.
- Refining all nodes.

---

## 5.6. Extension F — Domain Robustness

Evaluate on a different dataset or acquisition domain.

Tests:

- Intensity-normalization shift.
- Scanner or vendor shift.
- Spacing shift.
- Acquisition-protocol shift.
- KL-distribution shift.
- Bone-prediction quality shift.

Potential methods:

- Feature normalization along each ray.
- Domain-adversarial surface embeddings.
- Test-time feature normalization.
- Atlas-coordinate normalization.
- Uncertainty-based failure detection.

Do not add domain adaptation before the Main Model is stable internally.

---

## 5.7. Extension Go/No-Go Gates

Retain an extension only if it adds measurable value beyond the Main Model.

### Implicit field

It must:

- Reduce boundary mean absolute error.
- Do more than simply smooth the mask.
- Preserve defect detection.

### Multi-ray or slab

It must:

- Be clearly more robust under normal perturbation.
- Justify its additional computational cost.

### Learned normal correction

It must:

- Outperform multi-ray sampling or be substantially simpler.
- Avoid anatomically implausible rays.

### Probabilistic thickness

It must:

- Better distinguish absent cartilage from extremely thin cartilage.
- Produce calibrated uncertainty.

### Selective refinement

It must:

- Approach full-refinement accuracy while using less computation.

Do not include an extension in the final paper if it produces only a very small Dice gain without supporting a specific hypothesis.

---

# 6. Recommended Execution Order

## Stage 1 — MVP

1. Lock the baselines and data splits.
2. Construct ground-truth bone signed-distance fields and surfaces.
3. Validate surface normals.
4. Create the normal-ray dataset.
5. Run the coordinate round-trip reconstruction.
6. Analyse the single-interval ratio.
7. Train the ray-only model on femoral cartilage.
8. Run the tiny-set overfitting test.
9. Run negative-control direction experiments.
10. Run the surface-jitter test.
11. Test medial tibial cartilage.
12. Run a predicted-bone pilot.
13. Make the MVP go/no-go decision.

## Stage 2 — Main Model

1. Build a fold-specific articular atlas.
2. Generate predicted bone surfaces.
3. Add high-resolution ResEnc features.
4. Train the frozen-backbone ray model.
5. Add the presence head.
6. Add the intra-surface graph.
7. Add opposing-surface geometry.
8. Add cross-surface attention.
9. Reconstruct the volume.
10. Train gated fusion.
11. Run oracle-gap analysis.
12. Run the graph-oversmoothing test.
13. Run hard-negative analysis.
14. Perform partial fine-tuning.
15. Run complete five-fold validation.
16. Evaluate external data and the KL4 subset.
17. Make the go/no-go decision for Extensions.

## Stage 3 — Extension

1. Identify the largest remaining Main-Model failure mode.
2. Test the implicit decoder independently.
3. Test multi-ray or slab sampling independently.
4. Test zero-inflated thickness independently.
5. Test uncertainty modeling independently.
6. Combine only extensions that have demonstrated independent value.
7. Run the final ablation study.
8. Run domain-robustness evaluation.
9. Run clinical morphology evaluation.
10. Lock the final model and evaluate the held-out test set once.

---

# 7. Professional Experiment Tracking

Every run must record:

- Experiment ID.
- Git commit.
- YAML or JSON configuration.
- Dataset version.
- Fold.
- Random seed.
- Backbone checkpoint.
- Input channels.
- Surface source: ground truth or predicted.
- Atlas version.
- Mesh density.
- Ray length and sampling interval.
- Number of graph layers.
- Loss weights.
- Decision thresholds.
- Runtime.
- Peak GPU memory.
- Per-case metrics.
- Failure-case visualizations.

Example experiment IDs:

```text
MVP_FC_GTsurf_Ray64_MRI-SDF_Fold0_Seed1
MAIN_FC-PredSurf_Graph2_OppGeom_Fold0_Seed1
EXT_FC-Implicit_MultiRay5_Fold0_Seed1
```

---

# 8. Visual Quality-Control Checklist

For every major configuration, save visualizations of:

1. Bone surface and normals.
2. Articular domain.
3. Ground-truth and predicted occupancy profiles.
4. Presence probabilities.
5. Inner- and outer-boundary distributions.
6. Reconstructed cartilage surface.
7. ResEnc prediction.
8. Surface-model prediction.
9. Fused prediction.
10. Error map.
11. Uncertainty map.
12. KL4 failure cases.
13. Extremely thin-cartilage cases.
14. Full-thickness-loss cases.

Do not inspect only axial, sagittal, and coronal slices. Also inspect:

- Thickness maps projected onto the bone surface.
- Error maps on the articular manifold.
- Surface-distance heatmaps.

---

# 9. Final Success Criteria

The new architecture does not necessarily need to produce a very large increase in overall Dice. It should be considered successful if it demonstrates that it can:

1. Reduce boundary errors in thin cartilage.
2. Improve Surface Dice at a small tolerance.
3. Reduce confusion between cartilage absence and extremely thin cartilage.
4. Improve cartilage-thickness estimation.
5. Improve KL4 performance without degrading mild cases.
6. Remain robust when using predicted bone surfaces.
7. Show through negative controls and ablations that the gain comes from the surface-coordinate representation.
8. Produce repeatable results across folds and domains.
9. Maintain reasonable computational cost.
10. Quantify its failure modes clearly.

The final research narrative should be:

> ResEnc provides global volumetric understanding, while the bone-surface coordinate system transforms cartilage segmentation into a structured problem along the thickness direction and across the articular manifold. This representation is particularly useful in regions with extremely thin or completely absent cartilage, where conventional voxel-overlap segmentation may reach a performance ceiling.
