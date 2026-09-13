# Review and Recommendations on ASSD Metric Discrepancy and Gate 1 Interpretation

## 1. Executive Summary

The current interpretation is directionally reasonable, but two important points should be revised.

The observed femoral difference of approximately **0.060 mm** between the SciPy and SimpleITK implementations may represent a systematic, geometry-dependent implementation offset. However, it should not automatically be treated as a **0.060 mm uncertainty band** or “metric noise floor” for Gate 1.

The most important distinction is:

\[
\delta_B
=
ASSD_{\text{SciPy}}(B)
-
ASSD_{\text{SITK}}(B)
\]

is an absolute implementation difference measured on the baseline, whereas Gate 1 evaluates the within-implementation improvement:

\[
Prize_{\text{SciPy}}
=
ASSD_{\text{SciPy}}(B)
-
ASSD_{\text{SciPy}}(M)
\]

where:

- \(B\) is the baseline model.
- \(M\) is the candidate geometry-based model.

The relevant quantity for determining whether the model improvement depends on the metric definition is:

\[
\epsilon_{\text{metric}}
=
Prize_{\text{SciPy}}
-
Prize_{\text{SITK}}
\]

or equivalently:

\[
\epsilon_{\text{metric}}
=
\delta_B-\delta_M
\]

Therefore, the absolute SciPy–SimpleITK gap of 0.060 mm does not by itself invalidate a within-SciPy improvement of 0.04–0.06 mm.

---

# 2. Findings That Are Likely Correct

## 2.1. The Difference Is Probably Not a Basic File-Reading or Spacing Bug

The Dice values produced by the two implementations are effectively identical. This strongly suggests that both implementations are evaluating the same prediction and ground-truth masks.

The ASSD discrepancy is therefore more plausibly related to differences in:

- Surface extraction.
- Boundary-point definition.
- Distance-transform implementation.
- Implicit placement of the surface relative to the voxel grid.
- Treatment of anisotropic spacing.

The current comparison appears to involve two different surface-distance pipelines:

- A SciPy-based method that first defines a surface mask and then computes Euclidean distance transforms.
- A SimpleITK-based method using contour extraction and the Maurer distance transform.

These pipelines are not guaranteed to define the same physical surface, even when they use the same binary masks.

---

## 2.2. Absolute SciPy ASSD Should Not Be Compared Directly with Legacy SimpleITK ASSD

This conclusion is correct.

A single metric contract should be selected and used consistently for:

- The ResEnc baseline.
- The ROI-cascade baseline.
- The MVP surface-coordinate model.
- The Main Model.
- All Extensions.
- Final ablation studies.
- Final test-set reporting.

For example:

```text
Canonical ASSD contract
-----------------------
Surface extraction: explicitly specified
Distance transform: scipy.ndimage.distance_transform_edt
Spacing: physical spacing in the documented axis order
Symmetric aggregation: explicitly specified
Empty-mask handling: explicitly specified
```

The previously published value of approximately 0.21 mm may still be mentioned as historical context, but it should be accompanied by a clear warning:

> The published value was calculated using a legacy SimpleITK implementation and is not directly comparable with the SciPy ASSD values reported in the current study.

---

## 2.3. A Geometry-Dependent Implementation Gap Is Plausible

Femoral cartilage has a more complex surface-orientation distribution than tibial cartilage.

Potential contributing factors include:

- High curvature around the femoral condyles.
- A larger proportion of oblique surfaces relative to the image axes.
- More complex surface topology.
- Strong anisotropy between the through-plane and in-plane voxel spacing.
- Different behaviour of surface extraction on curved or oblique boundaries.

This makes a larger implementation gap for femoral cartilage plausible.

However, this explanation should currently be treated as a **hypothesis**, not a confirmed conclusion.

---

# 3. Interpretations That Should Be Revised

## 3.1. Avoid Saying That the Bias “Accumulates” Because Femoral Cartilage Is Larger

ASSD is a mean surface distance, not a sum:

\[
ASSD=
\frac{
\sum_{x\in S_G}d(x,S_P)
+
\sum_{y\in S_P}d(y,S_G)
}{
|S_G|+|S_P|
}
\]

A larger surface does not automatically produce a larger ASSD bias.

A more accurate interpretation is:

> Femoral cartilage contains a larger proportion of curved and oblique surface elements for which the two implementations may assign different physical boundary locations. This can increase the mean implementation offset.

The larger number of surface points may make the estimate more stable, but it does not itself cause the mean error to accumulate.

---

## 3.2. The 0.060 mm Difference Is Not Automatically a Gate 1 Noise Floor

Suppose:

\[
ASSD_{\text{SciPy}}(m)
=
ASSD_{\text{SITK}}(m)+0.060
\]

for both the baseline and the candidate model.

Then:

\[
\begin{aligned}
Prize_{\text{SciPy}}
&=
ASSD_{\text{SciPy}}(B)
-
ASSD_{\text{SciPy}}(M)\\
&=
[ASSD_{\text{SITK}}(B)+0.060]
-
[ASSD_{\text{SITK}}(M)+0.060]\\
&=
Prize_{\text{SITK}}
\end{aligned}
\]

The offset cancels completely.

Therefore, an improvement of 0.03 or 0.04 mm may still be measurable even when the absolute implementation difference is 0.060 mm.

The implementation discrepancy affects the prize only if:

\[
\delta_B\neq\delta_M
\]

where:

\[
\delta_M
=
ASSD_{\text{SciPy}}(M)
-
ASSD_{\text{SITK}}(M)
\]

The relevant robustness quantity is:

\[
\epsilon_{\text{metric}}
=
\delta_B-\delta_M
\]

A borderline result should be considered metric-definition-sensitive only when \(\epsilon_{\text{metric}}\) is large relative to the observed prize.

---

# 4. Limitations of the Current Comparison

## 4.1. Eight Cases Are Sufficient for Bug Detection, but Not for Characterizing the Offset

A sample of eight cases may be sufficient to detect:

- Incorrect spacing order.
- Incorrect label selection.
- Gross implementation errors.
- Distance errors at the millimetre scale.
- Inconsistent Dice values.

It is not sufficient to establish that the ASSD difference is:

- Stable.
- Always positive.
- Geometry dependent.
- Class specific.
- Consistent across severity levels.
- Consistent across prediction quality levels.

The current analysis should therefore be expanded.

Recommended minimum:

- A quick analysis using 30–50 stratified cases.
- A final analysis using the complete evaluation set.

The sample should be stratified by:

- KL grade.
- Cartilage thickness.
- Cartilage volume.
- Dataset or acquisition domain.
- Baseline ASSD.
- Bone and cartilage geometry.

---

## 4.2. The Mean Alone Is Insufficient

For each cartilage class, report the per-case implementation difference:

\[
\delta_i
=
ASSD_{\text{SciPy},i}
-
ASSD_{\text{SITK},i}
\]

Required summary statistics:

- Mean.
- Standard deviation.
- Median.
- Interquartile range.
- Minimum.
- Maximum.
- Bootstrap 95% confidence interval.
- Percentage of cases with positive \(\delta_i\).
- Percentage of cases with negative \(\delta_i\).

Two distributions may have the same mean of 0.060 mm but very different implications.

### Stable systematic offset

```text
0.055, 0.061, 0.059, 0.063, ...
```

### Unstable case-dependent offset

```text
-0.02, 0.01, 0.04, 0.05, 0.07, 0.09, 0.11, 0.13
```

Only the first pattern supports the interpretation of a stable implementation offset.

---

# 5. Recommended Revision of Gate 1

## 5.1. Primary Gate: Paired Improvement Using SciPy Only

Use one canonical metric implementation for the formal Gate 1 decision:

\[
Prize_i
=
ASSD_{\text{SciPy},i}(B)
-
ASSD_{\text{SciPy},i}(M)
\]

Calculate:

- Mean paired prize.
- Median paired prize.
- Bootstrap 95% confidence interval.
- Percentage of improved cases.
- Results by cartilage class.
- Results by KL grade.
- Results by thickness group.

The primary Gate 1 decision should be based on the paired distribution of \(Prize_i\), not on the absolute difference between SciPy and SimpleITK.

---

## 5.2. Suggested Gate Categories

### Strong proceed

Proceed when:

- Point estimate is at least approximately 0.08 mm.
- The bootstrap lower confidence bound remains positive.
- The improvement is present in thin-cartilage or KL4 cases.
- Dice, recall, and false-positive cartilage do not deteriorate materially.
- The result is repeated across folds or seeds.

### Strong stop

Stop when:

- Point estimate is below approximately 0.04 mm.
- The bootstrap upper confidence bound is also small.
- There is no clear benefit in thin-cartilage or KL4 groups.
- The model introduces substantial false-positive cartilage.

### Gray zone

Treat the result as inconclusive when:

- Confidence intervals are wide.
- The interval contains zero.
- Improvement occurs only in a few cases.
- Results differ substantially across folds or classes.
- Metric-definition robustness is poor.

The gray zone should be defined mainly by **paired statistical uncertainty**, rather than by the absolute 0.060 mm SciPy–SimpleITK difference.

---

## 5.3. Secondary Gate: Metric-Definition Robustness

For the same baseline and candidate model, calculate:

\[
Prize_{\text{SciPy}}
\]

and:

\[
Prize_{\text{SITK}}
\]

Then calculate:

\[
\epsilon_{\text{metric}}
=
Prize_{\text{SciPy}}
-
Prize_{\text{SITK}}
\]

The result is robust to metric definition when:

- Both prizes have the same sign.
- Model ranking is unchanged.
- \(|\epsilon_{\text{metric}}|\) is small relative to the prize.
- The bootstrap confidence interval of \(\epsilon_{\text{metric}}\) is close to zero.

Example of a robust result:

```text
SciPy prize: +0.090 mm
SITK prize:  +0.085 mm
Difference:  +0.005 mm
```

Example of a metric-sensitive result:

```text
SciPy prize: +0.070 mm
SITK prize:  +0.005 mm
Difference:  +0.065 mm
```

In the second case, the apparent improvement depends heavily on the metric definition.

---

# 6. Additional Side Tests

## 6.1. Geometry Phantom Test

Create synthetic surfaces with controlled properties:

- Flat planes.
- Spheres.
- Ellipsoids.
- Thin shells.
- Curved cartilage-like bands.
- Different thicknesses.
- Different orientations relative to the voxel grid.

Test surface orientations such as:

\[
0^\circ,\ 15^\circ,\ 30^\circ,\ 45^\circ,\ 60^\circ,\ 75^\circ
\]

Evaluate the SciPy–SimpleITK ASSD difference while keeping the physical prediction error constant.

Purpose:

> Determine whether the implementation gap increases with curvature, obliqueness, or anisotropic spacing.

---

## 6.2. Isotropic-Resampling Test

Calculate both metrics under:

1. Native anisotropic spacing.
2. An isotropically resampled representation.

If the femoral implementation gap decreases substantially after isotropic resampling, voxel anisotropy is likely a major contributor.

The resampled result should be used only as a diagnostic experiment, not necessarily as the final evaluation protocol.

---

## 6.3. Per-Case Implementation-Gap Distribution

For every class and case, calculate:

\[
\delta_i
=
ASSD_{\text{SciPy},i}
-
ASSD_{\text{SITK},i}
\]

Generate:

- Histogram.
- Box plot.
- Bland–Altman plot.
- SciPy-versus-SimpleITK scatter plot.
- Per-class violin plot.
- Bootstrap confidence interval.

This will determine whether the discrepancy is:

- A stable additive offset.
- Proportional to ASSD magnitude.
- Heteroscedastic.
- Class dependent.
- Case dependent.

---

## 6.4. Geometry-Correlation Analysis

Test whether \(\delta_i\) correlates with:

- Cartilage surface area.
- Mean curvature.
- Curvature variance.
- Surface-to-volume ratio.
- Fraction of oblique surface elements.
- Fraction of normals with a large through-plane component.
- Mean cartilage thickness.
- Minimum cartilage thickness.
- Ground-truth ASSD.
- Voxel-spacing anisotropy.

The geometry-dependent interpretation is stronger if the discrepancy correlates with curvature or orientation rather than only with surface size.

---

## 6.5. Difference-in-Differences Test

For each candidate model, calculate:

\[
\epsilon_i
=
\left[
ASSD_{\text{SciPy},i}(B)
-
ASSD_{\text{SciPy},i}(M)
\right]
-
\left[
ASSD_{\text{SITK},i}(B)
-
ASSD_{\text{SITK},i}(M)
\right]
\]

Report:

- Mean \(\epsilon_i\).
- Median \(\epsilon_i\).
- Bootstrap confidence interval.
- Percentage of cases where model ranking changes.
- Results by class and thickness group.

This is the most direct test of whether the Gate 1 conclusion depends on the ASSD implementation.

---

# 7. Dice Interpretation

The Dice implementations appear to agree for the same masks, so the implementation itself is reproducible.

However, a current Dice value should still not be treated as directly comparable with a published Dice value if any of the following differ:

- Model checkpoint.
- Training duration.
- Cross-validation split.
- Test population.
- Dataset domain.
- Fold ensemble.
- Post-processing.
- Label definitions.

Recommended wording:

> Dice implementation was reproduced consistently across the two evaluation pipelines. Nevertheless, historical and current Dice values remain contextual rather than directly comparable because the evaluated models, data splits, and test populations may differ.

---

# 8. Recommended Reporting Language

A rigorous interpretation would be:

> The approximately 0.060 mm femoral difference is a class-dependent implementation offset, plausibly associated with surface geometry and voxel anisotropy. This discrepancy prevents direct comparison of absolute SciPy ASSD values with legacy SimpleITK results. However, it does not by itself invalidate within-SciPy model improvements. Gate 1 should therefore be based on paired SciPy differences and their confidence intervals, while robustness to metric definition should be evaluated separately through the difference between SciPy- and SimpleITK-derived improvement estimates.

---

# 9. Recommended Immediate Actions

1. Select and document a canonical SciPy ASSD implementation.
2. Recalculate all current baselines using that implementation.
3. Expand the SciPy–SimpleITK comparison beyond eight cases.
4. Store per-case results rather than only class means.
5. Add bootstrap confidence intervals.
6. Run the geometry phantom test.
7. Run the isotropic-resampling diagnostic.
8. Calculate difference-in-differences once the first candidate model is available.
9. Base Gate 1 primarily on paired SciPy improvement.
10. Treat metric-definition robustness as a separate secondary criterion.

---

# 10. Final Assessment

The current reasoning is largely correct in the following respects:

- The discrepancy is unlikely to be a basic mask or spacing bug.
- Absolute SciPy ASSD should not be directly compared with legacy SimpleITK ASSD.
- A geometry-dependent implementation difference is plausible.
- Within-implementation paired improvements remain the appropriate basis for Gate 1.

The following points require revision:

- Femoral size does not cause ASSD bias to accumulate directly.
- The 0.060 mm absolute implementation gap is not automatically a 0.060 mm uncertainty band.
- Eight cases and a class mean are insufficient to characterize the discrepancy.
- Borderline Gate 1 decisions should be based on paired confidence intervals and difference-in-differences, not solely on the absolute implementation gap.
