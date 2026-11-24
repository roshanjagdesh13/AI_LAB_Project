# EDA findings

_All statements below are computed from the cleaned dataset by `src/eda.py`._

- **Class balance:** 18,905 events after cleaning; 12,332 gamma (65.2%) vs 6,573 hadron (34.8%). The data is imbalanced (ratio 1.88:1), so accuracy alone is misleading and the training set is oversampled.
- **Data quality:** 0 missing cells; 115 exact duplicate rows were removed (19,020 -> 18,905).
- **Best separating features:** `fAlpha` (single-feature AUC 0.785; median 9.71 for gamma vs 43.41 for hadron), `fLength` (single-feature AUC 0.615; median 34.56 for gamma vs 47.91 for hadron), `fAsym` (single-feature AUC 0.580; median 6.64 for gamma vs -1.48 for hadron).
- **Weakest separating features:** `fConc` (AUC 0.515), `fConc1` (AUC 0.509), `fM3Trans` (AUC 0.501) - the class distributions overlap heavily for these.
- **Strongly correlated features (|r| >= 0.7):** `fConc` & `fConc1` (r = +0.98); `fSize` & `fConc` (r = -0.85); `fSize` & `fConc1` (r = -0.81); `fLength` & `fWidth` (r = +0.77); `fWidth` & `fSize` (r = +0.72); `fLength` & `fSize` (r = +0.70). These are partly redundant, which can hurt linear models (coefficients become unstable) but matters little for tree ensembles.
- **Skew / outliers:** `fWidth` (skew +3.39, 8.1% IQR outliers), `fLength` (skew +2.02, 5.1% IQR outliers), `fM3Long` (skew -1.13, 6.4% IQR outliers), `fAsym` (skew -1.04, 9.3% IQR outliers). Feature scales differ by orders of magnitude, so standard scaling is applied before distance/gradient-based models.
- **Scale differences:** the widest-range feature is `fAsym` (range 1033.2) while the narrowest is `fConc1` (range 0.67).
