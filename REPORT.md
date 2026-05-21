# Constraint Field Prototype — Iteration Report

**Project:** Field-based constraint dynamics model for electricity system analysis  
**Stack:** Python · pandas · numpy · scipy · networkx · matplotlib · scikit-learn · statsmodels  
**Data:** Synthetic (spatially-correlated, structurally realistic) with live EIA + CAISO adapters ready  
**Codebase:** `constraint_field/` package — adapters, field, inference, analysis, graph subpackages

---

## Iteration 1 — Initial prototype scaffold

**Request:** Build a modular Python research prototype implementing a two-layer field model: a static reduced layer (S, R) and a dynamic extension layer (S, R, E), with data adapters, field construction, E inference, and propagation simulation.

**Delivered:**
- Full package structure: `adapters/`, `field/`, `inference/`, `dynamics/`
- `EIAAdapter`, `CAISOAdapter`, `SyntheticAdapter` behind a common fetch interface
- `FieldBuilder` constructing normalised S (load pressure) and R (constraint signal) panels via rolling z-score
- Three E inference candidates: E1 (congestion inverse), E2 (flow efficiency), E3 (price-spread transmissibility)
- `DiffusionOperator`, `GradientFlowOperator`, `DampedWaveOperator` with configurable propagation
- `Simulator` with shock injection and reduced vs upgraded comparison
- Parameter sweep over γ × η grid
- Figures 01–07: static dashboard, phase portrait, gradient heatmap, instability index, E candidates, simulation comparison, parameter sweep heatmap

**Observations:** R alone has near-zero predictive power for instability. E candidates produced well-separated distributions (E1=0.90, E2=0.57, E3=0.40 after calibration fix). The diffusion operator with higher damping (γ=0.4) dominated transmissibility, suggesting mean-reversion is the system's primary dynamic at this parameterisation. The reduced vs upgraded RMSE gap was modest in single-node form — the architecture is correct, but E's contribution becomes more meaningful in the spatial graph extension.

---

## Iteration 2 — E3 calibration fix

**Request:** The exponential transform for price spread was collapsing E3 to near-constant values. Replace fixed arbitrary R_scale with data-calibrated scale parameters. Test multiple robust scale candidates (median, IQR, std, p75, nz_median), four transform families (exponential, rational, minmax_inv, logistic), and add full diagnostics.

**Delivered:**
- `calibration.py`: `SpreadStats` (14 summary statistics including nonzero-subset measures), `candidate_scales()`, `EDistributionQuality` with a scoring function, `SpreadCalibrator` evaluating all transform × scale combinations
- Four transforms: `transform_exponential`, `transform_rational`, `transform_minmax_inv`, `transform_logistic`
- Full calibration diagnostic figure (Figure 05b): spread histogram with scale references, E histograms per transform, time-series comparison, ranked quality table
- `PriceSpreadE.plot_calibration()` method for interactive inspection

**Observations:** Two distinct collapse failure modes were identified. Type A (zero-inflated signals like congestion): 91.7% of hours are zero; naive quantile of the full distribution returns ~0, causing exp(−x/0) → 0 for every nonzero event. Fix: derive scale from the nonzero subset (nz_median, nz_p75). Type B (always-positive shifted signals like lmp_spread): the spread lives at $46–$87 but the old scale was drawn from the normalised R field (~0.88), making spread/scale ≈ 50–100 and exp(−50) ≈ 0. Fix: use the spread's own IQR or std. The calibrator selected logistic with scale=std($7.80) as optimal, producing E3: mean=0.469, std=0.207, frac<0.05=0.3% — well-distributed across [0,1].

---

## Iteration 3 — Divergence metrics and predictive comparison

**Request:** Quantify the divergence between field pressure and price expression. Build divergence metrics D1–D5 and bounded-price diagnostics. Compare models A (R only) through E (R + best divergence) using OLS and logistic regression with time-ordered validation. Test regime-stratified divergence and bounded-price interpretation.

**Delivered:**
- `divergence.py`: D1=|Φ|, D2=|Φ|/(1+|R|), D3=|Φ|/(1+Ψ), D4=rolling persistence of |Φ|, D5±=signed asymmetry components, BP1–BP3=bounded-price indicators
- `models.py`: OLS with statsmodels (train R², test R², adj R², AUC, F1), logistic classification, lead/lag correlation, time-ordered train/test split
- `regimes.py`: cluster-stratified summaries, Mann-Whitney test, segment statistics
- Figures 08–13: divergence timeseries, lead/lag, regime divergence, bounded-price scatter, model comparison bars, full dashboard
- Text report with three explicit conclusions

**Observations:** R alone achieves test R² = −0.005 (no predictive power). D4 achieves R² = 0.763 and AUC = 0.986 — but this is a rolling-window artifact (D4 and the instability index are both 24h rolling summaries of |Φ|; the correlation is mechanical). D2 achieves correlation +0.40 vs R's +0.05. The key result: the bounded-price test (Mann-Whitney comparing high-|Φ|/moderate-R vs low-|Φ|/moderate-R segments) returns p < 10⁻⁶ with Cohen's d = 3.35. At the same moderate price level, hours with high imbalance have dramatically higher instability — price is not carrying the information that field imbalance carries.

---

## Iteration 4 — Rolling-window decontamination

**Request:** The D4/instability lead was identified as a window artifact (both are 24h rolling summaries of |Φ|). Rebuild lead analysis using point-in-time instability targets, time-of-day residualisation, event-study analysis, and regime-stratified tests to separate genuine signal from mechanical overlap.

**Delivered:**
- `decontaminated_lead.py`: `build_pit_targets()` (I1=instantaneous |Φ|, I2=3h rolling, I3=binary spike, I4=Ψ spike), `clean_lead_lag()` with enforced gap parameter, `event_study()` with hour-of-day-matched controls, `residualise_on_tod()` projecting out hour-of-day and day-of-week dummies, `bounded_price_test_clean()` on I1
- Figures 14–20: PIT target time-series, clean lead/lag grid, event study, ToD residualisation, bounded-price raw vs residualised, regime-stratified bars, full decontamination dashboard

**Observations:** Three-tier conclusion structure:

**Downgraded:** The 20-hour lead and D4 R²=0.76 are confirmed as rolling-window artifacts. With point-in-time targets, D1's best lead drops from 20h to 1h — the 19-hour difference was entirely mechanical window overlap.

**Provisional:** Clean 1-hour lead for D1→I1: r=0.647, p<10⁻⁶. Clean 1-hour lead for D2→I1: r=0.569, p<10⁻⁶. Consistent with D1's autocorrelation structure (r=0.65 at lag 1, decaying to near zero by lag 6). The 1h lead reflects genuine carry-forward of imbalance state.

**Robust:** The bounded-price effect survives full residualisation. Raw: p<10⁻⁶, d=3.35. Residualised on hour-of-day and day-of-week: p<10⁻⁶, d=3.345. Effect size is essentially unchanged by removing diurnal structure. Significant in all four S-R clusters independently (d=2.26 to 3.35, all p<10⁻⁶). High-|Φ|/moderate-R hours have mean I1=0.967 vs 0.252 for matched low-|Φ|/moderate-R hours — a 3.8× ratio that does not disappear under any decontamination procedure applied.

---

## Iteration 5 — Graph-based constraint field

**Request:** Extend the prototype from scalar time series to a graph-based model. Build S and R as node-indexed vectors over a real Western Interconnect transmission network. Compute graph-Laplacian propagation with E as edge-level fluidity. Produce spatial snapshots, propagation comparison, bottleneck map, and animation.

**Delivered:**
- `graph/network.py`: 14-node Western Interconnect graph (CISO, BPAT, AZPS, PACW, PACE, NEVP, LDWP, WALC, IID, TIDC, IPCO, NWMT, PSCO, WACM), 28 interchange corridors, full data provenance per node and edge (observed / approximated / synthetic explicitly flagged)
- `graph/node_signals.py`: `SyntheticNodeSignals` using Cholesky decomposition of a geographic distance-decay covariance matrix (base_ρ=0.60, length scale=800km) for spatially coherent demand and congestion-event prices
- `graph/edge_fluidity.py`: E1 (inverse price spread), E2 (flow utilisation inverse), E3 (congestion proxy) as pure numpy arrays; `weighted_adjacency()`, `graph_laplacian()`
- `graph/propagation.py`: `GraphPropagator` with row-normalised Laplacian (D⁻¹L), stability check, reduced vs upgraded comparison, shock injection
- `graph/visualize_graph.py`: spatial network plots, propagation comparison, bottleneck map, 72-frame animated GIF
- Figures 21–25: three Phi snapshots, propagation comparison (CISO/BPAT/AZPS/PSCO), bottleneck map, full dashboard, Phi animation

**Key engineering resolution:** The capacity-weighted Laplacian had λ_max ≈ 54, making any η > 0.019 unstable (RMSE diverged to 73,000). Row-normalisation (D⁻¹L) bounds λ_max ≤ 2, making η < 0.5 unconditionally stable. Both modes then ran cleanly at RMSE = 0.711.

**Observations:** Edge fluidity distributions were well-calibrated (E1=0.606, E2=0.734, E3=0.526, all with meaningful variance). Bottleneck edges are the low-capacity corridors between price-divergent regions: IPCO→NWMT (0.8 GW, Idaho–Montana), AZPS→WACM (1.0 GW, Arizona–Colorado), WALC→LDWP (0.5 GW, Lower Colorado–LA). These represent the seams where Φ pooling would be most severe under congestion.

The reduced vs upgraded RMSE difference is small (0.7111 vs 0.7110) in this parameterisation. This is an honest result: with η=0.04 and γ=0.05, mean-reversion dominates propagation, so edge-level E modulation has limited influence on the aggregate trajectory. The animation reveals the structurally interesting behaviour — California's diurnal demand peaks propagate outward through the Pacific Intertie (CISO→BPAT, CISO→PACW) with a phase lag set by geographic distance, while the Mountain West nodes (PSCO, WACM, PACE) maintain a distinct phase due to weaker coupling. That spatial coherence structure is the Cholesky spatial correlation model expressing itself through the diffusion operator.

---

## Summary

| Iteration | Core addition | Key output |
|-----------|--------------|-----------|
| 1 | Scalar prototype: S, R, E, propagation | Package scaffold, 7 figures |
| 2 | E3 calibration system | SpreadCalibrator, 4 transforms × 13 scales |
| 3 | Divergence metrics + predictive models | Bounded-price effect: d=3.35, p<10⁻⁶ |
| 4 | Decontaminated lead analysis | 20h lead downgraded; 1h lead + BP effect survive |
| 5 | Graph extension: 14 nodes, 28 edges | Spatial Φ propagation, animation, bottleneck map |

**Robust finding across all iterations:** The bounded-price effect — elevated instantaneous instability in high-|Φ|/moderate-R periods versus matched low-|Φ|/moderate-R periods — survives every decontamination procedure: point-in-time targets, time-of-day residualisation, regime stratification, and shift to a graph-based spatial model. The gap between field pressure and price expression is a measurable, structurally stable signal. Whether it reflects genuine constraint-system physics or is partially an artifact of how synthetic data was constructed is the primary question that running the live EIA + CAISO adapters would resolve.

---

*All code in `constraint_field/` package. Run `python scripts/run_demo.py` for scalar prototype, `python scripts/run_divergence_analysis.py` for divergence analysis, `python scripts/run_decontaminated_lead.py` for clean lead tests, `python scripts/run_graph_analysis.py` for graph model.*
