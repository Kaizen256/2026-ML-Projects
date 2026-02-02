# Model 1: Baseline XGBoost (Lightcurve Statistics)

This notebook contains the second model for the competition. I used AI to create useful astronomical features, I do not have much knowledge in astronomy asides from a first year university elective.

This model is a structured upgrade over Model 1, focused on fixing two core issues:
- Astrophysical correctness (dust/extinction effects in flux measurements)
- Validation realism (data leakage and split structure)

The goal of this model is a higher score and impactful feature creation:
- correct observed flux for dust extinction (EBV)
- prevent leakage using split-aware validation
- stabilize predictions using fold ensembling
- tune hyperparameters using cross-validated Optuna

## Results

Best parameters:
- n_estimators: 3560
- learning_rate: 0.0240
- max_depth: 5
- min_child_weight: 11
- subsample: 0.5332
- colsample_bytree: 0.5563
- gamma: 0.7024
- reg_alpha: 5.8620
- reg_lambda: 9.4988
- max_delta_step: 9

OOF multiseed best threshold: 0.4583544303797469  
OOF multiseed best F1: 0.5102040816326531  

| Submission | Public LB F1 | Private LB F1 |
|-------------|--------------|----------------|
| 1 | 0.5921 | 0.5295 |

## Global (all-filters combined) features

These are computed using **all observations across all bands** for a given object.  
They summarize **time coverage, brightness distribution, cadence, variability, and context** (redshift + dust).

Differences vs Model 1:
 - Corrects flux/uncertainty for Milky Way dust extinction using EBV (more astrophysically correct flux features).
 - Adds rest-frame timing features by dividing time by (1 + z) so distant objects are compared on intrinsic timescales.
 - Expands from simple per-band stats to transient shape features (AUC above baseline, widths at 50/80%, slopes, Von Neumann eta).
 - Adds parametric Bazin curve-fit features for bands with enough data (captures rise/decay behavior).

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `n_obs` | Total number of observations across all filters | Captures overall sampling density and how well-measured the object is |
| `total_time_obs` | Observed-frame time baseline: `max(t_rel) - min(t_rel)` | Separates short transients vs long events and measures overall monitoring duration |
| `total_time_rest` | Rest-frame time baseline: `total_time_obs / (1+z)` | Removes time dilation so the model compares intrinsic evolution speed across redshifts |

### Flux distribution (dust-corrected `flux_corr`)

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `flux_mean` | Mean corrected flux | Measures average intrinsic brightness level (sensitive to sustained high flux) |
| `flux_median` | Median corrected flux | Robust typical brightness baseline (less sensitive to one-off spikes) |
| `flux_std` | Standard deviation of corrected flux | Captures variability strength (high = more change over time) |
| `flux_min` | Minimum corrected flux | Captures deep fades / dips / negative excursions from noise-subtraction artifacts |
| `flux_max` | Maximum corrected flux | Captures peak brightness or flare intensity (key transient signature) |
| `flux_range` | Flux range: `flux_max - flux_min` | Simple amplitude proxy for overall brightness swing |
| `flux_mad` | Median absolute deviation of corrected flux | Robust variability estimate that doesn’t get bullied by outliers |
| `flux_iqr` | Interquartile range of corrected flux | Another robust variability measure (spread of the middle 50%) |
| `flux_skew` | Skewness of corrected flux distribution | Detects asymmetric lightcurves (fast rise / slow decay vs vice versa) |
| `flux_kurt_excess` | Excess kurtosis of corrected flux distribution | Detects heavy tails/spiky behavior from rare bursts or sharp transients |
| `neg_flux_frac` | Fraction of corrected flux values `< 0` | Flags noise-dominated objects or weak detections where measurements hover around zero |

### SNR (using corrected errors `err_corr`)

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `snr_median` | Median SNR where `snr = \|flux_corr\| / (err_corr + 1e-8)` | Typical detection quality (separates clean signals from noisy junk) |
| `snr_max` | Maximum SNR | Captures the strongest detection event (some transients “light up” briefly) |

### Cadence / gaps

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `median_dt` | Median time gap between consecutive observations in `t_rel` | Describes typical cadence (important since sparse sampling hides shape) |
| `max_gap` | Maximum time gap between consecutive observations in `t_rel` | Detects missing windows (large gaps can explain unreliable peak/width estimates) |

### Time-series variability / shape

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `eta_von_neumann` | Von Neumann eta statistic on `flux_corr` (smoothness vs jumpiness) | Separates smooth evolving curves from noisy jitter or sudden jumps |
| `beyond_1p5std` | Fraction of points beyond `1.5 * std` from the center | Measures outlier / burstiness rate (transients often have extreme points) |
| `max_slope_global` | Maximum absolute slope in observed time (`t_rel`) | Captures fastest brightness change (sharp rise/fall events) |
| `med_abs_slope_global` | Median absolute slope in observed time (`t_rel`) | Captures typical rate of change (slow drifters vs active transients) |

### Context metadata

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `Z` | Redshift `z` | Encodes distance/epoch effects and shifts events into different observed regimes |
| `log1pZ` | `log(1+z)` | Stabilizes redshift scaling for models (less extreme leverage at high `z`) |
| `EBV` | Dust reddening used for extinction correction | Helps the model learn residual dust systematics and measurement conditions |

### Filter coverage

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `n_filters_present` | Number of filters with ≥ 1 observation | Multi-band coverage gives richer color/shape info; missing bands can correlate with class |
| `total_obs` | Total observations summed across all filters (same as `n_obs`) | Redundant but convenient sanity/coverage signal that some models exploit |

## Per-filter (band-wise) features

For each band `b ∈ {u,g,r,i,z,y}`, the following features are computed independently per filter.  
These capture color-dependent brightness behavior and band-specific temporal dynamics.

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `n_{b}` | Number of observations in band `b` | Band missingness and sampling density vary by object/class and affect reliability |
| `amp_{b}` | Amplitude above baseline: `max(fb) - median(fb)` | Measures event strength in that band (key for color-specific transient signatures) |
| `tpeak_{b}_obs` | Observed-frame time of peak flux in band `b` | Captures when the band reaches maximum brightness (timing is class-dependent) |
| `tpeak_{b}_rest` | Rest-frame time of peak flux: `tpeak_obs / (1+z)` | Removes time dilation so peak timing is comparable across redshifts |
| `width50_{b}_rest` | Rest-frame width above 50% amplitude (if measurable) | Measures event duration at mid-brightness (distinguishes fast vs slow transients) |
| `width80_{b}_rest` | Rest-frame width above 80% amplitude (if measurable) | Focuses on the high-brightness core duration (sharp vs broad peaks) |
| `auc_pos_{b}_rest` | Rest-frame AUC of positive signal: `∫ max(fb - baseline, 0) dt` | Measures total emitted “excess flux” over baseline (energy-like summary) |
| `snrmax_{b}` | Maximum SNR within band `b` | Strongest detection in that band (some classes peak strongly only in certain filters) |
| `eta_{b}` | Von Neumann eta within band `b` | Detects smooth evolution vs noise inside a single wavelength band |
| `maxslope_{b}` | Maximum slope within band `b` (rest-frame time) | Captures fastest intrinsic change rate per band (rise/decline sharpness) |

### Bazin fit features (only if `n_b >= 6`)

These are parametric shape features from fitting a **Bazin transient curve model** in each band.

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `bazin_A_{b}` | Bazin amplitude-like parameter | Encodes overall transient strength in a smooth, denoised way |
| `bazin_B_{b}` | Bazin baseline-like parameter | Captures persistent baseline flux level (helps separate steady sources vs transient-only) |
| `bazin_trise_{b}` | Bazin rise timescale | Learns how quickly brightness increases (very class-discriminative) |
| `bazin_tfall_{b}` | Bazin fall timescale | Learns decay speed (slow fades vs rapid drop-offs) |
| `bazin_redchi2_{b}` | Reduced chi-square of Bazin fit | Quality-of-fit measure: real transients fit well, noisy/non-transient behavior fits poorly |

## Cross-band pair features (adjacent pairs: `ug, gr, ri, iz, zy`)

For each adjacent filter pair `(a,b)`, these compare amplitude, timing, peak ratios, and energy-like signal across wavelengths.

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `ampdiff_{a}{b}` | Amplitude difference: `amp_a - amp_b` | Captures color gradients and temperature evolution signatures between bands |
| `tpeakdiff_{a}{b}_rest` | Rest-frame peak time difference: `tpeak_a_rest - tpeak_b_rest` | Measures chromatic peak lag/lead (some classes peak earlier in blue than red) |
| `peakratio_{a}{b}` | Peak flux ratio: `peak_flux_a / (peak_flux_b + 1e-8)` | Strong color/SED proxy without needing explicit magnitudes |
| `aucdiff_{a}{b}` | Positive-signal AUC difference: `auc_a - auc_b` | Measures which band dominates total emitted signal (helps separate spectral behaviors) |

## Training Setup

### Train / Validation Split
- 80/20 split using `train_test_split`
- stratified by target to preserve class balance

The dataset is imbalanced, so I use: scale_pos_weight to weight positive examples more strongly during training.

## Final training using the best CV hyperparameters

After Optuna tuning:
- retrain using GroupKFold across all split groups
- store out-of-fold probabilities for every training sample
- select a global F1-max threshold using OOF predictions

This provides:
- a fold-ensemble of models for inference
- a threshold optimized on realistic split-aware validation

## Takeaways

What worked:
- Additional features + de-extinction improved performance.
- Validation F1 score dropped, and leaderboard F1 score improved. Better generalization.
