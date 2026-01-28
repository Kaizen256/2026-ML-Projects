# Model 3: XGBoost/LightGBM Blend

This notebook contains the third model for the competition.

The goal was to improve generalization by:
- simulating realistic redshift uncertainty using `Z_err` (photo-z augmentation)
- training a two-model ensemble (XGBoost + LightGBM)
- tuning both the blend weight and the classification threshold using out-of-fold (OOF) predictions

## Results

| Model | OOF / CV Best | Public LB F1 |
|------:|---------------|--------------|
| Model 3 | Optuna best value: **0.5280** (trial 14) | **0.5613** and **0.5082** |

Which is a downgrade on the public LB from model #2 (0.5921)

OOF blend calibration:
- OOF best alpha: **0.03**
- OOF best threshold: **0.01**
- OOF blended best F1: **0.51875**

I used AI to help implement astronomy-specific preprocessing and feature functions. I am not an astronomer.

## Global (all-filters combined) features

These are computed using all observations across all bands for a given object.  
They summarize time coverage, brightness distribution, cadence, variability, and context (redshift + dust + redshift uncertainty).

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
| `flux_mad` | Median absolute deviation of corrected flux | Robust variability estimate that doesn’t get bullied by outliers |
| `flux_iqr` | Interquartile range of corrected flux | Another robust variability measure (spread of the middle 50%) |
| `flux_skew` | Skewness of corrected flux distribution | Detects asymmetric lightcurves (fast rise / slow decay vs vice versa) |
| `flux_kurt_excess` | Excess kurtosis of corrected flux distribution | Detects heavy tails/spiky behavior from rare bursts or sharp transients |
| `flux_p5` | 5th percentile of corrected flux | Robust low-end brightness level (less sensitive than min) |
| `flux_p25` | 25th percentile of corrected flux | Lower-quartile brightness level |
| `flux_p75` | 75th percentile of corrected flux | Upper-quartile brightness level |
| `flux_p95` | 95th percentile of corrected flux | Robust high-end brightness level (less sensitive than max) |
| `robust_amp_global` | Robust amplitude: `flux_p95 - flux_p5` | Outlier-resistant variability scale, often better than max-min |
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
| `chi2_const_global` | Chi-square vs constant-flux model using `err_corr` | Quantifies variability relative to measurement noise (true variability vs noise) |
| `stetsonJ_global_obs` | Stetson J index using observed-frame times | Captures correlated variability behavior (often strong for real transients) |
| `stetsonJ_global_rest` | Stetson J index using rest-frame times | Same idea, but corrected for time dilation so timing-related correlation is comparable |

### Slopes / rate of change (global)

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `max_slope_global_obs` | Maximum absolute slope in observed time (`t_rel`) | Captures fastest brightness change (sharp rise/fall events) |
| `max_slope_global_rest` | Maximum absolute slope in rest-frame time (`t_rest`) | Intrinsic fastest change rate (removes redshift stretching) |
| `med_abs_slope_global_obs` | Median absolute slope in observed time | Typical observed change rate (slow drifters vs active transients) |
| `med_abs_slope_global_rest` | Median absolute slope in rest-frame time | Typical intrinsic change rate |
| `slope_global_obs` | Best-fit linear slope over observed time | Captures long-term trend direction (rising vs fading overall) |
| `slope_global_rest` | Best-fit linear slope over rest-frame time | Same trend, but comparable across redshifts |

### Fractional variability

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `fvar_global` | Fractional variability accounting for measurement errors | Estimates intrinsic variability strength after subtracting noise contribution |

### Context metadata

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `Z` | Redshift `z` | Encodes distance/epoch effects and shifts events into different observed regimes |
| `log1pZ` | `log(1+z)` | Stabilizes redshift scaling for models (less extreme leverage at high `z`) |
| `Z_err` | Redshift uncertainty (clipped to `>= 0`) | Captures confidence in rest-frame correction; noisy redshifts degrade timing features |
| `log1pZerr` | `log(1+Z_err)` | Stabilizes uncertainty scaling and helps tree models split more smoothly |
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
| `robust_amp_{b}` | Robust amplitude: `p95_b - p5_b` | More stable amplitude estimate when peaks/outliers are noisy |
| `tpeak_{b}_obs` | Observed-frame time of peak flux in band `b` | Captures when the band reaches maximum brightness (timing is class-dependent) |
| `tpeak_{b}_rest` | Rest-frame time of peak flux: `tpeak_obs / (1+z)` | Removes time dilation so peak timing is comparable across redshifts |
| `width50_{b}_obs` | Observed-frame width above 50% amplitude | Measures mid-brightness duration in observed time (useful when timing is observationally relevant) |
| `width50_{b}_rest` | Rest-frame width above 50% amplitude | Intrinsic mid-brightness duration (fast vs slow transients) |
| `width80_{b}_obs` | Observed-frame width above 80% amplitude | Captures core peak width in observed time (sharp vs broad peaks) |
| `width80_{b}_rest` | Rest-frame width above 80% amplitude | Intrinsic core peak width (class-discriminative) |
| `rise50_{b}_obs` | Observed-frame time from 50% rise crossing to peak | Encodes rise speed in observed time (cadence-aware) |
| `decay50_{b}_obs` | Observed-frame time from peak to 50% decay crossing | Encodes decay speed in observed time |
| `asym50_{b}_obs` | Rise/decay asymmetry: `rise50 / (decay50 + 1e-8)` | Separates fast-rise slow-decay vs slow-rise fast-decay shapes |
| `rise50_{b}_rest` | Rest-frame time from 50% rise crossing to peak | Intrinsic rise timescale per band |
| `decay50_{b}_rest` | Rest-frame time from peak to 50% decay crossing | Intrinsic decay timescale per band |
| `asym50_{b}_rest` | Rest-frame rise/decay asymmetry | Intrinsic shape asymmetry (less biased by redshift) |
| `auc_pos_{b}_obs` | Observed-frame AUC of positive signal: `∫ max(fb - baseline, 0) dt` | Energy-like summary in observed time (cadence and time dilation included) |
| `auc_pos_{b}_rest` | Rest-frame AUC of positive signal | Energy-like summary comparable across redshifts |
| `snrmax_{b}` | Maximum SNR within band `b` | Strongest detection in that band (some classes peak strongly only in certain filters) |
| `eta_{b}` | Von Neumann eta within band `b` | Detects smooth evolution vs noise inside a single wavelength band |
| `chi2_const_{b}` | Chi-square vs constant-flux model within band | Measures variability significance relative to band-specific noise |
| `slope_{b}_obs` | Best-fit linear slope in band over observed time | Captures overall rise/fade trend per band |
| `slope_{b}_rest` | Best-fit linear slope in band over rest-frame time | Intrinsic trend per band (comparable across redshifts) |
| `maxslope_{b}_obs` | Maximum absolute slope in band (observed time) | Captures sharpest observed change (rise/fall) per band |
| `maxslope_{b}_rest` | Maximum absolute slope in band (rest time) | Captures sharpest intrinsic change rate per band |
| `stetsonJ_{b}_obs` | Stetson J in band using observed time | Detects correlated variability patterns per band |
| `stetsonJ_{b}_rest` | Stetson J in band using rest-frame time | Same, but corrected for time dilation |
| `p5_{b}` | 5th percentile of band flux `fb` | Robust low-end level per band |
| `p25_{b}` | 25th percentile of `fb` | Lower-quartile level per band |
| `p75_{b}` | 75th percentile of `fb` | Upper-quartile level per band |
| `p95_{b}` | 95th percentile of `fb` | Robust high-end level per band |
| `mad_{b}` | Median absolute deviation of `fb` | Robust band variability (outlier-resistant) |
| `iqr_{b}` | Interquartile range of `fb` | Robust spread of the middle 50% per band |
| `mad_over_std_{b}` | `mad_b / (std_b + 1e-8)` | Flags spike-dominated vs Gaussian-like variability (robustness/shape cue) |
| `fvar_{b}` | Fractional variability within band (noise-corrected) | Intrinsic variability strength per band |

## Cross-band pair features (adjacent pairs: `ug, gr, ri, iz, zy`)

For each adjacent filter pair `(a,b)`, these compare amplitude, timing, and peak ratios across wavelengths.

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `ampdiff_{a}{b}` | Amplitude difference: `amp_a - amp_b` | Captures color gradients and temperature evolution signatures between bands |
| `tpeakdiff_{a}{b}_obs` | Observed-frame peak time difference: `tpeak_a_obs - tpeak_b_obs` | Chromatic peak lag/lead in observed time (also reflects cadence + time dilation) |
| `tpeakdiff_{a}{b}_rest` | Rest-frame peak time difference: `tpeak_a_rest - tpeak_b_rest` | Measures intrinsic chromatic peak lag/lead (more physically comparable) |
| `peakratio_{a}{b}` | Peak flux ratio: `peak_flux_a / (peak_flux_b + 1e-8)` | Strong color/SED proxy without needing explicit magnitudes |

## Color features at r-peak (observed-frame) + 10-day color evolution

These interpolate `g`, `r`, `i` flux at the observed time when the r-band peaks (`tpeak_r_obs`), then compute log-flux colors.

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `color_gr_at_rpeak_obs` | `log1p(f_g) - log1p(f_r)` evaluated at `tpeak_r_obs` | Measures g-r color at peak, which is highly class-dependent |
| `color_ri_at_rpeak_obs` | `log1p(f_r) - log1p(f_i)` evaluated at `tpeak_r_obs` | Measures r-i color at peak (temperature / SED proxy) |
| `color_gr_slope10_obs` | `(color_gr(t+10) - color_gr(t)) / 10` days | Captures how color evolves after peak (cooling/heating signatures) |
| `color_ri_slope10_obs` | `(color_ri(t+10) - color_ri(t)) / 10` days | Another post-peak color evolution cue (very discriminative for transients) |

## What changed vs Model 2

### 1) Added `Z_err` and photo-z augmentation
Model 2 used redshift `Z` but did not explicitly model uncertainty.

Model 3 introduces:
- `Z_err` as a feature (plus log transforms)
- training-time augmentation where I resample each training object with a perturbed redshift:
  - sample `sigma` from the test `Z_err` distribution
  - generate `z_sim = z + Normal(0, sigma)` (clipped to `>= 0`)
  - recompute features with `(z_sim, sigma)`
  - mark augmented rows with `photoz_aug = 1`

### 2) Added a second learner (LightGBM)
Instead of only XGBoost, Model 3 trains:
- an XGBoost model per fold
- a LightGBM model per fold

### 3) OOF probability blending
I blend the two model families using:
- `p_blend = alpha * p_xgb + (1 - alpha) * p_lgb`

Then I tune:
- `alpha` (blend weight)
- `threshold` (probability cutoff)

using OOF predictions.

### 4) Objective mismatch experiment
XGBoost hyperparameters are tuned using Optuna on a CV metric (logged as the Optuna objective), and then a separate threshold search is used to optimize F1.
This was an experiment to see if improving ranking quality would help final classification performance.

## Notes on why this may have underperformed

This model produced inconsistent generalization:
- the best OOF blend weight was **alpha = 0.03**, meaning the blend mostly relied on one model family
- the best OOF threshold was extremely low (**0.01**), suggesting probability calibration mismatch
- photo-z augmentation may have added noise that improved CV behavior but did not transfer cleanly to leaderboard scoring

I also didn't train either model for very long at all. Longer training would yield better results, but it wouldn't be comparable to model 4 and 5.

## Best Optuna result (XGBoost)

Best is trial 14 with value: **0.528035337432086**

Best parameters:
- n_estimators: 4328
- learning_rate: 0.007079630495604182
- max_depth: 4
- min_child_weight: 1
- subsample: 0.5936362024881353
- colsample_bytree: 0.9146736072473098
- gamma: 0.6466321530023438
- reg_alpha: 4.130135428812432
- reg_lambda: 5.5649710878468825
- max_delta_step: 1
- grow_policy: depthwise

## Training and validation strategy

### Split-aware cross validation
This model uses a stratified grouped split to:
- prevent leakage across `split`
- keep class balance across folds

### Class imbalance handling
As in previous models, I compute `scale_pos_weight` inside each fold:
- `scale_pos_weight = neg / pos`

This weights the positive class more strongly during training.

## Ensemble and calibration

After training both model families across all folds:
- store OOF predictions for XGB and LGB separately
- search for the best blend weight `alpha` in `[0, 1]`
- for each `alpha`, search for the best F1 threshold
- select the best `(alpha, threshold)` pair by OOF F1

Final OOF blend parameters:
- alpha: 0.03
- threshold: 0.01
- OOF blended best F1: 0.51875

## Takeaways

What I learned:
- Photo-z augmentation is not automatically helpful and can introduce distribution mismatch.
- Blending can work, but OOF-optimal calibration (very low threshold and near-zero alpha) can signal instability.
- More complexity does not guarantee better generalization.

Next steps:
- refine augmentation policy (when to augment, how much, and for which objects)
- improve probability calibration and threshold stability
- keep validation strategy strict and aligned with leaderboard behavior
