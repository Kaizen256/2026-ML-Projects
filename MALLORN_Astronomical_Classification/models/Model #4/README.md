# Model 4: SpecType Teacher Features

This notebook contains my teacher–student style model for the MALLORN competition. I used AI to help design and sanity-check features, since my astronomy background is limited to an introductory university elective.

This model is a structured upgrade over earlier models and focuses on exploiting a key training-only signal:

- The training data includes `SpecType` / `SpecTypeGroup`
- These labels are not available in the test set
- `SpecTypeGroup` is strongly correlated with the final target classes
- A separate **teacher model** is trained to predict `SpecTypeGroup`
- Its out-of-fold predicted probabilities are used as features for the final model

Core goals:

- extract SpecTypeGroup signal without leakage
- add high-signal teacher probability features
- keep validation split-aware and realistic
- stabilize predictions with fold ensembling and threshold tuning

## Results

| Submission | Public LB F1 | Final LB F1 |
|-------------|--------------|-------------|
| 1 | 0.6309 | 0.5688 |
| 2 | 0.6024 | 0.5333 |
| 3 | 0.6009 | 0.5467 |

## Global features

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
| `flux_p5` | 5th percentile of corrected flux | Robust low-end level (less sensitive than min) |
| `flux_p25` | 25th percentile of corrected flux | Lower-quartile level |
| `flux_p75` | 75th percentile of corrected flux | Upper-quartile level |
| `flux_p95` | 95th percentile of corrected flux | Robust high-end level (less sensitive than max) |
| `robust_amp_global` | Robust amplitude: `flux_p95 - flux_p5` | Outlier-resistant variability scale, often better than max-min |
| `neg_flux_frac` | Fraction of corrected flux values `< 0` | Flags noise-dominated objects or weak detections where measurements hover around zero |

### SNR (using corrected errors `err_corr`)

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `snr_median` | Median SNR where `snr = \|flux_corr\| / (err_corr + EPS)` | Typical detection quality (separates clean signals from noisy junk) |
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
| `stetsonJ_global_obs` | Stetson J (consecutive-pairs) using observed-frame times | More cadence-aware correlation metric; reduces sensitivity to irregular sampling |
| `stetsonJ_global_rest` | Stetson J (consecutive-pairs) using rest-frame times | Same correlation idea, but corrected for time dilation |

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
This version adds **pre-peak baseline features** and richer **post-peak decay morphology**.

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `n_{b}` | Number of observations in band `b` | Band missingness and sampling density vary by object/class and affect reliability |
| `baseline_pre_{b}` | Estimated baseline flux before the main peak (from earliest fraction of points) | Gives a cleaner “true baseline” than median when post-peak tail biases the median |
| `amp_{b}` | Peak above median baseline: `peak_flux - median(fb)` | Simple band strength; works even if pre-peak baseline is unreliable |
| `amp_pre_{b}` | Peak above pre-peak baseline: `peak_flux - baseline_pre` | Physically better peak amplitude when baseline is stable; improves peak-related shape features |
| `robust_amp_{b}` | Robust amplitude: `p95_b - p5_b` | More stable amplitude estimate when peaks/outliers are noisy |
| `tpeak_{b}_obs` | Observed-frame time of peak flux in band `b` | Captures when the band reaches maximum brightness (timing is class-dependent) |
| `tpeak_{b}_rest` | Rest-frame time of peak flux: `tpeak_obs / (1+z)` | Removes time dilation so peak timing is comparable across redshifts |
| `snrmax_{b}` | Maximum SNR within band `b` | Strongest detection in that band (some classes peak strongly only in certain filters) |
| `eta_{b}` | Von Neumann eta within band `b` | Detects smooth evolution vs noise inside a single wavelength band |
| `chi2_const_{b}` | Chi-square vs constant-flux model within band | Measures variability significance relative to band-specific noise |
| `slope_{b}_obs` | Best-fit linear slope in band over observed time | Captures overall rise/fade trend per band |
| `slope_{b}_rest` | Best-fit linear slope in band over rest-frame time | Intrinsic trend per band (comparable across redshifts) |
| `maxslope_{b}_obs` | Maximum absolute slope in band (observed time) | Captures sharpest observed change per band |
| `maxslope_{b}_rest` | Maximum absolute slope in band (rest time) | Captures sharpest intrinsic change rate per band |
| `stetsonJ_{b}_obs` | Stetson J (consecutive-pairs) in band using observed time | Cadence-aware correlation metric per band |
| `stetsonJ_{b}_rest` | Stetson J (consecutive-pairs) in band using rest time | Same, but corrected for time dilation |
| `fvar_{b}` | Fractional variability within band (noise-corrected) | Intrinsic variability strength per band |
| `p5_{b}` | 5th percentile of band flux `fb` | Robust low-end level per band |
| `p25_{b}` | 25th percentile of `fb` | Lower-quartile level per band |
| `p75_{b}` | 75th percentile of `fb` | Upper-quartile level per band |
| `p95_{b}` | 95th percentile of `fb` | Robust high-end level per band |
| `mad_{b}` | Median absolute deviation of `fb` | Robust band variability (outlier-resistant) |
| `iqr_{b}` | Interquartile range of `fb` | Robust spread of the middle 50% per band |
| `mad_over_std_{b}` | `mad_b / (std_b + EPS)` | Flags spike-dominated vs Gaussian-like variability (robustness/shape cue) |

### Post-peak fall times, widths, and sharpness (only if `amp_pre_{b} > 0`)

These use `baseline_pre_{b}` and `amp_pre_{b}` to define levels as fractions of the peak amplitude.

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `t_fall50_{b}_obs` | Observed-frame time after peak to reach `baseline_pre + 0.50 * amp_pre` | Encodes decay speed in observed time (fast vs slow fall) |
| `t_fall20_{b}_obs` | Observed-frame time after peak to reach `baseline_pre + 0.20 * amp_pre` | Longer-tail decay behavior; distinguishes slow fade vs quick drop |
| `t_fall50_{b}_rest` | Rest-frame fall time to 50% level | Intrinsic decay speed comparable across redshifts |
| `t_fall20_{b}_rest` | Rest-frame fall time to 20% level | Intrinsic late-time fading timescale |
| `width50_{b}_obs` | Observed-frame width above 50% level (time span where `fb >= base + 0.5*amp`) | Measures how long the event stays bright in observed time |
| `width80_{b}_obs` | Observed-frame width above 80% level | Captures core peak width (sharp vs broad peak) |
| `width50_{b}_rest` | Rest-frame width above 50% level | Intrinsic duration at mid-brightness |
| `width80_{b}_rest` | Rest-frame width above 80% level | Intrinsic core-peak duration |
| `sharp50_{b}_obs` | Sharpness proxy: `amp_pre / (width50_obs + EPS)` | High = tall + narrow peaks (very class-discriminative) |
| `sharp50_{b}_rest` | Sharpness proxy in rest-frame | Same idea, but intrinsic (less redshift-biased) |
| `auc_pos_{b}_obs` | Observed-frame AUC above `baseline_pre`: `∫ max(fb - baseline_pre, 0) dt` | Energy-like summary tied to true baseline, not median-biased |
| `auc_pos_{b}_rest` | Rest-frame AUC above `baseline_pre` | Comparable across redshifts; strong spectral-energy cue |

### Peak structure and post-peak behavior (only if `amp_pre_{b} > 0`)

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `peak_dominance_{b}` | `amp_pre / (mad_pre + EPS)` where `mad_pre` is pre-peak baseline MAD | Measures how dominant the peak is relative to baseline noise (real transients pop out) |
| `std_ratio_prepost_{b}` | `std(pre_seg) / (std(post_seg) + EPS)` | Captures how variability changes after peak (e.g., noisy baseline vs smooth decay) |
| `n_peaks_{b}` | Count of significant peaks above baseline (sigma-thresholded) | Distinguishes single-peaked transients from multi-peaked/variable sources |
| `postpeak_monotone_frac_{b}` | Fraction of post-peak steps that are monotonic decreasing | Smooth decays (high) vs rebrightening/AGN-like variability (low) |
| `n_rebrighten_{b}` | Count of rebrightening events after peak (relative to `amp_pre`) | Strong discriminator: rebrightening often means non-simple transient behavior |

### Decay power-law fit (post-peak, only if `amp_pre_{b} > 0`)

A power-law fit is attempted on the post-peak decay segment (up to a max time window).

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `decay_pl_slope_{b}_obs` | Fitted power-law decay slope in observed time | Encodes decay physics/shape; different classes have different decay slopes |
| `decay_pl_r2_{b}_obs` | R² of the observed-frame power-law fit | Measures how well a clean power-law explains the decay (clean transient vs messy variability) |
| `decay_pl_npts_{b}_obs` | Number of points used in the observed-frame decay fit | Reliability indicator: more points = more trustworthy slope |
| `decay_pl_slope_{b}_rest` | Fitted power-law decay slope in rest-frame time | Intrinsic decay slope, comparable across redshifts |
| `decay_pl_r2_{b}_rest` | R² of the rest-frame power-law fit | Fit quality after time dilation correction |
| `decay_pl_npts_{b}_rest` | Number of points used in the rest-frame decay fit | Reliability indicator in rest-frame |

## Multi-band peak timing dispersion

These summarize how synchronized (or not) the band peaks are across filters.

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `tpeak_std_obs` | Standard deviation of `tpeak_b_obs` across bands with peaks | Measures chromatic timing spread in observed time (class-dependent) |
| `tpeak_std_rest` | Standard deviation of `tpeak_b_rest` across bands with peaks | Intrinsic chromatic peak spread (less redshift-biased) |

## Cross-band pair features (adjacent pairs: `ug, gr, ri, iz, zy`)

For each adjacent filter pair `(a,b)`, these compare peak timing and peak flux ratios across wavelengths.

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `tpeakdiff_{a}{b}_obs` | Observed-frame peak time difference: `tpeak_a_obs - tpeak_b_obs` | Chromatic peak lag/lead in observed time (includes cadence + dilation effects) |
| `tpeakdiff_{a}{b}_rest` | Rest-frame peak time difference: `tpeak_a_rest - tpeak_b_rest` | Intrinsic chromatic lag/lead; strong class signature (blue earlier than red, etc.) |
| `peakratio_{a}{b}` | Peak flux ratio: `peak_flux_a / (peak_flux_b + EPS)` | Strong color/SED proxy without needing explicit magnitudes |

## Color features at r-peak (observed-frame) + 20/40-day evolution

These interpolate `g`, `r`, `i` flux at the observed time when the r-band peaks (`tpeak_r_obs`), then compute log-flux colors.  
They also sample the same colors at `+20` and `+40` days to capture cooling/heating trends.

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `color_gr_at_rpeak_obs` | `log1p(f_g) - log1p(f_r)` evaluated at `tpeak_r_obs` | Measures g-r color at peak, highly class-dependent |
| `color_ri_at_rpeak_obs` | `log1p(f_r) - log1p(f_i)` evaluated at `tpeak_r_obs` | Measures r-i color at peak (temperature / SED proxy) |
| `color_gr_rpeak_p20_obs` | g-r color at `tpeak_r_obs + 20` days | Captures medium-term color evolution after peak |
| `color_ri_rpeak_p20_obs` | r-i color at `tpeak_r_obs + 20` days | Same, for redder color index |
| `color_gr_rpeak_p40_obs` | g-r color at `tpeak_r_obs + 40` days | Captures longer-term cooling/heating behavior |
| `color_ri_rpeak_p40_obs` | r-i color at `tpeak_r_obs + 40` days | Longer-term evolution in redder bands |
| `color_gr_slope20_obs` | `(color_gr(+20) - color_gr(0)) / 20` | Rate of color change over 20 days (cooling slope) |
| `color_ri_slope20_obs` | `(color_ri(+20) - color_ri(0)) / 20` | Rate of red color change over 20 days |
| `color_gr_slope40_obs` | `(color_gr(+40) - color_gr(0)) / 40` | Rate of color change over 40 days (more stable, less noisy) |
| `color_ri_slope40_obs` | `(color_ri(+40) - color_ri(0)) / 40` | Rate of red color change over 40 days |

## SpecType teacher stacking features (high-level)

`add_spectype_teacher_features()` adds *legal stacking* features by training a multiclass model on **train only** to predict a grouped version of `SpecType`, then appending the predicted class probabilities as new features.

Key steps:
- Map `SpecType` → `SpecTypeGroup` (TDE, AGN, SNIa, SNother, Other)
- Train a LightGBM multiclass model using CV splits by `split`
- Create:
  - OOF probabilities for train
  - full-fit probabilities for test
- Append probabilities + entropy as new features

### Per-class probability features

For every group label `c` in `classes`:

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `p_spec_{c}` | Predicted probability the object belongs to SpecTypeGroup `c` | Injects a strong “soft label” summary of transient type, which improves the final binary classifier |

### Probability-uncertainty feature

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `spec_entropy` | Entropy of the teacher probability vector | High entropy = teacher unsure (ambiguous object); low entropy = confident type signal (more reliable stacking) |

### Teacher Model

A LGBM Classifier is trained to predict `SpecTypeGroup` using the full feature set.

To prevent leakage:

- GroupKFold is used by split group
- Teacher predictions are generated out-of-fold
- Only OOF teacher probabilities are used as student features

### Student Model

The final classifier is trained to predict the competition target using:

- all base lightcurve/statistical features
- teacher OOF probability features
- teacher confidence and entropy features

## Best Optuna Result (Teacher)

Best AP: **0.6134734232399863**

Best parameters:

- n_estimators: 4770
- learning_rate: 0.009408
- max_depth: 5
- min_child_weight: 38
- subsample: 0.9580
- colsample_bytree: 0.5860
- gamma: 8.6793
- reg_alpha: 17.5374
- reg_lambda: 24.2249
- max_delta_step: 2
- grow_policy: depthwise

## Takeaways

What worked:

- Teacher probabilities added strong, structured signal.
- OOF teacher features improved generalization.
- Split-aware CV prevented leakage and fake validation gains.

What to watch:

- Teacher features must be strictly OOF.
- Proxy-task models help, but should not replace base features.
