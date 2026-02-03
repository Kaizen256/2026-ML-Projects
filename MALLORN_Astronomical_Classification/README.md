# MALLORN Astronomical Classification Challenge
Photometric TDE classification from LSST-like multi-band lightcurves

I competed in the MALLORN Classifier Challenge and placed **29th out of 2389 entrants (893 teams)** with limited time (about 2 weeks, versus the full ~4 months).

**I used AI to generate useful astronomical features, as many competitors did, since my astronomy background is limited to a first-year university elective.**

I have fully documented my progression across five models, each with detailed comments and its own README file.

## Competition Summary

Objective: Detect tidal disruption events (TDEs) from photometric lightcurves only, no spectra, using machine learning.

TDEs occur when a star gets torn apart by a supermassive black hole. They are rare, scientifically valuable, and difficult to identify at scale. Upcoming surveys like LSST (Vera C. Rubin Observatory) will discover massive numbers of transients, but only a small fraction can be followed up with spectroscopy. This makes accurate lightcurve-based classification essential.

## Dataset

The dataset contains simulated LSST-style lightcurves built from real nuclear transient observations (ZTF-based), measured across 6 filters:

- `u, g, r, i, z, y`

Each object includes:
- Multi-band time series flux measurements
- Measurement uncertainty
- Redshift (`Z`) and redshift error (`Z_err`)
- Dust extinction value (`EBV`)
- True spectroscopic type (train only)

## Task

**Binary classification**

- `target = 1` --> TDE  
- `target = 0` --> Non-TDE (AGN + multiple supernova subclasses)

Models must learn to distinguish TDE lightcurve behavior from other nuclear transients under realistic cadence gaps, noise, and multi-band variability.

## Key Columns

### Log files (`train_log.csv`, `test_log.csv`)
- `object_id`: unique identifier for each object
- `Z`: redshift
  - train: spectroscopic (negligible error)
  - test: photometric (has uncertainty)
- `Z_err`: redshift uncertainty (test-only, blank in train)
- `EBV`: extinction coefficient (dust), used for de-extinction correction
- `SpecType` (train-only): spectroscopic transient type
- `split`: which split folder contains the lightcurve rows
- `target` (train-only): binary TDE label

### Lightcurve files (`*_full_lightcurves.csv`)
- `object_id`
- `Time (MJD)`: observation time
- `Flux`: observed flux (can be negative due to baseline subtraction + noise)
- `Flux_err`: flux uncertainty
- `Filter`: `u,g,r,i,z,y`

## My Core Approach

1. Compress each irregular multi-band time series into a fixed-length feature vector
2. Train tree-based models (mainly XGBoost)
3. Use split-aware validation (`groups = split`) to avoid leakage and match dataset structure
4. Optimize a global decision threshold for F1

Key ideas that mattered most:
- Physics-aware preprocessing (dust de-extinction using `EBV`)
- Split-safe CV using `StratifiedGroupKFold`
- Teacher stacking using `SpecType` (train-only)
- Feature selection (gain-based Top-K) before Optuna
- Multiseed ensembling (should have tested this one a bit more as others said it had a negative impact after the competition finished)

## Results Overview

### Model timeline (what changed and what it scored)

| Model | Main idea | Validation style | Best Submission's public LB F1 | Best Submission's final LB F1 |
|------:|-----------|------------------|-------------:|--------------:|
| 1 | Baseline XGB + basic stats | Random stratified 80/20 | 0.4610 | 0.4540 |
| 2 | De-extinction + split-safe CV XGB | Group-aware CV | 0.5921 | 0.5295 |
| 3 | XGB + LGB blend + photo-z aug | Group-aware CV + blending | 0.5613 | 0.5313 |
| 4 | SpecType teacher features | Group-aware teacher OOF stacking | 0.6309 | 0.5688 |
| 5 | Richer features + Feature selection + multiseed | Full split-safe pipeline | 0.6304 | 0.6491 |

Best Final LB F1: **0.6491** (Model 5)
Unfortunately that was not the submission I kept for evaluation, the score that shows up on the leaderboard has an F1 score of **0.6413** which was my second best submission.

## Model-by-Model Summary

# Model 1: Baseline XGBoost (Lightcurve Statistics)

This notebook contains my first model for the competition.

The goal of this model was to build a relatively strong baseline:
- extract simple lightcurve features using AI
- train a strong XGBoost Classifier
- optimize hyperparameters and decision threshold for F1

### Results

Best threshold: 0.5147491638795987  
Best validation F1: 0.730769  

| Submission | Public LB F1 | Private LB F1 |
|-----------|--------------:|--------------:|
| 1 | 0.4582 | 0.4153 |
| 2 | 0.4610 | 0.4540 |

### Takeaways

What worked:
- Lightcurve summary features are enough to create a functional baseline
- XGBoost + Optuna finds strong parameter combinations quickly

Problems:
- Leaderboard F1 was far lower than validation. Validation was too easy.
- Needed split-safe evaluation and more meaningful features.

# Model 2: Physics-aware Features + Split-safe CV XGBoost Ensemble

This notebook contains the second model for the competition.

This model is a structured upgrade over Model 1, focused on fixing two core issues:
- Astrophysical correctness (dust/extinction effects in flux measurements)
- Validation realism (data leakage and split structure)

Goals:
- correct observed flux for dust extinction (EBV)
- prevent leakage using split-aware validation
- stabilize predictions using fold ensembling
- tune hyperparameters using cross-validated Optuna

### Results

Best threshold: 0.4583544303797469  
OOF best F1: 0.5102040816326531  

| Submission | Public LB F1 | Private LB F1 |
|-----------|--------------:|--------------:|
| 1 | 0.5921 | 0.5295 |

### What changed vs Model 1
- De-extinction: correct flux + uncertainty using `EBV`
- Rest-frame timing: divide time features by `(1 + Z)`
- Expanded transient-shape features (AUC above baseline, widths, slopes, Von Neumann eta)
- Bazin fits for bands with enough data (rise/decay structure)

### Takeaways
- Additional features + de-extinction improved performance.
- Validation F1 dropped but leaderboard improved (better generalization).

# Model 3: XGBoost / LightGBM Blend (plus photo-z augmentation)

This notebook contains the third model for the competition.

The goal was to improve generalization by:
- simulating realistic redshift uncertainty using `Z_err` (photo-z augmentation)
- training a two-model ensemble (XGBoost + LightGBM)
- tuning both the blend weight and the classification threshold using OOF predictions

### Results

Best threshold: 0.01 (Really weird)  
OOF Best F1: 0.51875  
Best alpha: 0.03  

These results are very strange, the threshold is extremely low. More importantly, alpha = 0.03 means the blend was ~97% LightGBM and ~3% XGBoost. So this outcome suggests XGBoost contributed little or even hurt which is surprising because XGBoost was tuned using optuna and LightGBM was not.

I did not spend much time debugging or tuning this blend. It was an overnight experiment to see whether a small ensemble improved stability or F1. Given the low alpha and odd threshold, this setup needs more experimentation. In later experiments, LightGBM often ended up with near-zero weight, so I dropped it from subsequent models. In future competitions, I plan to give LightGBM CatBoost a better evaluation instead of discarding them early. I already used LightGBM successfully for the SpecType teacher, and it likely has more value in the main pipeline than I explored here.

| Submission | Public LB F1 | Private LB F1 |
|-----------|--------------:|--------------:|
| 1 | 0.5613 | 0.5313 |
| 2 | 0.5082 | 0.5119 |

### What changed vs Model 2

#### 1) Added `Z_err` and photo-z augmentation
- Add `Z_err` as a feature (plus log transforms)
- Training-time augmentation:
  - sample `sigma` from the test `Z_err` distribution
  - generate `z_sim = z + Normal(0, sigma)` (clipped to `>= 0`)
  - recompute features with `(z_sim, sigma)`
  - mark augmented rows with `photoz_aug = 1`

#### 2) Added a second learner (LightGBM)
- Train XGB per fold
- Train LGB per fold

#### 3) OOF probability blending
- `p_blend = alpha * p_xgb + (1 - alpha) * p_lgb`
- Tune `alpha` and `threshold` on OOF predictions

### Notes on why this underperformed
- Best OOF alpha was `0.03` (mostly LGB, not XGB)
- Best threshold was extremely low (`0.01`)
- Photo-z augmentation may have added noise that improved CV but not leaderboard transfer
- Undertrained compared to later models

# Model 4: XGB SpecType Teacher Stacking

This notebook contains the fourth model for the competition.

This model was the first one that performed very well on the public leaderboard, but did not perform well on the final leaderboard.
The biggest change is using `SpecType` (train-only metadata) to generate features that can also be computed for the test set.

Core idea:
- Train an LGBM teacher to predict grouped `SpecType`
- Use the teacher's predicted probabilities as features for the TDE classifier

### Teacher pipeline (leakage-safe)
1. Train multiclass LightGBM to predict `SpecTypeGroup`:
   - TDE
   - AGN
   - SNIa
   - SNother
   - Other
2. Generate OOF teacher probabilities for train
3. Fit on full train and predict probabilities for test
4. Append:
   - `p_spec_<class>` for each class
   - `spec_entropy` as a confidence / ambiguity signal

### Results

OOF best threshold: 0.46798994974874375  
OOF best F1: 0.5531914893617021  
OOF AP (Approx of PR AUC): 0.6134734232399863  

| Submission | Public LB F1 | Private LB F1 |
|-----------|--------------:|--------------:|
| 1 | 0.6309 | 0.5688 |
| 2 | 0.6024 | 0.5333 |
| 3 | 0.6009 | 0.5467 |

# Model 5: Teacher + Feature Selection + Optuna (OOF F1) + Multiseed XGBoost

This notebook contains Model 5 for the MALLORN challenge and is the highest performing model.

Model 5 builds on the teacher-stacking concept from Model 4, but pushes further in three directions:
1) Richer time-series and cross-band physics-inspired features  
2) Feature selection using XGBoost gain importance  
3) Direct optimization for OOF F1 using split-aware Optuna, followed by multiseed training/inference  

### Results

OOF multiseed best threshold: 0.419  
OOF multiseed best F1: 0.6243705941591138  
OOF AP (Approx of PR AUC): 0.5164263302782434  

| Submission | Public LB F1 | Private LB F1 |
|-----------|--------------:|--------------:|
| 1 | 0.6222 | 0.6231 |
| 2 | 0.6358 | 0.6413 |
| 3 | 0.5830 | 0.5962 |
| 4 | 0.6304 | 0.6491 |
| 5 | 0.5840 | 0.6100 |

### Training Setup

#### Split-aware CV
- `StratifiedGroupKFold`
- `groups = split`
- stratified by target to preserve class balance per fold

#### Class imbalance
- `scale_pos_weight = (#neg / #pos)` computed per fold

#### Photo-z augmentation
- sample `sigma` from distribution of test `Z_err`
- create training rows with `z_sim = z0 + Normal(0, sigma)`
- mark augmented rows with `photoz_aug = 1`

#### Missingness handling
- add explicit missing indicators:
  - for each feature column `f`, add `f_isnan` (0/1)

## Pipeline Summary (Model 5)

1. **Build feature table**
   - per-object: global + per-band + cross-band features
   - apply de-extinction correction using EBV
   - optionally add photo-z augmentation rows

2. **Add SpecType teacher features**
   - train LGBM multiclass on `SpecTypeGroup` (train only)
   - append OOF probabilities to train and full-fit probabilities to test
   - add `spec_entropy` and `spec_topprob`

3. **Feature selection**
   - train a baseline XGB across folds
   - rank features by aggregated `gain`
   - keep Top-K features (FS_TOPK)

4. **Optuna hyperparameter tuning (OOF F1)**
   - split-aware CV
   - build OOF probabilities
   - choose global best threshold on OOF
   - return OOF F1 as the trial objective

5. **Final multiseed model**
   - train multiple XGB seeds and average probabilities
   - pick final threshold from OOF
   - produce submission CSV

## Lessons Learned

- Validation matters a lot. Model 1 validation was misleading.
- Teacher stacking was the biggest boost and helped with TDE vs AGN/SN confusion.
- Blending can silently break calibration under imbalance (Model 3 threshold weirdness).
- Feature quality + selection + split-safe tuning beat model variety.

## List of features used

Each feature with a {b} creates 6 features for each band. {a} and {b} is for comparing 2 bands.

| Feature | Meaning | Why it helps |
|---------|----------|--------------|
| `flux_mean_raw` | Mean flux before dust de-extinction correction | Lets the model compare raw vs corrected brightness scale and learn dust-impact patterns |
| `flux_std_raw` | Standard deviation of raw flux | Captures variability before correction to detect dust-driven distortions |
| `snr_max_raw` | Maximum SNR using raw flux and raw error | Measures best raw detection strength as a correction sanity check |
| `fvar_raw` | Fractional variability using raw flux and error | Provides intrinsic-variability proxy before dust adjustment |
| `flux_mean_deext_minus_raw` | Difference between corrected and raw mean flux | Direct signal of how strongly extinction correction shifts brightness |
| `snrmax_deext_minus_raw` | Difference between corrected and raw max SNR | Measures how much detectability improves after correction |
| `n_seasons_global` | Number of observing seasons inferred from large time gaps | Separates single-season vs multi-season coverage patterns |
| `gap_frac_gt90` | Fraction of time gaps greater than 90 days | Flags strongly seasonal sampling |
| `gap_frac_gt30` | Fraction of time gaps greater than 30 days | Captures moderate sampling fragmentation |
| `n_seasons_{b}` | Number of observing seasons in band b | Band-specific sampling structure can differ by class |
| `season_maxspan_{b}` | Longest continuous season span in band b | Measures longest uninterrupted coverage window |
| `season_meanspan_{b}` | Mean season span in band b | Captures typical continuous coverage length |
| `sf_medabs_5_{b}` | Median absolute flux difference at ~5-day lag | Measures short-timescale variability strength |
| `sf_n_5_{b}` | Number of pairs used for 5-day lag SF | Reliability indicator for short-lag estimate |
| `sf_medabs_10_{b}` | Median absolute flux difference at ~10-day lag | Captures slightly longer-timescale changes |
| `sf_n_10_{b}` | Pair count for 10-day lag SF | Reliability indicator |
| `sf_medabs_20_{b}` | Median absolute flux difference at ~20-day lag | Mid-scale variability measure |
| `sf_n_20_{b}` | Pair count for 20-day lag SF | Reliability indicator |
| `sf_medabs_50_{b}` | Median absolute flux difference at ~50-day lag | Long-timescale variability proxy |
| `sf_n_50_{b}` | Pair count for 50-day lag SF | Reliability indicator |
| `sf_medabs_100_{b}` | Median absolute flux difference at ~100-day lag | Very long-timescale variability proxy |
| `sf_n_100_{b}` | Pair count for 100-day lag SF | Reliability indicator |
| `bazin_A_{b}` | Bazin model amplitude parameter | Smooth transient strength estimate |
| `bazin_t0_{b}_obs` | Bazin peak-time parameter (observed frame) | Parametric peak timing estimate |
| `bazin_trise_{b}_obs` | Bazin rise timescale (observed frame) | Encodes rise speed |
| `bazin_tfall_{b}_obs` | Bazin decay timescale (observed frame) | Encodes decay speed |
| `bazin_B_{b}` | Bazin baseline parameter | Estimates underlying baseline level |
| `bazin_chi2red_{b}_obs` | Reduced chi-square of Bazin fit | Fit quality indicator |
| `bazin_trise_{b}_rest` | Bazin rise timescale (rest frame) | Intrinsic rise speed |
| `bazin_tfall_{b}_rest` | Bazin fall timescale (rest frame) | Intrinsic decay speed |
| `t_rise50_{b}_obs` | Time from baseline to 50% amplitude (observed) | Measures rise speed |
| `t_rise20_{b}_obs` | Time from baseline to 20% amplitude (observed) | Early-rise behavior |
| `t_rise50_{b}_rest` | Rise time to 50% amplitude (rest frame) | Intrinsic rise speed |
| `t_rise20_{b}_rest` | Rise time to 20% amplitude (rest frame) | Intrinsic early-rise behavior |
| `asym50_{b}_obs` | Fall50 / Rise50 ratio (observed) | Captures peak asymmetry |
| `asym50_{b}_rest` | Fall50 / Rise50 ratio (rest frame) | Intrinsic asymmetry measure |
| `amppreratio_{a}{b}` | Ratio of pre-baseline amplitudes between bands | Color-dependent peak strength comparison |
| `aucratio_{a}{b}_obs` | Ratio of positive AUC between bands | Relative emitted-energy proxy |
| `width50ratio_{a}{b}_obs` | Ratio of 50% widths between bands | Cross-band duration contrast |
| `asym50ratio_{a}{b}_obs` | Ratio of asymmetry metrics between bands | Cross-band shape contrast |
| `corr_gr_obs` | Correlation between g and r band lightcurves | Measures multi-band coherence |
| `corr_ri_obs` | Correlation between r and i bands | Same, redder wavelengths |
| `corr_iz_obs` | Correlation between i and z bands | Same, further red |
| `tpeak_vs_lambda_slope_obs` | Slope of peak-time vs wavelength fit | Detects chromatic timing trends |
| `tpeak_vs_lambda_intercept_obs` | Intercept of that regression | Baseline timing offset |
| `tpeak_vs_lambda_r2_obs` | R² of peak-time vs wavelength fit | Reliability of chromatic timing trend |
| `peakflux_vs_lambda_slope` | Slope of peak-flux vs wavelength fit | Spectral energy trend |
| `peakflux_vs_lambda_intercept` | Intercept of flux–wavelength fit | Baseline spectral level |
| `peakflux_vs_lambda_r2` | R² of flux–wavelength fit | Reliability of spectral slope |
| `sed_logflux_loglambda_slope_rpeak` | Slope of log(flux) vs log(wavelength) at r-peak | Spectral slope at peak |
| `sed_logflux_loglambda_r2_rpeak` | R² of SED fit at r-peak | Fit reliability |
| `sed_logflux_loglambda_nbands_rpeak` | Number of bands used in SED fit | Coverage reliability |
| `sed_slope_rpeak_p20` | SED slope at r-peak + 20 days | Spectral evolution rate |
| `sed_r2_rpeak_p20` | R² of SED fit at +20 days | Reliability indicator |
| `sed_nbands_rpeak_p20` | Bands used at +20 days | Coverage indicator |
| `spec_topprob` | Maximum teacher-model class probability | Teacher confidence summary for meta-learning |
| `n_obs` | Total number of observations across all filters | Coverage proxy; some classes are observed more densely |
| `total_time_obs` | Total observed duration (max time − min time) | Separates long-timescale variability from short transients |
| `total_time_rest` | Duration corrected by (1+z) time dilation | Makes durations comparable across redshift |
| `flux_mean` | Mean dust-corrected flux | Overall brightness level |
| `flux_median` | Median dust-corrected flux | Robust brightness estimate |
| `flux_std` | Standard deviation of corrected flux | Overall variability strength |
| `flux_min` | Minimum corrected flux | Captures deep dips / noise floor |
| `flux_max` | Maximum corrected flux | Captures peak brightness |
| `flux_mad` | Median absolute deviation | Robust variability measure |
| `flux_iqr` | Interquartile range | Robust spread measure |
| `flux_skew` | Skewness of flux distribution | Detects asymmetric burst-like shapes |
| `flux_kurt_excess` | Excess kurtosis | Detects heavy-tailed spike behavior |
| `flux_p5` | 5th percentile flux | Robust low level |
| `flux_p25` | 25th percentile flux | Lower quartile |
| `flux_p75` | 75th percentile flux | Upper quartile |
| `flux_p95` | 95th percentile flux | Robust high level |
| `robust_amp_global` | p95 − p5 | Stable global amplitude proxy |
| `neg_flux_frac` | Fraction of flux values below zero | Noise-dominated vs real detection signal |
| `snr_median` | Median signal-to-noise ratio | Typical detection quality |
| `snr_max` | Maximum signal-to-noise ratio | Strongest detection strength |
| `median_dt` | Median time gap between observations | Sampling cadence proxy |
| `max_gap` | Largest time gap | Detects large seasonal breaks |
| `eta_von_neumann` | Von Neumann eta statistic | Smoothness vs randomness indicator |
| `chi2_const_global` | Chi-square vs constant model | Detects variability vs flat signal |
| `stetsonJ_global_obs` | Stetson J index (observed frame) | Robust correlated variability measure |
| `stetsonJ_global_rest` | Stetson J index (rest frame) | Intrinsic variability measure |
| `max_slope_global_obs` | Maximum absolute slope (observed) | Fastest brightness change |
| `max_slope_global_rest` | Maximum slope (rest frame) | Intrinsic fastest change |
| `med_abs_slope_global_obs` | Median absolute slope (observed) | Typical change rate |
| `med_abs_slope_global_rest` | Median absolute slope (rest) | Intrinsic change rate |
| `slope_global_obs` | Linear trend slope (observed) | Long-term drift indicator |
| `slope_global_rest` | Linear trend slope (rest) | Intrinsic drift |
| `fvar_global` | Fractional variability | Noise-corrected variability strength |
| `Z` | Redshift | Distance and time-dilation proxy |
| `log1pZ` | log(1+Z) | Stabilized redshift scale |
| `Z_err` | Redshift uncertainty | Reliability of distance estimate |
| `log1pZerr` | log(1+Z_err) | Stabilized uncertainty scale |
| `EBV` | Dust extinction value | Measures dust impact |
| `n_filters_present` | Number of filters with data | Multi-band coverage indicator |
| `total_obs` | Total observations across bands | Coverage strength |
| `n_{b}` | Number of observations in band b | Band completeness differs by class |
| `p5_{b}` | 5th percentile flux in band b | Robust low level |
| `p25_{b}` | 25th percentile | Lower quartile |
| `p75_{b}` | 75th percentile | Upper quartile |
| `p95_{b}` | 95th percentile | Robust high level |
| `robust_amp_{b}` | p95 − p5 in band b | Stable band amplitude |
| `mad_{b}` | Median absolute deviation | Robust variability |
| `iqr_{b}` | Interquartile range | Robust spread |
| `mad_over_std_{b}` | MAD / std ratio | Outlier sensitivity indicator |
| `eta_{b}` | Von Neumann eta | Smoothness vs noise |
| `chi2_const_{b}` | Chi-square vs constant | Variability detector |
| `stetsonJ_{b}_obs` | Stetson J (observed) | Correlated variability |
| `stetsonJ_{b}_rest` | Stetson J (rest) | Intrinsic correlated variability |
| `fvar_{b}` | Fractional variability | Normalized variability strength |
| `snrmax_{b}` | Maximum SNR | Best detection strength |
| `baseline_pre_{b}` | Estimated pre-peak baseline | Reference level for amplitude |
| `amp_{b}` | Peak − median flux | Simple amplitude |
| `amp_pre_{b}` | Peak − pre-peak baseline | Cleaner transient amplitude |
| `tpeak_{b}_obs` | Peak time (observed frame) | Band timing behavior |
| `tpeak_{b}_rest` | Peak time (rest frame) | Intrinsic timing |
| `peak_dominance_{b}` | Peak / baseline noise scale | Peak significance |
| `std_ratio_prepost_{b}` | Pre/post peak std ratio | Stability vs post-peak chaos |
| `width50_{b}_obs` | Width above 50% amplitude (obs) | Duration at mid level |
| `width80_{b}_obs` | Width above 80% amplitude (obs) | Peak sharpness |
| `width50_{b}_rest` | Width50 (rest) | Intrinsic duration |
| `width80_{b}_rest` | Width80 (rest) | Intrinsic peak shape |
| `t_fall50_{b}_obs` | Fall time to 50% (obs) | Decay speed |
| `t_fall20_{b}_obs` | Fall time to 20% (obs) | Late decay |
| `t_fall50_{b}_rest` | Fall50 (rest) | Intrinsic decay |
| `t_fall20_{b}_rest` | Fall20 (rest) | Intrinsic late decay |
| `sharp50_{b}_obs` | Amplitude / width50 (obs) | Spike sharpness |
| `sharp50_{b}_rest` | Amplitude / width50 (rest) | Intrinsic sharpness |
| `postpeak_monotone_frac_{b}` | Fraction monotonic after peak | Smooth decay vs noisy |
| `n_peaks_{b}` | Significant peak count | Multi-peak vs single transient |
| `n_rebrighten_{b}` | Rebrightening count | Secondary bump behavior |
| `decay_pl_slope_{b}_obs` | Power-law decay slope (obs) | Decay steepness |
| `decay_pl_r2_{b}_obs` | Fit R² (obs) | Fit reliability |
| `decay_pl_npts_{b}_obs` | Points used (obs) | Support size |
| `decay_pl_slope_{b}_rest` | Power-law slope (rest) | Intrinsic decay |
| `decay_pl_r2_{b}_rest` | Fit R² (rest) | Reliability |
| `decay_pl_npts_{b}_rest` | Points used (rest) | Support size |
| `tpeak_std_obs` | Std of peak times across bands (obs) | Peak alignment indicator |
| `tpeak_std_rest` | Std of peak times (rest) | Intrinsic alignment |
| `tpeakdiff_{a}{b}_obs` | Peak time difference (obs) | Chromatic lag signal |
| `tpeakdiff_{a}{b}_rest` | Peak time difference (rest) | Intrinsic lag |
| `peakratio_{a}{b}` | Peak flux ratio | Peak color proxy |
| `color_gr_at_rpeak_obs` | g−r color at r-peak | Spectral color at peak |
| `color_ri_at_rpeak_obs` | r−i color at r-peak | Red color proxy |
| `color_gr_rpeak_p20_obs` | g−r at +20d | Color evolution |
| `color_ri_rpeak_p20_obs` | r−i at +20d | Color evolution |
| `color_gr_rpeak_p40_obs` | g−r at +40d | Slower evolution |
| `color_ri_rpeak_p40_obs` | r−i at +40d | Slower evolution |
| `color_gr_slope20_obs` | g−r slope over 20d | Early color change rate |
| `color_ri_slope20_obs` | r−i slope over 20d | Early color change |
| `color_gr_slope40_obs` | g−r slope over 40d | Longer color trend |
| `color_ri_slope40_obs` | r−i slope over 40d | Longer trend |
| `p_spec_{c}` | Teacher probability for class c | Soft-label prior signal |
| `spec_entropy` | Entropy of teacher probs | Teacher uncertainty |
| `spec_topprob` | Max teacher probability | Teacher confidence summary |