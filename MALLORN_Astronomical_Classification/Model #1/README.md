# Model 1: Baseline XGBoost (Lightcurve Statistics)

This notebook contains my first model for the competition.

The goal of this model was to build a relatively strong baseline:
- extract simple lightcurve features using AI
- train a strong XGBoost Classifier
- optimize hyperparameters and decision threshold for F1


## Results

Best parameters:
- n_estimators: 1556
- learning_rate: 0.011529
- max_depth: 5
- min_child_weight: 11
- subsample: 0.990469
- colsample_bytree: 0.964860
- colsample_bylevel: 0.931161
- gamma: 0.008020
- reg_alpha: 7.434848
- reg_lambda: 1.937161

Best threshold: 0.5147491638795987  
Best validation F1: 0.730769  

| Submission | Public LB F1 | Private LB F1 |
|-------------|--------------|----------------|
| 1 | 0.4582 | 0.4153 |
| 2 | 0.4610 | 0.4540 |



## Features
Each object has a time series (lightcurve) with observations in up to 6 filters: u, g, r, i, z, y

The majority of these features were created by tasking AI to go through astronomy research papers and create features. I am not an astronomer therefore I cannot create astronomy features with my knowledge.

- `Time (MJD)`: observation time in Modified Julian Date  
- `Flux`: measured brightness (can be negative due to noise/subtraction artifacts)  
- `Flux_err`: uncertainty in the flux measurement  
- `Filter`: which band the observation belongs to  

The goal of feature engineering here is to compress each irregular time series into a fixed-length numeric vector so a tabular model (like XGBoost) can learn patterns that separate classes.

### 1) Global features (all filters combined)
These are computed using all observations across all bands for a given object.  
They summarize the overall time coverage, brightness distribution, variability, uncertainty, and signal quality.

### 2) Per-filter features (computed separately per band)
These are computed **independently for each filter band**.  
They let the model detect color-dependent behavior (for example: strong variability in `g` but not in `i`).

## Global (all-filters combined) features

Below are the global feature columns and what each one represents:

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `total_time` | Total time span covered by the object's observations: `(max(Time) - min(Time))` after shifting to start at zero | Separates fast events vs slow events and distinguishes sparse vs long-baseline coverage |
| `n_obs` | Total number of observations across all filters | Captures sampling density and whether an object is well-measured |
| `median_flux` | Median flux across all observations | Robust estimate of typical brightness (less sensitive to spikes) |
| `mean_flux` | Mean flux across all observations | Captures average brightness but is more sensitive to outliers |
| `std_flux` | Standard deviation of flux across all observations | Measures overall variability (high = more change over time) |
| `min_flux` | Minimum observed flux | Captures dips, fading, or negative excursions from noise |
| `max_flux` | Maximum observed flux | Captures peak brightness or flare intensity |
| `range_flux` | Flux range: `max_flux - min_flux` | Simple variability amplitude proxy |
| `median_err` | Median flux uncertainty across all observations | Measures how noisy the measurements are overall |
| `median_snr` | Median signal-to-noise ratio: `median(\|Flux\| / Flux_err)` | Typical detection strength across observations |
| `max_snr` | Maximum signal-to-noise ratio: `max(\|Flux\| / Flux_err)` | Whether the object ever has a highly confident detection |
| `neg_flux_frac` | Fraction of observations where `Flux < 0` | Indicates low-SNR objects or subtraction-dominated measurements |

## Per-filter (band-wise) features

For each band in `filters = ["u", "g", "r", "i", "z", "y"]`, the following features are created:

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `n_obs_{band}` | Number of observations in this band | Some classes are observed more in specific bands; also captures missingness patterns |
| `total_time_{band}` | Time span covered within this band | Band-dependent cadence coverage (important if data is uneven across filters) |
| `median_flux_{band}` | Median flux in this band | Typical brightness in this band (captures spectral/color behavior) |
| `std_flux_{band}` | Flux standard deviation in this band | Variability strength in that band |
| `amp_{band}` | Simple amplitude proxy: `max(flux) - median(flux)` | Captures flare-like peaks or transient bursts without being too sensitive to one negative outlier |
| `median_err_{band}` | Median uncertainty in this band | Band-specific noise level (some filters are noisier) |
| `median_snr_{band}` | Median SNR in this band: `median(\|flux\| / err)` | Typical detection quality per filter |
| `max_snr_{band}` | Max SNR in this band: `max(\|flux\| / err)` | Best detection strength per filter |
| `neg_flux_frac_{band}` | Fraction of band observations with `flux < 0` | Band-specific low-SNR indicator |

## Additional summary features

These features describe how much band coverage exists overall:

| Feature | Meaning | Why it helps |
|--------|---------|--------------|
| `n_filters_present` | Count of how many filters have at least 1 observation | Objects with multi-band coverage have richer information; missing bands may correlate with class |
| `total_obs` | Total observations summed over all filters (same as `n_obs`) | Redundant but convenient for downstream logic or sanity checks |


## Training Setup

### Train / Validation Split
- 80/20 split using `train_test_split`
- stratified by target to preserve class balance

The dataset is imbalanced, so I use: scale_pos_weight to weight positive examples more strongly during training.

## Takeaways

What worked:
- Lightcurve summary features are enough to create a functional baseline
- XGBoost + Optuna finds strong parameter combinations quickly will stick to XGBoost and maybe an LGBM

Problems:
- Leaderboard F1 score was far lower than my validation score. I need to make the validation harder.
- Add more features
