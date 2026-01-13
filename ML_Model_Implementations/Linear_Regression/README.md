# Linear Regression From Scratch (NumPy)

This is for remembering the mechanics of linear regression and minibatch gradient descent, not on building a production model.

## Overview

- Dataset: California Housing
- Task: Regression
- Goal: Implement linear regression and ridge from scratch in NumPy and SKLearn.


## Results (Test Set)

### OLS
NumPy implementation
- MAE: ~0.528
- RMSE: ~0.724
- R^2: ~0.606

sklearn LinearRegression
- MAE: ~0.529
- RMSE: ~0.723
- R^2: ~0.608

### Ridge
NumPy implementation
- MAE: ~0.528
- RMSE: ~0.724
- R^2: ~0.607

sklearn Ridge
- MAE: ~0.529
- RMSE: ~0.723
- R^2: ~0.608

The close match confirms the from scratch implementation is correct.
