# surrogate-index

[![PyPI version](https://img.shields.io/pypi/v/surrogate-index.svg)](https://pypi.org/project/surrogate-index/)

## Introduction

This package provides an implementation of the **Surrogate Index Estimator** introduced by [Athey et al. (2016)](https://arxiv.org/pdf/1603.09326), a causal inference method for estimating long-term treatment effects using short-term randomized controlled trials (e.g., A/B tests).

The core idea is to **combine a randomized experimental dataset with an external observational dataset** to estimate the **Average Treatment Effect (ATE)** on a long-term outcome that is not directly observed in the experiment (e.g., annual revenue, long-term retention). This is particularly useful in settings where long-term metrics are delayed, costly, or infeasible to measure during the experiment window.

This package implements an estimator based on the **Efficient Influence Function (EIF)** derived by [Chen & Ritzwoller (2023)](https://arxiv.org/pdf/2107.14405), leveraging the **Double/Debiased Machine Learning (DML)** framework of [Chernozhukov et al. (2016)](https://arxiv.org/abs/1608.00060). EIF-based estimators enable valid inference while incorporating flexible machine learning models for nuisance components, such as short-term outcome regressions and propensity scores, without compromising asymptotic efficiency or introducing first-order bias.

## Brief Mathematical Background

Given the terms:
- $w\in\\{0,1\\}$: binary treatment indicator 
- $s$: a vector of an arbitrary number of short-term outcomes (typically used as the "metrics of interest" in an A/B Test)
- $x$: a vector of pre-treatment covariates.
- $y$: long-term outcome
- $g$: binary indicator for if the user is in the observational sample ($g=1$) or the experimental sample ($g=0$)

the corresponding influence function for the ATE $\tau_0$ is as follows: 

$$\xi_0(b,\tau_0,\varphi)=\frac{g}{1-\pi}\left[\frac{1-\gamma(s,x)}{\gamma(s,x)}\cdot\frac{(\varrho(s,x)-\varrho(x))(y-\nu(s,x))}{\varrho(x)(1-\varrho(x))}\right]+\frac{1-g}{1-\pi}\left[\frac{w(\nu(s,x)-\bar\nu_1(x))}{\varrho(x)}-\frac{(1-w)(\nu(s,x)-\bar\nu_0(x))}{1-\varrho(x)}+(\bar\nu_1(x)-\bar\nu_0(x))-\tau_0\right]$$

where:
- $\nu(s,x)=E[Y|S,X,G=1]$
- $\varrho(s,x)=P(W=1|S,X,G=0)$
- $\varrho(x)=P(W=1|X,G=0)$
- $\gamma(s,x)=P(G=1|S,X)$
- $\pi=P(G=1)$
- $\bar\nu_w(x)=E[\nu(S,X)|W=w, X,G=0]$
---
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Planned Features](#planned-features)
- [License](#license)

---

## Installation

```bash
# simplest
pip install surrogate-index

# with ML extras (e.g. XGBoost)
pip install "surrogate-index[ml]"

# Conda users
conda install -c conda-forge xgboost scikit-learn pandas numpy
pip install surrogate-index
```
## Usage

```python
from surrogate_index import efficient_influence_function

df_exp = ...  # experimental sample
df_obs = ...  # observational sample

results_df = efficient_influence_function(
    df_exp=df_exp,
    df_obs=df_obs,
    y="six_month_revenue",
    w="treatment",
    s_cols=[...],   # list of surrogate metrics
    x_cols=[...],   # list of covariate names
    classifier=..., # e.g., GradientBoostingClassifier()
    regressor=...,  # e.g., XGBRegressor()
)
print(results_df)
```

## Planned Features
- Convert structure to an Object-based one (scikit-learn style)
- Add diagnostic checks
- Add alternative estimators provided in Athey et al. 2016
- etc.

## License

Distributed under the MIT License. See `LICENSE` for details.