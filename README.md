# surrogate-index

<!-- Commented out for now since not on PyPI yet
[![PyPI - Version](https://img.shields.io/pypi/v/surrogate-index.svg)](https://pypi.org/project/surrogate-index)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/surrogate-index.svg)](https://pypi.org/project/surrogate-index)
-->

## Introduction

This package provides the first Python implementation of the **Surrogate Index Estimator** introduced by [Athey et al. (2016)](https://arxiv.org/pdf/1603.09326), a causal inference method for estimating long-term treatment effects using short-term randomized controlled trials (e.g., A/B tests).

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

Some industry examples of using this methodology (may differ in the estimation strategy) are:
- [Netflix](https://netflixtechblog.com/round-2-a-survey-of-causal-inference-applications-at-netflix-fd78328ee0bb)
- [Instacart](https://tech.instacart.com/instacarts-economics-team-using-surrogate-indices-to-estimate-long-run-heterogeneous-treatment-0bf7bc96c6e6)
---
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## Installation

**This package is not on PyPI yet. COMING SOON** For now, clone the repo and install locally:

```bash
git clone https://github.com/kideokkwon/surrogate-index.git
cd surrogate-index
pip install -e ".[dev,ml]"

## For Conda Users
conda install -c conda-forge xgboost scikit-learn pandas numpy
pip install -e ".[dev]"

## Usage
from surrogate_index import efficient_influence_function

df_exp = ...  # your experimental data
df_obs = ...  # your observational data

results_df = efficient_influence_function(
    df_exp=df_exp,
    df_obs=df_obs,
    y="six_month_revenue",
    w="treatment",
    s_cols=[...],  # surrogate metric names
    x_cols=[...],  # covariate names
    classifier=...,
    regressor=...,
)
```
