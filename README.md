# surrogate-index

<!-- Commented out for now since not on PyPI yet
[![PyPI - Version](https://img.shields.io/pypi/v/surrogate-index.svg)](https://pypi.org/project/surrogate-index)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/surrogate-index.svg)](https://pypi.org/project/surrogate-index)
-->

An efficient influence function (EIF) implementation of the Surrogate Index estimator  
from Chen & Ritzwoller (2023), used for estimating long-term causal effects using short-term surrogates.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## Installation

**This package is not on PyPI yet.** For now, clone the repo and install locally:

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