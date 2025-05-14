# surrogate-index

<!-- Commented out for now since not on PyPI yet
[![PyPI - Version](https://img.shields.io/pypi/v/surrogate-index.svg)](https://pypi.org/project/surrogate-index)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/surrogate-index.svg)](https://pypi.org/project/surrogate-index)
-->

This module is a Python implementation of [Athey et al (2016)](https://arxiv.org/pdf/1603.09326)'s Surrogate Index using the Efficient Influence Function derived by [Chen & Ritzwoller (2023)](https://arxiv.org/pdf/2107.14405), as seen below:

$$\xi_0(b,\tau_0,\varphi)=\frac{g}{1-\pi}\left[\frac{1-\gamma(s,x)}{\gamma(s,x)}\cdot\frac{(\varrho(s,x)-\varrho(x))(y-\nu(s,x))}{\varrho(x)(1-\varrho(x))}\right]+\frac{1-g}{1-\pi}\left[\frac{w(\nu(s,x)-\bar\nu_1(x))}{\varrho(x)}-\frac{(1-w)(\nu(s,x)-\bar\nu_0(x))}{1-\varrho(x)}+(\bar\nu_1(x)-\bar\nu_0(x))-\tau_0\right]$$

The Surrogate Index is a method for inferring Long-Term Outcomes using Short-term Experiments. Most AB Test are "Short-term Experiments" in the sense that while there exists a true Long-term Outcome $Y$ of interest (such as Revenue), we rely on *Short-term Outcomes* $S$ (called "Surrogates Metrics" in Athey et al) as our Outcome for our Average Treatment Effect. More specifically,

$$\text{ATE}_{S}=E[S(1)-S(0)]$$

This is for several reasons, one being that Long-term Outcomes are, well, unobserved in the Short-term. We do not want to run each AB Test for a long period of time as it would cripple product / business velocity, but in certain cases, we are still interested in the ATE of a Long-term Outcome. In other words,

$$\text{ATE}_{L}=E[Y(1)-Y(0)]$$

There seem to be several companies that have utilized this methodology for various applications. Notably, [Netflix](https://netflixtechblog.com/round-2-a-survey-of-causal-inference-applications-at-netflix-fd78328ee0bb) and [Instacart](https://tech.instacart.com/instacarts-economics-team-using-surrogate-indices-to-estimate-long-run-heterogeneous-treatment-0bf7bc96c6e6), but this is the first Python implementation available publicly.

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
