# src/surrogate_index/eif.py
"""
Efficient-Influence-Function (EIF) estimator for the surrogate-index ATE
in Chen & Ritzwoller (2023).

Target estimand (experimental units only):
    τ₀ = E[ Y(1) - Y(0) | G = 0 ]

Public API
----------
efficient_influence_function  - computes all nuisance functions + EIF vector
output_inference              - convenience wrapper for point estimate / CI
"""

from __future__ import annotations

from typing import List
import logging


import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from .learners import (
    fit_nuisance_function_primary,
    fit_nuisance_function_secondary,
)
from .preprocess import combine_dfs
from ._logging import enable_verbose_logging


# ---------------------------------------------------------------------
# Config: logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # library-style default


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def efficient_influence_function(
    df_exp: pd.DataFrame,
    df_obs: pd.DataFrame,
    y: str,
    w: str,
    s_cols: List[str],
    x_cols: List[str],
    classifier: BaseEstimator,  # should implement predict_proba
    regressor: BaseEstimator,
    unconfounded: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Computes all nuisance functions + EIF and returns a combined DataFrame.

    Parameters
    ----------
    df_exp : pd.DataFrame
        Experimental data (must contain treatment `w`)
    df_obs : pd.DataFrame
        Observational data (must contain outcome `y`)
    y, w   : str
        Column names for outcome (obs) and treatment (exp) respectively
    s_cols : list[str]
        Surrogate metric names
    x_cols : list[str]
        Pre-treatment covariate names
    classifier : BaseEstimator
        Any sklearn-style *classifier* with .predict_proba
    regressor  : BaseEstimator
        Any sklearn-style *regressor* with .predict
    unconfounded : bool, default True
        If True (e.g., A/B test) some nuisances collapse to empirical
        proportions; otherwise they are ML-estimated.

    Returns
    -------
    pd.DataFrame
        Original rows plus nuisance columns and `"eif"`.
    """

    # ---------------- Input validation --------------------------------
    if not isinstance(classifier, ClassifierMixin):
        raise TypeError("`classifier` must be an sklearn ClassifierMixin.")
    if not isinstance(regressor, RegressorMixin):
        raise TypeError("`regressor` must be an sklearn RegressorMixin.")

    # Ensure treatment is binary in experimental data
    unique_w = df_exp[w].dropna().unique()
    if set(unique_w) - {0, 1}:
        raise ValueError(f"Column '{w}' must be binary (0/1) in df_exp.")

    # --- turn package logging on if user asked for it ---
    if verbose:
        enable_verbose_logging()

    # ------------------------------------------------------------------
    # Combine experimental & observational data
    # ------------------------------------------------------------------
    df = combine_dfs(
        exp=df_exp,
        obs=df_obs,
        s_cols=s_cols,
        x_cols=x_cols,
        y_col=y,
        w_col=w,
    )

    # ------------------------------------------------------------------
    # Guard against duplicate columns
    # ------------------------------------------------------------------
    NUISANCE_COLS = {
        "nu_sx",
        "varrho_sx",
        "varrho_x",
        "gamma_sx",
        "pi",
        "bar_nu_1",
        "bar_nu_0",
        "eif",
    }
    dupes = NUISANCE_COLS.intersection(df.columns)
    if dupes:
        raise ValueError(
            f"DataFrame already contains nuisance column(s) {sorted(dupes)}."
        )

    # ------------------------------------------------------------------
    # 1. Estimate propensity-like nuisances
    # ------------------------------------------------------------------
    if unconfounded:
        n_exp = len(df[df["G"] == 0])
        n_obs = len(df[df["G"] == 1])
        treat_rate = len(df[(df[w] == 1) & (df["G"] == 0)]) / n_exp

        df["varrho_sx"] = treat_rate
        df["varrho_x"] = treat_rate
        df["gamma_sx"] = n_obs / len(df)  # cancels in EIF anyway
        pi_scalar = n_obs / len(df)
        df["pi"] = pi_scalar
    else:
        df["varrho_sx"] = fit_nuisance_function_primary(
            df,
            train_over="exp",
            dep_var=w,
            ind_vars=x_cols + s_cols,
            model_template=classifier,
            is_classifier=True,
            verbose=verbose,
        )
        df["varrho_x"] = fit_nuisance_function_primary(
            df,
            train_over="exp",
            dep_var=w,
            ind_vars=x_cols,
            model_template=classifier,
            is_classifier=True,
            verbose=verbose,
        )
        df["gamma_sx"] = fit_nuisance_function_primary(
            df,
            train_over="all",
            dep_var="G",
            ind_vars=x_cols + s_cols,
            model_template=classifier,
            is_classifier=True,
            verbose=verbose,
        )
        df["pi"] = len(df[df.G == 1]) / len(df)

    # ------------------------------------------------------------------
    # 2. Outcome & conditional means
    # ------------------------------------------------------------------
    df["nu_sx"] = fit_nuisance_function_primary(
        df,
        train_over="obs",
        dep_var=y,
        ind_vars=x_cols + s_cols,
        model_template=regressor,
        is_classifier=False,
        verbose=verbose,
    )
    df["bar_nu_1"] = fit_nuisance_function_secondary(
        df,
        w_col=w,
        w_value=1,
        ind_vars=x_cols,
        model_template=regressor,
        verbose=verbose,
    )
    df["bar_nu_0"] = fit_nuisance_function_secondary(
        df,
        w_col=w,
        w_value=0,
        ind_vars=x_cols,
        model_template=regressor,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # 3. Replace any remaining NaNs with 0 (safe: weights will zero them)
    # ------------------------------------------------------------------
    df[["varrho_sx", "varrho_x", "gamma_sx", "pi", "nu_sx", "bar_nu_1", "bar_nu_0"]] = (
        df[
            ["varrho_sx", "varrho_x", "gamma_sx", "pi", "nu_sx", "bar_nu_1", "bar_nu_0"]
        ].fillna(0.0)
    )

    df[[w, y]] = df[[w, y]].fillna(0.0)

    # ------------------------------------------------------------------
    # 4. Compute EIF
    # ------------------------------------------------------------------
    gamma_ratio = (1.0 - df["gamma_sx"]) / df["gamma_sx"]
    varrho_diff = df["varrho_sx"] - df["varrho_x"]
    outcome_dev = df[y] - df["nu_sx"]
    varrho_scale = df["varrho_x"] * (1.0 - df["varrho_x"])

    obs_part = gamma_ratio * varrho_diff * outcome_dev / varrho_scale

    treat_dev = df["nu_sx"] - df["bar_nu_1"]
    ctrl_dev = df["nu_sx"] - df["bar_nu_0"]

    exp_part = (
        df[w] * treat_dev / df["varrho_x"]
        - (1.0 - df[w]) * ctrl_dev / (1.0 - df["varrho_x"])
        + (df["bar_nu_1"] - df["bar_nu_0"])
    )

    weight_obs = df["G"] / (1.0 - df["pi"])
    weight_exp = (1.0 - df["G"]) / (1.0 - df["pi"])

    df["eif"] = weight_obs * obs_part + weight_exp * exp_part

    logger.info("EIF computation completed.")
    return df


def output_inference(
    eif: NDArray[np.floating],
    alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
    """
    Given an EIF vector, compute the point estimate, standard error,
    and confidence interval (default 95%).

    Parameters
    ----------
    eif : NDArray[np.floating]
        Efficient Influence Function vector (numeric array).
    alpha : float, default 0.05
        Significance level for confidence interval.

    Returns
    -------
    Tuple of (point_estimate, std_error, ci_lower, ci_upper)
    """
    est = eif.mean()
    se = np.sqrt(eif.var(ddof=1) / len(eif))
    z = abs(np.percentile(np.random.randn(10_000), (1 - alpha / 2) * 100))  # ≈ 1.96
    ci_lower = est - z * se
    ci_upper = est + z * se

    print("Surrogate-Index EIF results")
    print(f"Point estimate : {est:.5f}")
    print(f"Std. error     : {se:.5f}")
    print(f"{100 * (1 - alpha):.0f}% CI      : [{ci_lower:.5f}, {ci_upper:.5f}]")

    return est, se, ci_lower, ci_upper
