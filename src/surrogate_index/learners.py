# src/surrogate_index/learners.py
from __future__ import annotations

from typing import List, Literal, Optional
import logging

import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    ClassifierMixin,
    clone,
)
from sklearn.model_selection import KFold

from .preprocess import encode_categoricals

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # silence unless configured


# ----------------------------------------------------------------------
# Primary nuisance
# ----------------------------------------------------------------------
def fit_nuisance_function_primary(
    df: pd.DataFrame,
    train_over: Literal["exp", "obs", "all"],
    dep_var: str,
    ind_vars: List[str],
    model_template: BaseEstimator,
    is_classifier: bool = False,
    n_splits: int = 5,
    verbose: bool = False,
    tag: Optional[str] = None,
) -> pd.Series[float]:
    """
    Cross-fitted nuisance estimate.

    Parameters
    ----------
    df : pd.DataFrame
        Combined experimental + observational data.
    train_over : {'exp', 'obs', 'all'}
        Which subset to train on.
    dep_var : str
        Dependent variable name.
    ind_vars : list[str]
        Predictor columns.
    model_template : BaseEstimator
        Sklearn model to clone per fold.
    is_classifier : bool, default False
        If True, uses ``predict_proba(..., 1)``.
    n_splits : int, default 5
        Cross-fitting folds.
    verbose : bool, default False
        Emit an ``INFO`` log when finished.
    tag : str | None
        Optional label to include in the log message.

    Returns
    -------
    pd.Series
        Float64 predictions, index-aligned to **df** (NaN where not trained).
    """
    # ------- sanity: estimator type ------------------------------------
    if is_classifier and not isinstance(model_template, ClassifierMixin):
        raise TypeError("model_template must be a classifier when is_classifier=True")
    if not is_classifier and not isinstance(model_template, RegressorMixin):
        raise TypeError("model_template must be a regressor when is_classifier=False")

    # ------- subset selection -----------------------------------------
    if train_over == "obs":
        df_use = df[df["G"] == 1].copy()
    elif train_over == "exp":
        df_use = df[df["G"] == 0].copy()
    else:
        df_use = df.copy()

    df_use = df_use[[dep_var] + ind_vars]
    preds = pd.Series(index=df_use.index, dtype=np.float64)

    # ------- cross-fitting -------------------------------------------
    kf = KFold(n_splits=n_splits, shuffle=True)
    for train_idx, test_idx in kf.split(df_use):
        train_df = df_use.iloc[train_idx]
        test_df = df_use.iloc[test_idx]

        train_enc, test_enc, _ = encode_categoricals(train_df, test_df)
        X_train = train_enc.drop(columns=[dep_var])
        y_train = train_enc[dep_var]
        X_test = test_enc.drop(columns=[dep_var])

        model = clone(model_template).fit(X_train, y_train)
        preds.iloc[test_idx] = (
            model.predict_proba(X_test)[:, 1]
            if is_classifier
            else model.predict(X_test)
        )

    # ------- fill remainder (if any) ----------------------------------
    if len(df_use) != len(df):
        df_remain = df.loc[~df.index.isin(df_use.index), [dep_var] + ind_vars]
        df_rem_enc, df_use_enc, _ = encode_categoricals(df_remain, df_use)

        model = clone(model_template).fit(
            df_use_enc.drop(columns=[dep_var]), df_use_enc[dep_var]
        )
        remain_pred = (
            model.predict_proba(df_rem_enc.drop(columns=[dep_var]))[:, 1]
            if is_classifier
            else model.predict(df_rem_enc.drop(columns=[dep_var]))
        )

        preds = pd.concat(
            [preds, pd.Series(remain_pred, index=df_remain.index)]
        ).sort_index()

    # ------- logging --------------------------------------------------
    if verbose:
        label = tag or dep_var
        logger.info("Primary nuisance '%s' completed.", label)

    return preds


# ----------------------------------------------------------------------
# Secondary nuisance
# ----------------------------------------------------------------------
def fit_nuisance_function_secondary(
    df: pd.DataFrame,
    w_col: str,
    w_value: Literal[1, 0],
    ind_vars: List[str],
    model_template: BaseEstimator,
    n_splits: int = 5,
    verbose: bool = False,
    tag: Optional[str] = None,
) -> pd.Series[float]:
    """
    Fits E[nu(S,X) | W=w_value, X, G=0] and predicts on *all* experimental rows (G==0).
    """
    dep_var = "nu_sx"
    if dep_var not in df.columns:
        raise KeyError("nu_sx missing; run fit_nuisance_function_primary() first.")
    if not isinstance(model_template, RegressorMixin):
        raise TypeError("model_template must be a regressor.")

    mask_train = (df["G"] == 0) & (df[w_col] == w_value)  # training rows
    mask_pred = df["G"] == 0  # we need preds here

    df_train = df.loc[mask_train, [dep_var] + ind_vars]
    df_pred = df.loc[mask_pred, ind_vars]

    preds_full = pd.Series(index=df.index, dtype=np.float64)
    if df_train.empty:
        return preds_full

    # ---- cross-fitted predictions for the TRAINING rows (to curb over-fit) ---
    kf = KFold(n_splits=n_splits, shuffle=True)
    oof = pd.Series(index=df_train.index, dtype=np.float64)

    for tr_idx, te_idx in kf.split(df_train):
        tr, te = df_train.iloc[tr_idx], df_train.iloc[te_idx]
        tr_enc, te_enc, encoder = encode_categoricals(
            tr, te
        )  # you already have this helper
        model = clone(model_template).fit(
            tr_enc.drop(columns=[dep_var]), tr_enc[dep_var]
        )
        oof.iloc[te_idx] = model.predict(te_enc.drop(columns=[dep_var]))

    preds_full.loc[df_train.index] = oof

    # ---- fit once on ALL training data, predict on *every* experimental row ---
    train_enc, pred_enc, _ = encode_categoricals(df_train, df_pred)
    final_model = clone(model_template).fit(
        train_enc.drop(columns=[dep_var]), train_enc[dep_var]
    )
    preds_full.loc[mask_pred] = final_model.predict(pred_enc)

    if verbose:
        label = tag or f"{dep_var}_{w_value}"
        logger.info(
            "Secondary nuisance '%s' completed (train n=%d, pred n=%d).",
            label,
            mask_train.sum(),
            mask_pred.sum(),
        )

    return preds_full
