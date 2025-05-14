# src/surrogate_index/preprocess.py
from __future__ import annotations

from typing import List, Tuple, Optional
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------
# Combine experimental + observational data
# ---------------------------------------------------------------------
def combine_dfs(
    exp: pd.DataFrame,
    obs: pd.DataFrame,
    s_cols: List[str],
    x_cols: List[str],
    y_col: str,
    w_col: str,
) -> pd.DataFrame:
    """
    Merge *experimental* (G=0) and *observational* (G=1) datasets.

    Any column present in only one dataset is filled with NaN
    so that the final DataFrame has a consistent schema.

    Returns
    -------
    pd.DataFrame  (columns: [w_col, y_col, *s_cols, *x_cols, "G"])
    """

    # 0‒ Guard: do not overwrite an existing 'G'
    if "G" in exp.columns or "G" in obs.columns:
        raise ValueError(
            "Either input DataFrame already contains a column named 'G'. "
            "Rename it before calling combine_dfs()."
        )

    # 1‒ Validate required columns
    required_exp = {w_col, *s_cols, *x_cols}
    required_obs = {y_col, *s_cols, *x_cols}

    missing_exp = required_exp - set(exp.columns)
    missing_obs = required_obs - set(obs.columns)
    if missing_exp:
        raise KeyError(f"Experimental DF missing columns: {missing_exp}")
    if missing_obs:
        raise KeyError(f"Observational DF missing columns: {missing_obs}")

    # Optional: check treatment column binary in experimental data
    bad_vals = set(exp[w_col].dropna().unique()) - {0, 1}
    if bad_vals:
        raise ValueError(
            f"{w_col} must be binary (0/1) in experimental data; "
            f"found values {sorted(bad_vals)}."
        )

    # 2‒ Align schemas
    df_exp = exp.loc[:, [w_col, *s_cols, *x_cols]].copy()
    df_obs = obs.loc[:, [y_col, *s_cols, *x_cols]].copy()

    df_exp[y_col] = np.nan  # outcome not observed in exp
    df_obs[w_col] = np.nan  # treatment undefined in obs

    # 3‒ Add origin flag
    df_exp["G"] = 0
    df_obs["G"] = 1

    ordered_cols = [w_col, y_col, *s_cols, *x_cols, "G"]
    combined = pd.concat([df_exp, df_obs], ignore_index=True)
    return combined[ordered_cols]


# ---------------------------------------------------------------------
# One-hot-encode categoricals fold-by-fold
# ---------------------------------------------------------------------
def encode_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[OneHotEncoder]]:
    """
    One-hot-encode object / category columns.  For fold-specific encoding:
    fit on *train_df*, transform *both* train and test frames.

    Parameters
    ----------
    categorical_cols : list[str] | None
        If None, auto-detect ``object`` and ``category`` dtypes.

    Returns
    -------
    train_final : pd.DataFrame
    test_final  : pd.DataFrame
    encoder     : OneHotEncoder | None
        None if there were no categorical columns.
    """
    if categorical_cols is None:
        categorical_cols = train_df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    if not categorical_cols:
        return train_df.copy(), test_df.copy(), None

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    ).fit(train_df[categorical_cols])

    def _ohe(df: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.DataFrame(
            encoder.transform(df[categorical_cols]),
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index,
        )
        return pd.concat([df.drop(columns=categorical_cols), encoded], axis=1)

    train_final = _ohe(train_df)
    test_final = _ohe(test_df)

    logger.debug(
        "Encoded categorical cols %s into %d features",
        categorical_cols,
        train_final.shape[1] - (train_df.shape[1] - len(categorical_cols)),
    )

    return train_final, test_final, encoder
