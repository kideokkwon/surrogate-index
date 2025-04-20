# src/surrogate_index/preprocess.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def encode_categoricals(
        train_df: pd.DataFrame,
        apply_df: pd.DataFrame,
        categorical_cols: List[str] | None = None,
        *,
        drop_original: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    One-hot encode `categorical_cols` fitted on *train_df*, applied to *apply_df*.
    """
    if categorical_cols is None:
        categorical_cols = train_df.select_dtypes(["object", "category"]).columns.tolist()

    enc = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)
    ct = ColumnTransformer(
       transformers=[("ohe", enc, categorical_cols)],
       remainder="passthrough",
       verbose_feature_names_out=False, 
    )

    train_arr = ct.fit_transform(train_df)
    apply_arr = ct.transform(apply_df)

    cols = ct.get_feature_names_out()
    train_out = pd.DataFrame(train_arr, columns=cols, index=train_df.index)
    apply_out = pd.DataFrame(apply_arr, columns=cols, index=apply_df.index)

    return train_out, apply_out, enc