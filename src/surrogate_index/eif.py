from __future__ import annotations
from typing import Sequence
import pandas as pd


class SurrogateIndexEIF:
    """
    Placeholder estimator

    Parameters
    ----------
    y, w
        Column names: long-term outcome and treatment indicator
    s_cols, x_cols
        List of short-term outcomes and pre-treatment covariates
    """

    def __init__(
        self,
        *,
        y: str,
        w: str,
        s_cols: Sequence[str],
        x_cols: Sequence[str],
    ) -> None:
        self.y = y
        self.w = w
        self.s_cols = list(s_cols)
        self.x_cols = list(x_cols)
        # internal slots we'll fill later
        self._fitted = False
        self._ate_: float | None = None

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #
    def fit(self, df_obs: pd.DataFrame, df_exp: pd.DataFrame) -> "SurrogateIndexEIF":
        """
        Placeholder - right now just remembers row counts.
        """
        self.n_obs_ = len(df_obs)
        self.n_exp_ = len(df_exp)
        self._fitted = True
        return self

    def ate_(self) -> float:
        """
        Return the ATE estimate (raises if fit not called).
        """
        if not self._fitted or self._ate_ is None:
            raise RuntimeError("Call `fit` first.")
        return self._ate_
