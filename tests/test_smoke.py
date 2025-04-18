# tests/test_smoke.py
import pandas as pd
from surrogate_index import SurrogateIndexEIF


def test_smoke():
    df_obs = pd.DataFrame({"y": [1, 2], "w": [0, 1]})
    df_exp = pd.DataFrame({"y": [3], "w": [1]})
    est = SurrogateIndexEIF(y="y", w="w", s_cols=[], x_cols=[])
    est.fit(df_obs, df_exp)
    assert est.n_obs_ == 2 and est.n_exp_ == 1

