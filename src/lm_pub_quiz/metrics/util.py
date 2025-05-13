import numpy as np
import pandas as pd


def _accumulate(values: pd.Series, support: pd.Series):
    if isinstance(values.iloc[0], list):
        values = np.array(values.tolist())
        return list(np.average(np.array(values), weights=support, axis=0))
    else:
        return np.average(values, weights=support)



def accumulate_metrics(group: pd.DataFrame):
    return pd.Series(
        {
            k: (
                v.sum() if k == "support" else _accumulate(v, group["support"])
            )
            for k, v in group.items()
        }
    )
