import numpy as np
import pandas as pd


def accumulate_metrics(group: pd.DataFrame):
    return pd.Series(
        {k: (np.average(v, weights=group["support"]) if k != "support" else v.sum()) for k, v in group.items()}
    )
