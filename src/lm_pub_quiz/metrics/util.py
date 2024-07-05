import numpy as np
import pandas as pd


def accumulate_metrics(group: pd.DataFrame):
    return pd.Series(
        {
            k: (np.average(v, weights=group["num_instances"]) if k != "num_instances" else v.sum())
            for k, v in group.items()
        }
    )
