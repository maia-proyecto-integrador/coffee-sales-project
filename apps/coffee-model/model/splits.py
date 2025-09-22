from typing import List, Tuple
import pandas as pd

def rolling_origins(date_index: pd.Series, n_origins: int = 4, horizon: int = 7):
    unique_dates = pd.Series(pd.to_datetime(pd.unique(date_index))).sort_values()
    anchors = [unique_dates.iloc[-(i+1)*horizon] for i in range(n_origins)][::-1]
    splits: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for anchor in anchors:
        train_end = anchor - pd.Timedelta(days=1)
        test_start = anchor
        test_end = anchor + pd.Timedelta(days=horizon - 1)
        splits.append((train_end, test_start, test_end))
    return splits
