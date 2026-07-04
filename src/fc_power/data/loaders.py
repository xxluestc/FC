from pathlib import Path
import pandas as pd


def load_canonical_csv(path):
    return pd.read_csv(Path(path), parse_dates=["timestamp"])
