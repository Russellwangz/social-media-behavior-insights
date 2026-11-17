# data.py
# This file contains functions for loading and basic cleaning of the data.

from pathlib import Path
import pandas as pd

def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from column names
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Standardize string/categorical cells: strip whitespace
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # Drop duplicate rows if any
    df = df.drop_duplicates()

    return df