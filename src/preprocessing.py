"""
preprocessing.py
Raw CSV -> clean, ML-ready dataframe:
  1. load + name columns
  2. missing-value check (explicit summary)
  3. duplicate removal
  4. explicit label encoding (g -> 1, h -> 0)
"""
import pandas as pd

from src.config import DATA_PATH, FEATURES, TARGET


def load_raw(path=DATA_PATH):
    # The file has no header row, so column names are assigned manually.
    return pd.read_csv(path, names=FEATURES + [TARGET])


def check_missing(df, verbose=True):
    """Print a per-column missing-value summary; drop incomplete rows if any exist."""
    counts = df.isna().sum()
    total = int(counts.sum())
    if verbose:
        print("Missing values per column:")
        print(counts.to_string())
        print(f"-> total missing cells: {total}")
    if total:
        before = len(df)
        df = df.dropna()
        if verbose:
            print(f"-> dropped {before - len(df)} rows containing missing values")
    return df, total


def remove_duplicates(df, verbose=True):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(df)
    if verbose:
        print(f"Duplicate rows dropped: {dropped} ({before} -> {len(df)})")
    return df, dropped


def encode_labels(df):
    """Encode the class label: 'g' (gamma, signal) -> 1, 'h' (hadron, background) -> 0."""
    unexpected = set(df[TARGET].unique()) - {"g", "h"}
    if unexpected:
        raise ValueError(f"Unexpected class labels: {unexpected}")
    df = df.copy()
    df[TARGET] = (df[TARGET] == "g").astype(int)
    return df


def load_data(path=DATA_PATH, verbose=True):
    """Full cleaning pipeline. Quality stats are stored in df.attrs['quality']."""
    df = load_raw(path)
    n_raw = len(df)
    df, n_missing = check_missing(df, verbose)
    df, n_dupes = remove_duplicates(df, verbose)
    df = encode_labels(df)
    df.attrs["quality"] = {"rows_raw": n_raw, "missing_cells": n_missing,
                           "duplicates_dropped": n_dupes, "rows_clean": len(df)}
    return df
