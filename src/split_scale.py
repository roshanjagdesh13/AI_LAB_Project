"""
split_scale.py
Split -> fit scaler on TRAIN ONLY -> transform valid/test with that scaler ->
oversample the training set only.
"""
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, SEED, TARGET


def split_dataset(df, train_frac=0.6, valid_frac=0.2, seed=SEED):
    """Stratified 60/20/20 split so every part keeps the gamma/hadron ratio."""
    train_df, rest = train_test_split(
        df, train_size=train_frac, stratify=df[TARGET], random_state=seed)
    valid_size = valid_frac / (1 - train_frac)
    valid_df, test_df = train_test_split(
        rest, train_size=valid_size, stratify=rest[TARGET], random_state=seed)
    return train_df, valid_df, test_df


def fit_scaler(train_df):
    """Fit on the training features only (no leakage from valid/test)."""
    return StandardScaler().fit(train_df[FEATURES])


def transform(df, scaler):
    return scaler.transform(df[FEATURES]), df[TARGET].values


def oversample(X, y):
    return RandomOverSampler(random_state=SEED).fit_resample(X, y)
