# preprocess.py
# This file contains functions for building the preprocessing pipeline, which includes imputation, scaling, and encoding.

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

@dataclass
class PreprocessArtifacts:
    preprocessor: ColumnTransformer
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]


@dataclass
class CatBoostPreprocessArtifacts:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    cat_feature_cols: list[str]

def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> PreprocessArtifacts:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Numeric: if numeric missing, fills with median
    num_steps = [
        # keep_empty_features=True prevents dropping all-NaN columns during fit/transform
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler())) #standardize numeric features

    # Categorical: if missing, fills with most frequent category
    cat_steps = [
        # Use constant fill to avoid type-comparison issues in mixed object columns.
        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")), # encode into multiply binary columns
    ]

    # Ensure models won't error if NaNs

    # turn raw data into numeric matrix
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numeric_cols),
            ("cat", Pipeline(cat_steps), categorical_cols),
        ],
        remainder="drop", # drop any other columns
        sparse_threshold=0.3, # control sparse or dense matrix
    )

    return PreprocessArtifacts(
        preprocessor=preprocessor,
        feature_cols=X.columns.tolist(), # original feature columns (before encoding)
        numeric_cols=numeric_cols, # numeric columns (before encoding)
        categorical_cols=categorical_cols, # categorical columns (before encoding)
    )


def prepare_catboost_inputs(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> CatBoostPreprocessArtifacts:
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    X_train_out = X_train.copy()
    X_val_out = X_val.copy()
    X_test_out = X_test.copy()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_train_out[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
        X_val_out[numeric_cols] = num_imputer.transform(X_val[numeric_cols])
        X_test_out[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(
            strategy="constant",
            fill_value="__missing__",
            keep_empty_features=True,
        )
        X_train_out[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_val_out[categorical_cols] = cat_imputer.transform(X_val[categorical_cols])
        X_test_out[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

        # Keep categorical features as strings so CatBoost treats them as categories.
        X_train_out[categorical_cols] = X_train_out[categorical_cols].astype(str)
        X_val_out[categorical_cols] = X_val_out[categorical_cols].astype(str)
        X_test_out[categorical_cols] = X_test_out[categorical_cols].astype(str)

    return CatBoostPreprocessArtifacts(
        X_train=X_train_out,
        X_val=X_val_out,
        X_test=X_test_out,
        cat_feature_cols=categorical_cols,
    )
