from dataclasses import dataclass
from typing import Literal
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import RANDOM_SEED, TEST_SIZE, VAL_SIZE
from src.preprocess import build_preprocessor, prepare_catboost_inputs


@dataclass
class TrainResult:
    model: object
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


ModelName = Literal["logreg", "rf", "catboost"]


def _make_estimator(model_name: str):
    if model_name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model_name={model_name!r}. Use 'logreg', 'rf', or 'catboost'.")


def _make_catboost_estimator():
    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("CatBoost not installed. pip install catboost") from exc

    return CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=RANDOM_SEED,
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        verbose=False,
    )


def _cast_labels_if_integral(y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y):
        y_num = pd.to_numeric(y, errors="coerce")
        if y_num.notna().all():
            rounded = y_num.round()
            if (y_num == rounded).all():
                return rounded.astype(int)
    return y


def train_for_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_name: ModelName = "logreg",
) -> TrainResult:
    data = df.copy()
    data = data.dropna(subset=[target_col])

    X = data[feature_cols]
    y = data[target_col]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_trainval,
    )

    if model_name == "catboost":
        y_train = _cast_labels_if_integral(y_train)
        y_val = _cast_labels_if_integral(y_val)
        y_test = _cast_labels_if_integral(y_test)

        cb_data = prepare_catboost_inputs(X_train, X_val, X_test)
        model = _make_catboost_estimator()
        model.fit(
            cb_data.X_train,
            y_train,
            cat_features=cb_data.cat_feature_cols,
            eval_set=(cb_data.X_val, y_val),
            use_best_model=True,
        )

        return TrainResult(
            model=model,
            X_train=cb_data.X_train,
            y_train=y_train,
            X_val=cb_data.X_val,
            y_val=y_val,
            X_test=cb_data.X_test,
            y_test=y_test,
        )

    scale_numeric = model_name == "logreg"
    prep = build_preprocessor(X_train, scale_numeric=scale_numeric)
    estimator = _make_estimator(model_name)

    model = Pipeline(
        steps=[
            ("preprocess", prep.preprocessor),
            ("model", estimator),
        ]
    )

    model.fit(X_train, y_train)

    return TrainResult(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
