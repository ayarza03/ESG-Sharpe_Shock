#==========================================================================================================
#                                       Section: models.py
# In this section:
#==========================================================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    import statsmodels.formula.api as smf
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "statsmodels is required for the regression block. Install it via:\n"
        "pip install statsmodels"
    ) from e


@dataclass(frozen=True)
class RegressionSpec:
    name: str
    formula: str
    cov_type: str = "cluster"          # "cluster" or "HC3"
    cluster_col: Optional[str] = "ticker"


def fit_ols(spec: RegressionSpec, df: pd.DataFrame):
    """
    Fits an OLS regression using a formula, with robust covariance.
    Default: clustered SE by ticker (panel-friendly).
    """
    model = smf.ols(spec.formula, data=df)

    if spec.cov_type == "cluster":
        if spec.cluster_col is None or spec.cluster_col not in df.columns:
            raise ValueError("[REG] cluster_col missing for clustered SE.")
        res = model.fit(cov_type="cluster", cov_kwds={"groups": df[spec.cluster_col]})
    elif spec.cov_type.upper() in {"HC0", "HC1", "HC2", "HC3"}:
        res = model.fit(cov_type=spec.cov_type.upper())
    else:
        raise ValueError(f"[REG] Unsupported cov_type: {spec.cov_type}")

    return res

# =========================
# ML utilities (Step 8)
# =========================

@dataclass(frozen=True)
class TemporalSplit:
    train_start: int = 201809
    train_end: int = 202212
    test_start: int = 202301
    test_end: int = 202410


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Preprocessor:
      - Numeric: median impute + standardize
      - Categorical: most_frequent impute + one-hot
    """
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def get_ml_models(random_state: int = 42) -> Dict[str, object]:
    """
    Returns models required by the spec: Ridge, RF, GB.
    """
    models: Dict[str, object] = {
        "Ridge": Ridge(alpha=1.0),
        "RF": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=5,
        ),
        "GB": GradientBoostingRegressor(
            n_estimators=300,
            random_state=random_state,
            learning_rate=0.05,
            max_depth=3,
        ),
    }
    return models


def fit_predict_ml_suite(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Fits each model in a pipeline(preprocessor + model) and returns test predictions.
    """
    pre = build_preprocessor(numeric_features, categorical_features)
    models = get_ml_models(random_state=random_state)

    preds: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        preds[name] = pipe.predict(X_test)

    return preds