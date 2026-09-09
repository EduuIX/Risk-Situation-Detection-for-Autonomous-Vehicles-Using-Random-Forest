"""Transformações aprendidas somente dentro dos folds de treinamento."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import NUMERIC, CATEGORICAL


def make_preprocessor(scale: bool = False) -> ColumnTransformer:
    numeric = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if scale:
        numeric.append(('scale', StandardScaler()))
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('numeric', Pipeline(numeric), list(NUMERIC)),
        ('categorical', categorical, list(CATEGORICAL)),
    ], remainder='drop', verbose_feature_names_out=True)


def make_pipeline(model, scale: bool = False) -> Pipeline:
    return Pipeline([('preprocess', make_preprocessor(scale)), ('model', model)])
