import numpy as np
import pandas as pd
import pytest
from src.config import FEATURES, TARGET
from src.data_loader import load_data, transform_target, validate_features, inspect_data
from src.features import split_features_target


def test_load_ignores_serialized_index(tmp_path, sample_frame):
    path = tmp_path / 'data.csv'
    sample_frame.to_csv(path)
    frame = load_data(path)
    X, y = split_features_target(frame)
    assert tuple(X.columns) == FEATURES
    assert TARGET not in X and 'Unnamed: 0' not in X
    assert y.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert inspect_data(frame)['features'] == 14


def test_target_numeric_and_text():
    result = transform_target(pd.Series([1., 2., 3., 4., 'low', ' MEDIUM ', 'HIGH', 'VERY_HIGH']))
    assert result.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


@pytest.mark.parametrize('value', [None, 0, 5, 'danger', ''])
def test_unknown_target_rejected(value):
    with pytest.raises(ValueError, match='Target'):
        transform_target(pd.Series([value]))


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match='src.data_loader'):
        load_data(tmp_path / 'missing.csv')


def test_bad_schema_and_infinity(sample_frame):
    with pytest.raises(ValueError, match='ausentes'):
        validate_features(sample_frame.drop(columns='speed'))
    sample_frame.loc[0, 'speed'] = np.inf
    with pytest.raises(ValueError, match='infinitos'):
        validate_features(sample_frame)


def test_missing_features_allowed_but_target_not(tmp_path, sample_frame):
    sample_frame.loc[0, 'speed'] = np.nan
    path = tmp_path / 'data.csv'
    sample_frame.to_csv(path, index=False)
    assert load_data(path).speed.isna().sum() == 1
    sample_frame.loc[0, TARGET] = np.nan
    sample_frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Target'):
        load_data(path)
