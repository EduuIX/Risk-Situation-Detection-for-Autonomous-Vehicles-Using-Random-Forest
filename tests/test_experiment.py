import numpy as np
import pandas as pd
import pytest
from src.config import ExperimentConfig, FEATURES
from src.evaluate import binary_metrics
from src.train import prepare_data


def test_false_negatives_and_threshold():
    y, probability = [0, 0, 1, 1], [.1, .6, .4, .9]
    metrics = binary_metrics(y, probability)
    assert (metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']) == (1, 1, 1, 1)
    lower = binary_metrics(y, probability, .3)
    assert lower['fn'] == 0 and lower['recall'] == 1


def test_split_reproducible_and_duplicates_removed(tmp_path):
    frame = pd.DataFrame({name: np.arange(40, dtype=float) for name in FEATURES})
    frame['risk_level'] = [1, 2, 3, 4] * 10
    frame = pd.concat([frame, frame.iloc[:10]], ignore_index=True)
    path = tmp_path / 'input.csv'
    frame.to_csv(path, index=False)
    config = ExperimentConfig(data=path, output=tmp_path, folds=2)
    clean, split = prepare_data(config)
    _, other = prepare_data(config)
    assert len(clean) == 40
    assert len(split['X_test']) == 8
    assert set(split['train_indices']).isdisjoint(split['test_indices'])
    assert split['train_indices'] == other['train_indices']
    assert split['y_test'].value_counts().to_dict() == {0: 4, 1: 4}


def test_invalid_configuration():
    with pytest.raises(ValueError):
        ExperimentConfig(folds=1)
