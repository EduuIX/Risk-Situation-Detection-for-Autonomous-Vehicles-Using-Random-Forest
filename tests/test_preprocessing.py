import numpy as np
from src.config import NUMERIC
from src.features import split_features_target
from src.preprocessing import make_preprocessor


def test_imputer_does_not_learn_from_test(sample_frame):
    X, _ = split_features_target(sample_frame)
    train, test = X.iloc[:4].copy(), X.iloc[4:].copy()
    train.loc[0, 'speed'] = np.nan
    test.loc[4, 'speed'] = 100000
    test.loc[5, 'speed'] = np.nan
    test.loc[:, 'current_weather'] = 999
    preprocessor = make_preprocessor()
    preprocessor.fit(train)
    stats_before = preprocessor.named_transformers_['numeric']['impute'].statistics_.copy()
    transformed = preprocessor.transform(test)
    assert stats_before[list(NUMERIC).index('speed')] == 3.
    assert transformed[1, list(NUMERIC).index('speed')] == 3.
    np.testing.assert_array_equal(stats_before, preprocessor.named_transformers_['numeric']['impute'].statistics_)
    assert np.isfinite(transformed).all()


def test_empty_training_column_kept(sample_frame):
    X, _ = split_features_target(sample_frame)
    X['speed'] = np.nan
    preprocessor = make_preprocessor(scale=True)
    transformed = preprocessor.fit_transform(X)
    assert np.isfinite(transformed).all()
    assert 'numeric__speed' in preprocessor.get_feature_names_out()
