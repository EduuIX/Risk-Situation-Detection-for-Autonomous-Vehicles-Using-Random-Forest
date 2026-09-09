"""Inferência em CSV com as mesmas 14 features do treinamento."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ROOT
from .data_loader import validate_features


def predict_samples(samples: pd.DataFrame, model_path: Path = ROOT / 'models/random_forest_best.joblib') -> pd.DataFrame:
    if samples.empty:
        raise ValueError('Forneça pelo menos uma amostra.')
    model = joblib.load(model_path)
    features = validate_features(samples)
    probability = model.predict_proba(features)[:, 1]
    prediction = (probability >= .5).astype(int)
    return pd.DataFrame({'prediction': prediction,
                         'label': np.where(prediction == 1, 'HIGH RISK', 'LOW RISK'),
                         'probability_high_risk': probability}, index=samples.index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/random_forest_best.joblib')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = predict_samples(pd.read_csv(args.input), args.model)
    if args.output:
        result.to_csv(args.output, index=False)
    else:
        print(result.to_csv(index=False), end='')
