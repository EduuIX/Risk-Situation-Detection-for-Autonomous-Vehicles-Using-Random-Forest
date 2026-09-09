"""Treinamento reproduzível: python -m src.train [--data arquivo.csv]."""
import argparse
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, cross_validate, cross_val_predict, train_test_split,
)
from sklearn.tree import DecisionTreeClassifier

from .config import ExperimentConfig, FEATURES, TARGET
from .data_loader import load_data, inspect_data
from .eda import run_eda
from .evaluate import evaluate_artifacts, threshold_analysis
from .features import split_features_target
from .preprocessing import make_pipeline
from .utils import digest, output_dirs, write_json

SCORING = {
    'accuracy': 'accuracy', 'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'f1': make_scorer(f1_score, zero_division=0),
}


def prepare_data(config: ExperimentConfig):
    """Inspeção estrutural global; EDA e transformações aprendidas só no treino."""
    output_dirs(config.output)
    raw = load_data(config.data)
    write_json(config.output / 'reports/metrics/dataset_inspection.json', inspect_data(raw))
    # Duplicatas exatas não devem atravessar a fronteira treino/teste.
    frame = raw.drop_duplicates().copy()
    X, y = split_features_target(frame)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError('São necessárias as duas classes binárias com pelo menos 2 exemplos cada.')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.seed, stratify=y)
    if y_test.nunique() != 2 or y_train.value_counts().min() < config.folds:
        raise ValueError('Amostras insuficientes por classe para teste e validação cruzada configurados.')
    split = dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                 train_indices=X_train.index.tolist(), test_indices=X_test.index.tolist(),
                 data_sha256=digest(config.data))
    write_json(config.output / 'reports/metrics/split.json', {
        'train_indices': split['train_indices'], 'test_indices': split['test_indices'],
        'removed_exact_duplicates': len(raw) - len(frame),
        'train_classes': y_train.value_counts().to_dict(), 'test_classes': y_test.value_counts().to_dict(),
        'seed': config.seed, 'test_size': config.test_size, 'data_sha256': split['data_sha256'],
    })
    return frame, split


def run_training(config: ExperimentConfig = ExperimentConfig(), evaluate: bool = True):
    frame, split = prepare_data(config)
    root = config.output
    X_train, y_train = split['X_train'], split['y_train']
    run_eda(frame.loc[X_train.index], root)
    cv = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    models = {
        'dummy': make_pipeline(DummyClassifier(strategy='most_frequent', random_state=config.seed)),
        'decision_tree': make_pipeline(DecisionTreeClassifier(random_state=config.seed)),
        'random_forest': make_pipeline(RandomForestClassifier(random_state=config.seed, n_jobs=1)),
        'random_forest_balanced': make_pipeline(RandomForestClassifier(
            random_state=config.seed, class_weight='balanced', n_jobs=1)),
    }
    summaries, fold_rows = [], []
    for name, model in models.items():
        print(f'Validação cruzada: {name}', flush=True)
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING,
                                n_jobs=config.jobs, error_score='raise')
        row = {'model': name, 'selection_cv': False}
        for metric in SCORING:
            values = scores[f'test_{metric}']
            row[f'{metric}_mean'] = float(np.mean(values))
            row[f'{metric}_std'] = float(np.std(values))
            fold_rows.extend({'model': name, 'metric': metric, 'fold': i + 1, 'score': value}
                             for i, value in enumerate(values))
        summaries.append(row)
        model.fit(X_train, y_train)
    space = {
        'model__n_estimators': [100, 200, 400], 'model__max_depth': [None, 5, 10, 20],
        'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2', .8],
        'model__class_weight': [None, 'balanced'], 'model__bootstrap': [True, False],
    }
    search = RandomizedSearchCV(
        make_pipeline(RandomForestClassifier(random_state=config.seed, n_jobs=1)),
        space, n_iter=config.iterations, scoring=SCORING, refit='f1', cv=cv,
        random_state=config.seed, n_jobs=config.jobs, error_score='raise', return_train_score=True)
    print(f'Busca de hiperparâmetros: {config.iterations} candidatos × {config.folds} folds', flush=True)
    search.fit(X_train, y_train)
    models['random_forest_tuned'] = search.best_estimator_
    row = {'model': 'random_forest_tuned', 'selection_cv': True}
    for metric in SCORING:
        row[f'{metric}_mean'] = float(search.cv_results_[f'mean_test_{metric}'][search.best_index_])
        row[f'{metric}_std'] = float(search.cv_results_[f'std_test_{metric}'][search.best_index_])
        for i in range(config.folds):
            fold_rows.append({'model': 'random_forest_tuned', 'metric': metric, 'fold': i + 1,
                              'score': search.cv_results_[f'split{i}_test_{metric}'][search.best_index_]})
    summaries.append(row)
    summary = pd.DataFrame(summaries).set_index('model')
    candidates = summary.loc[[name for name in models if name.startswith('random_forest')]]
    selected = candidates.f1_mean.idxmax()
    print(f'RF selecionado pelo F1 de treino: {selected}', flush=True)
    summary.to_csv(root / 'reports/metrics/cv_summary.csv')
    pd.DataFrame(fold_rows).to_csv(root / 'reports/metrics/cv_folds.csv', index=False)
    pd.DataFrame(search.cv_results_).to_csv(root / 'reports/metrics/search_results.csv', index=False)
    write_json(root / 'reports/metrics/best_parameters.json', search.best_params_)
    oof = cross_val_predict(models[selected], X_train, y_train, cv=cv,
                           method='predict_proba', n_jobs=config.jobs)[:, 1]
    threshold_analysis(y_train, oof, root)
    pd.DataFrame({'row_index': X_train.index, 'risk': y_train.to_numpy(), 'probability': oof}).to_csv(
        root / 'reports/metrics/oof_predictions.csv', index=False)
    artifacts = {'models': models, 'selected_model': selected, 'threshold': .5,
                 'seed': config.seed, 'features': list(FEATURES),
                 'data_sha256': split['data_sha256'], 'train_indices': split['train_indices']}
    joblib.dump(split, root / 'data/processed/split.joblib')
    joblib.dump(artifacts, root / 'models/experiment.joblib')
    joblib.dump(models[selected], root / 'models/random_forest_best.joblib')
    write_json(root / 'models/metadata.json', {
        'selected_model': selected, 'threshold': .5, 'features': list(FEATURES),
        'data_sha256': split['data_sha256'], 'config': asdict(config),
        'python': platform.python_version(),
        'versions': {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib', 'matplotlib', 'seaborn']},
        'cv_warning': 'CV de seleção, não aninhada; threshold OOF exploratório usa configuração selecionada no treino.',
    })
    if evaluate:
        print(evaluate_artifacts(root, config.jobs).to_string())
    return artifacts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = ExperimentConfig()
    parser.add_argument('--data', type=Path, default=defaults.data)
    parser.add_argument('--output', type=Path, default=defaults.output)
    parser.add_argument('--folds', type=int, default=defaults.folds)
    parser.add_argument('--iterations', type=int, default=defaults.iterations)
    parser.add_argument('--jobs', type=int, default=defaults.jobs)
    parser.add_argument('--train-only', action='store_true')
    args = parser.parse_args()
    run_training(ExperimentConfig(data=args.data, output=args.output, folds=args.folds,
                                  iterations=args.iterations, jobs=args.jobs), evaluate=not args.train_only)
