"""Figuras do relatório, reproduzíveis a partir das métricas e modelos salvos."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

LABELS = {'dummy': 'Dummy', 'decision_tree': 'Decision Tree',
          'random_forest': 'RF padrão', 'random_forest_balanced': 'RF balanced',
          'random_forest_tuned': 'RF tuned'}
COLORS = dict(zip(LABELS, ['#777777', '#E69F00', '#0072B2', '#009E73', '#CC79A7']))
METRICS = {'accuracy': 'Acurácia', 'precision': 'Precisão', 'recall': 'Recall',
           'f1': 'F1', 'roc_auc': 'ROC-AUC', 'average_precision': 'AP'}
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
         'axes.spines.top': False, 'axes.spines.right': False,
         'pdf.fonttype': 42, 'ps.fonttype': 42}


def export(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(directory / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


@plt.rc_context(STYLE)
def plot_model_comparison(comparison, root):
    directory = root / 'reports/figures'
    values = comparison[list(METRICS)].to_numpy()
    fig, ax = plt.subplots(figsize=(8.2, 4.5), layout='constrained')
    im = ax.imshow(values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(METRICS)), METRICS.values())
    ax.set_yticks(range(len(comparison)), [LABELS.get(n, n) for n in comparison.index])
    for i, j in np.ndindex(values.shape):
        ax.text(j, i, f'{values[i, j]:.4f}', ha='center', va='center',
                color='white' if values[i, j] > .65 else '#222222')
    ax.set_title('Desempenho no teste', pad=14)
    fig.colorbar(im, ax=ax, label='Valor da métrica', shrink=.85)
    fig.supxlabel('Classe positiva: alto risco · limiar 0,5 para acurácia, precisão, recall e F1', fontsize=9)
    export(fig, directory, '11_model_comparison')
    for column, title in METRICS.items():
        fig, ax = plt.subplots(figsize=(7, 4.3), layout='constrained')
        for i, name in enumerate(comparison.index):
            value = comparison.loc[name, column]
            ax.scatter(i, value, color=COLORS.get(name, '#0072B2'), s=65, zorder=3)
            ax.annotate(f'{value:.4f}', (i, value), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=9)
        ax.set_xticks(range(len(comparison)), [LABELS.get(n, n) for n in comparison.index])
        ax.set(xlim=(-.5, len(comparison)-.5), ylim=(-.04, 1.1), ylabel=title,
               title=f'{title} — teste')
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.grid(axis='y', alpha=.2)
        export(fig, directory, f'11_model_comparison_{column}')


@plt.rc_context(STYLE)
def plot_curves(models, X, y, root):
    directory = root / 'reports/figures'
    for kind, filename in [('roc', '06_roc_comparison'), ('pr', '07_precision_recall_comparison')]:
        fig, ax = plt.subplots(figsize=(6.8, 5.8), layout='constrained')
        for i, (name, model) in enumerate(models.items()):
            p = model.predict_proba(X)[:, list(model.classes_).index(1)]
            if kind == 'roc':
                x, z, _ = roc_curve(y, p)
                score, metric = roc_auc_score(y, p), 'AUC'
            else:
                z, x, _ = precision_recall_curve(y, p)
                score, metric = average_precision_score(y, p), 'AP'
            ax.plot(x, z, color=COLORS.get(name), linestyle=[':', '-.', '-', '--', ':'][i % 5],
                    linewidth=1.8, drawstyle='steps-post' if kind == 'pr' else 'default',
                    label=f'{LABELS.get(name, name)} ({metric} = {score:.4f})')
        if kind == 'roc':
            ax.plot([0, 1], [0, 1], '--', color='#aaaaaa', linewidth=1, label='Referência aleatória')
            ax.set(xlabel='Taxa de falsos positivos', ylabel='Taxa de verdadeiros positivos (recall)', title='Curvas ROC — teste')
        else:
            ax.axhline(np.mean(y), color='#aaaaaa', linestyle='--', linewidth=1, label=f'Prevalência = {np.mean(y):.2f}')
            ax.set(xlabel='Recall', ylabel='Precisão', title='Curvas precisão–recall — teste')
        ax.set(xlim=(-.01, 1.01), ylim=(-.01, 1.02))
        ax.grid(alpha=.15)
        ax.legend(loc='lower right' if kind == 'roc' else 'lower left', frameon=False, fontsize=9)
        export(fig, directory, filename)


@plt.rc_context(STYLE)
def plot_cv(folds, directory):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    names = [n for n in LABELS if n in set(folds.model)]
    for ax, metric in zip(axes.flat, ('accuracy', 'precision', 'recall', 'f1')):
        for i, name in enumerate(names):
            scores = folds[(folds.model == name) & (folds.metric == metric)].sort_values('fold').score.to_numpy()
            ax.scatter(i + np.linspace(-.12, .12, len(scores)), scores, s=25,
                       color=COLORS[name], alpha=.7, zorder=3)
            ax.plot([i-.18, i+.18], [scores.mean()]*2, color='black', linewidth=2)
        ax.set_xticks(range(len(names)), [LABELS[n] for n in names], rotation=20, ha='right')
        ax.set(title=METRICS[metric], ylim=(-.04, 1.04), ylabel='Score')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Validação cruzada no treino — cinco folds')
    fig.supxlabel('Pontos: folds · traço: média · tuned: CV de seleção, não aninhada', fontsize=10)
    export(fig, directory, '13_cv_comparison')


@plt.rc_context(STYLE)
def plot_importances(impurity, permutation, directory):
    top = impurity.head(15).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    ax.hlines(top.index, 0, top.values, color='#0072B2', alpha=.4)
    ax.scatter(top.values, top.index, color='#0072B2', s=35, zorder=3)
    ax.set(title='Importância por impureza — RF selecionado', xlabel='Redução normalizada de impureza')
    ax.grid(axis='x', alpha=.15)
    export(fig, directory, '08_feature_importance')
    top = permutation.head(15).sort_values('mean')
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    ax.errorbar(top['mean'], top.feature, xerr=top['std'], fmt='o', color='#0072B2', capsize=3)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set(title='Importância por permutação — teste', xlabel='Redução de F1 (média ± desvio de 10 permutações)')
    ax.grid(axis='x', alpha=.15)
    export(fig, directory, '09_permutation_importance')
