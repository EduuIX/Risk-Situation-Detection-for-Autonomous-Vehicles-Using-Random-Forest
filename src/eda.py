"""EDA descritiva; gráficos usados para exploração são restritos ao treino."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import NUMERIC, TARGET
from .data_loader import inspect_data, transform_target
from .utils import write_json


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def run_eda(frame: pd.DataFrame, root: Path) -> None:
    figures = root / 'reports/figures'
    metrics = root / 'reports/metrics'
    figures.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    write_json(metrics / 'training_inspection.json', inspect_data(frame))
    y = transform_target(frame[TARGET])
    counts = y.value_counts().sort_index()
    distribution = pd.DataFrame({'count': counts, 'percentage': counts / len(y) * 100})
    distribution.to_csv(metrics / 'training_class_distribution.csv', index_label='risk')
    counts.plot.bar(title='Classes no treino', rot=0, figsize=(6, 4))
    plt.xlabel('Classe: 0 = baixo risco; 1 = alto risco'); plt.ylabel('Registros')
    save_figure(figures / '01_class_distribution.png')
    missing = pd.DataFrame({'missing_count': frame.isna().sum(),
                            'missing_percentage': frame.isna().mean() * 100})
    missing.to_csv(metrics / 'missing_values.csv', index_label='feature')
    missing.missing_percentage.plot.barh(title='Valores ausentes — treino', figsize=(9, 6))
    plt.xlabel('Percentual (%)'); plt.ylabel('Atributo')
    save_figure(figures / '02_missing_values.png')
    numeric = frame[list(NUMERIC)].copy()
    correlations = numeric.assign(risk=y).corr()
    correlations.to_csv(metrics / 'correlations.csv')
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlations, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Correlação de Pearson — treino; constantes sem correlação definida')
    plt.xlabel('Atributo'); plt.ylabel('Atributo')
    save_figure(figures / '03_correlation_matrix.png')
    numeric.hist(figsize=(16, 14), bins=25)
    for ax in plt.gcf().axes:
        ax.set_xlabel(ax.get_title()); ax.set_ylabel('Registros')
    plt.suptitle('Distribuições dos atributos — treino')
    save_figure(figures / '04_feature_distributions.png')
    fig, axes = plt.subplots(5, 3, figsize=(16, 18))
    for name, ax in zip(NUMERIC, axes.flat):
        sns.boxplot(x=y, y=frame[name], ax=ax)
        ax.set_title(name); ax.set_xlabel('Classe de risco'); ax.set_ylabel(name)
    for ax in list(axes.flat)[len(NUMERIC):]:
        ax.set_visible(False)
    save_figure(figures / '12_features_by_class.png')
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    iqr = q3 - q1
    outliers = ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum()
    pd.DataFrame({'iqr': iqr, 'outlier_count': outliers,
                  'outlier_percentage': outliers / len(frame) * 100,
                  'skewness': numeric.skew()}).to_csv(metrics / 'outliers.csv')
    frame.groupby(y)[list(NUMERIC)].agg(['mean', 'median', 'std']).to_csv(metrics / 'features_by_class.csv')
