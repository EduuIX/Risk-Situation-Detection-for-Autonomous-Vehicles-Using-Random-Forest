"""Avaliação final de modelos já selecionados no treino; sem retuning."""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
)

from .config import ROOT, RANDOM_STATE
from .eda import save_figure
from .utils import write_json
from .plots import plot_model_comparison, plot_curves, plot_cv, plot_importances


def binary_metrics(y, probability, threshold: float = .5) -> dict:
    prediction = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        'accuracy': accuracy_score(y, prediction),
        'precision': precision_score(y, prediction, zero_division=0),
        'recall': recall_score(y, prediction, zero_division=0),
        'f1': f1_score(y, prediction, zero_division=0),
        'roc_auc': roc_auc_score(y, probability),
        'average_precision': average_precision_score(y, probability),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def threshold_analysis(y, probability, root: Path) -> pd.DataFrame:
    table = pd.DataFrame([{'threshold': t, **binary_metrics(y, probability, t)}
                          for t in (.3, .4, .5, .6, .7)])
    table.to_csv(root / 'reports/metrics/threshold_analysis.csv', index=False)
    table.plot(x='threshold', y=['precision', 'recall', 'f1'], marker='o', ylim=(0, 1.05),
               title='Threshold — predições fora do fold, somente treino')
    plt.xlabel('Threshold'); plt.ylabel('Métrica')
    save_figure(root / 'reports/figures/10_threshold_analysis.png')
    return table


def importance_plots(model, X, y, root: Path, seed: int, jobs: int) -> tuple:
    figures, metrics = root / 'reports/figures', root / 'reports/metrics'
    impurity = pd.Series(model.named_steps['model'].feature_importances_,
                         index=model.named_steps['preprocess'].get_feature_names_out(),
                         name='importance').sort_values(ascending=False)
    impurity.to_csv(metrics / 'feature_importance.csv', index_label='feature')
    perm = permutation_importance(model, X, y, scoring='f1', n_repeats=10,
                                  random_state=seed, n_jobs=jobs)
    table = pd.DataFrame({'feature': X.columns, 'mean': perm.importances_mean,
                          'std': perm.importances_std}).sort_values('mean', ascending=False)
    table.to_csv(metrics / 'permutation_importance.csv', index=False)
    plot_importances(impurity, table, figures)
    return impurity, table


def plot_roc_comparison(models, X_test, y_test, root: Path = ROOT) -> None:
    """Exporta curvas ROC e PR comparativas dos modelos salvos."""
    plot_curves(models, X_test, y_test, root)


def evaluate_artifacts(root: Path = ROOT, jobs: int = 1) -> pd.DataFrame:
    artifacts = joblib.load(root / 'models/experiment.joblib')
    split = joblib.load(root / 'data/processed/split.joblib')
    if artifacts['data_sha256'] != split['data_sha256'] or artifacts['train_indices'] != split['train_indices']:
        raise ValueError('Modelo e split pertencem a execuções diferentes.')
    X_train, X_test, y_train, y_test = (split[k] for k in ('X_train', 'X_test', 'y_train', 'y_test'))
    rows, reports = [], {}
    for name, model in artifacts['models'].items():
        for subset, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
            probability = model.predict_proba(X)[:, 1]
            rows.append({'model': name, 'subset': subset, **binary_metrics(y, probability)})
            reports[f'{name}_{subset}'] = classification_report(
                y, (probability >= .5).astype(int), labels=[0, 1],
                target_names=['baixo_risco', 'alto_risco'], output_dict=True, zero_division=0)
    full = pd.DataFrame(rows)
    metrics, figures = root / 'reports/metrics', root / 'reports/figures'
    full.to_csv(metrics / 'train_test_comparison.csv', index=False)
    comparison = full[full.subset == 'test'].drop(columns='subset').set_index('model')
    comparison.to_csv(metrics / 'model_comparison.csv')
    write_json(metrics / 'classification_reports.json', reports)
    selected = artifacts['selected_model']
    model = artifacts['models'][selected]
    probability = model.predict_proba(X_test)[:, 1]
    ConfusionMatrixDisplay.from_predictions(y_test, probability >= .5,
        labels=[0, 1], display_labels=['Baixo risco', 'Alto risco'], cmap='Blues')
    plt.title(f'{selected} — TN/FP na linha superior; FN/TP na inferior')
    plt.xlabel('Classe predita'); plt.ylabel('Classe observada')
    save_figure(figures / '05_confusion_matrix.png')
    RocCurveDisplay.from_predictions(y_test, probability, name=selected)
    plt.title('ROC — conjunto de teste'); plt.xlabel('Taxa de falsos positivos'); plt.ylabel('Recall')
    save_figure(figures / '06_roc_curve.png')
    plot_roc_comparison(artifacts['models'], X_test, y_test, root)
    PrecisionRecallDisplay.from_predictions(y_test, probability, name=selected)
    plt.axhline(y_test.mean(), linestyle='--', color='gray', label='Prevalência de risco')
    plt.legend(); plt.title('Precision–Recall — conjunto de teste')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    save_figure(figures / '07_precision_recall_curve.png')
    plot_model_comparison(comparison, root)
    folds_path = metrics / 'cv_folds.csv'
    if folds_path.exists():
        plot_cv(pd.read_csv(folds_path), figures)
    impurity, permutation = importance_plots(model, X_test, y_test, root, artifacts['seed'], jobs)
    write_conclusion(root, comparison, full, selected, impurity, permutation)
    return comparison


def write_conclusion(root, comparison, full, selected, impurity, permutation):
    result = comparison.loc[selected]
    tuned_delta = comparison.loc['random_forest_tuned', 'f1'] - comparison.loc['random_forest', 'f1']
    gaps = []
    for name in ('decision_tree', 'random_forest', 'random_forest_tuned'):
        values = full[full.model == name].set_index('subset')
        gaps.append(f"- {name}: diferença treino − teste: F1={values.loc['train', 'f1'] - values.loc['test', 'f1']:.4f}; "
                    f"recall={values.loc['train', 'recall'] - values.loc['test', 'recall']:.4f}; "
                    f"accuracy={values.loc['train', 'accuracy'] - values.loc['test', 'accuracy']:.4f}.")
    text = f'''# Síntese automática do experimento

Modelo RF escolhido exclusivamente por F1 médio na validação de treino: **{selected}**.
No teste: accuracy={result.accuracy:.4f}, precision={result.precision:.4f},
recall={result.recall:.4f}, F1={result.f1:.4f}, ROC-AUC={result.roc_auc:.4f}.
TN={int(result.tn)}, FP={int(result.fp)}, FN={int(result.fn)}, TP={int(result.tp)}.

FP representa alerta preventivo desnecessário; FN representa risco rotulado que passou
sem alerta. FN pode ser mais crítico, mas custos não foram quantificados. Accuracy
isolada não descreve esse trade-off. O threshold final permaneceu 0,5; a análise
experimental em threshold_analysis.csv usa somente predições fora do fold no treino.
A curva PR evidencia precision/recall e deve ser comparada à prevalência, sobretudo
em dados desbalanceados. A amostra publicada é balanceada; isso não estima a prevalência real.

H1: F1 do RF selecionado menos Dummy = {result.f1 - comparison.loc['dummy', 'f1']:.4f}.
H2: F1 do RF padrão menos Decision Tree = {comparison.loc['random_forest', 'f1'] - comparison.loc['decision_tree', 'f1']:.4f}.
H3: diferença de F1 tuned − padrão no teste = {tuned_delta:.4f}. Uma diferença pontual
não comprova ganho estatisticamente significativo. Consulte também cv_summary.csv;
os scores da busca tuned são otimistas por serem usados na seleção (CV não aninhada).

Indícios de sobreajuste devem considerar conjuntamente as diferenças abaixo e a
variância entre folds; não há diagnóstico automático baseado apenas em accuracy.

{chr(10).join(gaps)}

H4: maior importância por impureza: {impurity.index[0]}; por permutação: {permutation.iloc[0]['feature']}.
Importâncias não demonstram causalidade ou significância. Impureza favorece atributos
com muitas divisões possíveis; correlações podem repartir a importância por permutação.
H5: consultar threshold_analysis.csv para o compromisso precision/recall; sem custos
explícitos, não é possível declarar um threshold operacional ótimo.

O experimento reproduz rótulos de um dataset offline; não valida segurança funcional.
Rótulos construídos por regras, dados sintéticos e possível reamostragem prévia podem
inflar resultados. O CSV não identifica a proveniência de cada linha nem condutor/viagem:
o split estratificado não garante independência temporal, por motorista ou por origem
sintética. Não se pode excluir leakage herdado da preparação pública dos dados.
Há poucos cenários, ausência de câmera/LiDAR no CSV e limitações de sensores, clima,
cidades, veículos e mudança de distribuição. Validar externamente com viagens reais
independentes e rótulos de eventos observados antes de qualquer aplicação prática.
'''
    (root / 'reports/conclusion.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    print(evaluate_artifacts(args.output, args.jobs).to_string())
