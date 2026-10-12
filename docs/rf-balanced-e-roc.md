# Verificação do RF balanced e da curva ROC

Verificação dos modelos salvos em `models/experiment.joblib` e do split em
`data/processed/split.joblib`, com scikit-learn 1.9.1.

O treinamento contém 792 registros de cada classe. A fórmula de peso balanceado,
`n_amostras / (n_classes * n_amostras_da_classe)`, resulta em 1 para ambas.
Os parâmetros dos estimadores salvos diferem apenas em `class_weight`: `None`
no RF padrão e `'balanced'` no RF balanced. Ambos usam 100 árvores,
`random_state=42`, `bootstrap=True` e o mesmo pré-processamento e split.

A diferença vem do sorteio do bootstrap na implementação instalada. Em
`sklearn.ensemble._forest._generate_sample_indices`, a ausência de pesos usa
`RandomState.randint`; pesos explícitos usam `RandomState.choice` com probabilidades
normalizadas. Mesmo com probabilidades uniformes e a mesma semente, esses caminhos
produzem sequências diferentes. Portanto, pesos iguais não implicam florestas
idênticas nessa versão.

Confirmação experimental, sem substituir os modelos originais:

- Clonar e retreinar o RF padrão reproduziu exatamente suas probabilidades no teste.
- Clonar o RF padrão e treiná-lo com `model__sample_weight=np.ones(len(y_train))`
  reproduziu exatamente as probabilidades do RF balanced no teste.
- As decisões no limiar 0,5 diferem em dois dos 396 registros de teste.
  O padrão tem 11 falsos negativos; o balanced, 9. Ambos têm 13 falsos positivos.

Isso explica a diferença observada, sem evidência de vantagem por correção de
desbalanceamento neste treino. Nos folds da validação cruzada, as contagens podem
diferir por um registro, de modo que os pesos também podem ficar ligeiramente
diferentes de 1; essa é uma questão distinta do ajuste final com 792/792.

Referências da implementação e dos parâmetros:
[código do scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py)
e [RandomForestClassifier 1.9](https://sklearn.org/1.9/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
A confirmação acima foi feita com o código local instalado, não depende do estado
futuro do ramo `main`.

## Curvas ROC

Já existe a curva do modelo selecionado em `reports/figures/06_roc_curve.png`.
A avaliação agora também exporta as curvas de todos os modelos no mesmo teste:

- `reports/figures/06_roc_comparison.png`
- `reports/figures/06_roc_comparison.pdf` (vetorial, para o Overleaf)

As curvas usam probabilidades da classe 1 (alto risco), variando o limiar, e mostram
a taxa de verdadeiros positivos contra a taxa de falsos positivos. A diagonal
representa a referência aleatória. AUC no teste: RF padrão 0,9865; balanced 0,9839;
tuned 0,9868. A barra de ROC-AUC resume a área; não substitui a curva.

Para reproduzir somente as novas figuras, sem retreinar ou reescrever as métricas:

```python
import joblib
from src.evaluate import plot_roc_comparison

artifacts = joblib.load('models/experiment.joblib')
split = joblib.load('data/processed/split.joblib')
plot_roc_comparison(artifacts['models'], split['X_test'], split['y_test'])
```

Legenda sugerida: “Curvas ROC dos classificadores no conjunto de teste
(396 registros, 198 por classe), calculadas a partir das probabilidades de alto
risco. Os valores entre parênteses indicam a área sob a curva (AUC).”
