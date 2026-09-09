# Projeto: Detecção de Situações de Risco para Veículos Autônomos utilizando Random Forest

Quero que você desenvolva um projeto completo de Machine Learning em Python com foco em:

**Detecção de situações de risco para veículos autônomos utilizando Random Forest.**

O objetivo é utilizar um dataset público contendo dados de condução, telemetria veicular, sensores e/ou condições ambientais para construir um modelo capaz de classificar situações potencialmente perigosas.

O projeto será utilizado em um trabalho acadêmico e deve priorizar:

* organização;
* clareza;
* reprodutibilidade;
* boas práticas de Machine Learning;
* análise crítica;
* visualizações;
* documentação;
* facilidade de execução.

Não implemente apenas um script monolítico. Estruture o projeto adequadamente e explique as principais decisões tomadas.

---

# 1. Objetivo geral

Construir um sistema de classificação baseado em **Random Forest** capaz de identificar situações de risco relacionadas à condução de um veículo.

A pergunta principal do projeto será:

> É possível identificar situações potencialmente perigosas para um veículo autônomo utilizando dados de telemetria, características do veículo e/ou condições ambientais?

O modelo deverá receber atributos relacionados ao estado do veículo e retornar uma classificação de risco.

Preferencialmente, trabalhar com uma classificação binária:

```text
0 = Situação normal / baixo risco
1 = Situação perigosa / alto risco
```

Caso o dataset escolhido originalmente possua múltiplos níveis de risco, como:

```text
LOW
MEDIUM
HIGH
VERY HIGH
```

criar também uma versão binária utilizando uma transformação coerente, por exemplo:

```text
LOW + MEDIUM      -> 0
HIGH + VERY HIGH  -> 1
```

Essa transformação deve ser explicitamente documentada e justificada.

---

# 2. Dataset

Utilizar um dataset público relacionado a:

* autonomous vehicles;
* driving risk;
* vehicle telemetry;
* road safety;
* driving behavior;
* vehicle sensors;
* intelligent transportation systems.

O dataset pode ser obtido de:

* Kaggle;
* UCI Machine Learning Repository;
* Zenodo;
* IEEE DataPort, caso seja publicamente acessível;
* outras fontes acadêmicas públicas.

Antes de implementar o treinamento:

1. identificar claramente o dataset utilizado;
2. informar a fonte;
3. documentar como obter os dados;
4. descrever o significado das principais features;
5. identificar a variável target;
6. informar quantidade de amostras;
7. informar quantidade de atributos;
8. analisar o balanceamento das classes.

Não invente atributos que não existam no dataset.

Caso o arquivo não esteja presente no repositório, implemente o projeto de maneira que o usuário possa colocá-lo em:

```text
data/raw/
```

e configurar facilmente o nome do arquivo.

---

# 3. Tecnologias

Utilizar preferencialmente:

```text
Python 3.11+
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
joblib
```

Pode utilizar também:

```text
scipy
imbalanced-learn
```

se houver justificativa.

Criar um arquivo:

```text
requirements.txt
```

ou:

```text
pyproject.toml
```

com as dependências.

---

# 4. Estrutura do projeto

Organize inicialmente o repositório aproximadamente da seguinte forma:

```text
autonomous-driving-risk/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── requirements.txt
├── README.md
└── .gitignore
```

A estrutura pode ser ajustada se houver justificativa técnica, mas mantenha clara separação entre:

* dados;
* notebooks;
* código-fonte;
* modelos;
* gráficos;
* métricas.

---

# 5. Etapa 1 — Carregamento e inspeção dos dados

Criar funções responsáveis pelo carregamento do dataset.

Realizar inicialmente:

```python
df.head()
df.tail()
df.shape
df.info()
df.describe()
```

Mostrar:

* número de registros;
* número de features;
* tipos das variáveis;
* valores ausentes;
* valores duplicados;
* quantidade de classes;
* distribuição da variável alvo.

Apresentar os resultados de forma organizada.

---

# 6. Etapa 2 — Exploratory Data Analysis

Criar um notebook:

```text
01_eda.ipynb
```

com uma EDA detalhada.

Incluir, quando aplicável:

## Distribuição das classes

Mostrar quantidade absoluta e percentual das classes.

Exemplo:

```text
Classe 0 — Normal
Classe 1 — Risco
```

Investigar se existe desbalanceamento.

---

## Distribuição das features

Criar gráficos apropriados para variáveis numéricas, como:

* histogramas;
* boxplots;
* density plots.

Avaliar:

* dispersão;
* assimetria;
* possíveis outliers;
* diferenças entre classes.

---

## Relação das features com o risco

Comparar as principais variáveis entre as classes.

Por exemplo:

```text
speed × risk
acceleration × risk
engine_load × risk
rpm × risk
throttle_position × risk
```

Somente utilizar variáveis realmente presentes no dataset.

---

## Correlação

Criar uma matriz de correlação para variáveis numéricas.

Investigar:

* features altamente correlacionadas;
* possíveis redundâncias;
* possíveis relações com a variável target.

---

## Dados faltantes

Criar uma análise mostrando:

```text
feature
missing_count
missing_percentage
```

---

## Outliers

Investigar possíveis outliers utilizando:

* boxplot;
* IQR;
* análise estatística.

Não remover automaticamente outliers.

Antes de removê-los, discutir se eles podem representar justamente situações reais de risco.

Isso é particularmente importante neste projeto.

---

# 7. Etapa 3 — Pré-processamento

Criar:

```text
02_preprocessing.ipynb
```

e funções reutilizáveis em:

```text
src/preprocessing.py
```

O pré-processamento deve considerar:

## Dados faltantes

Implementar tratamento apropriado.

Exemplos:

* mediana para variáveis numéricas;
* moda para categóricas;
* remoção apenas quando justificada.

---

## Variáveis categóricas

Caso existam, utilizar técnicas como:

```text
OneHotEncoder
OrdinalEncoder
```

de acordo com o significado da variável.

Evitar introduzir uma relação ordinal artificial quando ela não existir.

---

## Normalização

Random Forest não exige normalização das features.

Portanto, não normalizar automaticamente.

Explique no projeto que árvores de decisão particionam o espaço por limiares e não dependem da escala da mesma maneira que algoritmos baseados em distância ou gradiente.

Caso outros algoritmos sejam futuramente utilizados para comparação, permitir normalização através de um pipeline configurável.

---

# 8. Evitar Data Leakage

Toda transformação aprendida a partir dos dados deve ser ajustada apenas utilizando o conjunto de treinamento.

Sempre que possível utilizar:

```python
Pipeline
```

e:

```python
ColumnTransformer
```

do scikit-learn.

Nunca utilizar informações do conjunto de teste durante:

* imputação;
* feature selection;
* tuning;
* balanceamento;
* escolha de hiperparâmetros.

---

# 9. Train/Test Split

Separar os dados utilizando inicialmente:

```text
80% treinamento
20% teste
```

Utilizar:

```python
train_test_split(
    ...,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Utilizar `stratify=y` sempre que possível para preservar a distribuição das classes.

O conjunto de teste deve permanecer completamente isolado até a avaliação final.

---

# 10. Modelo baseline

Antes do Random Forest, implementar um modelo baseline.

Preferencialmente:

```python
DummyClassifier
```

e depois:

```python
DecisionTreeClassifier
```

O objetivo é demonstrar se o Random Forest realmente oferece ganho em relação a soluções mais simples.

Comparar pelo menos:

```text
Dummy Classifier
Decision Tree
Random Forest
Random Forest com tuning
```

---

# 11. Random Forest

Implementar:

```python
RandomForestClassifier
```

Criar inicialmente um modelo com parâmetros próximos dos padrões do scikit-learn.

Configurar:

```python
random_state=42
```

para garantir reprodutibilidade.

---

# 12. Hyperparameter Tuning

Realizar ajuste fino do Random Forest.

Avaliar hiperparâmetros como:

```python
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
class_weight
bootstrap
```

Utilizar preferencialmente:

```python
RandomizedSearchCV
```

para uma busca inicial.

Caso o espaço seja pequeno, utilizar:

```python
GridSearchCV
```

Utilizar validação cruzada estratificada:

```python
StratifiedKFold
```

com aproximadamente:

```text
5 folds
```

---

# 13. Métrica utilizada durante o tuning

Não utilizar obrigatoriamente apenas accuracy.

Como o projeto envolve identificação de situações perigosas, falsos negativos possuem grande importância.

Portanto, considerar como métrica principal:

```text
F1-score
```

ou:

```text
Recall da classe de risco
```

e justificar a escolha.

A escolha deve considerar que:

```text
False Negative:
situação perigosa classificada como segura.
```

Esse erro pode ser particularmente crítico em sistemas relacionados a veículos autônomos.

---

# 14. Class imbalance

Verificar se o dataset está desbalanceado.

Caso esteja, comparar pelo menos:

```text
Random Forest normal
```

contra:

```text
Random Forest class_weight="balanced"
```

Opcionalmente testar:

```text
SMOTE
```

mas somente aplicado ao conjunto de treinamento.

Nunca aplicar SMOTE antes do train/test split.

Caso SMOTE seja utilizado junto com cross-validation, utilizar pipeline do:

```text
imbalanced-learn
```

para evitar data leakage.

---

# 15. Avaliação obrigatória

No conjunto de teste calcular:

## Accuracy

```python
accuracy_score
```

---

## Precision

```python
precision_score
```

---

## Recall

```python
recall_score
```

---

## F1-score

```python
f1_score
```

---

## Matriz de Confusão

Utilizar:

```python
confusion_matrix
ConfusionMatrixDisplay
```

Apresentar claramente:

```text
True Positive
True Negative
False Positive
False Negative
```

Interpretar os resultados no contexto do problema.

---

# 16. Interpretação dos erros

A análise deve explicar explicitamente:

### False Positive

```text
O modelo identifica uma situação como perigosa,
mas ela era segura.
```

Possível consequência:

```text
o veículo poderia executar uma ação preventiva desnecessária,
como reduzir velocidade ou aumentar distância.
```

### False Negative

```text
Uma situação realmente perigosa é classificada
como segura.
```

Possível consequência:

```text
o sistema deixa de tomar uma ação preventiva.
```

Destacar que, dependendo do sistema, falsos negativos podem ser mais críticos.

---

# 17. ROC Curve e AUC

Calcular probabilidades utilizando:

```python
predict_proba()
```

Gerar:

```text
ROC Curve
```

e calcular:

```text
ROC-AUC
```

Mostrar graficamente o desempenho do modelo.

---

# 18. Precision-Recall Curve

Mesmo não sendo obrigatória, gerar também:

```text
Precision-Recall Curve
```

principalmente se houver forte desbalanceamento.

Calcular:

```text
Average Precision
```

Explicar por que PR Curve pode ser mais informativa que ROC em datasets muito desbalanceados.

---

# 19. Classification Report

Utilizar:

```python
classification_report
```

e apresentar:

```text
precision
recall
f1-score
support
```

por classe.

---

# 20. Comparação dos modelos

Gerar uma tabela semelhante a:

```text
Model                  Accuracy   Precision   Recall   F1      ROC-AUC

Dummy Classifier
Decision Tree
Random Forest
Random Forest Tuned
```

Não utilizar valores fictícios.

A tabela deve ser preenchida programaticamente com os resultados obtidos.

Salvar também em:

```text
reports/metrics/model_comparison.csv
```

---

# 21. Feature Importance

Uma das principais análises do projeto deverá ser a importância das features.

Utilizar inicialmente:

```python
model.feature_importances_
```

Gerar um ranking com as principais variáveis utilizadas pelo Random Forest.

Criar um gráfico com:

```text
Top 10
```

ou:

```text
Top 15
```

features.

Responder:

> Quais características mais influenciam o modelo na identificação de uma situação de risco?

---

# 22. Permutation Importance

Além da importância padrão do Random Forest, implementar:

```python
permutation_importance
```

no conjunto de teste.

Comparar:

```text
feature_importances_
```

com:

```text
Permutation Importance
```

Explicar brevemente que a importância baseada em impureza pode apresentar vieses.

---

# 23. Threshold Analysis

O Random Forest normalmente utiliza:

```text
threshold = 0.5
```

para classificação binária.

Criar uma análise experimental variando o threshold.

Exemplo:

```text
0.30
0.40
0.50
0.60
0.70
```

Para cada threshold calcular:

```text
Precision
Recall
F1
```

Gerar gráfico:

```text
Threshold × Precision
Threshold × Recall
Threshold × F1
```

Discutir o trade-off.

No contexto de segurança, investigar se reduzir o threshold aumenta Recall e diminui falsos negativos.

Não alterar o threshold final sem justificar com dados.

---

# 24. Cross-validation

Além do resultado final no test set, executar validação cruzada no conjunto de treinamento.

Mostrar:

```text
mean accuracy
mean precision
mean recall
mean F1
standard deviation
```

Isso deve permitir analisar se o modelo apresenta resultados consistentes ou alta variância.

---

# 25. Overfitting

Comparar desempenho entre:

```text
train set
```

e:

```text
test set
```

Principalmente para:

```text
Decision Tree
Random Forest
Random Forest Tuned
```

Mostrar se existe evidência de overfitting.

Exemplo de análise:

```text
Train accuracy muito elevada
+
Test accuracy significativamente inferior
=
possível overfitting
```

Não afirmar overfitting apenas olhando accuracy; considerar outras métricas.

---

# 26. Visualizações obrigatórias

Produzir e salvar pelo menos:

```text
01_class_distribution.png
02_missing_values.png
03_correlation_matrix.png
04_feature_distributions.png
05_confusion_matrix.png
06_roc_curve.png
07_precision_recall_curve.png
08_feature_importance.png
09_permutation_importance.png
10_threshold_analysis.png
11_model_comparison.png
```

Salvar em:

```text
reports/figures/
```

Os gráficos devem possuir:

* título;
* legenda quando aplicável;
* nomes dos eixos;
* tamanho adequado;
* boa legibilidade.

---

# 27. Salvamento do modelo

Salvar o melhor modelo utilizando:

```python
joblib
```

em:

```text
models/random_forest_best.joblib
```

Caso seja usado pipeline, salvar o pipeline completo.

Isso é importante para que o pré-processamento aplicado durante inferência seja exatamente o mesmo utilizado durante treinamento.

---

# 28. Script de inferência

Criar uma função ou script capaz de carregar o modelo e receber uma nova amostra.

Exemplo:

```python
prediction = model.predict(sample)
probability = model.predict_proba(sample)
```

Retornar algo semelhante a:

```text
Prediction: HIGH RISK
Probability: 0.87
```

Não criar uma API/web interface neste momento.

O foco do trabalho é Machine Learning.

---

# 29. Reprodutibilidade

Definir:

```python
RANDOM_STATE = 42
```

em arquivo central de configuração.

Aplicar sempre que a biblioteca permitir.

Criar:

```text
src/config.py
```

com configurações globais.

---

# 30. README

Criar um README completo contendo:

## Título

```text
Autonomous Driving Risk Detection using Random Forest
```

## Descrição

Explicar que o projeto investiga a utilização de Machine Learning para identificação de situações potencialmente perigosas em cenários relacionados à condução autônoma.

## Problema

Explicar o problema de classificação.

## Dataset

Informar:

* nome;
* fonte;
* link;
* tamanho;
* target;
* principais features.

## Arquitetura do projeto

Mostrar a estrutura de diretórios.

## Instalação

Exemplo:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Execução

Explicar como executar:

* notebooks;
* treinamento;
* avaliação.

## Modelos utilizados

```text
Dummy Classifier
Decision Tree
Random Forest
Random Forest Tuned
```

## Métricas

Explicar:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
```

## Resultados

Criar seção preparada para receber os resultados reais após treinamento.

## Conclusão

Criar automaticamente uma síntese dos resultados obtidos após execução, sem inventar resultados.

---

# 31. Análise crítica

O projeto não deve apenas mostrar métricas.

Produza também uma análise crítica abordando:

### 1. Accuracy é suficiente?

Discutir por que não.

### 2. Precision ou Recall?

Explicar o trade-off.

### 3. Por que False Negative é importante?

Contextualizar para veículos autônomos.

### 4. Existe desbalanceamento?

Mostrar o impacto.

### 5. Random Forest apresentou overfitting?

Comparar treino/teste.

### 6. O tuning trouxe ganho real?

Comparar antes/depois.

### 7. Quais features foram mais importantes?

Interpretar.

### 8. Qual seria uma limitação prática?

Discutir que um dataset offline simplificado não representa integralmente a complexidade de um veículo autônomo real.

### 9. O modelo pode ser utilizado diretamente em um veículo real?

Responder criticamente.

O modelo deve ser tratado como experimento acadêmico e não como componente validado de segurança funcional.

---

# 32. Limitações

Criar uma seção específica abordando possíveis limitações como:

* tamanho do dataset;
* representatividade dos cenários;
* desbalanceamento;
* qualidade dos sensores;
* dados ausentes;
* ausência de informação temporal;
* ausência de LiDAR/câmera;
* generalização para outras cidades;
* diferentes condições climáticas;
* diferentes modelos de veículo;
* mudança de distribuição dos dados;
* riscos de utilizar um modelo offline em ambientes reais.

---

# 33. Hipóteses de pesquisa

Utilizar como hipóteses iniciais:

### H1

Um modelo Random Forest consegue distinguir situações normais de situações de risco utilizando características relacionadas ao veículo e ao ambiente.

### H2

Random Forest apresenta desempenho superior a uma árvore de decisão individual.

### H3

O ajuste dos hiperparâmetros melhora a capacidade de generalização do modelo.

### H4

Algumas features possuem influência significativamente maior na identificação de situações de risco.

### H5

Otimizar apenas accuracy não necessariamente produz o melhor modelo quando falsos negativos possuem maior custo.

As hipóteses devem posteriormente ser confrontadas com os resultados reais.

---

# 34. Fluxo esperado

Implementar aproximadamente:

```text
Raw Dataset
     │
     ▼
Data Validation
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Preprocessing
     │
     ▼
Train/Test Split
     │
     ├───────────────┐
     │               │
     ▼               ▼
Decision Tree    Random Forest
                     │
                     ▼
              Hyperparameter Tuning
                     │
                     ▼
                Best Model
                     │
                     ▼
                 Test Set
                     │
           ┌─────────┼──────────┐
           ▼         ▼          ▼
      Metrics   ROC/PR     Confusion Matrix
                     │
                     ▼
               Feature Analysis
                     │
                     ▼
               Critical Analysis
```

---

# 35. Qualidade do código

Seguir boas práticas:

* funções pequenas;
* nomes de variáveis claros;
* evitar duplicação;
* comentários apenas quando necessários;
* docstrings;
* type hints quando útil;
* configuração centralizada;
* código reutilizável.

Evitar colocar toda a lógica exclusivamente nos notebooks.

Os notebooks devem importar as funções existentes em:

```text
src/
```

---

# 36. Testes básicos

Criar testes simples quando pertinente, principalmente para:

```text
data loading
target transformation
preprocessing
```

Criar diretório:

```text
tests/
```

Exemplo:

```text
tests/
├── test_data_loader.py
└── test_preprocessing.py
```

---

# 37. Resultado esperado

Ao final, quero conseguir executar o projeto e obter automaticamente:

```text
Dataset carregado
        ↓
EDA
        ↓
Pré-processamento
        ↓
Baseline
        ↓
Decision Tree
        ↓
Random Forest
        ↓
Hyperparameter Tuning
        ↓
Melhor modelo
        ↓
Avaliação no test set
        ↓
Accuracy
Precision
Recall
F1
ROC-AUC
        ↓
Confusion Matrix
        ↓
ROC Curve
        ↓
Precision-Recall Curve
        ↓
Feature Importance
        ↓
Threshold Analysis
        ↓
Comparação entre modelos
        ↓
Modelo salvo
```

---

# 38. Regra extremamente importante

Não invente resultados.

Não escreva valores fictícios como:

```text
Accuracy = 95%
AUC = 0.97
```

antes de executar o experimento.

Todos os resultados devem surgir exclusivamente da execução do código.

Se o dataset ainda não estiver disponível localmente:

1. prepare toda a infraestrutura;
2. indique onde colocá-lo;
3. implemente o carregamento;
4. documente o próximo passo;
5. não fabrique métricas.

---

# 39. Primeira etapa da implementação

Antes de escrever todo o código, faça o seguinte:

1. analise a estrutura do repositório atual;
2. identifique arquivos existentes;
3. não sobrescreva código útil sem necessidade;
4. proponha a estrutura final;
5. identifique o dataset e target;
6. apresente resumidamente o plano de implementação;
7. em seguida, implemente progressivamente o projeto.

Caso o repositório esteja vazio, crie a estrutura proposta.

---

# 40. Prioridades

Priorize, nesta ordem:

```text
1. Correção metodológica
2. Evitar data leakage
3. Reprodutibilidade
4. Avaliação adequada
5. Análise crítica
6. Organização do código
7. Visualização dos resultados
8. Otimização
```

O objetivo não é simplesmente obter a maior accuracy possível.

O objetivo é desenvolver um experimento de Machine Learning metodologicamente correto e conseguir discutir criticamente se o Random Forest é adequado para detectar situações de risco no contexto de veículos autônomos.
