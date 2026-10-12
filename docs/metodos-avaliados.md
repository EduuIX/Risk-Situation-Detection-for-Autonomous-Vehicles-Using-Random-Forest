# Métodos avaliados no experimento

Este documento explica os cinco classificadores implementados em
[`src/train.py`](../src/train.py), suas configurações e os resultados da execução
disponível. Os parâmetros foram conferidos nos estimadores salvos em
`models/experiment.joblib`; os números vêm de `reports/metrics/cv_summary.csv`,
`reports/metrics/model_comparison.csv` e `reports/metrics/best_parameters.json`.
A execução utilizou scikit-learn 1.9.1. Novos treinamentos podem alterar os resultados.

## 1. Problema e protocolo comum

O objetivo é classificar registros do dataset POLIDriving em duas classes:

| Classe | Rótulos originais | Interpretação no experimento |
|---|---|---|
| 0 | 1 e 2: LOW e MEDIUM | Grupo de menor risco, chamado de baixo risco nas figuras |
| 1 | 3 e 4: HIGH e VERY HIGH | Grupo de alto risco; classe positiva das métricas |

O modelo aprende a reproduzir esses rótulos. Sua saída não é uma estimativa validada
da probabilidade de acidente.

### Dados de entrada e pré-processamento

São utilizados 14 atributos: horário de observação, velocidade, RPM, aceleração,
posição do acelerador, temperatura e carga do motor, frequência cardíaca, clima,
visibilidade, precipitação, acidentes no local, velocidade de projeto e acidentes
no período. O alvo `risk_level` e o índice do CSV não entram como preditores.

O mesmo pipeline antecede todos os estimadores:

1. Nos 13 atributos numéricos, valores ausentes são preenchidos pela mediana.
2. Em `current_weather`, atributo categórico, ausências são preenchidas pela moda.
3. O clima passa por one-hot encoding: cada categoria conhecida vira uma coluna
   indicadora. Categorias desconhecidas são toleradas com `handle_unknown='ignore'`.
4. Não há padronização numérica nesta execução. Árvores tomam decisões por limiares
   e não exigem que as variáveis tenham a mesma escala.

Imputação e codificação são ajustadas exclusivamente no treino de cada fold.
No ajuste final, são aprendidas nos 1.584 registros de treinamento. Embora não
existam ausências neste recorte, o pipeline mantém esse tratamento disponível.

### Separação e validação

Após a verificação de duplicatas exatas, os 1.980 registros foram separados de forma
estratificada, com `random_state=42`: 80% para treino e 20% para teste. Não foram
removidas duplicatas nesta execução.

| Subconjunto | Classe 0 | Classe 1 | Total |
|---|---:|---:|---:|
| Treino | 792 | 792 | 1.584 |
| Teste | 198 | 198 | 396 |

A validação usa `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` somente
no treino. Cada modelo é ajustado em quatro folds e avaliado no restante, repetindo
o procedimento cinco vezes. As mesmas divisões são usadas na comparação dos modelos.
São calculadas médias e desvios de acurácia, precisão, recall e F1.

Após a validação, os modelos são ajustados no treino completo. A escolha final é
feita **entre as três variantes de Random Forest**, pelo maior F1 médio de validação.
Dummy e árvore individual são referências de comparação. O teste não escolhe o vencedor.

## 2. Dummy Classifier: referência sem aprendizagem dos atributos

**Identificador:** `dummy`.

O `DummyClassifier(strategy='most_frequent', random_state=42)` sempre retorna a
classe mais frequente nos dados usados no ajuste. Ele ignora os atributos do veículo,
do motorista e do ambiente. Sua função é mostrar o resultado de uma regra trivial:
um método mais complexo precisa justificar seu uso em relação a essa referência.

No treino completo, as classes empatam. O estimador ajustado escolhe a classe 0 e
atribui probabilidade zero à classe 1 para todos os registros. Consequentemente,
acerta todos os 198 negativos do teste e perde todos os 198 positivos.

| Métrica no teste | Valor |
|---|---:|
| Acurácia | 0,5000 |
| Precisão / recall / F1 da classe 1 | 0,0000 |
| ROC-AUC | 0,5000 |
| Average Precision | 0,5000 |

A precisão fica em zero pela convenção `zero_division=0`, pois nenhum alerta é
emitido. A AUC de 0,5 decorre de pontuações constantes, sem discriminação entre classes.

Na validação, o F1 médio do Dummy é 0,2661, embora no teste seja zero. Isso ocorre
porque os folds não mantêm necessariamente um empate exato: algumas partições de
treinamento têm um registro a mais de uma classe. Nelas, a classe constante pode
ser 1, produzindo recall 1 e F1 positivo. Não há aprendizagem de risco nesse efeito.

## 3. Decision Tree: árvore de decisão individual

**Identificador:** `decision_tree`.

A árvore divide sucessivamente os registros por regras como “atributo menor ou
igual a um limiar”. Em cada nó, busca uma divisão que reduza a mistura das classes.
Uma folha reúne registros que passaram pelo mesmo conjunto de regras; a proporção
de exemplos de cada classe nessa folha fornece a probabilidade estimada.

O critério utilizado é a impureza de Gini:

`Gini = 1 − Σ p_k²`, em que `p_k` é a proporção da classe `k` no nó.

Um nó com apenas uma classe tem Gini zero. O algoritmo procura reduzir a impureza
ponderada dos nós filhos, mas toma decisões locais; não procura exaustivamente a
melhor árvore completa.

| Parâmetro | Valor | Efeito |
|---|---|---|
| `criterion` | `'gini'` | Critério de qualidade das divisões |
| `splitter` | `'best'` | Busca a melhor divisão candidata no nó |
| `max_depth` | `None` | Sem limite explícito de profundidade |
| `min_samples_split` | 2 | Mínimo de registros para tentar dividir um nó |
| `min_samples_leaf` | 1 | Permite folhas com um único registro |
| `max_features` | `None` | Considera todos os atributos em cada divisão |
| `class_weight` | `None` | Sem ponderação adicional por classe |
| `ccp_alpha` | 0,0 | Sem poda por penalização de complexidade |
| `random_state` | 42 | Controla aleatoriedade interna e desempates |

A árvore é uma referência para avaliar o benefício de combinar várias árvores.
É mais fácil inspecionar suas regras, mas uma árvore profunda pode se adaptar a
particularidades do treino e mudar bastante com pequenas alterações dos dados.

No teste, obteve acurácia 0,9268, precisão 0,9122, recall 0,9444 e F1 0,9280,
com 18 falsos positivos e 11 falsos negativos. A ROC-AUC foi 0,9268. A proximidade
ou igualdade entre AUC e acurácia aqui não é uma identidade geral dessas métricas.

## 4. Random Forest padrão: combinação de árvores aleatorizadas

**Identificador:** `random_forest`.

O Random Forest combina árvores treinadas com fontes de aleatoriedade para reduzir
a dependência de uma única árvore. Nesta configuração, cada árvore recebe uma
amostra bootstrap: sorteiam-se registros do treino com reposição, permitindo que
um registro apareça várias vezes e que outros não apareçam naquela árvore.

Além disso, cada divisão considera um subconjunto aleatório dos atributos. Isso
ajuda a diversificar as árvores. A probabilidade final é a **média das probabilidades
das árvores**, e não necessariamente a fração de árvores que deram um voto positivo.

| Parâmetro | Valor | Efeito |
|---|---|---|
| `n_estimators` | 100 | Número de árvores |
| `criterion` | `'gini'` | Critério de divisão de cada árvore |
| `bootstrap` | `True` | Usa amostras com reposição |
| `max_samples` | `None` | Cada bootstrap tem o tamanho do treino recebido |
| `max_features` | `'sqrt'` | Considera aproximadamente a raiz quadrada do número de atributos transformados |
| `max_depth` | `None` | Sem limite explícito de profundidade |
| `min_samples_split` | 2 | Permite dividir nós pequenos |
| `min_samples_leaf` | 1 | Permite folhas unitárias |
| `class_weight` | `None` | Sem pesos adicionais de classe |
| `random_state` | 42 | Semente da floresta |
| `n_jobs` | 1 | Execução das árvores em um processo de trabalho |
| `oob_score` | `False` | Não usa avaliação out-of-bag |

O número de atributos considerado por `max_features` se refere à matriz **após**
o one-hot encoding, não necessariamente aos 14 atributos originais.

No teste, o RF padrão obteve acurácia 0,9394, precisão 0,9350, recall 0,9444,
F1 0,9397 e ROC-AUC 0,9865. Foram 13 falsos positivos e 11 falsos negativos.
Em relação à árvore individual, reduziu cinco falsos positivos, mantendo o número
de falsos negativos. Seu F1 médio de validação, 0,9530, foi o maior entre os RF;
por isso, **este foi o modelo selecionado**.

Combinar árvores reduz variabilidade, mas não elimina sobreajuste nem garante
generalização para outras cidades, veículos ou condições de coleta.

## 5. Random Forest balanced: ponderação automática das classes

**Identificador:** `random_forest_balanced`.

Usa a mesma configuração do RF padrão, alterando apenas `class_weight='balanced'`.
A intenção da ponderação é aumentar a influência de classes menos frequentes.
O peso da classe `k` é calculado no conjunto recebido pelo ajuste:

`w_k = N / (K × n_k)`

Aqui, `N` é o total de registros, `K` é o número de classes e `n_k` é a quantidade
de registros da classe. Não se trata de criar novos exemplos ou remover registros.
Também não é `balanced_subsample`, que recalcula pesos por amostra bootstrap.

No ajuste final deste experimento:

`w_0 = w_1 = 1584 / (2 × 792) = 1`.

Logo, nenhuma classe recebe maior peso que a outra. Mesmo assim, os resultados
podem diferir por um detalhe confirmado no scikit-learn 1.9.1 instalado: o bootstrap
sem pesos usa `RandomState.randint`, enquanto pesos explícitos usam
`RandomState.choice` com probabilidades normalizadas. Uma distribuição uniforme
não obriga esses dois algoritmos a produzir a mesma sequência com a mesma semente.

O retreinamento controlado do RF padrão com pesos unitários reproduziu exatamente
as probabilidades do RF balanced no teste. A diferença observada não é evidência
de ganho por corrigir um desbalanceamento no treino completo. Nos folds, pequenas
diferenças nas contagens também podem produzir pesos ligeiramente diferentes de 1.
Consulte a [verificação específica](rf-balanced-e-roc.md).

No teste, obteve acurácia 0,9444, precisão 0,9356, recall 0,9545, F1 0,9450 e
ROC-AUC 0,9839. Foram 13 falsos positivos e 9 falsos negativos. O F1 no limiar
adotado foi maior que o do padrão, mas a AUC foi menor: desempenho em um limiar
específico e ordenação das probabilidades são aspectos diferentes.

O F1 médio de validação foi 0,9464, inferior ao do padrão. Portanto, o resultado
posterior no teste não foi usado para substituir o modelo selecionado.

## 6. Random Forest tuned: busca de hiperparâmetros

**Identificador:** `random_forest_tuned`.

Não é uma nova família de algoritmos: é um Random Forest cuja configuração foi
escolhida por `RandomizedSearchCV`. Hiperparâmetros controlam a construção do modelo;
são diferentes dos limiares das árvores, aprendidos diretamente durante o ajuste.

A busca avaliou 20 configurações, com semente 42, usando os mesmos cinco folds.
Isso corresponde a 100 ajustes de candidatos nos folds, além do ajuste final da
configuração vencedora. O espaço contém 1.296 combinações possíveis; portanto,
não houve busca exaustiva.

| Hiperparâmetro | Valores candidatos |
|---|---|
| `n_estimators` | 100, 200, 400 |
| `max_depth` | `None`, 5, 10, 20 |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |
| `max_features` | `'sqrt'`, `'log2'`, 0,8 |
| `class_weight` | `None`, `'balanced'` |
| `bootstrap` | `True`, `False` |

Mais árvores aumentam o custo de treinamento e predição. Limitar profundidade ou
aumentar o tamanho mínimo das folhas restringe a complexidade. `max_features`
controla a quantidade de atributos candidatos em cada divisão; 0,8 representa uma
fração dos atributos transformados. `bootstrap` determina se cada árvore recebe
uma amostra com reposição ou o conjunto de treinamento completo.

A busca calcula acurácia, precisão, recall e F1, mas escolhe exclusivamente pelo
**F1 médio da classe 1**, com `refit='f1'`. A configuração vencedora foi:

| Hiperparâmetro | Valor escolhido |
|---|---|
| `n_estimators` | 400 |
| `max_depth` | `None` |
| `min_samples_split` | 5 |
| `min_samples_leaf` | 2 |
| `max_features` | `'sqrt'` |
| `class_weight` | `'balanced'` |
| `bootstrap` | `False` |

Os demais parâmetros permanecem nos valores do estimador base, incluindo
`criterion='gini'`, `random_state=42` e `n_jobs=1`. Sem bootstrap, todas as árvores
recebem o treino completo; a seleção aleatória de atributos ainda gera diversidade.
Os limites mínimos de divisão e folha são mais restritivos que no RF padrão.

O F1 médio de validação foi 0,9526, ligeiramente inferior ao 0,9530 do padrão.
Ser vencedor **dentro da busca** não significa vencer todas as variantes externas
comparadas pelo experimento.

No teste, teve acurácia 0,9394, precisão 0,9350, recall 0,9444 e F1 0,9397,
com 13 falsos positivos e 11 falsos negativos. Esses totais são iguais aos do RF
padrão, mas não demonstram, isoladamente, igualdade de decisões registro a registro.
A ROC-AUC foi 0,9868, indicando que as pontuações não precisam ser iguais para
produzir as mesmas métricas no limiar 0,5.

O score da configuração vencedora reutiliza os folds que orientaram a busca e
pode ser otimista. Não foi realizada validação cruzada aninhada. Tampouco há
evidência de ganho estatisticamente significativo apenas por diferenças pontuais.

## 7. Como interpretar as métricas

A avaliação final em [`src/evaluate.py`](../src/evaluate.py) converte probabilidades
em classe 1 quando `p >= 0,5`. A validação cruzada usa o `predict` dos estimadores,
cujo desempate pode favorecer a classe 0 quando `p = 0,5`. Essa diferença de
convenção deve ser considerada ao comparar validação e teste.

| Medida | Definição e interpretação |
|---|---|
| TP | Registro de alto risco corretamente sinalizado |
| TN | Registro da classe 0 corretamente classificado |
| FP | Alerta emitido para um registro da classe 0 |
| FN | Registro de alto risco que passou sem alerta |
| Acurácia | `(TP + TN) / N`: proporção total de acertos |
| Precisão | `TP / (TP + FP)`: proporção de alertas que correspondem à classe 1 |
| Recall | `TP / (TP + FN)`: proporção dos positivos detectados |
| F1 | `2 × precisão × recall / (precisão + recall)`: média harmônica das duas medidas |
| ROC-AUC | Área sob a curva de recall contra taxa de falsos positivos, variando o limiar |
| Average Precision (AP) | Resumo de precisão e recall ao variar o limiar, ponderando precisão pelos incrementos de recall |

A taxa de falsos positivos da ROC é `FP / (FP + TN)`. ROC-AUC e AP usam as
probabilidades, sem fixar o limiar em 0,5. AP não deve ser confundida com a
integração trapezoidal da curva precisão–recall. Uma pontuação constante tem AP
igual à prevalência positiva, aqui 0,5.

### Comparação consolidada no teste

| Método | Acurácia | Precisão | Recall | F1 | ROC-AUC | AP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy | 0,5000 | 0,0000 | 0,0000 | 0,0000 | 0,5000 | 0,5000 | 0 | 198 |
| Decision Tree | 0,9268 | 0,9122 | 0,9444 | 0,9280 | 0,9268 | 0,8893 | 18 | 11 |
| RF padrão | 0,9394 | 0,9350 | 0,9444 | 0,9397 | 0,9865 | 0,9850 | 13 | 11 |
| RF balanced | 0,9444 | 0,9356 | 0,9545 | 0,9450 | 0,9839 | 0,9814 | 13 | 9 |
| RF tuned | 0,9394 | 0,9350 | 0,9444 | 0,9397 | 0,9868 | 0,9857 | 13 | 11 |

### Critério usado para selecionar o RF

| Método | F1 médio na validação | Desvio entre folds |
|---|---:|---:|
| Dummy | 0,2661 | 0,3259 |
| Decision Tree | 0,9357 | 0,0029 |
| RF padrão | 0,9530 | 0,0102 |
| RF balanced | 0,9464 | 0,0112 |
| RF tuned | 0,9526 | 0,0070 |

Desvios entre folds descrevem variação das pontuações; não são intervalos de
confiança nem testes de significância. A comparação tuned também tem o viés de
seleção descrito anteriormente.

## 8. Análises complementares

Estas análises ajudam a interpretar os classificadores; não constituem modelos adicionais.

### Variação do limiar

São examinados limiares 0,3, 0,4, 0,5, 0,6 e 0,7 com probabilidades fora do fold
(OOF) do RF selecionado, apenas no treino. Cada registro recebe uma predição de
um modelo que não foi ajustado naquele registro. Diminuir o limiar tende a emitir
mais alertas e aumentar recall, podendo aumentar falsos positivos.

O procedimento é exploratório porque a configuração já foi selecionada usando
o treino. O limiar final permaneceu 0,5; não foram definidos custos que permitissem
escolher um limiar operacional ótimo.

### Importância por impureza

Resume a redução de impureza atribuída aos atributos nas divisões da floresta.
É calculada para o RF selecionado, sobre as colunas transformadas, incluindo as
indicadoras de clima. Pode favorecer atributos com muitas divisões possíveis;
não mede causalidade nem comprova significância estatística.

### Importância por permutação

Embaralha cada atributo original no teste e mede a redução de F1 do modelo já
treinado, com dez repetições e semente 42. Uma queda maior sugere maior dependência
do modelo em relação àquele atributo nesse conjunto. Atributos correlacionados
podem compensar uns aos outros, reduzindo a importância individual medida.

É uma interpretação posterior: seus resultados não foram usados para selecionar
atributos ou reajustar o modelo. O scorer utiliza `predict`, com a convenção de
desempate já mencionada.

### Curvas ROC e precisão–recall

A curva ROC mostra o compromisso entre detecção de positivos e falsos alarmes.
A curva precisão–recall mostra o compromisso entre a confiabilidade dos alertas
e a cobertura dos positivos. O gráfico de barras de AUC resume uma área e não
substitui a curva propriamente dita.

As figuras geradas ficam em `reports/figures/`: `06_roc_curve.png` para o modelo
selecionado, `06_roc_comparison.png` e `.pdf` para a comparação de modelos e
`07_precision_recall_curve.png` para a curva PR do selecionado.

## 9. Alcance dos resultados

O experimento avalia um dataset offline, pequeno e previamente preparado. A divisão
estratificada preserva proporções de classes, mas não garante independência temporal,
por motorista ou por viagem. O CSV não permite excluir dependências herdadas da
preparação pública dos dados, inclusive possível reamostragem anterior.

Rótulos construídos por regras podem ser recuperados pelos modelos sem representar
capacidade de antecipar acidentes reais. Probabilidades não foram calibradas, e o
balanceamento de 50%/50% não estima a frequência real de situações de alto risco.

Assim, os resultados sustentam uma comparação acadêmica dos métodos neste recorte.
Aplicações práticas exigiriam validação externa, dados independentes e avaliação
dos custos de falsos positivos e falsos negativos. Mais detalhes do protocolo estão
em [methodology.md](methodology.md) e do fluxo de execução em [pipeline.md](pipeline.md).
