# Apresentação das figuras

As figuras principais para o relatório estão em [`reports/figures/`](../reports/figures/).
Essa é a pasta atualizada pelo pipeline e utilizada para revisar e exportar os gráficos.

A revisão adota formatos adequados à informação apresentada. Não há um único
“estilo state-of-the-art”: a escolha depende da pergunta, do protocolo experimental
 e das exigências do periódico.

| Informação | Representação | Arquivo em `reports/figures/` |
|---|---|---|
| Discriminação em vários limiares | Curvas ROC com AUC e diagonal de referência | `06_roc_comparison` |
| Precisão e cobertura dos positivos | Curvas precisão–recall com AP e prevalência | `07_precision_recall_comparison` |
| Comparação compacta no teste | Matriz de métricas com valores anotados | `11_model_comparison` |
| Uma métrica de teste | Pontos por modelo, sem conectar categorias | `11_model_comparison_<métrica>` |
| Variação entre folds | Cinco pontos por modelo e traço da média | `13_cv_comparison` |
| Importância por impureza | Pontos ordenados com hastes | `08_feature_importance` |
| Importância por permutação | Pontos com média ± desvio das permutações | `09_permutation_importance` |

Todas essas figuras são exportadas em PNG a 300 dpi e PDF vetorial. As curvas e os
pontos usam cores consistentes entre modelos; as curvas também diferem pelo traçado.
As matrizes de confusão e de correlação continuam apropriadas como mapas de calor.
Barras de contagem e histogramas da análise exploratória continuam adequados às
respectivas distribuições.

Os cinco folds são mostrados individualmente porque há poucas observações para
caracterizar uma distribuição. Não foram adicionados intervalos de confiança ao
teste nem bandas às curvas: os artefatos disponíveis não contêm essa estimativa.
O desvio das permutações não é intervalo de confiança. A CV tuned é de seleção e
não aninhada. Nenhum modelo foi retreinado para a revisão visual.

A matriz de métricas usa escala comum de 0 a 1. Valores próximos devem ser comparados
pelas anotações e pela tabela, não apenas pela intensidade de cor. Os gráficos
individuais também mantêm a escala completa, evitando exagerar pequenas diferenças.

Referências técnicas para as representações de curvas:
[ROC e validação cruzada no scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html)
e [precisão–recall no scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html).
A primeira referência ilustra variabilidade quando há curvas por fold; esse tipo de
banda não foi aplicado às curvas de teste deste projeto.

`src/plots.py` centraliza os desenhos e `python -m src.evaluate` atualiza as figuras
em `reports/figures/` a partir dos modelos salvos. O snapshot em `docs/results/`
continua estático. Seu manifesto registra a revisão visual e a origem das curvas.
