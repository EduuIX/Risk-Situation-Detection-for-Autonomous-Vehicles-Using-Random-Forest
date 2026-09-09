# Protocolo e análise crítica

Para acompanhar a execução no código, consulte o [fluxo detalhado do pipeline](pipeline.md),
com diagrama, funções, comandos e artefatos de cada etapa.

1. Validar esquema, valores inválidos, ausências, duplicatas e target; registrar hash.
2. Remover somente duplicatas exatas; preservar índices originais para rastrear o split.
3. Separar 80%/20%, seed 42, estratificando o target binário.
4. Fazer EDA do treino; nenhuma seleção de atributos usa o teste.
5. Comparar Dummy, Decision Tree, RF padrão e RF balanced em StratifiedKFold (5 folds).
6. Buscar 20 configurações de RF com RandomizedSearchCV e refit por F1 da classe 1.
7. Selecionar entre os RF pelo F1 médio de treino; empates favorecem o primeiro (padrão).
8. Explorar thresholds com predições fora do fold no treino; manter 0,5 no modelo final.
9. Congelar modelos; comparar treino/teste e produzir interpretações finais.

A mediana numérica e a moda categórica são aprendidas no Pipeline de cada fold. One-hot
aceita categorias desconhecidas. Colunas inteiramente ausentes no treino são preservadas
com preenchimento de fallback do SimpleImputer. RF não precisa de normalização:
árvores dividem atributos por limiares. `make_preprocessor(scale=True)` permite testar
outros modelos com StandardScaler dentro do mesmo protocolo.

F1 equilibra precision e recall sem premiar uma regra que apenas alerte sempre.
Falsos negativos são riscos rotulados sem alerta; falsos positivos são alertas em
situações de risco baixo/médio. Custos operacionais não foram definidos, portanto não
há justificativa para otimizar uma função monetária ou escolher um threshold de produção.
Probabilidades da floresta não foram calibradas nem representam probabilidade de acidente.
Na inferência adotamos explicitamente classe 1 quando `p >= 0.5`, inclusive nos empates.
As métricas padrão de CV do scikit-learn usam `predict` (empate pode favorecer classe 0).

O teste serve para a avaliação final dos modelos definidos previamente. Mesmo se outra
variante tiver resultado melhor no teste, ela não substitui o modelo selecionado pelo
treino. Permutation importance no teste é somente interpretação posterior, sem seleção
de features ou novo treinamento a partir dela.

A CV do modelo tuned corresponde à configuração vencedora da busca e tende a ser
otimista; não é CV aninhada. A análise OOF de thresholds também reutiliza a configuração
selecionada no treino e tem caráter exploratório. O teste permanece fora de ambas.
Médias/desvios dos folds não são intervalos de confiança nem testes de significância.
Repetir escolhas depois de ver o teste prejudicaria sua independência.

## Hipóteses

- H1: RF distingue os rótulos de risco melhor que o baseline; isso não equivale a prever acidentes.
- H2: RF supera uma árvore individual; comparar F1, recall e estabilidade.
- H3: tuning melhora generalização; verificar diferenças sem presumir ganho.
- H4: atributos têm importâncias diferentes; rankings não provam significância ou causalidade.
- H5: accuracy isolada pode ocultar FN; confrontar confusion matrix, recall e thresholds.

`reports/conclusion.md` confronta essas hipóteses com números produzidos pelo experimento.
Diferenças de accuracy, F1 e recall entre treino e teste ajudam a investigar sobreajuste.
Curvas PR enfatizam o desempenho na classe positiva; sua referência depende da prevalência,
o que é especialmente útil em bases desbalanceadas. Importância por impureza favorece
atributos com muitas divisões possíveis; permutação pode diluir efeitos de atributos correlacionados.

## Limitações práticas

O recorte é pequeno, já preparado, com pouca diversidade geográfica. Sensores podem
ter ruído ou ausências; novos climas, cidades e veículos podem mudar a distribuição.
O CSV não contém câmera/LiDAR nem permite modelar adequadamente dependências temporais.
O uso de dados do motorista também limita a transferência a veículos sem condutor.
Rótulos definidos por regras podem ser reconstruídos a partir dos mesmos atributos,
produzindo bom desempenho sem aprender risco real. Não há validação de segurança
funcional, avaliação em malha fechada ou teste prospectivo. O artefato é acadêmico e
não deve ser utilizado diretamente como componente de um veículo real.
