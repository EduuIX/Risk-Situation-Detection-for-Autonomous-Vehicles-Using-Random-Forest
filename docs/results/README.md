# Evidências do relatório

Esta pasta preserva os gráficos e as métricas reais usados no relatório da raiz.
As métricas foram extraídas de `reports/metrics/`, sem novo treinamento.
Na revisão visual de 12/10/2026, os gráficos comparativos e de importância foram
redesenhados a partir dos CSV desta pasta. As curvas ROC/PR foram geradas dos modelos
locais salvos, cujas métricas de teste coincidem com as do snapshot. A revisão e as
versões desses modelos estão registradas separadamente em `figure_revision` no manifesto.
O [manifesto](manifest.json) registra a configuração, o hash dos dados, as versões e
os hashes dos arquivos copiados. Não contém dados brutos ou modelos serializados.

O relatório é estático: executar `src.train` ou `src.evaluate` atualiza os artefatos em
`reports/`, mas não substitui estas evidências nem os números e interpretações no README.
Ao documentar outro experimento, atualizar conjuntamente tabelas, texto, imagens e
manifesto, preservando a separação entre validação de treino e avaliação de teste.
