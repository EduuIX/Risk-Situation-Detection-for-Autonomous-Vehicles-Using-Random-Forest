# Evidências do relatório

Esta pasta preserva os gráficos e as métricas reais usados no relatório da raiz.
As cópias foram extraídas de `reports/figures/` e `reports/metrics/`, sem novo treinamento.
O [manifesto](manifest.json) registra a configuração, o hash dos dados, as versões e
os hashes dos arquivos copiados. Não contém dados brutos ou modelos serializados.

O relatório é estático: executar `src.train` ou `src.evaluate` atualiza os artefatos em
`reports/`, mas não substitui estas evidências nem os números e interpretações no README.
Ao documentar outro experimento, atualizar conjuntamente tabelas, texto, imagens e
manifesto, preservando a separação entre validação de treino e avaliação de teste.
