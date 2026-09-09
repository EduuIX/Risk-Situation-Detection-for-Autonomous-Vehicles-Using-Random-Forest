# Dataset e contrato dos dados

Fonte primária: [POLIDriving — repositório dos autores](https://github.com/laboratorioAI/polidriving).
Publicação: [Marcillo et al. (2024), Applied Sciences 14(14), 6300](https://doi.org/10.3390/app14146300).
Disponível também como [PDF local](polidriving-2024.pdf), com
[referência bibliográfica, BibTeX e licença](artigo-base.md).
Consulte a licença MIT do repositório original; a licença deste projeto não substitui a atribuição dos dados.

Usamos exclusivamente `labeled/20240208_120000_vld.csv`, descrito pelos autores como
validado por especialista, fixado no commit `47238f96a18cb58733353cb647a0af99d4b4f0b5`.
SHA-256: `c57a6e89f07fb7d38eb9055fc3bd7e172f736f8ab251713c5da4a2d663d38eca`.
O download verifica esse hash e não sobrescreve um arquivo local diferente.

A inspeção do arquivo confirmou **1.980 linhas, 14 preditores, um target e uma coluna
extra de índice serializado**, que é descartada. Os 32 atributos descritos para a
base completa não estão todos neste recorte. A contagem é recalculada a cada execução.

| Coluna no CSV | Significado | Unidade/tipo |
|---|---|---|
| observation_hour | Hora da observação | Hora numérica |
| speed | Velocidade | km/h |
| rpm | Rotação do motor | rpm |
| acceleration | Aceleração | m/s² |
| throttle_position | Posição do acelerador | % |
| engine_temperature | Temperatura do motor | °C |
| engine_load_value | Carga do motor | % |
| heart_rate | Frequência cardíaca do motorista | bpm |
| current_weather | Código de condição meteorológica | Categórica |
| visibility | Visibilidade | km |
| precipitation | Precipitação | mm |
| accidents_onsite | Indicador histórico de acidentes no local | Origem: estatísticas de mortes |
| design_speed | Velocidade de projeto da via | km/h |
| accidents_time | Indicador histórico de acidentes por horário | Origem: estatísticas de mortes |
| risk_level | Nível original de risco | 1 a 4 |

Os indicadores de acidentes vêm da documentação dos autores; não os interpretamos
como acidentes futuros observados ou contagens brutas inteiras. O arquivo publicado
contém valores fracionários. A disponibilidade dos indicadores no instante de
inferência precisaria ser auditada em uma aplicação prospectiva.

`current_weather` usa OneHotEncoder sem impor ordem. Arredondamos seus códigos a seis
casas decimais para unir ruídos como `8.999999999999998` e `9.0`; valores realmente
fracionários continuam categorias distintas. Não inventamos a descrição de cada código.
A origem desses valores fracionários não pode ser determinada por linha no CSV.

| Nível original | Interpretação adotada | Target binário | Registros |
|---|---|---|---|
| 1 / LOW | Baixo | 0 | 495 |
| 2 / MEDIUM | Médio | 0 | 495 |
| 3 / HIGH | Alto | 1 | 495 |
| 4 / VERY HIGH | Muito alto | 1 | 495 |

A interpretação dos códigos segue a ordenação dos níveis descrita pelos autores e é
explicitada também na [competição derivada](https://www.kaggle.com/competitions/distracted-driving-risk-detection-challenge).
Cada classe original representa 25%; cada classe binária tem 990 registros (50%).
Agrupar baixo e médio permite investigar risco elevado; **classe 0 não garante segurança**.
Um CSV binário 0/1 não é aceito como se já contivesse os níveis originais.

## Proveniência e validade

O estudo descreve construção dos rótulos por penalidades/regras, inclusão de exemplos
sintéticos e SMOTE no processo mais amplo de preparação. O arquivo selecionado não
permite rastrear quais linhas são naturais, sintéticas ou reamostradas. A distribuição
balanceada não pode ser interpretada como frequência natural de situações perigosas.
Não aplicamos SMOTE adicional. Duplicatas exatas são relatadas e removidas antes do split;
outliers são apenas descritos, pois podem ser justamente os eventos relevantes.

Não há identificadores de motorista/viagem nem timestamp completo neste CSV. Portanto,
o split aleatório solicitado é um **benchmark exploratório**, sem garantia de
independência entre trajetórias ou exemplos relacionados por geração sintética.
Mesmo mantendo todos os transformadores dentro dos folds, não podemos excluir
contaminação herdada da preparação upstream. Uma validação mais forte exigiria dados
originais com proveniência e separação por viagem/motorista, antes de qualquer reamostragem.
