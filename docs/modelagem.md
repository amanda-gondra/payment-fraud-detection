# Modelagem

Como o modelo é treinado, calibrado, transformado em decisão, explicado, auditado e salvo. As escolhas e as alternativas descartadas estão nas decisões D12 a D20 de [`decisoes-tecnicas.md`](decisoes-tecnicas.md). Os números saem em dois relatórios:

- [`reports/model-selection.md`](../reports/model-selection.md): o desenvolvimento, gerado antes de abrir o teste;
- [`reports/test-evaluation.md`](../reports/test-evaluation.md): a avaliação única no teste.

## Fluxo

| Comando | O que faz | Lê | Grava |
| --- | --- | --- | --- |
| `python -m pipeline.train` | Treina, compara, calibra, escolhe os limites, explica e audita usando só treino e validação | `data/processed/` (assinaturas conferidas) | `models/`, `mlruns/`, `reports/model-selection.md`, `reports/figures/` |
| `python -m pipeline.evaluate` | Avalia uma única vez o modelo congelado no `fraudTest.csv` | `models/` (assinaturas conferidas), `data/raw/fraudTest.csv` | `data/processed/test.parquet`, `models/test-evaluation.json`, `mlruns/`, `reports/test-evaluation.md` |

O código fica em `pipeline/modeling/`, um módulo por responsabilidade:

| Módulo | Responsabilidade |
| --- | --- |
| `data.py` | Conjuntos verificados, corte interno da validação e a matriz do modelo (`FeatureSpec`) |
| `baselines.py` | Regra do valor e regressão logística |
| `lgbm.py` | LightGBM: parâmetros, busca e treino com parada antecipada |
| `calibration.py` | Calibradores (nenhum, Platt, isotônico) e a validação cruzada por blocos de tempo |
| `decision.py` | Custos, os dois limites, a busca do menor custo e o resumo de cada estratégia |
| `metrics.py`, `evaluation.py` | Métricas, faixas de calibração e resultados por mês e por categoria |
| `explain.py` | Valores SHAP, importância global, gráfico e motivos em texto |
| `fairness.py` | Auditoria por gênero e faixa etária, com bootstrap por cartão |
| `artifacts.py` | Pacotes assinados, carregamento seguro e a resposta para cada transação |
| `tracking.py` | MLflow local, sem telemetria, e a versão do código |
| `plots.py`, `reporting.py`, `report_selection.py`, `report_test.py` | Figuras e relatórios |

## Conjuntos

Tudo vem do `fraudTrain.csv`, em ordem de tempo:

| Conjunto | Período | Uso |
| --- | --- | --- |
| Treino | 01/01/2019 a 31/12/2019 | Ajuste de todos os modelos |
| Seleção | 01/01/2020 a 31/03/2020 | Última janela da validação cruzada e parada antecipada do modelo final |
| Calibração | 01/04/2020 a 21/06/2020 | Calibradores, comparação dos modelos e limites de decisão |
| Teste | `fraudTest.csv`, 21/06/2020 a 31/12/2020 | Avaliação única, no fim |

A seleção e a calibração são as duas partes da validação da fase 3 (D9). A divisão interna separa quem escolhe o modelo de quem mede as probabilidades e escolhe os limites (D12).

## Modelos

- **Regra do valor:** alerta quando o valor passa de X. X é o corte de maior F1 no treino.
- **Regressão logística:** os sinais recebem tratamentos diferentes antes do modelo:
  - log nos sinais de cauda longa;
  - mediana com indicador para os ausentes;
  - padronização;
  - one-hot de categoria, hora e dia da semana;
  - regularização L2.

  A busca testa seis combinações de C e de peso das classes.
- **LightGBM**, com o sobreajuste controlado de forma explícita (D14):
  - **Parada:** pelo log loss dos dados de validação.
  - **Busca:** oito combinações de regularização (folhas, mínimo de transações por folha e L2).
  - **Validação cruzada temporal:** quatro janelas; cada uma treina com os meses anteriores e é medida nos seguintes.
  - **Escolha:** regra de um erro padrão; entre as configurações empatadas, vence a mais simples.
  - **Fora:** não usa reamostragem nem peso extra para as fraudes.

O relatório de seleção mostra a curva de aprendizado e a razão entre o log loss em dados novos e no treino: quanto mais perto de 1, menos o modelo decorou.

As referências são escolhidas pela PR-AUC (average precision) no período de seleção. A acurácia não é usada, porque aprovar tudo acerta 99,4%.

## Calibração

O modelo dá um escore em log-odds. Três leituras são comparadas: a probabilidade original do modelo, Platt e isotônica. Cada uma é avaliada por validação cruzada no período de calibração:
- o período é cortado em cinco blocos consecutivos;
- cada bloco é previsto por um calibrador ajustado nos outros quatro.

Vence o menor log loss, com Brier e erro de calibração por faixa (ECE) como confirmação. O vencedor é então ajustado no período inteiro (D15). Os calibradores são salvos como números em JSON:
- o Platt guarda dois coeficientes;
- a isotônica guarda os pontos da função.

## Decisão em três faixas

Sejam p a probabilidade calibrada, t₁ o limite de revisão e t₂ o limite de bloqueio:

- **Aprovar:** p < t₁;
- **Revisão humana:** t₁ ≤ p < t₂;
- **Bloquear:** p ≥ t₂.

Os dois limites minimizam o custo total no período de calibração e respeitam a capacidade da equipe de revisão. A busca usa as probabilidades da validação cruzada e testa todos os cortes possíveis entre probabilidades diferentes. Quando há cortes demais, mantém todos os da parte mais alta da escala, onde as decisões são tomadas (D16).

As hipóteses de custo, na unidade dos valores da base:

| Situação | Custo | Tipo |
| --- | --- | --- |
| Fraude aprovada | valor da transação | financeiro |
| Fraude bloqueada | 0 (a perda é evitada) | — |
| Fraude em revisão | 5 + 10% do valor (a revisão detecta 90%) | operacional e financeiro |
| Legítimo em revisão | 5 + 2 | operacional e experiência do cliente |
| Legítimo bloqueado | 25 | experiência do cliente e comercial |
| Legítimo aprovado | 0 | — |

Outras restrições e verificações:
- **Capacidade de revisão:** no máximo 1% das transações.
- **Sensibilidade:** a busca é repetida mudando uma hipótese de cada vez, para mostrar quanto os limites e o custo dependem delas.

## Como os motivos são escritos

Para cada transação, o LightGBM calcula os valores SHAP (TreeSHAP, exatos para árvores) dos 28 sinais. A soma desses valores com o valor esperado dá o escore do modelo em log-odds. Os testes conferem essa soma e comparam os valores com um cálculo de Shapley por força bruta. A calibração é crescente, então um sinal que aumenta o escore também aumenta a probabilidade calibrada.

Os motivos são os sinais com as maiores contribuições em valor absoluto: quatro na resposta do modelo e cinco nos relatórios. Contribuições iguais a zero não viram motivo. Cada motivo tem duas partes:

1. **O fato**, escrito a partir do valor real do sinal naquela transação;
2. **O efeito no modelo**: "aumentou o risco" se a contribuição é positiva, "reduziu o risco" se é negativa.

| Sinal | Frase | Sem valor |
| --- | --- | --- |
| `amount` | Valor da compra: 950,00 | — |
| `category` | Categoria do lojista: shopping_net | — |
| `hour` | Compra às 23h | — |
| `day_of_week` | Compra num sábado | — |
| `is_late_night` | Compra na madrugada (22h às 3h59) / fora da madrugada | — |
| `is_weekend` | Compra no fim de semana / em dia útil | — |
| `distance_km` | 120 km entre o endereço do cliente e o lojista | Distância indisponível (coordenada inválida) |
| `tx_count_<janela>` | 3 compras do cartão na última hora | — |
| `amount_sum_<janela>` | Soma das compras do cartão nas últimas 24 horas: 1.200,00 | — |
| `amount_mean_<janela>` | Média das compras do cartão nos últimos 7 dias: 85,30 | Nenhuma compra do cartão nos últimos 7 dias, sem média para comparar |
| `has_history` | O cartão tem compras anteriores / Primeira compra registrada do cartão | — |
| `history_mean_amount` | Média das compras anteriores do cartão: 64,10 | Sem compras anteriores do cartão |
| `amount_to_history_mean` | Valor 6,1 vezes a média das compras anteriores do cartão | Sem compras anteriores para comparar o valor |
| `amount_minus_history_mean` | Valor 1.234,50 acima da média das compras anteriores do cartão | Sem compras anteriores para comparar o valor |
| `amount_log_zscore` | Valor 3,2 desvios padrão acima do padrão de valores do cartão | Menos de 10 compras anteriores: desvio não calculado |
| `seconds_since_previous` | 90 segundos desde a compra anterior do cartão | Sem compra anterior do cartão |
| `is_new_category` | Primeira compra do cartão nesta categoria / O cartão já comprou nesta categoria | — |
| `history_share_same_hour` | 25% das compras anteriores do cartão foram nesta hora do dia | Sem compras anteriores do cartão |
| `history_share_late_night` | Nenhuma compra anterior do cartão na madrugada | Sem compras anteriores do cartão |

Limites que valem para todas as explicações:
- **Não é causa:** os motivos descrevem o que pesou no modelo, não o que causou a fraude.
- **Crédito dividido:** sinais correlacionados, como o valor e a razão entre o valor e a média do cartão, dividem o crédito entre si.
- **Escala:** a contribuição está em log-odds, não em pontos percentuais de probabilidade.

## Auditoria de equidade

Gênero e idade nunca entram na matriz do modelo:
- a `FeatureSpec` só lê os 28 sinais, mesmo quando a tabela traz outras colunas;
- no treino e na validação, esses atributos ficam no arquivo de auditoria; no teste, só em memória.

Depois das decisões, eles são juntados para comparar, por grupo, as seguintes taxas:
- falsos positivos;
- fraudes não detectadas;
- precisão e recall;
- legítimos bloqueados;
- envio para revisão.

**Faixas de idade:** até 29 anos, 30 a 44 anos, 45 a 59 anos e 60 anos ou mais.

**Intervalos de 95%:** saem de bootstrap por cartão, ou seja, sorteando clientes inteiros, porque as transações de um mesmo cliente não são independentes.

**Como ler os resultados:**
- Uma diferença cujo intervalo exclui zero é indício para investigar, não prova de discriminação.
- Grupos com menos de 30 fraudes recebem um aviso de amostra pequena.
- Os dados são simulados, então os resultados não valem automaticamente para pessoas reais (D20).

## Artefatos e integridade

`models/fraud-detector/` guarda tudo o que a inferência precisa, sem pickle:

| Arquivo | Conteúdo |
| --- | --- |
| `model.txt` | Modelo no formato de texto do próprio LightGBM: árvores e números, sem código |
| `features.json` | Ordem das 28 colunas e o código de cada categoria |
| `calibrator.json` | Método de calibração e seus números |
| `decision.json` | Limites, nomes das decisões e hipóteses de custo |
| `metadata.json` | Versão, assinaturas dos dados, parâmetros, bibliotecas, commit e versão do código, execução do MLflow |
| `manifest.json` | SHA-256 de cada arquivo e do pacote inteiro |

`models/baselines/` guarda a regra do valor e a regressão logística, com seu calibrador e seus limites. A regressão logística é salva com skops e só é carregada se todos os tipos estiverem numa lista explícita de confiança.

O carregamento (`FraudModel.load`) segue cinco passos:
1. lê cada arquivo uma única vez;
2. confere o SHA-256 dos bytes lidos antes de interpretá-los;
3. recusa arquivo a mais, a menos ou alterado;
4. confere que as colunas do modelo são as de `features.json` e que o modelo lido reproduz `model.txt`;
5. verifica as assinaturas de novo no fim.

Nenhum segredo fica nos artefatos: o modelo não usa `card_key`, e a chave do pseudônimo fica só no `.env`. A probabilidade devolvida em cada resposta é arredondada a duas casas, para dificultar a sondagem fina dos limites.

## Uso único do teste

O `pipeline.evaluate` grava `models/test-evaluation.json` antes de abrir o teste, e esse registro impõe duas regras:

1. **Avaliação concluída:** ele se recusa a rodar de novo.
2. **Execução interrompida:** ele só aceita repetir com o mesmo pacote de modelo, conferido pela assinatura. Assim, nada ajustado depois de ver o teste pode ser avaliado nele.

As métricas, a meta de precisão (recall com precisão de pelo menos 90%) e a leitura sim/não das decisões foram fixadas em `reports/model-selection.md` antes da avaliação (D17).

## Rastreamento

O MLflow fica em `mlruns/`, fora do Git, e roda com a telemetria desligada. Cada configuração testada é uma execução filha da execução `model-selection`; a avaliação do teste é a execução `test-evaluation`.

O que é registrado:
- a versão dos dados (assinaturas da fase 3);
- o commit e a assinatura do código;
- os sinais, os hiperparâmetros e a estratégia de balanceamento;
- as métricas e os tempos;
- a calibração, os limites e os custos;
- os artefatos.

Nunca são registradas transações individuais, pseudônimos de cartão ou atributos pessoais. Para abrir o MLflow:

```
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```
