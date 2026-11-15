# Dados processados

Gerados por `python -m pipeline.build` a partir do `fraudTrain.csv`. Ficam em `data/processed/`, fora do Git. As regras estão nas decisões D8 a D11 de [`decisoes-tecnicas.md`](decisoes-tecnicas.md); as datas de corte e as contagens, em [`reports/temporal-split.md`](../reports/temporal-split.md).

## Arquivos

| Arquivo | Conteúdo | Uso |
| --- | --- | --- |
| `data/processed/transactions.parquet` | Base minimizada e pseudonimizada, em ordem de data e hora | Fonte dos sinais e dos testes contra vazamento |
| `data/processed/train.parquet` | Identificadores, 28 sinais e o rótulo das transações de 01/01/2019 a 31/12/2019 | Treino dos modelos |
| `data/processed/validation.parquet` | O mesmo, para 01/01/2020 a 21/06/2020 | Escolha do modelo, dos limites e da calibração |
| `data/processed/audit/attributes.parquet` | Gênero e idade, ligados pelo `transaction_id` | Só a auditoria de viés da fase 4; nunca entra no modelo |
| `data/processed/manifest.json` | Origem, assinatura SHA-256 de cada arquivo, impressão da chave e versão do código | Rastreabilidade |

O `fraudTest.csv` não é lido nesta fase (D11).

**Formato:** Parquet com compressão zstd. Os tipos são os mesmos em todas as etapas:

- data e hora em `datetime64[ms]`, sempre com segundos inteiros;
- categorias e `card_key` como categóricas;
- contagens em `int32`, indicadores em `bool` e valores em `float64`;
- o rótulo em `int8`.

O build lê cada arquivo de volta e para com erro se algum valor ou tipo mudar.

## O que acontece com cada coluna original

| Coluna | Destino | Motivo |
| --- | --- | --- |
| índice sem nome | `source_row` | Posição no arquivo original: rastreabilidade e desempate de horários iguais (D6) |
| `trans_num` | `transaction_id` | Identificador aleatório da transação, sem dado pessoal; liga os arquivos entre si |
| `trans_date_trans_time` | `timestamp` | O relógio do projeto (D6) |
| `cc_num` | `card_key` | Pseudônimo HMAC-SHA256 com chave secreta (D10); o número do cartão nunca é salvo |
| `category` | `category` | Sinal de decisão |
| `amt` | `amount` | Sinal de decisão |
| `lat`, `long`, `merch_lat`, `merch_long` | `distance_km`, e são descartadas | As coordenadas do cliente mostram onde ele mora; o modelo só precisa da distância |
| `gender` | Só no arquivo de auditoria | Não decide nada (D5); serve para comparar erros entre grupos |
| `dob` | `age_years`, só no arquivo de auditoria | A idade basta para a auditoria; a data de nascimento não é guardada |
| `first`, `last` | Descartadas | Nome do cliente |
| `street`, `city`, `state`, `zip` | Descartadas | Endereço do cliente |
| `city_pop` | Descartada | Derivada do endereço e sem uso na análise |
| `job` | Descartada | Atributo pessoal (D5), com 494 valores: grupos pequenos demais até para a auditoria |
| `merchant` | Descartada | 693 valores e nenhuma evidência na análise exploratória; pode voltar com justificativa |
| `unix_time` | Descartada | Não marca o mesmo instante que a data e hora (D6) |
| `is_fraud` | `is_fraud` | Rótulo. Nunca é usado para calcular sinais |

Nos arquivos de treino e validação, `transaction_id`, `timestamp` e `card_key` acompanham os sinais, mas não são sinais. Eles servem para:

- ligar os arquivos entre si;
- avaliar o modelo por período;
- identificar a primeira fraude de cada episódio.

## Sinais

**Histórico** é o conjunto das transações do mesmo cartão com horário *estritamente anterior* ao da transação avaliada (D8):

- a própria transação nunca entra;
- transações do mesmo cartão no mesmo segundo não entram no histórico umas das outras;
- uma janela de 1 hora cobre o intervalo [t − 1 h, t): inclui o limite inferior e exclui o instante t;
- o rótulo de fraude nunca é usado.

Quando o cartão não tem histórico, contagens e somas valem 0, médias e razões ficam ausentes (NaN) e `has_history` é falso. A fase 4 decide como cada modelo trata as ausências: o LightGBM as aceita como estão; a regressão logística precisa de imputação.

| Sinal | Definição | Sem histórico |
| --- | --- | --- |
| `amount` | Valor da transação | — |
| `category` | Categoria do lojista (14 valores) | — |
| `hour` | Hora do dia, de 0 a 23 | — |
| `day_of_week` | Dia da semana, de 0 (segunda) a 6 (domingo) | — |
| `is_late_night` | Compra entre 22h e 3h59, faixa com 84,8% das fraudes do treino (fase 2) | — |
| `is_weekend` | Compra no sábado ou no domingo | — |
| `distance_km` | Distância em linha reta entre o endereço do cliente e o lojista (Haversine, raio de 6.371,0088 km). Ausente se alguma coordenada for inválida. Não é o deslocamento real | — |
| `tx_count_1h` | Compras do cartão na última 1 hora | 0 |
| `amount_sum_1h` | Soma dos valores dessas compras | 0 |
| `amount_mean_1h` | Média dos valores dessas compras | Ausente |
| `tx_count_24h` | Compras do cartão nas últimas 24 horas | 0 |
| `amount_sum_24h` | Soma dos valores dessas compras | 0 |
| `amount_mean_24h` | Média dos valores dessas compras | Ausente |
| `tx_count_7d` | Compras do cartão nos últimos 7 dias | 0 |
| `amount_sum_7d` | Soma dos valores dessas compras | 0 |
| `amount_mean_7d` | Média dos valores dessas compras | Ausente |
| `tx_count_30d` | Compras do cartão nos últimos 30 dias | 0 |
| `amount_sum_30d` | Soma dos valores dessas compras | 0 |
| `amount_mean_30d` | Média dos valores dessas compras | Ausente |
| `has_history` | O cartão tem pelo menos uma compra anterior | Falso |
| `history_mean_amount` | Média de todas as compras anteriores do cartão | Ausente |
| `amount_to_history_mean` | Valor dividido pela média anterior | Ausente |
| `amount_minus_history_mean` | Valor menos a média anterior: a diferença absoluta, com sinal | Ausente |
| `amount_log_zscore` | Quantos desvios padrão o log(1 + valor) está da média dos log(1 + valor) anteriores. Só existe com pelo menos 10 compras anteriores e desvio maior que zero; o log reduz o peso dos valores extremos | Ausente |
| `seconds_since_previous` | Segundos desde a compra anterior do cartão | Ausente |
| `is_new_category` | Primeira compra do cartão nesta categoria | Verdadeiro |
| `history_share_same_hour` | Fração das compras anteriores feitas na mesma hora do dia | Ausente |
| `history_share_late_night` | Fração das compras anteriores feitas entre 22h e 3h59 | Ausente |

**Fora dos sinais, de propósito:**

- a contagem de todas as compras anteriores, porque mede quando o cartão aparece no arquivo, e não o comportamento do cliente (fase 2);
- `unix_time`, nome, endereço, profissão, gênero e idade.

## Como a ausência de vazamento é verificada

Os testes ficam em `tests/test_features.py` e `tests/test_processed_data.py`:

1. **Referência independente.** Uma segunda implementação, linha por linha e sem nenhuma lógica em comum com o pipeline, recalcula os 28 sinais e os compara aos gravados. Ela roda em casos de borda, em dados aleatórios e em 300 transações reais sorteadas com semente fixa.
2. **Invariâncias:**
   - mudar ou acrescentar transações futuras não altera nenhum sinal do passado;
   - mudar a transação atual não altera o histórico dela;
   - mudar os rótulos não altera nada.
3. **Vazamentos de propósito.** Quatro versões erradas do pipeline são testadas, e a referência precisa denunciar todas:
   - a transação vê a si mesma;
   - empates no mesmo segundo ficam visíveis;
   - o futuro do cartão fica visível;
   - a janela perde o limite inferior.
4. **Limites e empates** conferidos com valores calculados à mão.
