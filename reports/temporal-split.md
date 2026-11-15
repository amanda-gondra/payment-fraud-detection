# Divisão temporal e sinais

Gerado por `python -m pipeline.build` a partir do `fraudTrain.csv` (SHA-256 `fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d`). As regras estão nas decisões D8 a D11 de [`docs/decisoes-tecnicas.md`](../docs/decisoes-tecnicas.md) e cada campo em [`docs/dados-processados.md`](../docs/dados-processados.md).

## Conjuntos

| Conjunto | Primeira transação | Última transação | Transações | Fraudes | Taxa de fraude | Cartões |
|---|---|---|---:|---:|---:|---:|
| Treino | 01/01/2019 00:00:18 | 31/12/2019 23:59:53 | 924.850 | 5.220 | 0,56% | 960 |
| Validação | 01/01/2020 00:00:57 | 21/06/2020 12:13:37 | 371.825 | 2.286 | 0,61% | 931 |

- **Corte:** a validação começa em 01/01/2020 00:00:00. Tudo antes desse instante é treino; tudo a partir dele é validação.
- **Teste:** o `fraudTest.csv` não é aberto nesta fase. Ele fica intocado para a avaliação final (decisão D11); seus sinais serão calculados com o mesmo código, usando o histórico de treino e validação como passado.

## Sinais

28 sinais por transação: `amount`, `category`, `hour`, `day_of_week`, `is_late_night`, `is_weekend`, `distance_km`, `tx_count_1h`, `amount_sum_1h`, `amount_mean_1h`, `tx_count_24h`, `amount_sum_24h`, `amount_mean_24h`, `tx_count_7d`, `amount_sum_7d`, `amount_mean_7d`, `tx_count_30d`, `amount_sum_30d`, `amount_mean_30d`, `has_history`, `history_mean_amount`, `amount_to_history_mean`, `amount_minus_history_mean`, `amount_log_zscore`, `seconds_since_previous`, `is_new_category`, `history_share_same_hour`, `history_share_late_night`.

## Conteúdo dos arquivos

Assinatura dos valores de todas as colunas, menos `card_key` (que depende da chave secreta). Quem rodar o pipeline com outra chave obtém as mesmas assinaturas.

| Arquivo | Linhas | Assinatura do conteúdo |
|---|---:|---|
| `data/processed/transactions.parquet` | 1.296.675 | `8d8d6020ecb3f088` |
| `data/processed/train.parquet` | 924.850 | `64c2972faf97b803` |
| `data/processed/validation.parquet` | 371.825 | `32e016e22821da83` |
