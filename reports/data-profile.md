# Perfil dos dados

Gerado por `python -m pipeline.profile`, a partir dos arquivos conferidos com `data/manifest.json`. O relatório só tem contagens e estatísticas agregadas: nenhum nome, número de cartão, endereço ou data de nascimento.

## Origem

| Arquivo | Bytes | SHA-256 |
|---|---:|---|
| `fraudTrain.csv` | 351.238.196 | `fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d` |
| `fraudTest.csv` | 150.354.339 | `12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0` |

Bibliotecas: pandas 3.1.1, pyarrow 26.0.1, numpy 2.5.4.

## Visão geral

| | treino | teste |
|---|---:|---:|
| Transações | 1.296.675 | 555.719 |
| Fraudes | 7.506 | 2.145 |
| Taxa de fraude | 0,58% | 0,39% |
| Primeira transação | 01/01/2019 00:00:18 | 21/06/2020 12:14:25 |
| Última transação | 21/06/2020 12:13:37 | 31/12/2020 23:59:34 |
| Cartões | 983 | 924 |
| Lojistas | 693 | 693 |
| Categorias | 14 | 14 |
| Linhas repetidas (sem contar `trans_num`) | 0 | 0 |
| `trans_num` repetido | 0 | 0 |
| Grupos do mesmo cartão no mesmo segundo | 20 (40 transações) | 24 (48 transações) |
| Linhas em ordem de data e hora | não | sim |

## Entre os arquivos

- O teste começa 48 segundos depois da última transação do treino; os períodos se sobrepõem: não.
- `trans_num` presentes nos dois arquivos: 0.
- Cartões do teste que também aparecem no treino: 908 de 924.
- Mesmas categorias nos dois arquivos: sim.

## Relógio

- `unix_time` fica de 2.556 a 2.557 dias antes de `trans_date_trans_time`, cerca de 7 anos. As duas colunas não marcam o mesmo instante.

## Valores

| | treino | teste |
|---|---:|---:|
| Menor valor (`amt`) | 1,00 | 1,00 |
| Mediana do valor | 47,52 | 47,29 |
| Maior valor | 28.948,90 | 22.768,11 |
| Valores menores ou iguais a zero | 0 | 0 |
| Coordenadas fora da faixa válida | 0 | 0 |
| `city_pop` menor ou igual a zero | 0 | 0 |
| Valores de `gender` | F, M | F, M |
| Idade na data da transação (anos) | 13,9 a 95,6 | 15,4 a 96,2 |
| Nascimento depois da transação | 0 | 0 |

Categorias: `entertainment`, `food_dining`, `gas_transport`, `grocery_net`, `grocery_pos`, `health_fitness`, `home`, `kids_pets`, `misc_net`, `misc_pos`, `personal_care`, `shopping_net`, `shopping_pos`, `travel`.

## Colunas

| Coluna | Tipo | Ausentes treino | Ausentes teste | Distintos treino | Distintos teste |
|---|---|---:|---:|---:|---:|
| `trans_date_trans_time` | datetime64[s] | 0 | 0 | 1.274.791 | 544.760 |
| `cc_num` | int64 | 0 | 0 | 983 | 924 |
| `merchant` | str | 0 | 0 | 693 | 693 |
| `category` | str | 0 | 0 | 14 | 14 |
| `amt` | float64 | 0 | 0 | 52.928 | 37.256 |
| `first` | str | 0 | 0 | 352 | 341 |
| `last` | str | 0 | 0 | 481 | 471 |
| `gender` | str | 0 | 0 | 2 | 2 |
| `street` | str | 0 | 0 | 983 | 924 |
| `city` | str | 0 | 0 | 894 | 849 |
| `state` | str | 0 | 0 | 51 | 50 |
| `zip` | int64 | 0 | 0 | 970 | 912 |
| `lat` | float64 | 0 | 0 | 968 | 910 |
| `long` | float64 | 0 | 0 | 969 | 910 |
| `city_pop` | int64 | 0 | 0 | 879 | 835 |
| `job` | str | 0 | 0 | 494 | 478 |
| `dob` | object | 0 | 0 | 968 | 910 |
| `trans_num` | str | 0 | 0 | 1.296.675 | 555.719 |
| `unix_time` | int64 | 0 | 0 | 1.274.823 | 544.760 |
| `merch_lat` | float64 | 0 | 0 | 1.247.805 | 546.490 |
| `merch_long` | float64 | 0 | 0 | 1.275.745 | 551.770 |
| `is_fraud` | int64 | 0 | 0 | 2 | 2 |

## Fraude por mês

| Mês | Arquivo | Transações | Fraudes | Taxa |
|---|---|---:|---:|---:|
| 01/2019 | treino | 52.525 | 506 | 0,96% |
| 02/2019 | treino | 49.866 | 517 | 1,04% |
| 03/2019 | treino | 70.939 | 494 | 0,70% |
| 04/2019 | treino | 68.078 | 376 | 0,55% |
| 05/2019 | treino | 72.532 | 408 | 0,56% |
| 06/2019 | treino | 86.064 | 354 | 0,41% |
| 07/2019 | treino | 86.596 | 331 | 0,38% |
| 08/2019 | treino | 87.359 | 382 | 0,44% |
| 09/2019 | treino | 70.652 | 418 | 0,59% |
| 10/2019 | treino | 68.758 | 454 | 0,66% |
| 11/2019 | treino | 70.421 | 388 | 0,55% |
| 12/2019 | treino | 141.060 | 592 | 0,42% |
| 01/2020 | treino | 52.202 | 343 | 0,66% |
| 02/2020 | treino | 47.791 | 336 | 0,70% |
| 03/2020 | treino | 72.850 | 444 | 0,61% |
| 04/2020 | treino | 66.892 | 302 | 0,45% |
| 05/2020 | treino | 74.343 | 527 | 0,71% |
| 06/2020 | treino | 57.747 | 334 | 0,58% |
| 06/2020 | teste | 30.058 | 133 | 0,44% |
| 07/2020 | teste | 85.848 | 321 | 0,37% |
| 08/2020 | teste | 88.759 | 415 | 0,47% |
| 09/2020 | teste | 69.533 | 340 | 0,49% |
| 10/2020 | teste | 69.348 | 384 | 0,55% |
| 11/2020 | teste | 72.635 | 294 | 0,40% |
| 12/2020 | teste | 139.538 | 258 | 0,18% |
