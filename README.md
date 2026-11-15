# Detecção de Fraude em Pagamentos

Modelo que estima, para cada transação de cartão, a probabilidade de ser fraude e recomenda uma decisão (aprovar, revisar ou bloquear), mostrando os sinais que mais pesaram. Inclui o caminho completo: dados com origem comprovada, sinais de comportamento sem vazamento, modelos comparados por custo, explicações, API e publicação.

> **Status: fase 2 concluída** — análise exploratória publicada no Kaggle. Próxima: sinais de comportamento e divisão temporal.

## Dados

[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Kaggle, licença CC0): 1.852.394 transações simuladas de 999 cartões em 693 lojistas, de 2019 a 2020 ([perfil dos dados](reports/data-profile.md)). Os dados são simulados; ainda assim, campos pessoais são descartados na entrada e não entram nas decisões.

## Análise exploratória

📓 **[Notebook no Kaggle](https://www.kaggle.com/code/amandagondra/payment-fraud-detection-eda-pt-br)** · [versão no repositório](notebooks/eda.ipynb)

O notebook investiga o que diferencia uma transação fraudulenta de uma legítima, para orientar a escolha dos sinais e das métricas do modelo. Ele usa só o arquivo de treino (01/01/2019 a 21/06/2020); o arquivo de teste fica guardado para a avaliação final. Em seis frentes:

| Frente | Principal achado |
| --- | --- |
| Distribuição geral | 0,58% de fraude, uma para cada 172 transações legítimas: quem aprova tudo acerta 99,42%, então a acurácia não serve como métrica |
| Horário | 84,8% das fraudes acontecem entre 22h e 3h59, contra 23,2% das compras legítimas |
| Categoria | `shopping_net`, `misc_net` e `grocery_pos` reúnem 58,2% das fraudes com 21,9% das transações; entre 22h e 23h, 43,8% das compras em `grocery_pos` são fraude |
| Valor | mediana de 396,50 nas fraudes e de 47,28 nas legítimas; o sentido da diferença muda com a categoria |
| Distância | não separa fraude de compra legítima nesta base |
| Histórico do cartão | com o valor 10 vezes acima da média anterior do cartão, 27,9% das transações são fraude; a primeira fraude de cada episódio se comporta diferente das seguintes |

Todo sinal de histórico é calculado só com transações anteriores, e o notebook verifica isso. As limitações estão escritas no próprio notebook, inclusive os artefatos da simulação: categorias com horário fixo e nenhuma fraude acima de 1.376,04.

## Como reproduzir

Requer Python 3.12. No Windows:

```
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pipeline.download
python -m pipeline.profile
```

No Linux e no macOS, as duas primeiras linhas são `python3.12 -m venv .venv` e `source .venv/bin/activate`.

O `pipeline.download` baixa a base (cerca de 480 MB) sem pedir conta no Kaggle e confere cada arquivo com o `data/manifest.json`; se algum não conferir, ele para com erro. O `pipeline.profile` gera o [perfil dos dados](reports/data-profile.md).

Para rodar os testes, que usam só dados sintéticos e não precisam da base:

```
python -m pip install -r requirements-dev.txt
python -m pytest
```

A fonte do notebook é `notebooks/eda.py`. Para gerar o `notebooks/eda.ipynb` de novo e executá-lo do zero (depois do `pipeline.download`):

```
python -m jupytext --to ipynb notebooks/eda.py
python -m jupyter execute --inplace notebooks/eda.ipynb
```

No Windows, rode antes `$env:PYTHONUTF8 = "1"` (PowerShell); sem isso, os acentos do notebook saem corrompidos.

## Documentação

| Documento | Conteúdo |
| --- | --- |
| [Escopo](docs/escopo-projeto.md) | Objetivo, dados, método, segurança, stack e fases |
| [Decisões técnicas](docs/decisoes-tecnicas.md) | O que foi escolhido, as alternativas e o motivo |
| [Perfil dos dados](reports/data-profile.md) | Colunas, ausentes, repetições, período e fraude por mês, gerado por `python -m pipeline.profile` |
| [Análise exploratória](notebooks/eda.ipynb) | Horário, categoria, valor, distância e histórico do cartão; também [no Kaggle](https://www.kaggle.com/code/amandagondra/payment-fraud-detection-eda-pt-br) |

## Autora

Amanda Gondra — [github.com/amanda-gondra](https://github.com/amanda-gondra)
