# Detecção de Fraude em Pagamentos

Modelo que estima, para cada transação de cartão, a probabilidade de ser fraude e recomenda uma decisão (aprovar, revisar ou bloquear), mostrando os sinais que mais pesaram. Inclui o caminho completo: dados com origem comprovada, sinais de comportamento sem vazamento, modelos comparados por custo, explicações, API e publicação.

> **Status: fase 1 concluída** — base baixada com origem comprovada, perfil dos dados e testes. Próxima: análise exploratória.

## Dados

[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Kaggle, licença CC0): 1.852.394 transações simuladas de 999 cartões em 693 lojistas, de 2019 a 2020 ([perfil dos dados](reports/data-profile.md)). Os dados são simulados; ainda assim, campos pessoais são descartados na entrada e não entram nas decisões.

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

## Documentação

| Documento | Conteúdo |
| --- | --- |
| [Escopo](docs/escopo-projeto.md) | Objetivo, dados, método, segurança, stack e fases |
| [Decisões técnicas](docs/decisoes-tecnicas.md) | O que foi escolhido, as alternativas e o motivo |
| [Perfil dos dados](reports/data-profile.md) | Colunas, ausentes, repetições, período e fraude por mês, gerado por `python -m pipeline.profile` |

## Autora

Amanda Gondra — [github.com/amanda-gondra](https://github.com/amanda-gondra)
