# Detecção de Fraude em Pagamentos

Modelo que estima, para cada transação de cartão, a probabilidade de ser fraude e recomenda uma decisão (aprovar, revisar ou bloquear), mostrando os sinais que mais pesaram. Inclui o caminho completo: dados com origem comprovada, sinais de comportamento sem vazamento, modelos comparados por custo, explicações, API e publicação.

> **Status: fase 1 em andamento** — download da base com registro de origem e perfil dos dados.

## Dados

[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Kaggle, licença CC0): transações simuladas de 1.000 clientes em 800 lojistas, de 2019 a 2020. Os dados são simulados; ainda assim, campos pessoais são descartados na entrada e não entram nas decisões.

## Documentação

| Documento | Conteúdo |
| --- | --- |
| [Escopo](docs/escopo-projeto.md) | Objetivo, dados, método, segurança, stack e fases |
| [Decisões técnicas](docs/decisoes-tecnicas.md) | O que foi escolhido, as alternativas e o motivo |

## Autora

Amanda Gondra — [github.com/amanda-gondra](https://github.com/amanda-gondra)
