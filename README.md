# Detecção de Fraude em Pagamentos

Modelo que estima, para cada transação de cartão, a probabilidade de ser fraude e recomenda uma decisão (aprovar, revisar ou bloquear), mostrando os sinais que mais pesaram. Inclui o caminho completo: dados com origem comprovada, sinais de comportamento sem vazamento, modelos comparados por custo, explicações, API e publicação.

> **Status: fase 3 concluída** — sinais de comportamento e divisão temporal prontos, com testes contra vazamento. Próxima: modelos, limites de decisão e explicações.

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

## Sinais e divisão temporal

O `python -m pipeline.build` transforma o arquivo de treino numa base pronta para os modelos, sem alterar o arquivo original:

- **Minimização:** nome, endereço e número do cartão são descartados na leitura. O cartão vira um pseudônimo HMAC-SHA256 com chave secreta, que não pode ser revertido sem ela. Gênero e idade ficam num arquivo separado, só para a auditoria de viés.
- **28 sinais:** valor, horário, categoria, distância e histórico do cartão em janelas de 1 hora, 24 horas, 7 dias e 30 dias. O histórico usa só transações estritamente anteriores: nem a própria transação, nem as do mesmo segundo, nem os rótulos entram no cálculo.
- **Divisão por data**, sem embaralhar:

| Conjunto | Período | Transações | Fraudes |
| --- | --- | --- | --- |
| Treino | 01/01/2019 a 31/12/2019 | 924.850 | 5.220 (0,56%) |
| Validação | 01/01/2020 a 21/06/2020 | 371.825 | 2.286 (0,61%) |
| Teste | `fraudTest.csv` | não é aberto nesta fase | — |

O teste fica intocado até a avaliação final. Uma segunda implementação, independente do pipeline, recalcula os sinais de transações sorteadas e precisa chegar aos mesmos valores. Quatro vazamentos introduzidos de propósito são todos detectados. Detalhes em [dados processados](docs/dados-processados.md) e no [relatório da divisão](reports/temporal-split.md).

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

Para gerar os sinais, crie antes a chave do pseudônimo do cartão, uma única vez: copie o `.env.example` para `.env`, gere a chave com o segundo comando e cole o resultado em `CARD_PSEUDONYM_KEY`. O `.env` nunca vai para o Git. Se ele já existir, pule as duas primeiras linhas, porque o `copy` o substituiria e a chave se perderia.

```
copy .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"
python -m pipeline.build
```

No Linux e no macOS, use `cp` no lugar de `copy`. O `pipeline.build` confere o `fraudTrain.csv` com o `data/manifest.json`, leva cerca de um minuto e grava os arquivos em `data/processed/`, fora do Git. Guarde a chave: com outra, os pseudônimos mudam, mas o restante dos dados fica igual.

Para rodar os testes (os de dados sintéticos não precisam da base; os de regressão sobre os dados reais só rodam depois do `pipeline.build` e, sem ele, são pulados):

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
| [Dados processados](docs/dados-processados.md) | Campos removidos, transformados e mantidos, dicionário dos 28 sinais e testes contra vazamento |
| [Divisão temporal](reports/temporal-split.md) | Datas de corte, transações e fraudes por conjunto, gerado por `python -m pipeline.build` |

## Autora

Amanda Gondra — [github.com/amanda-gondra](https://github.com/amanda-gondra)
