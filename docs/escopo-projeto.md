# Escopo do projeto

## Objetivo

Estimar, para cada transação de cartão, a probabilidade de ser fraude e recomendar uma de três decisões: **aprovar**, **mandar para revisão humana** ou **bloquear**, mostrando os sinais que mais pesaram em cada caso. O projeto reproduz, em escala de portfólio, o trabalho de um time de prevenção a fraudes: transformar o histórico de transações em sinais de comportamento, treinar um modelo que reconheça padrões raros, escolher com critério onde cortar entre "suspeito" e "normal" e entregar tudo por uma API que outro sistema consiga consultar.

## O que o sistema entrega

- A **probabilidade de fraude** de uma transação, arredondada.
- A **decisão** (aprovar, revisar ou bloquear), a partir de limites escolhidos pelo custo dos erros.
- Os **motivos** em linguagem humana, por exemplo "valor 6× acima da média deste cartão" ou "compra às 3h, fora do horário habitual".
- A **versão do modelo** que tomou a decisão.

## Perguntas que o projeto responde

1. Qual a chance desta transação ser fraude?
2. Por que o modelo achou isso?
3. Com o limite escolhido, quantas fraudes são detectadas e quantos clientes honestos são incomodados?
4. O modelo continua bom em meses que ele não viu no treinamento?

## Dados

| Item | Valor |
| --- | --- |
| Base | [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection), gerada com a ferramenta Sparkov |
| Licença | CC0: Public Domain, conferida na API do Kaggle em 15/11/2026 |
| Versão | 1, publicada em 05/08/2020 |
| Arquivos | `fraudTrain.csv` (335 MB) e `fraudTest.csv` (143 MB) |
| Cobertura | Transações simuladas de 1.000 clientes em 800 lojistas, de 01/01/2019 a 31/12/2020 |

Os dados são simulados: nenhuma pessoa real aparece neles. Mesmo assim, o projeto trata cada coluna pessoal como se fosse real (veja Segurança).

Como contexto brasileiro, o projeto pode citar os dados abertos do Banco Central sobre transações Pix contestadas por suspeita de fraude (Mecanismo Especial de Devolução), sem usá-los no treino.

### Bases avaliadas e descartadas

| Base | Motivo |
| --- | --- |
| IEEE-CIS Fraud Detection | Dados reais, mas a licença é só para pesquisa não comercial e proíbe redistribuir os dados ou os modelos treinados: o modelo não poderia ser publicado |
| Credit Card Fraud Detection (ULB) | As colunas já passaram por PCA: não sobra nada para criar sinais nem para explicar decisões |
| PaySim | Alternativa válida (transferências de carteira digital), mas as colunas de saldo não podem ser usadas, porque as transações detectadas como fraude são canceladas |

## Método

- **Ordem do tempo:** todas as transações são ordenadas por data e hora antes de qualquer cálculo.
- **Divisão temporal:** treino nos meses mais antigos, validação nos seguintes e teste nos mais recentes. O teste é usado uma única vez, no fim.
- **Sinais sem vazamento:** cada transação é avaliada só com o histórico anterior a ela (contagens e valores do cartão em janelas de 1 h, 24 h, 7 d e 30 d, tempo desde a compra anterior, valor em relação à média do cartão, distância entre cliente e lojista, horário). Um teste automatizado confere que nenhum sinal usa informação do futuro.
- **Modelos:** regressão logística como referência e LightGBM como modelo principal, ambos comparados também com uma regra simples (valor acima de um limite).
- **Métricas:** área sob a curva de precisão e revocação (PR-AUC), revocação com precisão fixa e custo dos erros. A acurácia não é usada: com fraude rara, um modelo que aprova tudo acertaria quase sempre.
- **Limites de decisão:** escolhidos na validação a partir do custo de cada erro (deixar passar uma fraude custa o valor da transação; bloquear um cliente honesto custa um valor fixo de atrito, documentado).
- **Explicações:** SHAP, traduzido em motivos legíveis.
- **Rastreabilidade:** experimentos no MLflow, com parâmetros, métricas e a versão dos dados (SHA-256).

## Segurança e IA responsável

**Dados (LGPD).** O modelo nunca recebe nome, endereço ou número do cartão. O número vira um identificador interno, usado só para montar o histórico, e os demais campos pessoais são descartados na entrada (minimização). Gênero, idade e profissão não são usados como sinais de decisão; servem apenas para uma auditoria que compara os erros entre grupos.

**Integridade do modelo e do código.** O modelo é salvo num formato que não executa código ao ser carregado (nada de pickle), com SHA-256 registrado; a API só sobe com a versão publicada cujo hash confere, fixada pelo identificador exato da versão no Hugging Face. As dependências têm versões fixas e passam por `pip-audit` e `bandit` no CI. A imagem Docker roda sem privilégios de administrador e sem segredos.

**API.** Entradas validadas em tipo, faixa e formato, com campos extras recusados e tamanho de requisição limitado; limite de pedidos por visitante, identificado de forma que não possa ser falsificado atrás do proxy; CORS só para a demonstração; erros sem detalhes internos; logs sem os dados das transações. A probabilidade sai arredondada, o que, junto com o limite de pedidos, dificulta sondar o modelo em massa.

**Uso responsável.** A faixa intermediária vai para revisão humana em vez de decisão automática (a LGPD, art. 20, garante a revisão de decisões tomadas só por tratamento automatizado). Cada decisão vem com os motivos. A API, a demonstração e a ficha do modelo deixam claro que ele foi treinado com dados simulados e serve para demonstração, não para decidir sobre pagamentos reais.

## Stack

| Área | Tecnologia |
| --- | --- |
| Linguagem | Python 3.12 |
| Dados | Pandas, NumPy, PyArrow (Parquet) |
| Análise | Jupyter, Matplotlib, Seaborn; notebook publicado no Kaggle |
| Modelos | scikit-learn, LightGBM |
| Explicações | SHAP |
| Experimentos | MLflow |
| API | FastAPI, Pydantic |
| Publicação | Hugging Face Hub (modelo) e Hugging Face Spaces (demonstração), Docker |
| Qualidade | pytest, ruff, GitHub Actions, pip-audit, bandit |

Custo zero: todas as ferramentas e hospedagens usam planos gratuitos.

## Fases

| Fase | Entrega | Previsão |
| --- | --- | --- |
| 1 | Download com registro de origem e SHA-256, perfil dos dados e licença conferida | 15–16/11 |
| 2 | Análise exploratória, publicada como notebook no Kaggle | 16–17/11 |
| 3 | Sinais de comportamento sem vazamento e divisão temporal | 18/11 |
| 4 | Modelos, limites por custo, explicações e auditoria de viés | 19–20/11 |
| 5 | API com testes e Docker | 21/11 |
| 6 | Publicação: modelo e ficha no Hugging Face, demonstração, README e CI | 22–23/11 |
| 7 | Post de lançamento no LinkedIn | 24/11 |

Cada fase é feita um arquivo por vez, com explicação, e só avança com aprovação.

## Fora do escopo da primeira versão

- Monitoramento do modelo ao longo do tempo (detecção de mudança de comportamento) e retreinamento automático.
- Processamento em tempo real (streaming).
- Redes neurais e modelos de grafo.
- Dados reais de pagamentos do Brasil.

## Limitações conhecidas

- Dados simulados, de cidades dos Estados Unidos, em 2019 e 2020: os padrões não representam necessariamente fraudes reais nem o mercado brasileiro.
- A base traz só transações de cartão; não cobre Pix, boletos ou golpes de engenharia social.
- O custo de um falso positivo é uma hipótese do projeto, documentada, e não um valor de mercado.
