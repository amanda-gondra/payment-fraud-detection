# Decisões técnicas

Cada decisão registra o que foi escolhido, as alternativas e o motivo.

### D1 — Base de dados: Sparkov (Kaggle, CC0)

- **Decisão:** usar o *Credit Card Transactions Fraud Detection Dataset* (`kartik2112/fraud-detection`, versão 1).
- **Alternativas:** IEEE-CIS (dados reais, licença não comercial que proíbe redistribuir dados e modelos treinados); ULB (colunas já transformadas por PCA); PaySim (carteira digital, colunas de saldo inutilizáveis).
- **Justificativa:** licença de domínio público, que permite publicar o modelo; colunas com data e hora, lojista, categoria, valor e localização, que permitem criar sinais de comportamento e explicar as decisões.
- **Verificado em:** 15/11/2026, na API pública do Kaggle (licença "CC0: Public Domain", versão 1, publicada em 05/08/2020).

### D2 — Python 3.12

- **Decisão:** o projeto usa Python 3.12 (`.python-version`).
- **Alternativa:** Python 3.14, também instalado.
- **Justificativa:** bibliotecas de ML com partes compiladas (LightGBM, SHAP e o numba, que o SHAP usa) costumam demorar a publicar versões para o Python mais recente; a 3.12 tem suporte estável em todas.

### D3 — Demonstração no Hugging Face Spaces

- **Decisão:** a demonstração pública fica num Space do Hugging Face.
- **Alternativa:** site em React, como na Análise de Crédito Rural.
- **Justificativa:** cabe no prazo da primeira versão, é gratuito e fica junto do modelo publicado no Hugging Face.

### D4 — Dados fora do Git, com manifesto

- **Decisão:** os arquivos CSV ficam em `data/raw/`, fora do Git. O `data/manifest.json`, versionado, registra origem, licença, versão, tamanho e SHA-256 de cada arquivo.
- **Justificativa:** os arquivos somam cerca de 480 MB; o manifesto permite provar que qualquer pessoa treinou com exatamente os mesmos dados, e o download falha se os arquivos não conferirem.

### D5 — Atributos pessoais fora das decisões

- **Decisão:** nome, endereço e número do cartão são descartados na entrada (o número vira um identificador interno). Gênero, idade e profissão não entram como sinais de decisão; são usados só na auditoria de viés.
- **Justificativa:** minimização de dados (LGPD) e para que o modelo não decida com base em características pessoais.

### D6 — Relógio e ordem das transações

- **Decisão:** o relógio do projeto é `trans_date_trans_time`. A coluna `unix_time` não é usada. Toda etapa ordena as transações por data e hora antes de calcular qualquer sinal.
- **Empates:** a ordenação é estável. Transações com a mesma data e hora mantêm a ordem em que aparecem no arquivo, e o resultado é sempre o mesmo. Essa ordem não define o histórico: transações do mesmo cartão no mesmo segundo não entram no histórico umas das outras (D8).
- **Justificativa:** o perfil dos dados ([`reports/data-profile.md`](../reports/data-profile.md)) mostrou que `unix_time` fica cerca de 7 anos antes de `trans_date_trans_time`, então as duas colunas não marcam o mesmo instante. Também mostrou que o arquivo de treino não está em ordem de data e hora e que há grupos de transações do mesmo cartão no mesmo segundo (20 grupos no treino, com 40 transações, e 24 no teste, com 48). Os sinais de histórico precisam de uma ordem estável para não contar a própria transação nem uma transação futura.

### D7 — Análise exploratória só no arquivo de treino

- **Decisão:** a análise exploratória ([`notebooks/eda.ipynb`](../notebooks/eda.ipynb), também [no Kaggle](https://www.kaggle.com/code/amandagondra/payment-fraud-detection-eda-pt-br)) usa só o `fraudTrain.csv`. O `fraudTest.csv` não é olhado até a avaliação final.
- **Alternativa:** analisar os dois arquivos juntos, com mais dados.
- **Justificativa:** as escolhas de sinais, limites e métricas saem desta análise. Se o arquivo de teste influenciasse essas escolhas, a avaliação final mediria o modelo num período que ele já "conhece", e o resultado sairia otimista.

### D8 — Histórico estritamente anterior ao instante da transação

- **Decisão:** o histórico de uma transação feita no instante t são as transações do mesmo cartão com horário estritamente anterior a t. A própria transação nunca entra. Transações do mesmo cartão no mesmo segundo não entram no histórico umas das outras. Uma janela de duração w cobre [t − w, t). O rótulo de fraude não é usado em nenhum sinal.
- **Alternativa:** usar a ordem do arquivo como desempate e deixar a primeira transação do segundo entrar no histórico da segunda (a ordenação estável da D6).
- **Justificativa:** com horário em segundos, não há como saber qual de duas transações do mesmo segundo veio antes, e o arquivo não está em ordem de tempo. Tratar a ordem do arquivo como ordem real seria usar uma informação que o sistema não teria. A ordenação estável da D6 continua valendo para guardar as linhas sempre na mesma ordem, mas não define disponibilidade. A regra fica num único lugar do código (`_SortedHistory.earlier`) e é verificada por uma implementação independente (`tests/reference_history.py`).

### D9 — Divisão temporal: validação a partir de 01/01/2020 00:00:00

- **Decisão:** do `fraudTrain.csv`, treino vai de 01/01/2019 até antes de 01/01/2020 00:00:00, e validação vai desse instante até o fim do arquivo (21/06/2020 12:13:37). O `fraudTest.csv` é o teste (D11). As datas exatas e as contagens estão em [`reports/temporal-split.md`](../reports/temporal-split.md).
- **Alternativas:**
  - validar só nos últimos 3 meses do arquivo: teria cerca de 1.163 fraudes, metade das 2.286 atuais;
  - um corte proporcional, como 80/20: cairia no meio de um mês e deixaria dezembro de 2019 dividido;
  - um intervalo vazio entre treino e validação.
- **Justificativa:**
  - o treino cobre 12 meses completos, inclusive dezembro, quando o volume dobra (fase 2);
  - a validação tem 2.286 fraudes em quase 6 meses, o suficiente para escolher limites com estabilidade, e fica colada no período do teste;
  - não há intervalo entre os conjuntos porque os sinais não usam rótulos (D8): o histórico de uma transação de validação pode olhar para o período de treino, como aconteceria em produção, mas nenhuma transação de treino olha para frente.

### D10 — Pseudônimo do cartão por HMAC-SHA256 com chave secreta

- **Decisão:** o número do cartão vira `card_key`, os 16 primeiros caracteres hexadecimais do HMAC-SHA256 do número com uma chave secreta de 32 bytes. A chave fica na variável `CARD_PSEUDONYM_KEY` (arquivo `.env`, fora do Git) e nunca é impressa nem gravada. O manifesto guarda só uma impressão curta que muda junto com a chave.
- **Alternativas:**
  - um número sequencial por cartão: muda se a ordem dos dados mudar e pode ser refeito por quem tem a base original;
  - um hash simples, sem chave: pode ser revertido testando todos os números de cartão possíveis.
- **Justificativa:** é estável enquanto a chave for a mesma e não pode ser revertido sem ela. Com outra chave, só o `card_key` muda: os testes confirmam que o conteúdo de todas as outras colunas continua idêntico, e o relatório publica uma assinatura desse conteúdo que qualquer pessoa consegue reproduzir. A API da fase 5 vai precisar da mesma chave, como segredo, para achar o histórico de um cartão.

### D11 — Arquivo de teste fora da fase 3

- **Decisão:** o `fraudTest.csv` não é aberto na fase 3, nem para conferir o SHA-256. Os sinais do teste serão calculados uma única vez, na avaliação final, com o mesmo código, sobre treino e teste juntos em ordem de tempo. Assim o histórico de uma transação de teste inclui as transações anteriores de treino e validação, como em produção.
- **Alternativa:** calcular agora os sinais do teste e guardar o arquivo.
- **Justificativa:** nada do teste, nem contagens, nem distribuições, nem rótulos, pode influenciar a escolha de sinais, modelos e limites. Calcular os sinais depois não muda nenhum valor, porque eles só dependem do passado (D8).
