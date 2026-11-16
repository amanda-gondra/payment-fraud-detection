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

### D12 — Validação dividida em seleção e calibração

- **Decisão:** a validação da fase 3 (D9) é cortada em 01/04/2020 00:00:00:
  - **seleção** (01/01 a 31/03/2020): última janela da validação cruzada temporal (D14) e parada antecipada do modelo final;
  - **calibração** (01/04 a 21/06/2020): calibradores, comparação dos três modelos e limites de decisão.
- **Alternativa:** usar a validação inteira para tudo.
- **Justificativa:**
  - Escolher o modelo, ajustar a calibração e escolher os limites nas mesmas transações deixaria cada etapa otimista sobre a anterior.
  - Com o corte, o período de calibração nunca participa da escolha do modelo.
  - Cada parte tem cerca de 1.100 fraudes, o suficiente para as duas funções.
  - O corte cai no início de um mês, sem dividir nenhum mês.

### D13 — Duas referências: regra do valor e regressão logística

- **Decisão:**
  - **Regra do valor:** alerta quando o valor passa de X. X é o corte de maior F1 no treino.
  - **Regressão logística:** usa os 28 sinais, com estes tratamentos:
    - log nos sinais de cauda longa;
    - mediana com indicador para os ausentes;
    - padronização;
    - one-hot de categoria, hora e dia da semana;
    - L2.

    C e o peso das classes são escolhidos pela PR-AUC na seleção.
  - **Na avaliação:** as duas são medidas com as mesmas métricas do LightGBM. A regressão logística ganha calibrador e limites próprios, escolhidos do mesmo jeito.
- **Alternativa:** comparar o LightGBM só com a própria versão anterior, ou com uma regra escolhida olhando a validação.
- **Justificativa:**
  - Sem referência não dá para dizer quanto o modelo principal acrescenta.
  - O F1 escolhe X sem hipótese de custo e sem olhar a validação.
  - A regressão logística salva usa o skops, que não executa código arbitrário (D18).
- **Resultado:** X = 713,69. Regressão logística com C = 0,1 e sem peso de classes, PR-AUC de 0,7945 na seleção; o peso balanceado piorou todas as combinações ([`model-selection.md`](../reports/model-selection.md)).

### D14 — LightGBM com controle explícito de sobreajuste

- **Decisão (versão atual):**
  - **Fixo:** taxa de aprendizado 0,05, até 5.000 árvores, 80% das linhas e das colunas por árvore, semente fixa com `deterministic=True`, sem reamostragem e sem peso extra para as fraudes.
  - **Parada:** pelo log loss dos dados de validação, depois de 100 rodadas sem melhora. A PR-AUC continua registrada.
  - **Busca de regularização:** oito combinações de:
    - folhas: 7 ou 15;
    - mínimo de transações por folha: 300 ou 1.000;
    - L2: 1 ou 10.
  - **Validação cruzada temporal:** quatro janelas.
    - Nas três primeiras, o modelo treina com tudo de 2019 antes da janela e é medido nos dois meses dela: jul-ago, set-out e nov-dez.
    - A quarta treina em 2019 inteiro e valida no período de seleção. O modelo dela é o modelo final.
  - **Escolha:** regra de um erro padrão. Entre as configurações cuja PR-AUC média nas janelas fica a no máximo um erro padrão da melhor, vence a mais simples: menos folhas, depois mais transações por folha, depois mais L2, depois menos árvores.
- **Primeira versão (substituída antes de abrir o teste):**
  - **Como era:** parada pela PR-AUC na seleção, num único período. A busca combinava folhas (15 ou 63), mínimo por folha (50 ou 300) e peso das fraudes (1, raiz da razão ou razão completa).
  - **O que escolheu:** 15 folhas, 300 por folha, sem peso e 2.175 árvores.
  - **O que a curva de aprendizado mostrou:** o modelo decorava.

    | Árvores | Log loss treino | Log loss seleção | Seleção ÷ treino |
    | --- | --- | --- | --- |
    | 300 | 0,00168 | 0,00228 | 1,4× |
    | 1.000 | 0,00042 | 0,00182 | 4,3× |
    | 1.500 | 0,00020 | 0,00178 | 8,8× |
    | 2.175 | 0,00009 | 0,00184 | 19,8× |

  - **Onde estava o problema:** depois de 1.500 árvores, o log loss em dados novos piorava, enquanto a PR-AUC continuava subindo por décimos de milésimo. A parada pela PR-AUC escolheu um ponto em que o modelo já estava confiante demais em dados novos.
- **Alternativas:**
  - manter a primeira versão, que tinha boas métricas em dados novos;
  - SMOTE ou subamostragem;
  - uma busca grande de hiperparâmetros.
- **Justificativa:**
  - Uma métrica alta em dados novos não basta. Um modelo que decora detalhes do treino tende a ser menos estável quando o comportamento muda, e o teste é o semestre seguinte.
  - O log loss é sensível a confiança excessiva; por isso ele decide a parada.
  - Várias janelas de tempo mostram se o padrão aprendido se repete ao longo do ano, e não só num período. O erro padrão entre janelas dá a margem de empate.
  - Entre modelos empatados, o mais simples é preferível: tem menos capacidade de decorar, explicações mais estáveis e inferência mais rápida.
  - O peso das fraudes saiu da busca porque não ajudou em nenhuma combinação da primeira versão.
  - A reamostragem continua de fora, porque inventa ou descarta transações e distorce as probabilidades.
  - O treino é reprodutível: os testes treinam duas vezes e exigem o mesmo modelo, byte a byte. Nos dados reais, dois treinos completos da primeira versão deram os mesmos números.
- **Resultado:**
  - **Escolhido:** 7 folhas, no mínimo 1.000 transações por folha e L2 = 10. A PR-AUC média nas janelas foi de 0,9764 ± 0,0055; a maior entre as oito configurações foi 0,9767.
  - **Modelo final:** 2.592 árvores. O log loss em dados novos ficou 2,3 vezes o do treino, contra 19,8 vezes na primeira versão.
  - **PR-AUC:** 0,9900 na seleção e 0,9849 no período de calibração.
  - **Reprodutibilidade:** dois treinos completos da versão atual deram o mesmo `model.txt`, byte a byte.

  Detalhes em [`model-selection.md`](../reports/model-selection.md).

### D15 — Calibração escolhida por validação cruzada em blocos de tempo

- **Decisão:** compara três leituras do escore:
  - a probabilidade original do modelo;
  - Platt;
  - isotônica.

  Cada uma é avaliada por validação cruzada no período de calibração: cinco blocos consecutivos, e cada bloco é previsto por um calibrador ajustado nos outros quatro. Vence o menor log loss, com Brier e ECE como confirmação. O vencedor é ajustado no período inteiro e salvo como números em JSON.
- **Alternativa:** ajustar e avaliar o calibrador nos mesmos dados; ou escolher a isotônica sem comparar.
- **Justificativa:**
  - Medir um calibrador nos dados em que ele foi ajustado favorece o mais flexível, a isotônica.
  - Os blocos de tempo respeitam a ordem dos dados.
  - A probabilidade original precisa ser medida antes de virar decisão; por isso a calibração vem depois do treino, e o método é escolhido por evidência.
- **Resultado:**
  - **Vencedor:** a probabilidade do próprio LightGBM, sem calibrador. Teve log loss de 0,00180, contra 0,00184 do Platt e 0,00216 da isotônica. No Brier e no ECE, o Platt e a probabilidade original ficaram praticamente iguais.
  - **Faixas de 30% a 80%:** com poucas transações por faixa, a taxa observada fica em geral acima da prevista. Todas essas transações já estão acima do limite de bloqueio (D16).

  Detalhes em [`model-selection.md`](../reports/model-selection.md).

### D16 — Três faixas de decisão escolhidas por custo hipotético

- **Decisão:** probabilidade de fraude (a do método escolhido em D15) abaixo de t₁: aprovar; entre t₁ e t₂: revisão humana; a partir de t₂: bloquear. Os dois limites minimizam o custo total no período de calibração, com estas hipóteses na unidade dos valores da base:

  | Situação | Custo |
  | --- | --- |
  | Fraude aprovada | valor da transação |
  | Revisão | 5 |
  | Legítimo em revisão | mais 2 de atrito |
  | Fraude em revisão | 10% do valor, porque a revisão detecta 90% |
  | Legítimo bloqueado | 25 |

  - **Capacidade:** no máximo 1% das transações em revisão.
  - **Probabilidades usadas:** as da validação cruzada (D15).
  - **Sensibilidade:** a busca é refeita mudando uma hipótese por vez.
- **Alternativas:**
  - um limite único de 0,5;
  - um limite escolhido por F1;
  - uma regra que dependa do valor de cada transação.
- **Justificativa:**
  - O limite é uma decisão de negócio: quanto custa cada tipo de erro. O valor de 0,5 não tem relação com esses custos.
  - Não há dados reais de custo numa base simulada, então os valores são hipóteses explícitas, com a análise de sensibilidade ao lado.
  - Dois limites fixos de probabilidade são mais simples de auditar e de explicar do que uma regra por valor. A regra por valor fica como melhoria possível.
- **Resultado:**
  - **Limites:** revisão a partir de 0,0436 e bloqueio a partir de 0,2624.
  - **No período de calibração:**
    - 1.137 das 1.163 fraudes identificadas;
    - 0,12% das transações em revisão, abaixo do teto de 1%;
    - 64 clientes legítimos bloqueados;
    - custo total de 6.374, contra 30.588 de só bloquear a partir de 0,5 e 69.355 da regressão logística.
  - **Sensibilidade:**
    - o limite de revisão (0,0436) só muda quando a revisão passa a custar 20;
    - o de bloqueio vai de 0,1494 a 0,6888, conforme o custo da revisão, o custo do bloqueio indevido e a eficácia da revisão;
    - o teto de capacidade não chegou a limitar.

  Detalhes em [`model-selection.md`](../reports/model-selection.md).

### D17 — Teste usado uma única vez, com tudo pré-registrado

- **Decisão:** o `python -m pipeline.train` não lê o `fraudTest.csv`, o que foi verificado com um gancho de auditoria do Python. Antes de abrir o teste, o `reports/model-selection.md` congela:
  - o modelo, o calibrador e os limites, com as assinaturas;
  - as métricas;
  - a meta: recall com precisão de pelo menos 90%;
  - a leitura sim/não das decisões: alerta é revisão ou bloqueio.

  O `python -m pipeline.evaluate` grava `models/test-evaluation.json` antes de abrir o teste. Ele se recusa a rodar depois de concluído, ou para um modelo diferente do que abriu o teste. Os experimentos ficam no MLflow local (`mlruns/`), sem telemetria e sem transações individuais.
- **Alternativa:** avaliar no teste durante o desenvolvimento, ou permitir várias avaliações.
- **Justificativa:**
  - Cada olhada no teste vira uma escolha de modelagem disfarçada.
  - A trava ligada à assinatura do modelo impede avaliar no teste algo que foi ajustado depois de vê-lo.
  - A meta de 90% de precisão significa no máximo um cliente legítimo a cada dez alertas, o nível em que o bloqueio automático ainda é defensável.

### D18 — Modelo salvo sem pickle, com assinatura

- **Decisão:**
  - **Formatos:** o LightGBM vai no seu formato de texto (`model.txt`). A ordem das colunas, as categorias, o calibrador, os limites e os metadados vão em JSON. A regressão logística vai no skops, com lista explícita de tipos confiáveis.
  - **Manifesto:** guarda o SHA-256 de cada arquivo e do pacote.
  - **Carregamento:** confere os bytes antes de interpretá-los, recusa arquivo a mais ou alterado, confere as colunas do modelo e repete a verificação no fim.
- **Alternativas:** pickle ou joblib; ONNX.
- **Justificativa:**
  - Pickle e joblib executam código ao carregar.
  - O ONNX exigiria bibliotecas de conversão a mais, e o suporte às variáveis categóricas do LightGBM é limitado. O formato de texto do LightGBM já não executa código, e os testes confirmam que o modelo lido é idêntico ao salvo.
  - Nenhum segredo fica no pacote: o modelo não usa `card_key`.

### D19 — Explicações pelo TreeSHAP do próprio LightGBM

- **Decisão:**
  - **Cálculo:** os valores SHAP vêm do próprio LightGBM (`pred_contrib=True`), em log-odds, antes da calibração.
  - **Explicação global:** média de |SHAP| numa amostra aleatória de 20.000 transações, comparada com a importância por ganho, e o gráfico de resumo.
  - **Explicação individual:** os sinais de maior contribuição absoluta (quatro na resposta, cinco nos relatórios). O texto é escrito a partir do valor real do sinal e diz se a contribuição aumentou ou reduziu o risco.
  - **Documentação:** as frases de cada sinal estão em [`modelagem.md`](modelagem.md).
- **Alternativas:** a biblioteca `shap`; textos fixos por sinal, sem o valor.
- **Justificativa:**
  - O TreeSHAP do LightGBM é exato para árvores e evita uma dependência pesada.
  - Os testes conferem que as contribuições somam o escore e que batem com um cálculo de Shapley por força bruta.
  - Escrever a partir do valor real impede motivos inventados.
  - Calibração crescente preserva o sentido de cada contribuição.

### D20 — Auditoria de equidade com intervalos por cartão

- **Decisão:**
  - **Separação:** gênero e faixa etária ficam fora da matriz do modelo e só são juntados às decisões depois de tomadas.
  - **Faixas de idade:** até 29, 30 a 44, 45 a 59, 60 ou mais.
  - **Métricas por grupo:** falsos positivos, fraudes não detectadas, precisão, recall, legítimos bloqueados e envio para revisão.
  - **Intervalos de 95%:** bootstrap que sorteia cartões inteiros.
  - **Diferenças:** cada grupo contra o maior, também com intervalo.
  - **Amostra pequena:** grupos com menos de 30 fraudes recebem aviso.
- **Alternativa:** comparar só percentuais; intervalos que tratam cada transação como independente.
- **Justificativa:**
  - As transações de um cliente não são independentes, e um episódio de fraude atinge várias de uma vez. Intervalos por transação seriam estreitos demais.
  - Um intervalo que exclui zero é indício para investigar, não prova de discriminação; e um intervalo que inclui zero não prova igualdade.
  - Os dados são simulados, então nada aqui vale automaticamente para pessoas reais.
