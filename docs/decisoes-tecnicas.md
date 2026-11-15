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
