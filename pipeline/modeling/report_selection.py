"""reports/model-selection.md: the development results and the choices frozen before the test."""

from __future__ import annotations

from typing import Any, Final

from pipeline.modeling import calibration, decision, fairness, metrics
from pipeline.modeling.reporting import (
    DASH,
    band_label,
    decimal,
    integer,
    interval,
    moment,
    pct,
    table,
    target_recall,
    times,
)

SET_LABELS: Final = {"train": "Treino", "selection": "Seleção", "calibration": "Calibração"}
SET_USE: Final = {
    "train": "ajuste de todos os modelos",
    "selection": "parada antecipada e escolha dos hiperparâmetros",
    "calibration": "calibradores, comparação dos modelos e limites de decisão",
}
METHOD_LABELS: Final = {"none": "Sem calibração", "platt": "Platt", "isotonic": "Isotônica"}
METHOD_PHRASES: Final = {"none": "probabilidade do próprio modelo, sem calibrador",
                         "platt": "calibração de Platt",
                         "isotonic": "calibração isotônica"}
CLASS_WEIGHT_LABELS: Final = {"none": "nenhum", "balanced": "balanceado"}
MODEL_LABELS: Final = {"rule": "Regra do valor", "logistic": "Regressão logística",
                       "lightgbm": "LightGBM"}
GAP_LABELS: Final = {"fpr": "Taxa de falsos positivos", "fnr": "Fraudes não detectadas",
                     "precision": "Precisão", "legit_block_rate": "Legítimos bloqueados",
                     "review_rate": "Enviados para revisão"}
ATTRIBUTE_LABELS: Final = {"gender": "Gênero", "age_band": "Faixa etária"}


def period(row: dict[str, Any]) -> str:
    return f"{row['first']:%d/%m/%Y} a {row['last']:%d/%m/%Y}"


def sets_section(sets: dict[str, dict[str, Any]]) -> list[str]:
    rows = [[SET_LABELS[name], period(row), integer(row["transactions"]), integer(row["frauds"]),
             pct(row["fraud_rate"], 2), SET_USE[name]] for name, row in sets.items()]
    return [
        "## Dados",
        "",
        "Os três conjuntos vêm do `fraudTrain.csv`, em ordem de tempo (decisão D12). A validação "
        "da fase 3 foi cortada em 01/04/2020: a primeira parte escolhe os modelos e a segunda, "
        "que nunca participa dessa escolha, calibra as probabilidades e define os limites.",
        "",
        *table(["Conjunto", "Período", "Transações", "Fraudes", "Taxa de fraude", "Uso"], rows,
               "llrrrl"),
        "",
    ]


def rule_section(v: dict[str, Any]) -> list[str]:
    rows = []
    for name, result in v["rule_results"].items():
        rows.append([SET_LABELS[name], integer(result["tp"] + result["fp"]), integer(result["tp"]),
                     integer(result["fp"]), pct(result["precision"]), pct(result["recall"]),
                     pct(result["f1"])])
    return [
        "## Modelos de referência",
        "",
        "### Regra do valor",
        "",
        f"Sinaliza a transação quando o valor é pelo menos **X = {decimal(v['rule'].threshold)}**. "
        "X é o corte com o maior F1 no treino (2019): o F1 pesa igualmente precisão e recall e "
        "não depende de hipóteses de custo. A regra é fixada sem olhar a validação.",
        "",
        *table(["Conjunto", "Sinalizadas", "Fraudes identificadas", "Falsos positivos",
                "Precisão", "Recall", "F1"], rows, "lrrrrrr"),
        "",
    ]


def logistic_section(v: dict[str, Any]) -> list[str]:
    best = v["logistic_best"]
    rows = []
    for row in v["logistic_rows"]:
        mark = "**" if row is best else ""
        rows.append([f"{mark}{decimal(row['C'], 2)}{mark}",
                     CLASS_WEIGHT_LABELS[row["class_weight"]],
                     integer(row["iterations"]), decimal(row["train_average_precision"], 4),
                     f"{mark}{decimal(row['selection_average_precision'], 4)}{mark}",
                     f"{decimal(row['seconds'], 0)} s"])
    return [
        "### Regressão logística",
        "",
        "Sinais de cauda longa em log (valores, contagens e tempo desde a compra anterior), "
        "ausentes imputados pela mediana do treino com um indicador de ausência, padronização, "
        "categoria, hora e dia da semana em one-hot e regularização L2. Seis combinações de C e "
        "de peso das classes; vence a de maior PR-AUC (average precision) no período de seleção.",
        "",
        *table(["C", "Peso das classes", "Iterações", "PR-AUC treino", "PR-AUC seleção",
                "Tempo"], rows, "rlrrrr"),
        "",
        f"**Escolhida:** C = {decimal(best['C'], 2)}, peso das classes "
        f"{CLASS_WEIGHT_LABELS[best['class_weight']]}.",
        "",
    ]


def lightgbm_section(v: dict[str, Any]) -> list[str]:
    best, rows_in = v["lightgbm_best"], v["lightgbm_rows"]
    windows = [[w["window"], f"{w['fit_first']:%d/%m/%Y} a {w['fit_last']:%d/%m/%Y}",
                f"{w['validation_first']:%d/%m/%Y} a {w['validation_last']:%d/%m/%Y}",
                integer(w["fit_transactions"]), integer(w["validation_transactions"]),
                integer(w["validation_frauds"])] for w in v["windows"]]
    rows = []
    for row in rows_in:
        mark = "**" if row is best else ""
        rows.append([
            f"{mark}{row['num_leaves']}{mark}", integer(row["min_child_samples"]),
            decimal(row["lambda_l2"], 0),
            f"{mark}{decimal(row['cv_average_precision'], 4)}{mark} ± "
            f"{decimal(row['cv_average_precision_se'], 4)}",
            decimal(row["cv_log_loss"], 5), times(row["cv_gap"]),
            integer(row["cv_trees"]), f"{decimal(row['seconds'] / 60, 1)} min",
        ])
    top = max(rows_in, key=lambda row: row["cv_average_precision"])
    floor = top["cv_average_precision"] - top["cv_average_precision_se"]
    final = best["folds"][-1]
    curve = [[integer(point["trees"]), decimal(point["train_log_loss"], 5),
              decimal(point["selection_log_loss"], 5),
              times(point["selection_log_loss"] / point["train_log_loss"]),
              decimal(point["train_average_precision"], 4),
              decimal(point["selection_average_precision"], 4)] for point in v["learning_curve"]]
    return [
        "## Modelo principal: LightGBM",
        "",
        "Taxa de aprendizado 0,05; 80% das linhas e das colunas sorteadas por árvore; sem SMOTE, "
        "sem outra reamostragem e sem peso extra para as fraudes (decisão D14). O controle do "
        "sobreajuste é explícito:",
        "",
        "- **parada pelo log loss** dos dados de validação, depois de 100 rodadas sem melhora: ele "
        "piora assim que o modelo fica confiante demais em dados novos;",
        "- **busca só de regularização:** folhas por árvore, mínimo de transações por folha e "
        "penalidade L2;",
        "- **validação cruzada temporal:** cada configuração é treinada e medida em quatro "
        "janelas; em cada uma, aprende com os meses anteriores e é medida nos seguintes;",
        "- **regra de um erro padrão:** entre as configurações cuja PR-AUC média fica a no máximo "
        "um erro padrão da melhor, vence a mais simples (menos folhas, depois mais transações "
        "por folha, depois mais L2).",
        "",
        "### Janelas da validação cruzada",
        "",
        *table(["Janela", "Treino", "Validação", "Transações de treino",
                "Transações de validação", "Fraudes na validação"], windows, "lllrrr"),
        "",
        "### Configurações",
        "",
        *table(["Folhas", "Mín. por folha", "L2", "PR-AUC média ± erro padrão",
                "Log loss médio", "Validação ÷ treino", "Árvores (média)", "Tempo"],
               rows, "rrrrrrrr"),
        "",
        "*Validação ÷ treino* é quantas vezes o log loss em dados novos é maior que nos dados de "
        "treino, na média das janelas: quanto mais perto de 1, menos o modelo decorou.",
        "",
        f"A maior PR-AUC média é {decimal(top['cv_average_precision'], 4)}; com o erro padrão, o "
        f"limite da regra fica em {decimal(floor, 4)}. **Escolhido:** {best['num_leaves']} "
        f"folhas, no mínimo {integer(best['min_child_samples'])} transações por folha, L2 = "
        f"{decimal(best['lambda_l2'], 0)}. O modelo final é o da última janela: treinado em "
        f"todo o 2019, com parada no período de seleção, {integer(final['trees'])} árvores e "
        f"PR-AUC de {decimal(final['average_precision'], 4)} na seleção.",
        "",
        "### Aprendizado e sobreajuste",
        "",
        "Log loss e PR-AUC do modelo final no treino e no período de seleção, conforme as árvores "
        "são somadas:",
        "",
        *table(["Árvores", "Log loss treino", "Log loss seleção", "Seleção ÷ treino",
                "PR-AUC treino", "PR-AUC seleção"], curve, "rrrrrr"),
        "",
        f"No modelo final, o log loss em dados novos é {times(final['gap'])} o do treino. "
        "Uma diferença perto de 1 indica que o modelo aprendeu padrões que se repetem, e não "
        "detalhes de cada transação do treino.",
        "",
        "![Curva de aprendizado do LightGBM escolhido](figures/learning-curve.png)",
        "",
    ]


def comparison_section(v: dict[str, Any]) -> list[str]:
    rows = []
    for model, row in v["comparison"].items():
        rows.append([
            MODEL_LABELS[model], decimal(row["selection_average_precision"], 4),
            decimal(row["average_precision"], 4),
            target_recall(row["recall_at_target_precision"], row["target_precision_reached"]),
            decimal(row.get("brier", float("nan")), 5),
            decimal(row.get("log_loss", float("nan")), 4), integer(row["frauds_flagged"]),
            integer(row["false_positives"]), decimal(row["total_cost"], 0),
        ])
    return [
        "## Comparação dos três modelos",
        "",
        "PR-AUC no período de seleção e, no período de calibração, que não participou da escolha "
        "dos hiperparâmetros: PR-AUC, recall com precisão de pelo menos 90%, qualidade das "
        "probabilidades finais (validação cruzada) e o resultado das decisões com os limites "
        "de cada modelo. A regra do valor manda para revisão tudo o que passa de "
        f"{decimal(v['rule'].threshold)}.",
        "",
        *table(["Modelo", "PR-AUC seleção", "PR-AUC calibração", "Recall com precisão ≥ 90%",
                "Brier", "Log loss", "Fraudes identificadas", "Falsos positivos", "Custo total"],
               rows, "lrrrrrrrr"),
        "",
        "Os limites da regressão logística e do LightGBM foram escolhidos neste mesmo período, "
        "então os custos acima são otimistas para os dois; a medida honesta virá do teste.",
        "",
        "![Curvas de precisão e recall dos três modelos](figures/precision-recall.png)",
        "",
    ]


def calibration_section(v: dict[str, Any]) -> list[str]:
    main, logistic = v["main_cal"], v["logistic_cal"]
    rows = []
    for label, calibrated in (("LightGBM", main), ("Regressão logística", logistic)):
        for method in calibration.METHODS:
            quality = calibrated.quality[method]
            mark = "**" if method == calibrated.method else ""
            rows.append([label, f"{mark}{METHOD_LABELS[method]}{mark}",
                         decimal(quality["brier"], 6), decimal(quality["log_loss"], 5),
                         decimal(quality["ece"], 5)])
    bands = [[band_label(band["band"]), integer(band["transactions"]),
              pct(band["mean_predicted"], 2), pct(band["observed_rate"], 2),
              interval(band["low"], band["high"], 1)]
             for band in main.reliability[main.method].to_dict("records")]
    return [
        "## Calibração",
        "",
        "Cada método foi avaliado por validação cruzada no período de calibração: o período foi "
        f"cortado em {calibration.BLOCKS} blocos consecutivos e cada bloco foi previsto por um "
        "calibrador ajustado nos outros, para nenhum método ser medido nos dados em que foi "
        "ajustado (decisão D15). Vence o menor log loss; o Brier e o erro de calibração (ECE, "
        "diferença média entre previsto e observado por faixa) confirmam ou não.",
        "",
        *table(["Modelo", "Método", "Brier", "Log loss", "ECE"], rows, "llrrr"),
        "",
        chosen_text(main.method),
        "",
        f"Faixas de probabilidade ({METHOD_PHRASES[main.method]}, validação cruzada):",
        "",
        *table(["Faixa prevista", "Transações", "Média prevista", "Taxa observada", "IC 95%"],
               bands, "lrrrl"),
        "",
        "![Curvas de calibração do LightGBM](figures/calibration.png)",
        "",
    ]


def chosen_text(method: str) -> str:
    if method == "none":
        return ("**Escolhida para o LightGBM: a probabilidade do próprio modelo, sem "
                "calibrador.** Ela errou menos em dados que não viu do que as duas calibrações, "
                "então um calibrador só acrescentaria uma peça sem melhorar a probabilidade.")
    return (f"**Escolhida para o LightGBM: {METHOD_LABELS[method]}**, ajustada depois em todo "
            "o período de calibração. A calibração é medida, não presumida: vence o método que "
            "menos erra em dados que não viu.")


def costs_table(costs: decision.Costs) -> list[str]:
    review = decimal(costs.review, 0)
    rows = [
        ["Fraude aprovada", "valor da transação", "financeiro"],
        ["Fraude bloqueada", "0 (a perda é evitada)", DASH],
        ["Fraude em revisão", f"{review} + {pct(1 - costs.review_detection, 0)} do valor "
         f"(a revisão detecta {pct(costs.review_detection, 0)})", "operacional e financeiro"],
        ["Legítimo em revisão", f"{review} + {decimal(costs.review_friction, 0)}",
         "operacional e experiência do cliente"],
        ["Legítimo bloqueado", decimal(costs.block_legit, 0),
         "experiência do cliente e comercial"],
        ["Legítimo aprovado", "0", DASH],
    ]
    return table(["Situação", "Custo", "Tipo"], rows, "lll")


def decision_section(v: dict[str, Any]) -> list[str]:
    thresholds = v["thresholds"]
    rows = []
    for label, limits, summary in v["decision"]["strategies"]:
        shown = label.replace("a partir de X", f"a partir de {decimal(v['rule'].threshold)}")
        rows.append([
            shown, DASH if limits is None else decimal(limits.review, 4),
            DASH if limits is None else decimal(limits.block, 4),
            integer(summary["approve_transactions"]), integer(summary["review_transactions"]),
            integer(summary["block_transactions"]), integer(summary["frauds_flagged"]),
            integer(summary["review_legit"]), integer(summary["block_legit"]),
            decimal(summary["financial_cost"], 0), decimal(summary["operational_cost"], 0),
            decimal(summary["experience_cost"], 0), decimal(summary["total_cost"], 0),
        ])
    sensitivity = []
    for label, _, result, summary in v["decision"]["sensitivity"]:
        sensitivity.append([
            label, decimal(result.thresholds.review, 4), decimal(result.thresholds.block, 4),
            pct(summary["review_share"], 2), integer(summary["block_transactions"]),
            integer(summary["frauds_flagged"]), integer(summary["block_legit"]),
            decimal(summary["total_cost"], 0),
        ])
    costs = decision.BASE_COSTS
    return [
        "## Custos e limites de decisão",
        "",
        "### Hipóteses de custo",
        "",
        "Não há dados reais de custo para uma base simulada; os valores abaixo são hipóteses "
        "explícitas, na unidade dos valores da base (decisão D16):",
        "",
        *costs_table(costs),
        "",
        f"Capacidade da equipe de revisão: no máximo {pct(costs.max_review_share, 0)} das "
        "transações. A análise de sensibilidade, mais abaixo, mostra o efeito de mudar cada "
        "hipótese.",
        "",
        "### Limites escolhidos",
        "",
        f"- **Aprovar:** probabilidade de fraude menor que {decimal(thresholds.review, 4)};",
        f"- **Revisão humana:** de {decimal(thresholds.review, 4)} até antes de "
        f"{decimal(thresholds.block, 4)};",
        f"- **Bloquear:** a partir de {decimal(thresholds.block, 4)}.",
        "",
        "Os dois limites minimizam o custo total no período de calibração, usando as "
        "probabilidades da validação cruzada e respeitando a capacidade de revisão. Comparação "
        "com outras estratégias, no mesmo período:",
        "",
        *table(["Estratégia", "Limite inferior", "Limite superior", "Aprovadas", "Em revisão",
                "Bloqueadas", "Fraudes identificadas", "Legítimos em revisão",
                "Legítimos bloqueados", "Custo financeiro", "Custo operacional",
                "Custo de experiência", "Custo total"], rows, "lrrrrrrrrrrrr"),
        "",
        "### Sensibilidade às hipóteses",
        "",
        "Os limites procurados de novo sob cada hipótese, uma de cada vez:",
        "",
        *table(["Hipótese", "Limite inferior", "Limite superior", "Em revisão", "Bloqueadas",
                "Fraudes identificadas", "Legítimos bloqueados", "Custo total"], sensitivity,
               "lrrrrrrr"),
        "",
    ]


def importance_lines(shap: dict[str, Any], figure: str) -> list[str]:
    """Top signals by mean |SHAP|, next to LightGBM's gain rank, and the summary figure."""
    rows = [[integer(row["shap_rank"]), f"`{row['feature']}`", decimal(row["mean_abs_shap"], 3),
             pct(row["shap_share"]), integer(row["gain_rank"])]
            for row in shap["importance"].head(12).to_dict("records")]
    return [
        *table(["Posição", "Sinal", "SHAP absoluto médio", "Participação",
                "Posição por ganho"],
               rows, "rlrrr"),
        "",
        f"![Resumo SHAP: contribuição de cada sinal por transação](figures/{figure})",
        "",
        "SHAP explica o modelo, não as causas no mundo real; sinais correlacionados dividem o "
        "crédito entre si (por exemplo, o valor e a razão entre o valor e a média do cartão).",
        "",
    ]


def example_lines(examples: list[dict[str, Any]]) -> list[str]:
    """Each typical case with its reasons and their SHAP contributions."""
    lines = []
    for example in examples:
        label = decision.DECISION_LABELS[example["decision"]]
        lines += [f"**{example['case']}** (probabilidade {pct(example['probability'], 1)}, "
                  f"decisão: {label}):", ""]
        lines += [f"- {reason.text} [{'+' if reason.contribution > 0 else ''}"
                  f"{decimal(reason.contribution, 2)}]" for reason in example["reasons"]]
        lines.append("")
    return lines


def shap_section(v: dict[str, Any]) -> list[str]:
    shap = v["shap"]
    return [
        "## Explicações (SHAP)",
        "",
        f"Valores SHAP (TreeSHAP) calculados pelo próprio LightGBM numa amostra aleatória de "
        f"{integer(shap['sample'])} transações do período de calibração (decisão D19). A "
        "contribuição está em log-odds do modelo. A probabilidade final é uma função crescente "
        "desse escore, com ou sem calibrador, então um sinal que aumenta o escore também aumenta "
        "a probabilidade. A última coluna compara com a importância por ganho do próprio "
        "LightGBM.",
        "",
        *importance_lines(shap, "shap-summary.png"),
        "### Exemplos individuais",
        "",
        "Um caso típico de cada resultado no período de calibração, com os cinco sinais que mais "
        "pesaram. O texto vem do valor real do sinal; o número entre colchetes é a contribuição "
        "SHAP.",
        "",
        *example_lines(shap["examples"]),
    ]


def fairness_lines(results: dict[str, dict[str, Any]]) -> list[str]:
    """Per attribute: error rates by group with intervals, and the differences that stand out."""
    lines: list[str] = []
    order = [label for _, _, label in fairness.AGE_BANDS]
    for attribute, result in results.items():
        frame = result["table"]
        if attribute == "age_band":
            frame = frame.set_index("group").reindex(
                [group for group in order if group in set(frame["group"])]).reset_index()
        rows = [[row["group"], integer(row["transactions"]), integer(row["cards"]),
                 integer(row["frauds"]), pct(row["fraud_rate"], 2),
                 f"{pct(row['fpr'], 2)} ({interval(row['fpr_low'], row['fpr_high'], 2)})",
                 f"{pct(row['fnr'])} ({interval(row['fnr_low'], row['fnr_high'])})",
                 f"{pct(row['precision'])} "
                 f"({interval(row['precision_low'], row['precision_high'])})",
                 pct(row["legit_block_rate"], 2), pct(row["review_rate"], 2),
                 "" if row["reliable"] else "amostra pequena"]
                for row in frame.to_dict("records")]
        gaps = result["differences"]
        standing = gaps[gaps["excludes_zero"]] if not gaps.empty else gaps
        flagged = standing[standing["reliable"]]
        small = len(standing) - len(flagged)
        lines += [
            f"### {ATTRIBUTE_LABELS[attribute]}",
            "",
            *table(["Grupo", "Transações", "Cartões", "Fraudes", "Taxa de fraude",
                    "Falsos positivos (IC 95%)", "Fraudes não detectadas (IC 95%)",
                    "Precisão (IC 95%)", "Legítimos bloqueados", "Em revisão", "Aviso"],
                   rows, "lrrrrrrrrrl"),
            "",
        ]
        if small:
            plural = small > 1
            lines += [f"{small} diferença{'s' if plural else ''} com intervalo que exclui zero "
                      f"envolve{'m' if plural else ''} grupos com menos de 30 fraudes e "
                      f"não sustenta{'m' if plural else ''} conclusão.", ""]
        if flagged.empty:
            lines += ["Entre grupos com pelo menos 30 fraudes, nenhuma diferença em relação ao "
                      f"grupo de referência ({result['reference']}) tem intervalo de 95% que "
                      "exclua zero.", ""]
            continue
        lines += [f"Diferenças entre grupos com pelo menos 30 fraudes cujo intervalo de 95% "
                  f"exclui zero (referência: "
                  f"{result['reference']}):", ""]
        lines += [f"- {row['group']}, {GAP_LABELS[row['metric']].lower()}: "
                  f"{'+' if row['difference'] > 0 else ''}{decimal(100 * row['difference'], 2)}"
                  f" pontos percentuais ({decimal(100 * row['low'], 2)} a "
                  f"{decimal(100 * row['high'], 2)})" for row in flagged.to_dict("records")]
        lines += ["", "São indícios para investigar, não prova de discriminação.", ""]
    return lines


def fairness_section(v: dict[str, Any]) -> list[str]:
    return [
        "## Auditoria de equidade (desenvolvimento)",
        "",
        "Gênero e faixa etária não entram no modelo; foram juntados às decisões do período de "
        "calibração só para comparar erros entre grupos (decisão D20). Os intervalos de 95% vêm "
        "de bootstrap por cartão, porque as transações de um mesmo cliente não são "
        "independentes. A avaliação definitiva será no teste.",
        "",
        *fairness_lines(v["fairness"]),
    ]


def frozen_section(v: dict[str, Any]) -> list[str]:
    manifest, baseline = v["manifest"], v["baseline_manifest"]
    thresholds = v["thresholds"]
    files = [[f"`{name}`", f"`{digest}`"] for name, digest in manifest["files"].items()]
    files += [[f"`baselines/{name}`", f"`{digest}`"] for name, digest in baseline["files"].items()]
    return [
        "## O que fica congelado para o teste",
        "",
        f"- **Modelo:** `models/fraud-detector/`, versão {manifest['model_version']}, LightGBM "
        f"({METHOD_PHRASES[v['main_cal'].method]}); assinatura do pacote "
        f"`{manifest['bundle_sha256']}`.",
        f"- **Limites:** revisão a partir de {decimal(thresholds.review, 4)}, bloqueio a partir "
        f"de {decimal(thresholds.block, 4)}, com as hipóteses de custo acima.",
        f"- **Referências:** `models/baselines/`, assinatura `{baseline['bundle_sha256']}`.",
        "- **Meta de precisão:** recall com precisão de pelo menos "
        f"{pct(metrics.PRECISION_TARGET, 0)}. Com essa precisão, no máximo um a cada dez alertas "
        "atinge um cliente legítimo, o nível em que um bloqueio automático ainda é defensável.",
        "- **Leitura sim/não das decisões:** *alerta* é revisão ou bloqueio. Precisão, recall, F1 "
        "e matriz de confusão são calculados para o alerta e, à parte, para o bloqueio.",
        "- **Métricas:** PR-AUC, precisão, recall, F1, matriz de confusão, Brier, log loss, ECE, "
        "fraudes identificadas, recall com precisão ≥ 90% e custos financeiro, operacional e de "
        "experiência com as hipóteses base; por mês e por categoria, com aviso para grupos com "
        "menos de 30 fraudes; equidade por gênero e faixa etária.",
        "- **Procedimento:** `python -m pipeline.evaluate`, uma única vez. Ele confere as "
        "assinaturas abaixo antes de abrir o `fraudTest.csv` e se recusa a rodar de novo depois "
        "de gravar `models/test-evaluation.json`. Nada do teste volta para o modelo nesta fase.",
        "",
        *table(["Arquivo", "SHA-256"], files, "ll"),
        "",
        "## Rastreamento",
        "",
        f"Experimento `payment-fraud-detection` no MLflow, execução `{v['run_id']}` (com uma "
        "execução filha por configuração testada). Para abrir: `mlflow ui --backend-store-uri "
        "sqlite:///mlruns/mlflow.db`. Tempo total de desenvolvimento: "
        f"{decimal(v['timings']['total_seconds'] / 60, 1)} minutos.",
        "",
    ]


def render(results: Any) -> str:
    v = results.values
    lines = [
        "# Seleção do modelo (fase 4, antes do teste)",
        "",
        f"Gerado por `python -m pipeline.train` em {moment(v['created'])}. Usa só o "
        "`fraudTrain.csv`: o `fraudTest.csv` não foi aberto. Este relatório registra as escolhas "
        "que a avaliação única no teste (`python -m pipeline.evaluate`) vai medir. As decisões "
        "D12 a D20 de [`docs/decisoes-tecnicas.md`](../docs/decisoes-tecnicas.md) explicam cada "
        "uma.",
        "",
        "Os dados são simulados e os custos são hipóteses: os números servem para comparar "
        "estratégias, não como estimativa de um negócio real.",
        "",
        *sets_section(results.sets),
        *rule_section(v),
        *logistic_section(v),
        *lightgbm_section(v),
        *comparison_section(v),
        *calibration_section(v),
        *decision_section(v),
        *shap_section(v),
        *fairness_section(v),
        *frozen_section(v),
    ]
    return "\n".join(lines).rstrip() + "\n"
