"""reports/test-evaluation.md: the single evaluation on fraudTest.csv."""

from __future__ import annotations

from typing import Any, Final

import pandas as pd

from pipeline.modeling import metrics
from pipeline.modeling.report_selection import (
    METHOD_PHRASES,
    MODEL_LABELS,
    costs_table,
    example_lines,
    fairness_lines,
    importance_lines,
)
from pipeline.modeling.reporting import (
    band_label,
    decimal,
    integer,
    interval,
    moment,
    pct,
    table,
    target_recall,
)

MONTHS: Final = {"01": "jan", "02": "fev", "03": "mar", "04": "abr", "05": "mai", "06": "jun",
                 "07": "jul", "08": "ago", "09": "set", "10": "out", "11": "nov", "12": "dez"}
SMALL: Final = "amostra pequena"


def month_label(value: str) -> str:
    year, month = value.split("-")
    return f"{MONTHS[month]}/{year}"


def header(v: dict[str, Any]) -> list[str]:
    record, period = v["record"], v["period"]
    model = v["model"]
    return [
        "# Avaliação final no teste (fase 4)",
        "",
        f"Avaliação única, gerada por `python -m pipeline.evaluate` em {moment(v['completed'])}, "
        f"do modelo **{model.version}** (pacote `{record['bundle_sha256']}`), congelado antes em "
        "[`model-selection.md`](model-selection.md). Nenhuma escolha de modelagem foi feita "
        "depois de abrir o teste, e nada deste relatório volta para o modelo nesta fase.",
        "",
        f"- **Teste:** `fraudTest.csv` (SHA-256 `{record['test_source_sha256']}`), de "
        f"{period['first']:%d/%m/%Y} a {period['last']:%d/%m/%Y}: "
        f"{integer(period['transactions'])} transações, {integer(period['frauds'])} fraudes "
        f"({pct(period['frauds'] / period['transactions'], 2)}), {integer(period['cards'])} "
        "cartões. Os sinais usam o arquivo de treino como passado (decisões D8 e D11).",
        f"- **Decisão congelada:** revisão a partir de {decimal(v['thresholds'].review, 4)}, "
        f"bloqueio a partir de {decimal(v['thresholds'].block, 4)} de probabilidade de fraude "
        f"({METHOD_PHRASES[model.calibrator.method]}).",
        f"- **Meta pré-registrada:** recall com precisão de pelo menos "
        f"{pct(metrics.PRECISION_TARGET, 0)}. *Alerta* é revisão ou bloqueio.",
        "",
        "Os dados são simulados e os custos são hipóteses: os números comparam estratégias, não "
        "estimam um negócio real.",
        "",
    ]


def results_section(v: dict[str, Any]) -> list[str]:
    ranking, probability = [], []
    for model in ("rule", "logistic", "lightgbm"):
        r = v["overall"][model]
        ranking.append([
            MODEL_LABELS[model], decimal(r["average_precision"], 4),
            target_recall(r["recall_at_target_precision"], r["target_precision_reached"]),
            pct(r["alert_precision"]), pct(r["alert_recall"]), pct(r["alert_f1"]),
            pct(r["block_precision"]), pct(r["block_recall"]),
        ])
        probability.append([
            MODEL_LABELS[model], decimal(r.get("brier", float("nan")), 5),
            decimal(r.get("log_loss", float("nan")), 4), decimal(r.get("ece", float("nan")), 5),
            integer(r["frauds_flagged"]), integer(r["review_legit"]), integer(r["block_legit"]),
            decimal(r["total_cost"], 0),
        ])
    return [
        "## Resultados principais",
        "",
        "Cada modelo com os seus próprios limites, escolhidos no período de calibração. A regra "
        f"do valor manda para revisão tudo a partir de {decimal(v['rule'].threshold)}.",
        "",
        *table(["Modelo", "PR-AUC", "Recall com precisão ≥ 90%", "Precisão do alerta",
                "Recall do alerta", "F1 do alerta", "Precisão do bloqueio", "Recall do bloqueio"],
               ranking, "lrrrrrrr"),
        "",
        *table(["Modelo", "Brier", "Log loss", "ECE", "Fraudes identificadas",
                "Legítimos em revisão", "Legítimos bloqueados", "Custo total"], probability,
               "lrrrrrrr"),
        "",
        "![Curvas de precisão e recall no teste](figures/precision-recall-test.png)",
        "",
    ]


def confusion_section(v: dict[str, Any]) -> list[str]:
    r = v["overall"]["lightgbm"]
    bands = [[label, integer(r[f"{name}_transactions"]), integer(r[f"{name}_frauds"]),
              integer(r[f"{name}_legit"])]
             for name, label in (("approve", "Aprovar"), ("review", "Revisão humana"),
                                 ("block", "Bloquear"))]
    return [
        "## Matriz de confusão do LightGBM",
        "",
        *table(["Decisão", "Transações", "Fraudes", "Legítimas"], bands, "lrrr"),
        "",
        *table(["", "Fraude", "Legítima"], [
            ["Alerta (revisão ou bloqueio)", integer(r["alert_tp"]), integer(r["alert_fp"])],
            ["Aprovada", integer(r["alert_fn"]), integer(r["alert_tn"])],
        ], "lrr"),
        "",
        *table(["", "Fraude", "Legítima"], [
            ["Bloqueada", integer(r["block_tp"]), integer(r["block_fp"])],
            ["Não bloqueada", integer(r["block_fn"]), integer(r["block_tn"])],
        ], "lrr"),
        "",
    ]


def cost_section(v: dict[str, Any]) -> list[str]:
    r = v["overall"]["lightgbm"]
    rows = [[label, decimal(summary["financial_cost"], 0), decimal(summary["operational_cost"], 0),
             decimal(summary["experience_cost"], 0), decimal(summary["total_cost"], 0)]
            for label, summary in v["sensitivity"]]
    return [
        "## Custos estimados",
        "",
        *costs_table(v["costs"]),
        "",
        f"Com o LightGBM, as fraudes do teste somam {decimal(r['fraud_amount'], 0)}: é o que se "
        f"perderia aprovando tudo. A perda evitada é de {decimal(r['loss_avoided'], 0)}, e o custo "
        f"total fica em {decimal(r['total_cost'], 0)}: {decimal(r['financial_cost'], 0)} "
        f"financeiro, {decimal(r['operational_cost'], 0)} operacional e "
        f"{decimal(r['experience_cost'], 0)} de experiência do cliente. "
        f"{pct(r['review_share'], 2)} das transações foram para revisão.",
        "",
        "Com os limites congelados, o custo sob as outras hipóteses da análise de "
        "sensibilidade:",
        "",
        *table(["Hipótese", "Financeiro", "Operacional", "Experiência", "Total"], rows, "lrrrr"),
        "",
    ]


def calibration_section(v: dict[str, Any]) -> list[str]:
    bands = v["reliability"]["LightGBM"]
    predicted = float((bands["transactions"] * bands["mean_predicted"]).sum()
                      / bands["transactions"].sum())
    observed = v["period"]["frauds"] / v["period"]["transactions"]
    rows = [[band_label(band["band"]), integer(band["transactions"]),
             pct(band["mean_predicted"], 2), pct(band["observed_rate"], 2),
             interval(band["low"], band["high"], 1)] for band in bands.to_dict("records")]
    return [
        "## Calibração no teste",
        "",
        f"Probabilidade média prevista: {pct(predicted, 3)}; taxa de fraude observada: "
        f"{pct(observed, 3)}.",
        "",
        *table(["Faixa prevista", "Transações", "Média prevista", "Taxa observada", "IC 95%"],
               rows, "lrrrl"),
        "",
        "![Curvas de calibração no teste](figures/calibration-test.png)",
        "",
    ]


def group_rows(frame: pd.DataFrame, label: Any) -> list[list[str]]:
    return [[label(row["group"]), integer(row["transactions"]), integer(row["frauds"]),
             pct(row["fraud_rate"], 2), decimal(row["average_precision"], 3),
             pct(row["precision"]), pct(row["recall"]), integer(row["false_positives"]),
             pct(row["review_share"], 2), pct(row["block_share"], 2),
             decimal(row["total_cost"], 0), "" if row["reliable"] else SMALL]
            for row in frame.to_dict("records")]


GROUP_HEADERS: Final = ["Transações", "Fraudes", "Taxa de fraude", "PR-AUC", "Precisão",
                        "Recall", "Falsos positivos", "Em revisão", "Bloqueadas", "Custo",
                        "Aviso"]


def extremes(frame: pd.DataFrame, label: Any) -> str:
    """Lowest and highest recall and precision among the groups with enough frauds."""
    reliable = frame[frame["reliable"]]
    if len(reliable) < 2:
        return "Há menos de dois grupos com pelo menos 30 fraudes: sem comparação confiável."
    parts = []
    for metric, name in (("recall", "o recall"), ("precision", "a precisão")):
        known = reliable[reliable[metric].notna()]
        low, high = known.loc[known[metric].idxmin()], known.loc[known[metric].idxmax()]
        parts.append(f"{name} vai de {pct(low[metric])} ({label(low['group'])}) a "
                     f"{pct(high[metric])} ({label(high['group'])})")
    return f"Entre os grupos com pelo menos 30 fraudes, {parts[0]}; {parts[1]}."


def months_section(v: dict[str, Any]) -> list[str]:
    frame = v["months"]
    return [
        "## Desempenho por mês",
        "",
        *table(["Mês", *GROUP_HEADERS], group_rows(frame, month_label), "lrrrrrrrrrrl"),
        "",
        extremes(frame, month_label),
        "",
    ]


def categories_section(v: dict[str, Any]) -> list[str]:
    frame = v["categories"].sort_values("transactions", ascending=False)
    small = ", ".join(f"`{group}`" for group in frame.loc[~frame["reliable"], "group"])
    return [
        "## Desempenho por categoria",
        "",
        *table(["Categoria", *GROUP_HEADERS],
               group_rows(frame, lambda group: f"`{group}`"), "lrrrrrrrrrrl"),
        "",
        extremes(frame, lambda group: f"`{group}`"),
        "",
        f"Com menos de 30 fraudes, sem conclusão confiável: {small}." if small else
        "Todas as categorias têm pelo menos 30 fraudes.",
        "",
    ]


def shap_section(v: dict[str, Any]) -> list[str]:
    shap = v["shap"]
    return [
        "## Explicações (SHAP) no teste",
        "",
        f"Mesmo cálculo da seleção, numa amostra aleatória de {integer(shap['sample'])} "
        "transações do teste. Contribuições em log-odds do modelo.",
        "",
        *importance_lines(shap, "shap-summary-test.png"),
        "### Exemplos individuais",
        "",
        "Um caso típico de cada resultado no teste, com os cinco sinais que mais pesaram e a "
        "contribuição SHAP entre colchetes.",
        "",
        *example_lines(shap["examples"]),
    ]


def fairness_section(v: dict[str, Any]) -> list[str]:
    return [
        "## Auditoria de equidade",
        "",
        "Gênero e faixa etária do teste ficaram só em memória e foram juntados às decisões "
        "depois de tomadas (decisão D20). Intervalos de 95% por bootstrap de cartões. Uma "
        "diferença com intervalo que exclui zero é indício para investigar, não prova de "
        "discriminação; um intervalo que inclui zero não prova igualdade. Os dados são "
        "simulados: nada aqui vale automaticamente para uma população real.",
        "",
        *fairness_lines(v["fairness"]),
    ]


def limitations() -> list[str]:
    return [
        "## Limitações",
        "",
        "- Os dados são simulados (Sparkov): os padrões de fraude seguem as regras do simulador "
        "e não a variedade de uma operação real.",
        "- Os custos são hipóteses; a análise de sensibilidade mostra quanto o resultado depende "
        "delas.",
        "- O modelo só vê os 28 sinais; não vê dispositivo, endereço de entrega, histórico do "
        "lojista nem contestações.",
        "- Mudanças de comportamento depois de dezembro de 2020 não estão medidas aqui.",
        "",
    ]


def render(v: dict[str, Any]) -> str:
    lines = [
        *header(v),
        *results_section(v),
        *confusion_section(v),
        *cost_section(v),
        *calibration_section(v),
        *months_section(v),
        *categories_section(v),
        *shap_section(v),
        *fairness_section(v),
        *limitations(),
    ]
    return "\n".join(lines).rstrip() + "\n"
