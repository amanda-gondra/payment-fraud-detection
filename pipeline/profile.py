"""Profile the raw Sparkov files before any modelling.

The profile answers the questions that shape the next phases:

- which columns exist, with which types, and how many values are missing;
- whether rows or transaction ids repeat;
- whether the values are plausible: amounts, coordinates, ages, categories;
- which period each file covers, and whether the test file starts after the training file ends;
- whether the rows are in time order, and which column is the real clock;
- how rare fraud is, overall and month by month.

Only counts and aggregates leave this script. Names, card numbers, streets and birth dates are
never printed or written: the report says how many distinct values a column has, never what
they are.

The profile describes and does not correct. A structural problem (unexpected columns, a first
column that is not a row index, a target other than 0 and 1) stops it; every other finding goes
to the report, and the preparation step decides what to do with it.

    python -m pipeline.profile

The files are first checked against data/manifest.json, so the profile always describes the exact
files the manifest records. The result goes to reports/data-profile.md. The report has no
timestamp: the same files and the same library versions always produce the same report.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from pipeline import download

SPLITS: Final = {"train": "fraudTrain.csv", "test": "fraudTest.csv"}
LABELS: Final = {"train": "treino", "test": "teste"}
LIBRARIES: Final = ("pandas", "pyarrow", "numpy")

LEFTOVER_INDEX: Final = "Unnamed: 0"
TIMESTAMP: Final = "trans_date_trans_time"
UNIX_TIME: Final = "unix_time"
CARD: Final = "cc_num"
MERCHANT: Final = "merchant"
CATEGORY: Final = "category"
AMOUNT: Final = "amt"
GENDER: Final = "gender"
CITY_POPULATION: Final = "city_pop"
BIRTH_DATE: Final = "dob"
TRANSACTION_ID: Final = "trans_num"
TARGET: Final = "is_fraud"
EXPECTED_COLUMNS: Final = (
    TIMESTAMP, CARD, MERCHANT, CATEGORY, AMOUNT, "first", "last", GENDER, "street", "city",
    "state", "zip", "lat", "long", CITY_POPULATION, "job", BIRTH_DATE, TRANSACTION_ID, UNIX_TIME,
    "merch_lat", "merch_long", TARGET,
)
# Largest absolute value a valid latitude or longitude can have.
COORDINATE_LIMITS: Final = {"lat": 90, "merch_lat": 90, "long": 180, "merch_long": 180}

REPORT: Final = download.ROOT / "reports" / "data-profile.md"
SECONDS_PER_DAY: Final = 86_400
DAYS_PER_YEAR: Final = 365.25
PT_BR_SEPARATORS: Final = str.maketrans(",.", ".,")


@dataclass(frozen=True)
class SplitProfile:
    """Counts that describe one file."""

    rows: int
    frauds: int
    first: pd.Timestamp
    last: pd.Timestamp
    in_time_order: bool
    cards: int
    merchants: int
    category_names: tuple[str, ...]
    duplicated_rows: int
    duplicated_ids: int
    tie_groups: int
    tie_rows: int
    clock_gap_days: tuple[float, float]

    @property
    def fraud_rate(self) -> float:
        return self.frauds / self.rows


@dataclass(frozen=True)
class ValueProfile:
    """Checks that the values of one file are plausible."""

    amount_min: float
    amount_median: float
    amount_max: float
    non_positive_amounts: int
    coordinates_out_of_range: int
    non_positive_city_population: int
    genders: tuple[str, ...]
    age_min: float
    age_max: float
    born_after_transaction: int


@dataclass(frozen=True)
class Overlap:
    """How the test file relates to the training file."""

    gap: pd.Timedelta
    shared_ids: int
    test_cards: int
    test_cards_seen_in_train: int


@dataclass(frozen=True)
class DataProfile:
    """Everything the report shows, gathered before it is written."""

    files: dict[str, download.FileInfo]
    splits: dict[str, SplitProfile]
    values: dict[str, ValueProfile]
    overlap: Overlap
    columns: list[dict[str, object]]
    months: pd.DataFrame


def load(path: Path) -> pd.DataFrame:
    """Read one raw file, check its columns and drop the leftover row index."""
    frame = pd.read_csv(path, engine="pyarrow", parse_dates=[TIMESTAMP])
    columns = tuple(frame.columns)
    if columns != (LEFTOVER_INDEX, *EXPECTED_COLUMNS):
        raise SystemExit(f"{path.name}: unexpected columns {columns}")
    if not np.array_equal(frame[LEFTOVER_INDEX].to_numpy(), np.arange(len(frame))):
        raise SystemExit(f"{path.name}: the first column is not a plain row index")
    if not set(frame[TARGET].unique()) <= {0, 1}:
        raise SystemExit(f"{path.name}: {TARGET} has values other than 0 and 1")
    return frame.drop(columns=LEFTOVER_INDEX)


def profile_split(frame: pd.DataFrame) -> SplitProfile:
    """Count rows, frauds, repeats and the period covered by one file."""
    times = frame[TIMESTAMP]
    epoch_seconds = (times - pd.Timestamp(0)) // pd.Timedelta(seconds=1)
    gap_days = (frame[UNIX_TIME] - epoch_seconds) / SECONDS_PER_DAY
    # Several transactions of the same card in the same second: their order is ambiguous.
    per_card_second = frame.groupby([CARD, TIMESTAMP]).size()
    ties = per_card_second[per_card_second > 1]
    return SplitProfile(
        rows=len(frame),
        frauds=int(frame[TARGET].sum()),
        first=times.min(),
        last=times.max(),
        in_time_order=bool(times.is_monotonic_increasing),
        cards=frame[CARD].nunique(),
        merchants=frame[MERCHANT].nunique(),
        category_names=tuple(sorted(str(name) for name in frame[CATEGORY].unique())),
        # trans_num is unique by construction, so a repeated row must be looked for without it.
        duplicated_rows=int(frame.drop(columns=TRANSACTION_ID).duplicated().sum()),
        duplicated_ids=int(frame[TRANSACTION_ID].duplicated().sum()),
        tie_groups=len(ties),
        tie_rows=int(ties.sum()),
        clock_gap_days=(float(gap_days.min()), float(gap_days.max())),
    )


def profile_values(frame: pd.DataFrame) -> ValueProfile:
    """Ranges and impossible values: amounts, coordinates, population, gender and age."""
    amounts = frame[AMOUNT]
    birth = pd.to_datetime(frame[BIRTH_DATE])
    age_years = (frame[TIMESTAMP] - birth).dt.days / DAYS_PER_YEAR
    return ValueProfile(
        amount_min=float(amounts.min()),
        amount_median=float(amounts.median()),
        amount_max=float(amounts.max()),
        non_positive_amounts=int((amounts <= 0).sum()),
        coordinates_out_of_range=sum(
            int((frame[column].abs() > limit).sum()) for column, limit in COORDINATE_LIMITS.items()
        ),
        non_positive_city_population=int((frame[CITY_POPULATION] <= 0).sum()),
        genders=tuple(sorted(str(value) for value in frame[GENDER].unique())),
        age_min=float(age_years.min()),
        age_max=float(age_years.max()),
        born_after_transaction=int((birth > frame[TIMESTAMP]).sum()),
    )


def compare(train: pd.DataFrame, test: pd.DataFrame) -> Overlap:
    """Check that the test file comes after the training file and shares no transaction."""
    test_cards = pd.Index(test[CARD].unique())
    return Overlap(
        gap=test[TIMESTAMP].min() - train[TIMESTAMP].max(),
        shared_ids=int(test[TRANSACTION_ID].isin(train[TRANSACTION_ID]).sum()),
        test_cards=len(test_cards),
        test_cards_seen_in_train=int(test_cards.isin(train[CARD]).sum()),
    )


def profile_columns(frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    """Type, missing values and number of distinct values of each column, per file."""
    rows = []
    for column in EXPECTED_COLUMNS:
        types = sorted({str(frame[column].dtype) for frame in frames.values()})
        row: dict[str, object] = {"column": column, "dtype": " / ".join(types)}
        for split, frame in frames.items():
            row[f"missing_{split}"] = int(frame[column].isna().sum())
            row[f"distinct_{split}"] = int(frame[column].nunique())
        rows.append(row)
    return rows


def monthly_fraud(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Transactions, frauds and fraud rate per calendar month, per file."""
    parts = []
    for split, frame in frames.items():
        month = frame[TIMESTAMP].dt.to_period("M").rename("month")
        counts = frame.groupby(month)[TARGET].agg(transactions="size", frauds="sum")
        parts.append(counts.assign(split=split).reset_index())
    by_month = pd.concat(parts, ignore_index=True)
    return by_month.assign(rate=by_month["frauds"] / by_month["transactions"])


def build_profile(
    files: dict[str, download.FileInfo], frames: dict[str, pd.DataFrame]
) -> DataProfile:
    """Run every check on the loaded files ("train" and "test")."""
    return DataProfile(
        files=files,
        splits={split: profile_split(frame) for split, frame in frames.items()},
        values={split: profile_values(frame) for split, frame in frames.items()},
        overlap=compare(frames["train"], frames["test"]),
        columns=profile_columns(frames),
        months=monthly_fraud(frames),
    )


def number(value: int) -> str:
    return f"{value:,}".translate(PT_BR_SEPARATORS)


def decimal(value: float, places: int = 2) -> str:
    return f"{value:,.{places}f}".translate(PT_BR_SEPARATORS)


def percent(value: float) -> str:
    return f"{value:.2%}".translate(PT_BR_SEPARATORS)


def moment(value: pd.Timestamp) -> str:
    return value.strftime("%d/%m/%Y %H:%M:%S")


def yes_no(value: bool) -> str:
    return "sim" if value else "não"


def table(rows: list[tuple[str, str, str]]) -> list[str]:
    """Two-column Markdown table, training file on the left and test file on the right."""
    return [
        "| | treino | teste |",
        "|---|---:|---:|",
        *(f"| {label} | {left} | {right} |" for label, left, right in rows),
    ]


def render(data: DataProfile) -> str:
    """Write the profile as a Markdown report in Portuguese."""
    train, test = data.splits["train"], data.splits["test"]
    train_values, test_values = data.values["train"], data.values["test"]
    overlap = data.overlap
    gaps = [abs(day) for split in data.splits.values() for day in split.clock_gap_days]
    low_gap, high_gap = min(gaps), max(gaps)
    same_categories = train.category_names == test.category_names
    libraries = ", ".join(f"{name} {version(name)}" for name in LIBRARIES)
    summary = [
        ("Transações", number(train.rows), number(test.rows)),
        ("Fraudes", number(train.frauds), number(test.frauds)),
        ("Taxa de fraude", percent(train.fraud_rate), percent(test.fraud_rate)),
        ("Primeira transação", moment(train.first), moment(test.first)),
        ("Última transação", moment(train.last), moment(test.last)),
        ("Cartões", number(train.cards), number(test.cards)),
        ("Lojistas", number(train.merchants), number(test.merchants)),
        ("Categorias", number(len(train.category_names)), number(len(test.category_names))),
        (
            f"Linhas repetidas (sem contar `{TRANSACTION_ID}`)",
            number(train.duplicated_rows),
            number(test.duplicated_rows),
        ),
        (f"`{TRANSACTION_ID}` repetido", number(train.duplicated_ids), number(test.duplicated_ids)),
        (
            "Grupos do mesmo cartão no mesmo segundo",
            f"{number(train.tie_groups)} ({number(train.tie_rows)} transações)",
            f"{number(test.tie_groups)} ({number(test.tie_rows)} transações)",
        ),
        ("Linhas em ordem de data e hora", yes_no(train.in_time_order), yes_no(test.in_time_order)),
    ]
    checks = [
        (
            f"Menor valor (`{AMOUNT}`)",
            decimal(train_values.amount_min),
            decimal(test_values.amount_min),
        ),
        (
            "Mediana do valor",
            decimal(train_values.amount_median),
            decimal(test_values.amount_median),
        ),
        ("Maior valor", decimal(train_values.amount_max), decimal(test_values.amount_max)),
        (
            "Valores menores ou iguais a zero",
            number(train_values.non_positive_amounts),
            number(test_values.non_positive_amounts),
        ),
        (
            "Coordenadas fora da faixa válida",
            number(train_values.coordinates_out_of_range),
            number(test_values.coordinates_out_of_range),
        ),
        (
            f"`{CITY_POPULATION}` menor ou igual a zero",
            number(train_values.non_positive_city_population),
            number(test_values.non_positive_city_population),
        ),
        (f"Valores de `{GENDER}`", ", ".join(train_values.genders), ", ".join(test_values.genders)),
        (
            "Idade na data da transação (anos)",
            f"{decimal(train_values.age_min, 1)} a {decimal(train_values.age_max, 1)}",
            f"{decimal(test_values.age_min, 1)} a {decimal(test_values.age_max, 1)}",
        ),
        (
            "Nascimento depois da transação",
            number(train_values.born_after_transaction),
            number(test_values.born_after_transaction),
        ),
    ]
    lines = [
        "# Perfil dos dados",
        "",
        "Gerado por `python -m pipeline.profile`, a partir dos arquivos conferidos com "
        "`data/manifest.json`. O relatório só tem contagens e estatísticas agregadas: nenhum "
        "nome, número de cartão, endereço ou data de nascimento.",
        "",
        "## Origem",
        "",
        "| Arquivo | Bytes | SHA-256 |",
        "|---|---:|---|",
        *(
            f"| `{name}` | {number(int(info['bytes']))} | `{info['sha256']}` |"
            for name, info in data.files.items()
        ),
        "",
        f"Bibliotecas: {libraries}.",
        "",
        "## Visão geral",
        "",
        *table(summary),
        "",
        "## Entre os arquivos",
        "",
        f"- O teste começa {number(int(overlap.gap.total_seconds()))} segundos depois da última "
        f"transação do treino; os períodos se sobrepõem: {yes_no(overlap.gap <= pd.Timedelta(0))}.",
        f"- `{TRANSACTION_ID}` presentes nos dois arquivos: {number(overlap.shared_ids)}.",
        f"- Cartões do teste que também aparecem no treino: "
        f"{number(overlap.test_cards_seen_in_train)} de {number(overlap.test_cards)}.",
        f"- Mesmas categorias nos dois arquivos: {yes_no(same_categories)}.",
        "",
        "## Relógio",
        "",
        f"- `{UNIX_TIME}` fica de {number(round(low_gap))} a {number(round(high_gap))} dias antes "
        f"de `{TIMESTAMP}`, cerca de {round(high_gap / DAYS_PER_YEAR)} anos. As duas colunas não "
        "marcam o mesmo instante.",
        "",
        "## Valores",
        "",
        *table(checks),
        "",
        f"Categorias: {', '.join(f'`{name}`' for name in train.category_names)}.",
        "",
        "## Colunas",
        "",
        "| Coluna | Tipo | Ausentes treino | Ausentes teste | Distintos treino | Distintos teste |",
        "|---|---|---:|---:|---:|---:|",
        *(
            f"| `{row['column']}` | {row['dtype']} | {number(row['missing_train'])} | "
            f"{number(row['missing_test'])} | {number(row['distinct_train'])} | "
            f"{number(row['distinct_test'])} |"
            for row in data.columns
        ),
        "",
        "## Fraude por mês",
        "",
        "| Mês | Arquivo | Transações | Fraudes | Taxa |",
        "|---|---|---:|---:|---:|",
        *(
            f"| {row.month.strftime('%m/%Y')} | {LABELS[row.split]} | "
            f"{number(row.transactions)} | {number(row.frauds)} | {percent(row.rate)} |"
            for row in data.months.itertuples()
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    paths = {split: download.RAW_DIR / name for split, name in SPLITS.items()}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(
            f"missing in data/raw/: {', '.join(missing)}; run python -m pipeline.download first"
        )
    files = download.describe_files()
    download.verify(files)

    data = build_profile(files, {split: load(path) for split, path in paths.items()})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(data), encoding="utf-8", newline="\n")
    for split, profile in data.splits.items():
        print(
            f"{split}: {profile.rows:,} rows, {profile.frauds:,} frauds "
            f"({profile.fraud_rate:.2%}), {profile.first} to {profile.last}"
        )
    print(f"wrote {REPORT.relative_to(download.ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
