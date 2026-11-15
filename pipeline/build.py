"""Build the processed data: prepared base, train and validation features, audit attributes.

    python -m pipeline.build

Steps:

1. check fraudTrain.csv against data/manifest.json (fraudTest.csv is not opened, D11);
2. prepare it: minimise, pseudonymise the card, order by time (pipeline.prepare);
3. compute the behaviour signals from earlier transactions only (pipeline.features, D8);
4. split by time into train and validation (pipeline.split, D9);
5. write Parquet files (zstd), read each one back to confirm nothing was lost, and record their
   SHA-256 in data/processed/manifest.json and the cut dates in reports/temporal-split.md.

Everything under data/processed/ stays out of Git. The card pseudonym needs the secret key in
CARD_PSEUDONYM_KEY (see .env.example); the key is never printed or written anywhere.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Final

import pandas as pd

from pipeline import download, features, prepare, profile, split

PROCESSED: Final = download.ROOT / "data" / "processed"
BASE_FILE: Final = PROCESSED / "transactions.parquet"
SPLIT_FILES: Final = {name: PROCESSED / f"{name}.parquet" for name in split.SPLITS}
AUDIT_FILE: Final = PROCESSED / "audit" / "attributes.parquet"
MANIFEST: Final = PROCESSED / "manifest.json"
REPORT: Final = download.ROOT / "reports" / "temporal-split.md"
ID_COLUMNS: Final = ("transaction_id", "timestamp", "card_key")
TARGET: Final = "is_fraud"
LIBRARIES: Final = ("pandas", "pyarrow", "numpy")
SPLIT_LABELS: Final = {"train": "Treino", "validation": "Validação"}
PT_BR: Final = str.maketrans(",.", ".,")


def content_digest(frame: pd.DataFrame) -> str:
    """SHA-256 of the values of every column except card_key.

    card_key depends on the secret key, so two people with different keys get different files;
    this digest still matches when everything else is the same.
    """
    values = frame.drop(columns="card_key", errors="ignore")
    hashes = pd.util.hash_pandas_object(values, index=False).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def code_digest() -> str:
    """SHA-256 of the pipeline source, with line endings normalised (Windows and Linux agree)."""
    digest = hashlib.sha256()
    for path in sorted((download.ROOT / "pipeline").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode())
    return digest.hexdigest()


def write_parquet(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    """Write with zstd compression, read back to confirm the round trip, return the record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.reset_index(drop=True)
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    try:
        pd.testing.assert_frame_equal(pd.read_parquet(path, engine="pyarrow"), frame)
    except AssertionError as error:
        raise SystemExit(f"{path.name} changed when read back: {error}") from error
    return {
        "path": path.relative_to(download.ROOT).as_posix(),
        "rows": len(frame),
        "sha256": download.sha256(path),
        "content_sha256": content_digest(frame),
    }


def number(value: float) -> str:
    return f"{value:,.0f}".translate(PT_BR)


def percent(value: float) -> str:
    return f"{value:.2%}".translate(PT_BR)


def render_report(source_sha256: str, summary: pd.DataFrame, outputs: dict[str, dict]) -> str:
    """The cut dates and the content of each set, in Portuguese (reports/temporal-split.md)."""
    rows = [
        f"| {SPLIT_LABELS[name]} | {row['first']:%d/%m/%Y %H:%M:%S} | "
        f"{row['last']:%d/%m/%Y %H:%M:%S} | {number(row['transactions'])} | "
        f"{number(row['frauds'])} | {percent(row['fraud_rate'])} | {number(row['cards'])} |"
        for name, row in summary.iterrows()
    ]
    feature_list = ", ".join(f"`{name}`" for name in features.FEATURE_COLUMNS)
    digests = [
        f"| `{outputs[name]['path']}` | {number(outputs[name]['rows'])} | "
        f"`{outputs[name]['content_sha256'][:16]}` |"
        for name in ("base", *split.SPLITS)
    ]
    return "\n".join(
        [
            "# Divisão temporal e sinais",
            "",
            "Gerado por `python -m pipeline.build` a partir do `fraudTrain.csv` "
            f"(SHA-256 `{source_sha256}`). As regras estão nas decisões D8 a D11 de "
            "[`docs/decisoes-tecnicas.md`](../docs/decisoes-tecnicas.md) e cada campo em "
            "[`docs/dados-processados.md`](../docs/dados-processados.md).",
            "",
            "## Conjuntos",
            "",
            "| Conjunto | Primeira transação | Última transação | Transações | Fraudes | Taxa de "
            "fraude | Cartões |",
            "|---|---|---|---:|---:|---:|---:|",
            *rows,
            "",
            f"- **Corte:** a validação começa em {split.VALIDATION_START:%d/%m/%Y %H:%M:%S}. Tudo "
            "antes desse instante é treino; tudo a partir dele é validação.",
            "- **Teste:** o `fraudTest.csv` não é aberto nesta fase. Ele fica intocado para a "
            "avaliação final (decisão D11); seus sinais serão calculados com o mesmo código, "
            "usando o histórico de treino e validação como passado.",
            "",
            "## Sinais",
            "",
            f"{len(features.FEATURE_COLUMNS)} sinais por transação: {feature_list}.",
            "",
            "## Conteúdo dos arquivos",
            "",
            "Assinatura dos valores de todas as colunas, menos `card_key` (que depende da chave "
            "secreta). Quem rodar o pipeline com outra chave obtém as mesmas assinaturas.",
            "",
            "| Arquivo | Linhas | Assinatura do conteúdo |",
            "|---|---:|---|",
            *digests,
            "",
        ]
    )


def main() -> int:
    key = prepare.load_key()
    source_sha256 = prepare.verify_train_file()
    raw = profile.load(download.RAW_DIR / prepare.TRAIN_FILE)
    base, audit = prepare.prepare(raw, key)
    del raw  # personal columns leave memory as soon as the base is built

    dataset = base[list(ID_COLUMNS)].join(features.build_features(base)).join(base[[TARGET]])
    dataset["split"] = split.assign(dataset["timestamp"])
    split.check(dataset["timestamp"], dataset["split"])

    outputs = {"base": write_parquet(base, BASE_FILE)}
    for name, path in SPLIT_FILES.items():
        rows = dataset[dataset["split"].eq(name)].drop(columns="split")
        outputs[name] = write_parquet(rows, path)
    outputs["audit"] = write_parquet(audit, AUDIT_FILE)

    summary = split.summarize(dataset)
    manifest = {
        "built_by": "python -m pipeline.build",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {"file": prepare.TRAIN_FILE, "sha256": source_sha256},
        "test_file": "fraudTest.csv was not read (decision D11)",
        "card_key": {
            "method": f"HMAC-SHA256, first {prepare.CARD_KEY_LENGTH} hex characters",
            "key_variable": prepare.KEY_VARIABLE,
            "key_fingerprint": prepare.key_fingerprint(key),
        },
        "validation_start": f"{split.VALIDATION_START:%Y-%m-%d %H:%M:%S}",
        "features": list(features.FEATURE_COLUMNS),
        "code_sha256": code_digest(),
        "libraries": {name: version(name) for name in LIBRARIES},
        "outputs": outputs,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = render_report(source_sha256, summary, outputs)
    REPORT.write_text(report, encoding="utf-8", newline="\n")

    for name, row in summary.iterrows():
        print(f"{name}: {row['transactions']:,} rows, {row['frauds']:,} frauds "
              f"({row['fraud_rate']:.2%}), {row['first']} to {row['last']}")
    print(f"{len(features.FEATURE_COLUMNS)} features; wrote {len(outputs)} Parquet files, "
          f"{MANIFEST.relative_to(download.ROOT).as_posix()} and "
          f"{REPORT.relative_to(download.ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
