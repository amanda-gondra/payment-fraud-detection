"""Tests for pipeline/modeling/fairness.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.modeling import fairness
from pipeline.modeling.decision import APPROVE, BLOCK, REVIEW


def decisions_frame() -> pd.DataFrame:
    """Group A: 2 cards; group B: 2 cards. Counts chosen so every rate is easy by hand."""
    rows = []
    # group A, card a1: 1 fraud blocked, 1 fraud approved, 4 legit approved, 1 legit reviewed
    rows += [("A", "a1", 1, BLOCK), ("A", "a1", 1, APPROVE)]
    rows += [("A", "a1", 0, APPROVE)] * 4 + [("A", "a1", 0, REVIEW)]
    # group A, card a2: 3 legit approved
    rows += [("A", "a2", 0, APPROVE)] * 3
    # group B, card b1: 2 frauds reviewed, 2 legit blocked, 6 legit approved
    rows += [("B", "b1", 1, REVIEW)] * 2 + [("B", "b1", 0, BLOCK)] * 2
    rows += [("B", "b1", 0, APPROVE)] * 6
    rows += [("B", "b2", 0, APPROVE)] * 2
    return pd.DataFrame(rows, columns=["group", "card_key", "is_fraud", "decision"])


def test_rates_by_hand() -> None:
    table, _ = fairness.audit(decisions_frame(), boot=50)
    a = table.set_index("group").loc["A"]
    b = table.set_index("group").loc["B"]
    assert (a["transactions"], a["cards"], a["frauds"]) == (10, 2, 2)
    assert a["fpr"] == pytest.approx(1 / 8)  # 1 of 8 legit got an alert
    assert a["fnr"] == pytest.approx(1 / 2)
    assert a["precision"] == pytest.approx(1 / 2)  # 2 alerts, 1 fraud
    assert a["legit_block_rate"] == 0
    assert a["review_rate"] == pytest.approx(1 / 10)
    assert b["fpr"] == pytest.approx(2 / 10)
    assert b["fnr"] == 0
    assert b["legit_block_rate"] == pytest.approx(2 / 10)
    assert not a["reliable"] and not b["reliable"]  # far fewer than 30 frauds


def test_bootstrap_is_reproducible_and_brackets_the_point() -> None:
    first, _ = fairness.audit(decisions_frame(), boot=200, seed=9)
    second, _ = fairness.audit(decisions_frame(), boot=200, seed=9)
    pd.testing.assert_frame_equal(first, second)
    for row in first.to_dict("records"):
        assert row["review_rate_low"] <= row["review_rate"] <= row["review_rate_high"]


def test_one_card_gives_no_spread() -> None:
    frame = decisions_frame()
    table, _ = fairness.audit(frame[frame["card_key"] == "b1"], boot=100)
    row = table.iloc[0]
    assert row["fpr_low"] == row["fpr"] == row["fpr_high"]


def test_differences_against_the_reference_group() -> None:
    table, draws = fairness.audit(decisions_frame(), boot=200)
    gaps = fairness.differences(table, draws, reference="A").set_index("metric")
    assert gaps.loc["legit_block_rate", "difference"] == pytest.approx(0.2)
    assert gaps.loc["fpr", "difference"] == pytest.approx(0.2 - 1 / 8)
    assert set(gaps["group"]) == {"B"}


def test_attributes_keep_the_order_and_refuse_missing_rows() -> None:
    audit = pd.DataFrame({"transaction_id": ["t1", "t2", "t3", "t4"],
                          "gender": ["F", "M", "F", "M"], "age_years": [29, 30, 59, 60]})
    found = fairness.attributes(audit, pd.Series(["t4", "t1", "t3", "t2"]))
    assert found["gender"].tolist() == ["Masculino", "Feminino", "Feminino", "Masculino"]
    assert found["age_band"].astype(str).tolist() == [
        "60 anos ou mais", "Até 29 anos", "45 a 59 anos", "30 a 44 anos"]
    with pytest.raises(ValueError, match="missing"):
        fairness.attributes(audit, pd.Series(["t1", "t9"]))


def test_the_audit_never_reads_the_model_inputs() -> None:
    frame = decisions_frame().assign(amount=np.arange(len(decisions_frame())))
    with_extra, _ = fairness.audit(frame, boot=20)
    without, _ = fairness.audit(decisions_frame(), boot=20)
    pd.testing.assert_frame_equal(with_extra, without)
