"""Unit tests for temporal-claim verification (v3)."""

from src.pipeline_v2.faithfulness_v3 import check

TRAJ = {"events": [
    {"type": "lab", "t": 3.2, "label": "Lactate", "value": 2.1},
    {"type": "lab", "t": 14.8, "label": "Lactate", "value": 4.8},
    {"type": "vital", "t": 10.0, "label": "MAP", "value": 58.0},
    {"type": "vital", "t": 0.0, "label": "Heart_Rate", "value": 88.0},
]}


def test_correct_value_and_hour():
    r = check("Lactate rose from 2.1 at ICU hour +3 to 4.8 at ICU hour +15.", TRAJ)
    assert r["n_temporal_claims"] == 2 and r["n_temporal_supported"] == 2


def test_right_value_wrong_hour():
    r = check("Lactate was 4.8 at ICU hour +2.", TRAJ)
    assert r["n_value_supported"] == 1 and r["n_temporal_claims"] == 1 and r["n_temporal_supported"] == 0


def test_vital_bin_tolerance_and_untimed_claim():
    r = check("MAP fell to 58 at hour 11. Heart rate was 88.", TRAJ)
    assert r["n_value_claims"] == 2 and r["n_temporal_claims"] == 1 and r["n_temporal_supported"] == 1


def test_hour_number_not_a_value():
    r = check("At ICU hour +15 the lactate was 4.8 mmol/L.", TRAJ)
    assert r["n_value_claims"] == 1 and r["n_temporal_supported"] == 1
