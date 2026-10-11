"""Unit tests for trajectory-citation verification (written before computing any rates)."""

from src.pipeline_v2.faithfulness import check, extract_claims

TRAJ = {"events": [
    {"type": "lab", "label": "WBC", "value": 19.1}, {"type": "lab", "label": "WBC", "value": 18.7},
    {"type": "lab", "label": "Creatinine", "value": 3.9}, {"type": "lab", "label": "Creatinine", "value": 4.1},
    {"type": "lab", "label": "Lactate", "value": 1.2},
    {"type": "vital", "label": "Heart_Rate", "value": 67.5}, {"type": "vital", "label": "Heart_Rate", "value": 81.5},
    {"type": "vital", "label": "SpO2", "value": 98.0}, {"type": "vital", "label": "SpO2", "value": 98.5},
    {"type": "med", "label": "Heparin", "value": None},
]}


def test_supported_values_and_ranges():
    r = check("WBC fell from 19.1 to 18.7 K/uL; heart rate ranged 67.5-81.5; SpO2 98-98.5%.", TRAJ)
    assert r["n_claims"] == 5 and r["n_supported"] == 5


def test_rounding_and_stated_change():
    r = check("Creatinine rose from 3.9 to 4.1 mg/dL, an increase of 0.2.", TRAJ)
    assert r["n_supported"] == r["n_claims"] == 3


def test_fabricated_value_is_unsupported():
    r = check("Lactate was 4.5 mmol/L, and the heart rate reached 130.", TRAJ)
    assert r["n_claims"] == 2 and r["n_supported"] == 0
    assert set(r["unsupported"]) == {"Lactate=4.5", "Heart_Rate=130"}


def test_variable_absent_from_data_is_unsupported():
    r = check("Troponin T was 0.5 ng/mL.", TRAJ)
    assert r["n_claims"] == 1 and r["n_supported"] == 0


def test_times_percentages_and_age_are_not_claims():
    claims, unattributed = extract_claims(
        "A 77-year-old with a 55.7% predicted risk (threshold 30%); at ICU hour +3 lactate was 1.2.")
    assert [(v, x) for v, x, _ in claims] == [("Lactate", 1.2)]
    assert unattributed == 0


def test_number_far_from_variable_is_unattributed():
    claims, unattributed = extract_claims(
        "Lactate was reviewed carefully along with many other routine measurements today, and 7 doses were given.")
    assert claims == [] and unattributed == 1


TRAJ2 = {"events": [
    {"type": "lab", "label": "Base_Excess", "value": -14.0}, {"type": "lab", "label": "Base_Excess", "value": -4.0},
    {"type": "lab", "label": "pH", "value": 7.19}, {"type": "lab", "label": "pH", "value": 7.45},
    {"type": "lab", "label": "Platelets", "value": 348.0}, {"type": "lab", "label": "Platelets", "value": 324.0},
    {"type": "lab", "label": "Hemoglobin", "value": 8.0},
]}


def test_underscored_names_plurals_and_respectively():
    text = ("Base_Excess of -14 and a pH of 7.19, which improved to -4 and 7.45, respectively. "
            "Hemoglobin: 8, platelets: 348 to 324.")
    r = check(text, TRAJ2)
    assert r["n_claims"] == 7 and r["n_supported"] == 7


def test_duration_with_hyphen_is_not_a_claim():
    claims, _ = extract_claims("Troponin trends beyond the 5-hour mark are missing.")
    assert claims == []
