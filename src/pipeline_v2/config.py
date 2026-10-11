"""Configuration for the corrected TAXAI pipeline (v2, MIMIC-IV).

Design changes relative to v1 (MIMIC-III), see plan/root-cause-fix-plan.md:
  * ICD diagnosis/procedure codes are never model or LLM inputs; discharge
    diagnoses are kept only as the hidden gold standard for DC.
  * Discharge summaries are excluded; only radiology report references are
    used as note events (they carry a real charttime).
  * Prediction time is 24 h after first ICU admission; events come from the
    window [max(admittime, intime - 24 h), intime + 24 h].
  * Temporal split by MIMIC-IV anchor_year_group.
  * LLM calls are deterministic (temperature 0, fixed seed).

Secrets: the database password is read from the environment (PGPASSWORD).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("TAXAI_V2_DATA", ROOT / "data_v2"))


@dataclass(frozen=True)
class DBConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: str = os.getenv("DB_PORT", "5432")
    name: str = os.getenv("DB_NAME", "mimiciv")
    user: str = os.getenv("DB_USER", "postgres")

    @property
    def uri(self) -> str:
        password = os.environ["PGPASSWORD"]
        return f"postgresql+psycopg2://{self.user}:{password}@{self.host}:{self.port}/{self.name}"


@dataclass(frozen=True)
class CohortConfig:
    min_age: int = 18
    window_before_icu_h: float = 24.0
    window_after_icu_h: float = 24.0  # prediction time = ICU intime + 24 h
    train_groups: Tuple[str, ...] = ("2008 - 2010", "2011 - 2013", "2014 - 2016")
    val_groups: Tuple[str, ...] = ("2017 - 2019",)
    test_groups: Tuple[str, ...] = ("2020 - 2022",)


LAB_ITEMIDS: Dict[int, str] = {
    50813: "Lactate", 50912: "Creatinine", 51002: "Troponin_I", 51003: "Troponin_T",
    50963: "NTproBNP", 51300: "WBC", 51301: "WBC", 50971: "Potassium",
    50983: "Sodium", 50931: "Glucose", 51222: "Hemoglobin", 51265: "Platelets",
    50902: "Chloride", 50882: "Bicarbonate", 51006: "BUN", 50893: "Calcium_Total",
    50804: "Calculated_Total_CO2", 50821: "PO2", 50820: "pH", 50818: "pCO2",
    50802: "Base_Excess",
}

VITAL_ITEMIDS: Dict[int, str] = {
    220045: "Heart_Rate",
    220050: "Arterial_BP_Systolic", 220051: "Arterial_BP_Diastolic", 220052: "Arterial_BP_Mean",
    220179: "Non_Invasive_BP_Systolic", 220180: "Non_Invasive_BP_Diastolic",
    220181: "Non_Invasive_BP_Mean",
    220210: "Respiratory_Rate", 224690: "Respiratory_Rate",
    220277: "SpO2",
    223761: "Temperature_F", 223762: "Temperature_C",
}

FEATURE_LABS: Tuple[str, ...] = (
    "Lactate", "Creatinine", "Troponin_I", "Troponin_T", "NTproBNP", "WBC",
    "Potassium", "Sodium", "Glucose", "Hemoglobin", "Platelets", "Chloride",
    "Bicarbonate", "BUN", "Calcium_Total",
)
FEATURE_VITALS: Tuple[str, ...] = (
    "Heart_Rate", "Arterial_BP_Systolic", "Arterial_BP_Diastolic", "Arterial_BP_Mean",
    "Non_Invasive_BP_Systolic", "Non_Invasive_BP_Diastolic", "Non_Invasive_BP_Mean",
    "Respiratory_Rate", "SpO2", "Temperature_C",
)

# Event types available before the prediction time. ICD codes are excluded.
EVENT_TYPES: Tuple[str, ...] = ("lab", "vital", "med", "note")


@dataclass(frozen=True)
class TrajectoryConfig:
    vital_bin_h: int = 8          # one median value per vital per 8-hour bin in the LLM timeline
    timeline_vitals: Tuple[str, ...] = ("Heart_Rate", "SBP", "MAP", "Respiratory_Rate", "SpO2")
    budget: Tuple[Tuple[str, int], ...] = (("lab", 36), ("vital", 30), ("med", 10), ("note", 4))
    type_priority: Tuple[str, ...] = ("lab", "vital", "med", "note")


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 1000
    max_depth: int = 5
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50
    seed: int = 42


@dataclass(frozen=True)
class LLMConfig:
    generator: str = "qwen2.5:14b"
    judges: Tuple[str, ...] = ("qwen2.5:14b", "phi4")
    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    options: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 0.0, "seed": 42, "num_ctx": 8192,
    })


@dataclass(frozen=True)
class StudyConfig:
    n_per_label: int = 250        # explanation sample: 250 predicted HIGH + 250 predicted LOW
    n_multilevel: int = 25        # cases that also get patient/regulator outputs
    cu_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)  # DC', TAS, BAR
    seed: int = 42


DB = DBConfig()
COHORT = CohortConfig()
TRAJ = TrajectoryConfig()
MODEL = ModelConfig()
LLM = LLMConfig()
STUDY = StudyConfig()
