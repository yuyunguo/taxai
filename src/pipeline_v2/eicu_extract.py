"""Extract an eICU-CRD cohort in the same schema as the MIMIC-IV extraction.

The cohort mirrors the MIMIC-IV definition: adults with a cardiovascular
diagnosis code (ICD-9 390-459 or ICD-10 I*), first ICU unit stay of the
hospital stay, first eligible hospital stay per patient, alive and in
hospital 24 h after ICU admission. Outcome: in-hospital death.

eICU stores times as minute offsets from ICU admission. They are converted
to synthetic timestamps around a fixed reference so that build.py runs
unchanged. Labs and vitals are mapped onto the MIMIC-IV item identifiers
used in config.py. Known gaps relative to MIMIC-IV:
  * eICU reports BNP, not NT-proBNP; BNP is not mapped (NT-proBNP missing).
  * There are no radiology report references; the note table is empty.
  * Periodic vitals (5-minute) are aggregated to hourly means.
  * The ICU unit type is mapped to the closest MIMIC-IV care unit.

Usage (on the database host):
    PGPASSWORD=... DB_NAME=eicu TAXAI_V2_DATA=.../data_eicu \
        python -m src.pipeline_v2.eicu_extract
"""

import logging
from typing import Dict

import pandas as pd
from sqlalchemy import create_engine, text

from .config import COHORT, DATA_DIR, DB

logger = logging.getLogger(__name__)

REF_TIME = pd.Timestamp("2015-01-01 00:00:00")   # synthetic ICU admission time
WINDOW_MIN = int(COHORT.window_before_icu_h * 60)
PRED_MIN = int(COHORT.window_after_icu_h * 60)

LAB_MAP: Dict[str, int] = {
    "lactate": 50813, "creatinine": 50912, "troponin - I": 51002, "troponin - T": 51003,
    "WBC x 1000": 51301, "potassium": 50971, "sodium": 50983, "glucose": 50931,
    "Hgb": 51222, "platelets x 1000": 51265, "chloride": 50902, "bicarbonate": 50882,
    "BUN": 51006, "calcium": 50893, "Total CO2": 50804, "paO2": 50821, "pH": 50820,
    "paCO2": 50818, "Base Excess": 50802,
}
PERIODIC_MAP: Dict[str, int] = {
    "heartrate": 220045, "respiration": 220210, "sao2": 220277, "temperature": 223762,
    "systemicsystolic": 220050, "systemicdiastolic": 220051, "systemicmean": 220052,
}
APERIODIC_MAP: Dict[str, int] = {
    "noninvasivesystolic": 220179, "noninvasivediastolic": 220180, "noninvasivemean": 220181,
}
UNIT_MAP: Dict[str, str] = {
    "Cardiac ICU": "Coronary Care Unit (CCU)",
    "CCU-CTICU": "Coronary Care Unit (CCU)",
    "CSICU": "Cardiac Vascular Intensive Care Unit (CVICU)",
    "CTICU": "Cardiac Vascular Intensive Care Unit (CVICU)",
    "MICU": "Medical Intensive Care Unit (MICU)",
    "SICU": "Surgical Intensive Care Unit (SICU)",
    "Med-Surg ICU": "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "Neuro ICU": "Neuro Surgical Intensive Care Unit (Neuro SICU)",
}

COHORT_SQL = """
CREATE TEMP TABLE cohort_tmp AS
WITH cv AS (
    SELECT DISTINCT patientunitstayid FROM (
        SELECT patientunitstayid, TRIM(UNNEST(STRING_TO_ARRAY(icd9code, ','))) AS code
        FROM eicu_crd.diagnosis WHERE icd9code IS NOT NULL AND icd9code <> ''
    ) d
    WHERE code ~ '^(39[0-9]|4[0-5][0-9])' OR code ~ '^I[0-9]'
),
elig AS (
    SELECT p.*,
           CASE WHEN p.age = '> 89' THEN 90 ELSE NULLIF(p.age, '')::int END AS age_num,
           ROW_NUMBER() OVER (PARTITION BY p.uniquepid
                              ORDER BY p.hospitaldischargeyear, p.patienthealthsystemstayid) AS rn
    FROM eicu_crd.patient p
    JOIN cv USING (patientunitstayid)
    WHERE p.unitvisitnumber = 1
      AND p.hospitaldischargestatus IN ('Alive', 'Expired')
      AND p.hospitaldischargeoffset > :pred
      AND p.age <> ''
)
SELECT patientunitstayid AS hadm_id, uniquepid, hospitalid, age_num AS age, gender,
       unittype, hospitaladmitsource, hospitaladmitoffset, hospitaldischargeyear,
       (hospitaldischargestatus = 'Expired')::int AS hospital_expire_flag,
       GREATEST(hospitaladmitoffset, -:win) AS win_start
FROM elig WHERE rn = 1 AND age_num >= :min_age;
CREATE INDEX ON cohort_tmp (hadm_id);
ANALYZE cohort_tmp;
"""

LAB_SQL = """
SELECT l.patientunitstayid AS hadm_id, l.labname, l.labresultoffset AS offset_min,
       l.labresult AS value, l.labmeasurenamesystem AS uom
FROM eicu_crd.lab l JOIN cohort_tmp c ON l.patientunitstayid = c.hadm_id
WHERE l.labname = ANY(:names) AND l.labresult IS NOT NULL
  AND l.labresultoffset BETWEEN c.win_start AND :pred
"""

PERIODIC_SQL = """
SELECT v.patientunitstayid AS hadm_id, FLOOR(v.observationoffset / 60.0) * 60 AS offset_min,
       AVG(v.heartrate) AS heartrate, AVG(v.respiration) AS respiration, AVG(v.sao2) AS sao2,
       AVG(v.temperature) AS temperature, AVG(v.systemicsystolic) AS systemicsystolic,
       AVG(v.systemicdiastolic) AS systemicdiastolic, AVG(v.systemicmean) AS systemicmean
FROM eicu_crd.vitalperiodic v JOIN cohort_tmp c ON v.patientunitstayid = c.hadm_id
WHERE v.observationoffset BETWEEN 0 AND :pred
GROUP BY 1, 2
"""

APERIODIC_SQL = """
SELECT v.patientunitstayid AS hadm_id, v.observationoffset AS offset_min,
       v.noninvasivesystolic, v.noninvasivediastolic, v.noninvasivemean
FROM eicu_crd.vitalaperiodic v JOIN cohort_tmp c ON v.patientunitstayid = c.hadm_id
WHERE v.observationoffset BETWEEN 0 AND :pred
"""

MED_SQL = """
SELECT m.patientunitstayid AS hadm_id, m.drugstartoffset AS offset_min, m.drugname AS drug,
       m.routeadmin AS route
FROM eicu_crd.medication m JOIN cohort_tmp c ON m.patientunitstayid = c.hadm_id
WHERE m.drugordercancelled = 'No' AND m.drugname IS NOT NULL AND m.drugname <> ''
  AND m.drugstartoffset BETWEEN c.win_start AND :pred
"""

GOLD_DX_SQL = """
SELECT d.patientunitstayid AS hadm_id, d.diagnosispriority AS seq_num,
       d.icd9code AS icd_code,
       TRIM(REGEXP_REPLACE(d.diagnosisstring, '^.*\\|', '')) AS long_title
FROM eicu_crd.diagnosis d JOIN cohort_tmp c ON d.patientunitstayid = c.hadm_id
"""


def _to_time(offset_min: pd.Series) -> pd.Series:
    return REF_TIME + pd.to_timedelta(offset_min.astype(float), unit="m")


def _melt_vitals(df: pd.DataFrame, mapping: Dict[str, int]) -> pd.DataFrame:
    long = df.melt(id_vars=["hadm_id", "offset_min"], value_vars=list(mapping),
                   var_name="var", value_name="value").dropna(subset=["value"])
    long["itemid"] = long["var"].map(mapping)
    long["charttime"] = _to_time(long["offset_min"])
    return long[["hadm_id", "itemid", "charttime", "value"]]


def extract_all() -> Dict[str, int]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(DB.uri)
    params = {"pred": PRED_MIN, "win": WINDOW_MIN}
    counts: Dict[str, int] = {}
    with engine.connect() as conn:
        conn.execute(text(COHORT_SQL), {**params, "min_age": COHORT.min_age})
        c = pd.read_sql(text("SELECT * FROM cohort_tmp"), conn)
        c["intime"] = REF_TIME
        c["admittime"] = _to_time(c["hospitaladmitoffset"])
        c["window_start"] = _to_time(c["win_start"])
        c["pred_time"] = REF_TIME + pd.Timedelta(minutes=PRED_MIN)
        c["gender"] = c["gender"].map({"Male": "M", "Female": "F"}).fillna("U")
        c["admission_type"] = c["hospitaladmitsource"].fillna("UNKNOWN").replace(
            {"Emergency Department": "EMERGENCY"})
        c["first_careunit"] = c["unittype"].map(UNIT_MAP).fillna(c["unittype"])
        c["split"] = "test"   # whole eICU cohort is an external test set
        c["anchor_year_group"] = c["hospitaldischargeyear"].astype(str)
        c.to_parquet(DATA_DIR / "cohort.parquet", index=False)
        counts["cohort"] = len(c)
        logger.info(f"eICU cohort: {len(c):,} stays from {c['hospitalid'].nunique()} hospitals; "
                    f"mortality {c['hospital_expire_flag'].mean():.1%}")

        labs = pd.read_sql(text(LAB_SQL), conn, params={**params, "names": list(LAB_MAP)})
        labs["itemid"] = labs["labname"].map(LAB_MAP)
        labs["charttime"] = _to_time(labs["offset_min"])
        labs[["hadm_id", "itemid", "charttime", "value", "uom"]].to_parquet(DATA_DIR / "labs.parquet", index=False)
        counts["labs"] = len(labs)

        per = _melt_vitals(pd.read_sql(text(PERIODIC_SQL), conn, params=params), PERIODIC_MAP)
        aper = _melt_vitals(pd.read_sql(text(APERIODIC_SQL), conn, params=params), APERIODIC_MAP)
        vit = pd.concat([per, aper], ignore_index=True)
        vit.to_parquet(DATA_DIR / "vitals.parquet", index=False)
        counts["vitals"] = len(vit)

        meds = pd.read_sql(text(MED_SQL), conn, params=params)
        meds["charttime"] = _to_time(meds["offset_min"])
        meds[["hadm_id", "charttime", "drug", "route"]].to_parquet(DATA_DIR / "meds.parquet", index=False)
        counts["meds"] = len(meds)

        pd.DataFrame({"hadm_id": pd.Series(dtype="int64"), "charttime": pd.Series(dtype="datetime64[ns]"),
                      "note_type": pd.Series(dtype="object")}).to_parquet(DATA_DIR / "notes.parquet", index=False)
        gold = pd.read_sql(text(GOLD_DX_SQL), conn)
        gold.to_parquet(DATA_DIR / "gold_dx.parquet", index=False)
        counts["gold_dx"] = len(gold)
    for k, v in counts.items():
        logger.info(f"  {k}: {v:,}")
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_all()
