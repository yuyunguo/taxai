"""Extract the MIMIC-IV cardiovascular ICU cohort and in-window events.

All event queries are restricted to the observation window
[window_start, prediction_time] defined per admission, so no post-prediction
information enters the trajectory. Discharge ICD diagnoses are extracted
separately as the hidden gold standard for Diagnostic Concordance.

Usage (on the database host):
    PGPASSWORD=... python -m src.pipeline_v2.extract
"""

import logging
from typing import Dict

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection

from .config import COHORT, DATA_DIR, DB, LAB_ITEMIDS, VITAL_ITEMIDS

logger = logging.getLogger(__name__)

COHORT_SQL = """
CREATE TEMP TABLE cohort_tmp AS
WITH cv AS (
    SELECT DISTINCT hadm_id FROM mimic_hosp.diagnoses_icd
    WHERE (icd_version = 9 AND icd_code BETWEEN '390' AND '459ZZ')
       OR (icd_version = 10 AND LEFT(icd_code, 1) = 'I')
),
first_icu AS (
    SELECT DISTINCT ON (hadm_id) hadm_id, stay_id, intime, first_careunit
    FROM mimic_icu.icustays ORDER BY hadm_id, intime
),
elig AS (
    SELECT a.subject_id, a.hadm_id, f.stay_id, a.admittime, a.dischtime, a.deathtime,
           f.intime, f.first_careunit, a.admission_type, a.hospital_expire_flag,
           p.gender, p.anchor_year_group,
           p.anchor_age + (EXTRACT(YEAR FROM a.admittime) - p.anchor_year) AS age,
           GREATEST(a.admittime, f.intime - make_interval(hours => CAST(:before AS integer))) AS window_start,
           f.intime + make_interval(hours => CAST(:after AS integer)) AS pred_time,
           ROW_NUMBER() OVER (PARTITION BY a.subject_id ORDER BY a.admittime) AS rn
    FROM mimic_core.admissions a
    JOIN cv USING (hadm_id)
    JOIN first_icu f USING (hadm_id)
    JOIN mimic_core.patients p ON p.subject_id = a.subject_id
    WHERE p.anchor_age >= :min_age
      AND a.dischtime > f.intime + make_interval(hours => CAST(:after AS integer))
      AND (a.deathtime IS NULL OR a.deathtime > f.intime + make_interval(hours => CAST(:after AS integer)))
)
SELECT * FROM elig WHERE rn = 1;
CREATE INDEX ON cohort_tmp (subject_id);
CREATE INDEX ON cohort_tmp (hadm_id);
CREATE INDEX ON cohort_tmp (stay_id);
ANALYZE cohort_tmp;
"""

LAB_SQL = """
SELECT c.hadm_id, l.itemid, l.charttime, l.valuenum AS value, l.valueuom AS uom
FROM mimic_hosp.labevents l
JOIN cohort_tmp c ON l.subject_id = c.subject_id
WHERE l.itemid = ANY(:itemids)
  AND l.valuenum IS NOT NULL
  AND l.charttime >= c.window_start AND l.charttime <= c.pred_time
"""

VITAL_SQL = """
SELECT c.hadm_id, ce.itemid, ce.charttime, ce.valuenum AS value
FROM mimic_icu.chartevents ce
JOIN cohort_tmp c ON ce.stay_id = c.stay_id
WHERE ce.itemid = ANY(:itemids)
  AND ce.valuenum IS NOT NULL
  AND ce.charttime >= c.window_start AND ce.charttime <= c.pred_time
"""

MED_SQL = """
SELECT c.hadm_id, rx.starttime AS charttime, rx.drug, rx.route
FROM mimic_hosp.prescriptions rx
JOIN cohort_tmp c ON rx.hadm_id = c.hadm_id
WHERE rx.starttime >= c.window_start AND rx.starttime <= c.pred_time
"""

NOTE_SQL = """
SELECT c.hadm_id, r.charttime, r.note_type
FROM mimiciv_note.radiology r
JOIN cohort_tmp c ON r.subject_id = c.subject_id
WHERE r.charttime >= c.window_start AND r.charttime <= c.pred_time
"""

GOLD_DX_SQL = """
SELECT c.hadm_id, d.seq_num, d.icd_code, d.icd_version, dd.long_title
FROM mimic_hosp.diagnoses_icd d
JOIN cohort_tmp c ON d.hadm_id = c.hadm_id
LEFT JOIN mimic_hosp.d_icd_diagnoses dd
  ON dd.icd_code = d.icd_code AND dd.icd_version = d.icd_version
"""


def _split_label(group: str) -> str:
    if group in COHORT.train_groups:
        return "train"
    if group in COHORT.val_groups:
        return "val"
    if group in COHORT.test_groups:
        return "test"
    raise ValueError(f"Unmapped anchor_year_group: {group}")


def _query(conn: Connection, sql: str, name: str, **params) -> pd.DataFrame:
    logger.info(f"Extracting {name} ...")
    df = pd.read_sql(text(sql), conn, params=params)
    out = DATA_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"  {name}: {len(df):,} rows -> {out}")
    return df


def extract_all() -> Dict[str, int]:
    """Run all extraction queries in one session and write parquet files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(DB.uri)
    counts: Dict[str, int] = {}
    with engine.connect() as conn:
        conn.execute(text(COHORT_SQL), {
            "before": COHORT.window_before_icu_h,
            "after": COHORT.window_after_icu_h,
            "min_age": COHORT.min_age,
        })
        cohort = pd.read_sql(text("SELECT * FROM cohort_tmp"), conn)
        cohort["split"] = cohort["anchor_year_group"].map(_split_label)
        cohort.to_parquet(DATA_DIR / "cohort.parquet", index=False)
        counts["cohort"] = len(cohort)
        logger.info(f"Cohort: {len(cohort):,} admissions, mortality "
                    f"{cohort['hospital_expire_flag'].mean():.1%}; "
                    f"splits {cohort['split'].value_counts().to_dict()}")

        counts["labs"] = len(_query(conn, LAB_SQL, "labs", itemids=list(LAB_ITEMIDS)))
        counts["vitals"] = len(_query(conn, VITAL_SQL, "vitals", itemids=list(VITAL_ITEMIDS)))
        counts["meds"] = len(_query(conn, MED_SQL, "meds"))
        counts["notes"] = len(_query(conn, NOTE_SQL, "notes"))
        counts["gold_dx"] = len(_query(conn, GOLD_DX_SQL, "gold_dx"))
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for k, v in extract_all().items():
        logger.info(f"{k}: {v:,}")
