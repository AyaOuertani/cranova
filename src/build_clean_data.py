from pathlib import Path
import numpy as np
import pandas as pd
from load_data import (CRANIAL_MEASURES, CRANIAL_NAMES, FACIAL_LANDMARK_NAMES, ancestry_from_id,
                       load_cranial, load_facial, load_tissue_depth)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT/ "data" / "processed"
TAB = ROOT / "reports" / "tables"

#CLeaning decisions
PRIMARY_CRANIAL = [m for m in CRANIAL_MEASURES if m != "OBBL"]
UNDOCUMENTED_TD = ["me", "sg", "az L", "az R", "gn - z"]            
IMPLAUSIBLE_ROBUST_Z = 20.0                                          

#One row per subject ID seen anywhere, with membership flags and the modelling status
def build_subject_registry(cranial, facial, tissue):
    """One row per subject ID seen anywhere, with membership flags and the modelling status."""
    ids = sorted(set(cranial.index) | set(facial.index) | set(tissue.index))
    reg = pd.DataFrame(index=pd.Index(ids, name="id"))
    reg["ancestry"] = [ancestry_from_id(i) for i in ids]
    reg["sex"] = pd.concat([cranial["sex"], facial["sex"], tissue["sex"]]).groupby(level=0).first()
    reg["in_cranial"] = reg.index.isin(cranial.index)
    reg["in_facial"] = reg.index.isin(facial.index)
    reg["in_tissue_depth"] = reg.index.isin(tissue.index)
    split = pd.read_csv(ROOT / "splits" / "split_v1.csv", index_col="id")
    reg["split_role"] = split["role"].reindex(reg.index)
    
    
    def status(row):
        if row.in_cranial and row.in_facial:
            return "modelling set"
        if row.in_cranial:
            return "cranial only (no facial data)"
        if row.in_facial:
            return "facial only (no cranial data)"
        return "tissue depth only"
    
    reg["status"] = reg.apply(status, axis=1)
    return reg

#Keep documented columns; set physically implausible values to missing and log them.
def clean_tissue_depth(tissue):
    documented = [c for c in tissue.columns if c != "sex" and c not in UNDOCUMENTED_TD]
    td = tissue[documented].copy()
    median = td.median()
    mad = (td - median).abs().median()
    robust_z = 0.6745 * (td - median) / mad
    log_rows = []
    for (subject, column), z in robust_z.stack().items():
        if abs(z) > IMPLAUSIBLE_ROBUST_Z:
            log_rows.append({"id": subject, "dataset": "tissue_depth", "column": column,
                             "original_value_mm": td.loc[subject, column],
                             "new_value": np.nan,
                             "reason": f"robust z = {z:.1f} > {IMPLAUSIBLE_ROBUST_Z:.0f}: implausible, set to missing"})
            td.loc[subject, column] = np.nan
    return td, pd.DataFrame(log_rows, columns=["id", "dataset", "column", "original_value_mm",
                                               "new_value", "reason"])
    
def build_data_dictionary(facial_columns, td_columns):
    rows = []
    for m in CRANIAL_MEASURES:
        role = "input" if m in PRIMARY_CRANIAL else "excluded (missing for all ASNM subjects)"
        rows.append(("cranial", m, CRANIAL_NAMES[m], "mm", role))
    rows += [("cranial", "sex", "Recorded sex (F/M)", "category", "input"),
             ("cranial", "ancestry", "AFR, ASN or EUR", "category", "input")]
    for d in facial_columns:
        a, b = d.split("-")
        text = f"Distance between {FACIAL_LANDMARK_NAMES[a]} and {FACIAL_LANDMARK_NAMES[b]}"
        rows.append(("facial_known", d, text + " (CT-derived face)", "mm", "target"))
        rows.append(("facial_reface", d, text + " (ReFace approximation)", "mm",
                     "baseline only, never a model input"))
    for c in td_columns:
        rows.append(("tissue_depth", c, "Soft-tissue depth at this landmark (workbook Landmarks_Ref)",
                     "mm", "auxiliary training target only"))
    for c in UNDOCUMENTED_TD:
        rows.append(("tissue_depth_excluded", c, "Not defined in the workbook's landmark reference",
                     "mm", "excluded"))
    return pd.DataFrame(rows, columns=["dataset", "column", "description", "unit", "role"])

def main():
    cranial, facial_k, facial_r = load_cranial(), load_facial("k"), load_facial("r")
    tissue = load_tissue_depth()

    # Consistency checks: stop the pipeline if the raw data violate what the analysis assumes
    assert set(facial_k.index) == set(facial_r.index), "CT and ReFace faces must cover the same subjects"
    assert list(facial_k.columns) == list(facial_r.columns), "distance names must match"
    assert not cranial.index.duplicated().any() and not facial_k.index.duplicated().any()
    both = sorted(set(cranial.index) & set(facial_k.index))
    assert (cranial.loc[both, "sex"] == facial_k.loc[both, "sex"]).all(), "sex must agree across files"

    registry = build_subject_registry(cranial, facial_k, tissue)
    td_clean, cleaning_log = clean_tissue_depth(tissue)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    cranial[["sex", "ancestry"] + PRIMARY_CRANIAL].to_csv(PROCESSED / "cranial_primary.csv")
    facial_k.to_csv(PROCESSED / "facial_known.csv")
    facial_r.to_csv(PROCESSED / "facial_reface.csv")
    td_clean.to_csv(PROCESSED / "tissue_depth_clean.csv")
    registry.to_csv(TAB / "subject_registry.csv")
    cleaning_log.to_csv(TAB / "cleaning_log.csv", index=False)
    dictionary = build_data_dictionary([c for c in facial_k.columns if c != "sex"], list(td_clean.columns))
    dictionary.to_csv(TAB / "data_dictionary.csv", index=False)

    print("Subjects by status:")
    print(registry["status"].value_counts().to_string())
    print(f"\nModelling set: {int((registry['status'] == 'modelling set').sum())} subjects "
          f"| with tissue depth too: {int(((registry['status'] == 'modelling set') & registry['in_tissue_depth']).sum())}")
    print(f"Tissue-depth columns kept: {td_clean.shape[1]} | values set to missing: {len(cleaning_log)}")
    if len(cleaning_log):
        print(cleaning_log.to_string(index=False))
    print(f"Data dictionary rows: {len(dictionary)}")

if __name__ == "__main__":
    main()