#Loaders for the raw datasets. Raw files are only read, never modified
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parent.parent /"data"/"raw"

CRANIAL_MEASURES = [
    "AUB", "BBH", "BNL", "BPL", "FRC", "GOL", "MAB", "NLB", "NLHL",
    "OBBL", "OBHL", "OCC", "PAC", "UFHT", "WFB", "XCB", "ZYB",
]
CRANIAL_NAMES = {
    "AUB": "Auricular breadth", "BBH": "Basion-bregma height", "BNL": "Basion-nasion length",
    "BPL": "Basion-prosthion length", "FRC": "Frontal chord", "GOL": "Glabello-occipital length",
    "MAB": "Maxillo-alveolar breadth", "NLB": "Nasal breadth", "NLHL": "Nasal height",
    "OBBL": "Orbital breadth", "OBHL": "Orbital height", "OCC": "Occipital chord",
    "PAC": "Parietal chord", "UFHT": "Upper facial height", "WFB": "Minimum frontal breadth",
    "XCB": "Max cranial breadth", "ZYB": "Bizygomatic breadth",
}

FACIAL_LANDMARK_NAMES = {
    "se": "sellion", "exL": "left exocanthion", "exR": "right exocanthion", "prn": "pronasale",
    "acL": "left alare curvature point", "acR": "right alare curvature point", "sn": "subnasale",
    "ls": "labrale superius", "li": "labrale inferius", "sto": "stomion",
    "chL": "left cheilion", "chR": "right cheilion",
}
#Return one row per subject: index 'id', columns sex, ancestry + 17 measures (mm).
def load_cranial() -> pd.DataFrame:
    path = RAW / "ALL_Cranial_ILD_data.xlsx"
    sheets = pd.read_excel(path, sheet_name=None)
    parts = []
    for name, df in sheets.items():
        #citation sheet, no data
        if name == "Ref":
            continue
    
        part = pd.DataFrame(index=df.iloc[:, 0].astype(str).str.lower())
        part.index.name="id"
        part["sex"]= df["Sex"].values
        part["ancestry"] = name[:3]
        for m in CRANIAL_MEASURES:
            part[m] = df[m].values if m in df.columns else float("nan")
        parts.append(part)
    return pd.concat(parts)
    
ANCESTRY_BY_PREFIX = {"afr": "AFR", "asn": "ASN", "cau": "EUR", "his": "HIS",
                      "hwf": "HWF/HWM", "hwm": "HWF/HWM"}
#Ancestry label from the ID prefix
def ancestry_from_id(subject_id: str) -> str:
    return ANCESTRY_BY_PREFIX[subject_id[:3]]

#Facial distances (mm). version 'k' = CT-measured, 'r' = ReFace approximation.
def load_facial(version: str) -> pd.DataFrame:
    if version not in ("k", "r"):
        raise ValueError("version must be 'k' or 'r'")
    
    path = RAW / "ALL_Facial_ILD_data.xlsx"
    sheets = pd.read_excel(path, sheet_name=None, index_col=0)
    parts = []
    
    for name, df in sheets.items():
        if not name.endswith("_" + version):
            continue
        
        # sheets have distances as rows and subjects as columns
        part = df.T
        part.index = part.index.astype(str).str.lower()
        
        # ReFace IDs carry an extra 'r' prefix
        if version == "r":
            part.index = part.index.str[1:]
        part.index.name = "id"
        part.insert(0, "sex", name[3].upper())
        parts.append(part)
    return pd.concat(parts) 

# Known typos in the raw data, documented in the dataset documentation.
ID_FIXES = {"ans012": "asn012"}

#Tissue depths (mm): index 'id', columns sex + 41 depths. Summary rows removed.
def load_tissue_depth() -> pd.DataFrame:
    path = RAW / "ALL_Tissue_Depth_data.xlsx"
    df = pd.read_excel(path, sheet_name="Tissue Depths_ReFace")
    df["id"] = df["ReFace ID"].astype(str).str.lower().replace(ID_FIXES)
    
    # keep only real subject rows (drops Mean/STD/Median/Count rows and the empty row)
    df = df[df["id"].str.match(r"^(afr|asn|cau|his)\d+$")]
    df = df.drop(columns="ReFace ID").set_index("id")
    return df[["Sex"] + [c for c in df.columns if c != "Sex"]].rename(columns={"Sex": "sex"})
    
if __name__ == "__main__":
    c=load_cranial()
    print("shape:", c.shape)
    print("duplicate ids:", c.index.duplicated().sum())
    print(c.groupby(["ancestry", "sex"]).size())
    print("non-missing OBBL:", c["OBBL"].notna().sum())
    print(c.head(3))