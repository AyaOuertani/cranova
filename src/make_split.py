#Create the frozen dev/test split
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from load_data import load_cranial, load_facial

SEED = 42
Test_Fraction = 0.20
OUT = Path(__file__).resolve().parent.parent/ "splits" /"split_v1.csv"

cranial = load_cranial()
facial = load_facial("k")

#Modelling set : objects that have cranial and facial data
matched = sorted(set(cranial.index) & set(facial.index))
meta = cranial.loc[matched, ["ancestry", "sex"]].copy()
#keep ancestry x sex balance in both parts
strata = meta["ancestry"] +"_"+ meta["sex"]

dev_ids, test_ids = train_test_split(
    matched, test_size=Test_Fraction, stratify=strata, random_state=SEED
)
meta["role"] = ["test" if i in set (test_ids) else "dev" for i in meta.index]

OUT.parent.mkdir(exist_ok=True)
meta.to_csv(OUT)
print("subjects:", len(meta), "| dev:", len(dev_ids), "| test:", len(test_ids))
print(meta.groupby(["ancestry", "sex", "role"]).size().unstack())