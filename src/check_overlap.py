#Counts how many subjects appear in each dataset and in combination of them
import pandas as pd
from load_data import ancestry_from_id, load_cranial, load_facial, load_tissue_depth

cranial = load_cranial()
facial = load_facial("k")
tissue = load_tissue_depth()

ids = {"Cranial": set(cranial.index), 
       "Facial" : set(facial.index),
       "TD" : set(tissue.index)}
all_ids = set().union(*ids.values())

rows = []
for anc in ["AFR", "ASN", "EUR", "HIS", "HWF/HWM"]:
    sub = {i for i in all_ids if ancestry_from_id(i) == anc}
    c, f, t = (ids[k] & sub for k in ("Cranial", "Facial", "TD"))
    rows.append([anc, len(c), len(f), len(t), len(c & f), len(c & t), len(f & t), len(c & f & t)])

table = pd.DataFrame(rows, columns=["ancestry", "Cranial", "Facial", "TD", "C+F", "C+TD", "F+TD", "All three"])

table.loc[len(table)] = ["Total"] + table.iloc[:, 1:].sum().tolist()
print(table.to_string(index=False))