# CRANOVA

**A machine-learning tool to assist facial geometry estimation from cranial measurements.**

Enter 16 measurements of a skull, plus sex and ancestry. CRANOVA estimates 66 distances
between 12 points of the face and places those points in 3D, each with a calibrated 90% uncertainty range.
It is an approximation to guide an investigation, never to confirm an identity.

**Try the app:** [link coming after deployment]

## How it works
- **Model:** ridge regression (a regularized linear model). Neural networks were tested and did no better.
- **Uncertainty:** split-conformal 90% ranges, calibrated separately for each ancestry × sex group.
- **3D:** the predicted distances are turned into 3D landmark positions and shown on a generic head model (GNM).
- **Evaluation:** 219 people for development, 55 people held out and tested once: **2.3 mm** average error.

## Run locally (Python 3.12)
```
pip install -r requirements.txt
python src/reconstruction_engine.py   # rebuilds models/reconstruction_bundle.joblib
streamlit run app/app.py
```

## Data
Cranial and facial measurements from CT scans, by Simmons-Ehrhardt, Parks & Monson (2022), licensed **CC BY 4.0**:

> Simmons-Ehrhardt, T. L., Parks, C. L., & Monson, K. L. (2022). Cranial and facial inter-landmark distances and
> tissue depth dataset from computed tomography scans of 388 living persons. *Data in Brief*, 43, 108334.
> https://doi.org/10.1016/j.dib.2022.108334

This repository includes the two original data files the app reads (`data/raw/`), unchanged.

## Credits and licences
- **3D head model:** GNM Head v3 by Google, Apache 2.0
- **Fonts:** Cormorant Garamond and Jost, SIL Open Font License
- **Project:** Aya Ouertani, supervised by Sirine Marrakchi. Master's in Data Science, ISIMa, University of Monastir, 2026–2027