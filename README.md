# CMSE 492 Project — OATP Inhibition Prediction

This project predicts the inhibitory effect of small molecules on the liver transporter OATP1B1 using machine learning models (Ridge, Random Forest, and MLP). The dataset originates from Karlgren et al. (2012), and analysis includes feature correlations, model comparisons, and performance visualization.

---

## Directory Structure
cmse492_project/
├── README.md
├── .gitignore
├── data/
│ ├── raw/
│   └── processed/
├── notebooks/
│   └── exploratory/
├── src/
│   ├── preprocessing/
│ ├── models/
│   └── evaluation/
├── figures/
├── docs/
└── requirements.txt

---

## Setup
```bash
conda create -n cmse492 python=3.11 -y
conda activate cmse492
pip install pandas numpy scikit-learn matplotlib openpyxl