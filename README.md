# CMSE 492 Project — OATP Inhibition Prediction

This project investigates the use of machine learning to predict whether a small molecule inhibits the liver transporter **OATP1B1**, a protein responsible for hepatic drug uptake and a major contributor to drug–drug interactions (DDIs). Poor prediction of OATP-mediated DDIs can lead to toxicity during drug development and clinical treatment.

Three machine learning models trained on **SMILES-based Morgan fingerprints** and a neural network trained on **PLIP-generated Protein Ligand Interactions** are compared:

- **Ridge Classifier** (SMILES linear baseline)  
- **Random Forest Classifier** (SMILES nonlinear ensemble)  
- **Multilayer Perceptron (MLP)** (SMILES neural network)
- **Graph-Neural Network (GNN)** (PLIP neural network)

The project includes data processing, model training, performance evaluation, and visualization.  

---

## Directory Structure
---

cmse492_project/
-   README.md
-   .gitignore
-   data/
    -   raw/
    -   processed/
-   notebooks/
    -   exploratory/
-   src/
    -   preprocessing/
    -   models/
    -   evaluation/
-   figures/
-   docs/
-   requirements.txt

---

## ⚙️ Setup & Installation

### 1. Create the conda environment
```bash
conda create -n cmse492 python=3.11 -y
conda activate cmse492
