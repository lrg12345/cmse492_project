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

## ⚙️ Setup, Installation, and Operation

### 1. Create the conda environment
conda create -n cmse492 python=3.11 -y

conda activate cmse492

### 2. Install Python dependencies
pip install -r requirements.txt

### 3. Running the SMILES models
Each model can be trained by running its corresponding script in src/models/ 
- python train_ridge.py
- python train_random_forest.py
- python train_mlp.py

All models load:
- X_smiles_morgan2048.npy
- y_binary.npy
- compound_names.npy

These scripts output:
- Test metrics (*_metrics.csv)
- Test predictions (*_test_predictions.csv)
- Performance plots (saved in figures/)

### 4. Running the PLIP-GNN model
The GNN operates on protein–ligand interaction graphs generated from PLIP XML files. Train the final GNN with:
- python train_plip_gnn.py

This produces:
- gnn_plip_metrics.csv
- gnn_plip_test_predictions.csv
- ROC & PR curves in /figures

Hyperparameter tuning for the GNN is implemented in:
- python train_plip_gnn_sweep.py

## Acknowledgments

This work was performed as a final project for CMSE 492: Applied Machine Learning at Michigan State University.
