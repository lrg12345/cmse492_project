import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

###############################################
#               CONFIG
###############################################
INPUT_FILE = "karlgren_with_smiles.xlsx"
OUTPUT_FEATURES_NPY = "X_smiles_morgan2048.npy"
OUTPUT_TARGET_NPY = "y_binary.npy"
OUTPUT_NAMES_NPY = "compound_names.npy"
OUTPUT_CSV = "smiles_morgan2048_features.csv"

MORGAN_BITS = 2048
THRESHOLD = 50.0   # % inhibition threshold for binary label
###############################################

print("Loading dataset...")
df = pd.read_excel(INPUT_FILE)

required_cols = ["compound", "inhib_1b1_pct", "SMILES"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

print("Parsing SMILES and computing Morgan fingerprints...")

fingerprints = []
bad_smiles = []

for smi in df["SMILES"]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        bad_smiles.append(smi)
        fp = np.zeros(MORGAN_BITS, dtype=np.int8)
    else:
        fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=MORGAN_BITS)
        fp = np.array(fp_vec)
    fingerprints.append(fp)

if bad_smiles:
    print("⚠️ WARNING: Could not parse the following SMILES:")
    for b in bad_smiles:
        print("  -", b)
else:
    print("All SMILES parsed successfully.")

X = np.vstack(fingerprints)
print("Feature matrix shape:", X.shape)

###############################################
#            Create Binary Target
###############################################

y_binary = (df["inhib_1b1_pct"] > THRESHOLD).astype(int).values
print(f"Binary inhibition labels created using threshold > {THRESHOLD}%")
print("Positive class count:", y_binary.sum())
print("Negative class count:", len(y_binary) - y_binary.sum())

compound_names = df["compound"].astype(str).values

###############################################
#                  Save Outputs
###############################################

np.save(OUTPUT_FEATURES_NPY, X)
np.save(OUTPUT_TARGET_NPY, y_binary)
np.save(OUTPUT_NAMES_NPY, compound_names)

# Also create a CSV for human inspection
csv_df = pd.DataFrame(X, columns=[f"bit_{i}" for i in range(MORGAN_BITS)])
csv_df.insert(0, "compound", compound_names)
csv_df.insert(1, "binary_label", y_binary)

csv_df.to_csv(OUTPUT_CSV, index=False)

print("\n💾 Saved:")
print(" -", OUTPUT_FEATURES_NPY)
print(" -", OUTPUT_TARGET_NPY)
print(" -", OUTPUT_NAMES_NPY)
print(" -", OUTPUT_CSV)
print("\n✔️ Morgan fingerprint + binary label dataset created successfully!")