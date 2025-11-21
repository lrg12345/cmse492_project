#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from rdkit import Chem

INPUT_FILE = "/Users/logangarland/Desktop/cmse492_project/data/processed/karlgren_with_smiles.xlsx"
PLIP_DIR = "/Users/logangarland/Desktop/VSCode/HOLIgraph/Examples/Example_XMLs"       # <-- adjust if needed
OUTPUT_X = "X_plip_simple.npy"
OUTPUT_Y = "y_binary.npy"
OUTPUT_NAMES = "compound_names.npy"
OUTPUT_CSV = "plip_simple_features.csv"


def canonicalize_smiles(smiles):
    """Return canonical SMILES or None."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def compute_inchikey(smiles):
    """Compute InChIKey from SMILES using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.inchi.MolToInchiKey(mol)
    except:
        return None


def extract_plip_info(xml_file):
    """Extract SMILES, InChIKey and interaction counts from a PLIP XML."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract SMILES + InChIKey from XML
    smiles = None
    inchikey = None

    ident = root.find(".//identifiers")
    if ident is not None:
        sm = ident.find("smiles")
        ik = ident.find("inchikey")
        if sm is not None and sm.text:
            smiles = sm.text.strip()
        if ik is not None and ik.text:
            inchikey = ik.text.strip()

    # Count interactions
    interactions = root.find(".//interactions")
    if interactions is None:
        return smiles, inchikey, np.zeros(8)

    def count(tag):
        node = interactions.find(tag)
        if node is None:
            return 0
        return len(list(node))

    # Features: (same 8 as before)
    f_hydrophobic = count("hydrophobic_interactions")
    f_hbonds = count("hydrogen_bonds")
    f_waters = count("water_bridges")
    f_saltbridges = count("salt_bridges")
    f_pi_stack = count("pi_stacks")
    f_pi_cation = count("pi_cation_interactions")
    f_halogen = count("halogen_bonds")
    f_metal = count("metal_complexes")

    features = np.array([
        f_hydrophobic,
        f_hbonds,
        f_waters,
        f_saltbridges,
        f_pi_stack,
        f_pi_cation,
        f_halogen,
        f_metal
    ], dtype=float)

    return smiles, inchikey, features


# -------------------------------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_excel(INPUT_FILE)

compound_names = df["compound"].astype(str).values
smiles_list = df["SMILES"].astype(str).values
inhibition = df["inhib_1b1_pct"].values  # already % 0–100

print(f"Total compounds: {len(compound_names)}")

y_binary = (inhibition > 50.0).astype(int)
print(f"Positive class: {y_binary.sum()}")
print(f"Negative class: {(y_binary==0).sum()}")


# Canonicalize Excel compounds
canonical_excel_smiles = []
excel_inchikeys = []

print("\nCanonicalizing Excel SMILES...")
for sm in smiles_list:
    can = canonicalize_smiles(sm)
    canonical_excel_smiles.append(can)
    excel_inchikeys.append(compute_inchikey(can) if can else None)


# Load all PLIP XML files
xml_files = glob.glob(os.path.join(PLIP_DIR, "*.xml"))
print(f"\nFound {len(xml_files)} XML files in PLIP directory.\n")

# Parse all PLIP XMLs and store by InChIKey
plip_data = {}  # inchikey → (canonical_smiles, feature_vector)

print("Parsing PLIP XMLs...")
for xml in xml_files:
    sm_xml, ik_xml, feat = extract_plip_info(xml)

    if sm_xml is None and ik_xml is None:
        continue

    can_xml = canonicalize_smiles(sm_xml) if sm_xml else None
    if ik_xml:
        plip_data[ik_xml] = (can_xml, feat)
    elif can_xml:
        plip_data[can_xml] = (can_xml, feat)

print(f"Parsed {len(plip_data)} PLIP entries.\n")


# Match Excel compounds to PLIP features
X = []
missing = []

print("Matching compounds to PLIP entries...\n")
for name, ik, sm in zip(compound_names, excel_inchikeys, canonical_excel_smiles):

    matched = False

    # Match by InChIKey first
    if ik and ik in plip_data:
        X.append(plip_data[ik][1])
        matched = True
    # Fallback: match by canonical SMILES
    elif sm and sm in plip_data:
        X.append(plip_data[sm][1])
        matched = True

    if not matched:
        missing.append(name)
        X.append(np.zeros(8))  # placeholder: no interactions found


X = np.array(X, dtype=float)
print(f"\nPLIP feature matrix shape: {X.shape}")
print(f"Missing compounds: {len(missing)}")

if missing:
    print("⚠ Missing:")
    for m in missing:
        print(f"  - {m}")


# Save outputs
np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y_binary)
np.save(OUTPUT_NAMES, compound_names)

pd.DataFrame(X, columns=[
    "hydrophobic", "hbond", "water", "saltbridge",
    "pi_stack", "pi_cation", "halogen", "metal"
]).to_csv(OUTPUT_CSV, index=False)

print("\n✔ Finished! Saved:")
print(f" - {OUTPUT_X}")
print(f" - {OUTPUT_Y}")
print(f" - {OUTPUT_NAMES}")
print(f" - {OUTPUT_CSV}")
