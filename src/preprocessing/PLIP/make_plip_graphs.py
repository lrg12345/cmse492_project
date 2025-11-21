#!/usr/bin/env python3
"""
Build PLIP-based graphs for OATP1B1 inhibitors, with geometric edge features.

Nodes:
  - Ligand atoms (from SMILES)
  - Protein residues (from PLIP XML)

Node features (unchanged from previous version):
  - First ATOM_DIM entries: atom-specific features
  - Last  RES_DIM entries: residue-specific features
    (atoms fill residue part with zeros; residues fill atom part with zeros)

Edges:
  - Ligand–ligand bonds from RDKit
  - Ligand–residue edges from PLIP interactions
    (hydrophobic, hbond, salt, pi_stack, pi_cation, metal, halogen, water_bridge)

Edge features (edge_attr, new with geometry):
  Dimension = 18, layout:

    0   : is_lig_lig   (1 if ligand–ligand, else 0)
    1   : is_lig_prot  (1 if ligand–protein, else 0)

    2-5 : bond type one-hot (ligand–ligand edges only)
          [single, double, triple, aromatic]

    6   : bond_conjugated (ligand–ligand edges only)
    7   : bond_in_ring    (ligand–ligand edges only)

    8-15: interaction type one-hot (ligand–protein edges only)
          ["hydrophobic", "hbond", "salt_bridge", "pi_stack",
           "pi_cation", "metal", "halogen", "water_bridge"]

    16  : distance (Å), if known; else 0
    17  : 1 / distance, if >0; else 0

Labels:
  - Binary inhibition: 1 if inhib_1b1_pct > 50, else 0

Output:
  plip_graphs.pkl  (dict with keys: graphs, feature_dim, atom_dim, res_dim, compounds_total)

Each graph dict:
  {
    "x": node_feature_matrix (num_nodes, FEAT_DIM),
    "edge_index": (2, num_edges),
    "edge_attr": (num_edges, EDGE_DIM),
    "y": 0/1,
    "name": compound_name
  }
"""

import os
import glob
import pickle
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdchem

# -----------------------------
# Config
# -----------------------------

EXCEL_FILE = "/Users/logangarland/Desktop/cmse492_project/data/processed/karlgren_with_smiles.xlsx"
PLIP_DIR = "/Users/logangarland/Desktop/VSCode/HOLIgraph/Examples/Example_XMLs"
OUTPUT_FILE = "plip_graphs.pkl"

INHIB_COL = "inhib_1b1_pct"
COMPOUND_COL = "compound"
SMILES_COL = "SMILES"
INHIB_THRESHOLD = 50.0  # percent

# -----------------------------
# Node feature dimensions
# -----------------------------

# Atom features:
#   - atom type one-hot (C, N, O, S, P, F, Cl, Br, I, other) = 10
#   - aromatic flag (1)
#   - degree one-hot (0..5) = 6
#   - formal charge one-hot (-1, 0, +1, other) = 4
ATOM_TYPES = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
N_ATOM_TYPES = len(ATOM_TYPES) + 1  # + "other"
N_DEGREES = 6
N_CHARGE_BUCKETS = 4
ATOM_DIM = N_ATOM_TYPES + 1 + N_DEGREES + N_CHARGE_BUCKETS  # 10 + 1 + 6 + 4 = 21

# Residue features:
#   - residue one-hot (20 standard aa) + "OTHER" = 21
#   - interaction flags: hydrophobic, hbond, salt, pi, pi_cation, metal, halogen, water = 8
RES_TYPES = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
]
N_RES_TYPES = len(RES_TYPES) + 1  # + OTHER
INTERACTION_FLAGS = [
    "hydrophobic",
    "hbond",
    "salt_bridge",
    "pi_stack",
    "pi_cation",
    "metal",
    "halogen",
    "water_bridge",
]
N_INTER_FLAGS = len(INTERACTION_FLAGS)
RES_DIM = N_RES_TYPES + N_INTER_FLAGS  # 21 + 8 = 29

FEAT_DIM = ATOM_DIM + RES_DIM

# -----------------------------
# Edge feature dimensions
# -----------------------------

# Edge layout:
#   [is_lig_lig, is_lig_prot,
#    bond_type_onehot(4), bond_conj, bond_ring,
#    interaction_type_onehot(8),
#    distance, inv_distance]
BOND_TYPES = ["single", "double", "triple", "aromatic"]
N_BOND_TYPES = len(BOND_TYPES)

INT_TYPES = [
    "hydrophobic",
    "hbond",
    "salt_bridge",
    "pi_stack",
    "pi_cation",
    "metal",
    "halogen",
    "water_bridge",
]
N_INT_TYPES = len(INT_TYPES)

EDGE_DIM = 2 + N_BOND_TYPES + 1 + 1 + N_INT_TYPES + 2  # 2 + 4 + 1 + 1 + 8 + 2 = 18


# -----------------------------
# Helper: atom features
# -----------------------------

def atom_feature_vector(atom: rdchem.Atom) -> np.ndarray:
    """Return fixed-length atom feature vector of size FEAT_DIM."""
    vec = np.zeros(FEAT_DIM, dtype=np.float32)

    # Slice for atom-specific part
    f = np.zeros(ATOM_DIM, dtype=np.float32)
    idx = 0

    # 1) Atom type
    symbol = atom.GetSymbol()
    if symbol in ATOM_TYPES:
        type_idx = ATOM_TYPES.index(symbol)
    else:
        type_idx = len(ATOM_TYPES)  # "other"
    type_onehot = np.zeros(N_ATOM_TYPES, dtype=np.float32)
    type_onehot[type_idx] = 1.0
    f[idx:idx+N_ATOM_TYPES] = type_onehot
    idx += N_ATOM_TYPES

    # 2) Aromatic
    f[idx] = 1.0 if atom.GetIsAromatic() else 0.0
    idx += 1

    # 3) Degree one-hot (0..5, clip >5 to bucket 5)
    degree = atom.GetDegree()
    if degree > 5:
        degree = 5
    deg_onehot = np.zeros(N_DEGREES, dtype=np.float32)
    deg_onehot[degree] = 1.0
    f[idx:idx+N_DEGREES] = deg_onehot
    idx += N_DEGREES

    # 4) Formal charge bucket
    q = atom.GetFormalCharge()
    if q == -1:
        charge_idx = 0
    elif q == 0:
        charge_idx = 1
    elif q == +1:
        charge_idx = 2
    else:
        charge_idx = 3
    ch_onehot = np.zeros(N_CHARGE_BUCKETS, dtype=np.float32)
    ch_onehot[charge_idx] = 1.0
    f[idx:idx+N_CHARGE_BUCKETS] = ch_onehot

    # Place into first segment of full vector
    vec[:ATOM_DIM] = f
    # residue part remains zero
    return vec


# -----------------------------
# Helper: residue features
# -----------------------------

def residue_feature_vector(resname: str, flags: set) -> np.ndarray:
    """Return fixed-length residue feature vector of size FEAT_DIM."""
    vec = np.zeros(FEAT_DIM, dtype=np.float32)

    # Slice for residue-specific part
    g = np.zeros(RES_DIM, dtype=np.float32)
    idx = 0

    # 1) Residue type one-hot
    rname = resname.upper()
    if rname in RES_TYPES:
        r_idx = RES_TYPES.index(rname)
    else:
        r_idx = len(RES_TYPES)  # OTHER
    res_onehot = np.zeros(N_RES_TYPES, dtype=np.float32)
    res_onehot[r_idx] = 1.0
    g[idx:idx+N_RES_TYPES] = res_onehot
    idx += N_RES_TYPES

    # 2) Interaction flags
    for i, flag_name in enumerate(INTERACTION_FLAGS):
        if flag_name in flags:
            g[idx + i] = 1.0

    # Place into second segment of full vector
    vec[ATOM_DIM:ATOM_DIM + RES_DIM] = g
    return vec


# -----------------------------
# Helper: edge feature vector
# -----------------------------

def make_edge_attr(
    is_lig_lig: bool,
    bond_type: str | None = None,
    is_conjugated: bool = False,
    is_in_ring: bool = False,
    interaction_type: str | None = None,
    distance: float | None = None,
) -> np.ndarray:
    """
    Build edge feature vector of length EDGE_DIM with the schema defined above.
    """
    v = np.zeros(EDGE_DIM, dtype=np.float32)

    # Ligand–ligand vs ligand–protein
    if is_lig_lig:
        v[0] = 1.0
    else:
        v[1] = 1.0

    # Bond-type info (for ligand–ligand edges)
    if bond_type in BOND_TYPES:
        idx = BOND_TYPES.index(bond_type)
        v[2 + idx] = 1.0

    if is_conjugated:
        v[2 + N_BOND_TYPES] = 1.0  # index 6

    if is_in_ring:
        v[2 + N_BOND_TYPES + 1] = 1.0  # index 7

    # Interaction type (for ligand–protein edges)
    if interaction_type in INT_TYPES:
        int_idx = INT_TYPES.index(interaction_type)
        v[8 + int_idx] = 1.0

    # Distance features
    if distance is not None and distance > 0:
        v[-2] = float(distance)
        v[-1] = 1.0 / float(distance)

    return v


# -----------------------------
# Helper: parse PLIP XML (residues + atom-level interactions with geometry)
# -----------------------------

def _get_float_from_tags(elem, tags):
    """
    Try several tag names to fetch a float distance.
    """
    for t in tags:
        txt = elem.findtext(t)
        if txt is not None:
            try:
                return float(txt)
            except ValueError:
                continue
    return None


def parse_plip_xml(xml_path: str):
    """
    Parse a PLIP XML file and extract:

    - residues: dict keyed by residue identifier (e.g. "42A"), value:
          {"aa": "THR", "flags": set([...])}

    - interactions: list of tuples
          (lig_atom_idx, resid_text, interaction_type, distance)

      where lig_atom_idx is the 0-based RDKit atom index.
    """
    residues = {}
    interactions = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"  !! Failed to parse {os.path.basename(xml_path)}: {e}")
        return residues, interactions

    # Build mapping from PDB atom index -> RDKit atom index via smiles_to_pdb
    pdb2rdkit = {}
    s2p_text = root.findtext(".//smiles_to_pdb")
    if s2p_text:
        pairs = s2p_text.strip().split(",")
        for pair in pairs:
            if ":" not in pair:
                continue
            smi_idx_str, pdb_idx_str = pair.split(":")
            try:
                smi_idx = int(smi_idx_str) - 1  # SMILES atom index (1-based) -> RDKit index
                pdb_idx = int(pdb_idx_str)
            except ValueError:
                continue
            pdb2rdkit[pdb_idx] = smi_idx

    # Collect basic residue info from <bs_residues>
    bs_residues = root.findall(".//bs_residue")
    for r in bs_residues:
        aa = r.attrib.get("aa", "UNK")
        resid_text = r.text.strip() if r.text else ""
        if not resid_text:
            continue
        residues[resid_text] = {
            "aa": aa,
            "flags": set(),
        }

    # Helper to ensure residue exists before adding flags or interactions
    def ensure_resid(resnr: str, reschain: str):
        resid_text = f"{resnr}{reschain}"
        if resid_text not in residues:
            residues[resid_text] = {"aa": "UNK", "flags": set()}
        return resid_text

    def add_flag(resid_text: str, flag: str):
        if resid_text not in residues:
            residues[resid_text] = {"aa": "UNK", "flags": set()}
        residues[resid_text]["flags"].add(flag)

    def add_interaction(pdb_idx: int, resnr: str, reschain: str,
                        int_type: str, distance: float | None):
        if pdb_idx not in pdb2rdkit:
            return
        lig_idx = pdb2rdkit[pdb_idx]
        resid_text = ensure_resid(resnr, reschain)
        interactions.append((lig_idx, resid_text, int_type, distance))

    # --- Hydrophobic interactions ---
    for hphob in root.findall(".//hydrophobic_interaction"):
        resnr = hphob.findtext("resnr")
        reschain = hphob.findtext("reschain")
        ligcarbonidx = hphob.findtext("ligcarbonidx")
        if resnr and reschain and ligcarbonidx:
            resid_text = ensure_resid(resnr, reschain)
            add_flag(resid_text, "hydrophobic")
            dist = _get_float_from_tags(hphob, ["dist"])
            try:
                pdb_idx = int(ligcarbonidx)
                add_interaction(pdb_idx, resnr, reschain, "hydrophobic", dist)
            except ValueError:
                pass

    # --- Hydrogen bonds ---
    for hb in root.findall(".//hydrogen_bond"):
        resnr = hb.findtext("resnr")
        reschain = hb.findtext("reschain")
        if not (resnr and reschain):
            continue
        resid_text = ensure_resid(resnr, reschain)
        add_flag(resid_text, "hbond")

        dist = _get_float_from_tags(hb, ["dist_d-a", "dist_h-a"])
        protisdon = hb.findtext("protisdon")
        donoridx = hb.findtext("donoridx")
        acceptoridx = hb.findtext("acceptoridx")

        lig_pdb_idx = None
        if protisdon == "True" and acceptoridx is not None:
            # Protein is donor, ligand is acceptor
            try:
                lig_pdb_idx = int(acceptoridx)
            except ValueError:
                pass
        elif protisdon == "False" and donoridx is not None:
            # Ligand is donor
            try:
                lig_pdb_idx = int(donoridx)
            except ValueError:
                pass

        if lig_pdb_idx is not None:
            add_interaction(lig_pdb_idx, resnr, reschain, "hbond", dist)

    # --- Salt bridges ---
    for sb in root.findall(".//salt_bridge"):
        resnr = sb.findtext("resnr")
        reschain = sb.findtext("reschain")
        if not (resnr and reschain):
            continue
        resid_text = ensure_resid(resnr, reschain)
        add_flag(resid_text, "salt_bridge")
        dist = _get_float_from_tags(sb, ["dist"])

        lig_list = sb.find("lig_idx_list")
        if lig_list is not None:
            for idx_elem in lig_list.findall("idx"):
                txt = idx_elem.text
                if not txt:
                    continue
                try:
                    lig_pdb_idx = int(txt)
                except ValueError:
                    continue
                add_interaction(lig_pdb_idx, resnr, reschain, "salt_bridge", dist)

    # --- Pi stacks ---
    for ps in root.findall(".//pi_stack"):
        resnr = ps.findtext("resnr")
        reschain = ps.findtext("reschain")
        if not (resnr and reschain):
            continue
        resid_text = ensure_resid(resnr, reschain)
        add_flag(resid_text, "pi_stack")
        dist = _get_float_from_tags(ps, ["dist"])

        lig_list = ps.find("lig_idx_list")
        if lig_list is not None:
            for idx_elem in lig_list.findall("idx"):
                txt = idx_elem.text
                if not txt:
                    continue
                try:
                    lig_pdb_idx = int(txt)
                except ValueError:
                    continue
                add_interaction(lig_pdb_idx, resnr, reschain, "pi_stack", dist)

    # --- Pi-cation ---
    for pc in root.findall(".//pi_cation_interaction"):
        resnr = pc.findtext("resnr")
        reschain = pc.findtext("reschain")
        if not (resnr and reschain):
            continue
        resid_text = ensure_resid(resnr, reschain)
        add_flag(resid_text, "pi_cation")
        dist = _get_float_from_tags(pc, ["dist"])

        lig_list = pc.find("lig_idx_list")
        if lig_list is not None:
            for idx_elem in lig_list.findall("idx"):
                txt = idx_elem.text
                if not txt:
                    continue
                try:
                    lig_pdb_idx = int(txt)
                except ValueError:
                    continue
                add_interaction(lig_pdb_idx, resnr, reschain, "pi_cation", dist)

    # --- Halogen bonds ---
    for hb in root.findall(".//halogen_bond"):
        resnr = hb.findtext("resnr")
        reschain = hb.findtext("reschain")
        if not (resnr and reschain):
            continue
        resid_text = ensure_resid(resnr, reschain)
        add_flag(resid_text, "halogen")
        dist = _get_float_from_tags(hb, ["dist"])

        lig_list = hb.find("lig_idx_list")
        if lig_list is not None:
            for idx_elem in lig_list.findall("idx"):
                txt = idx_elem.text
                if not txt:
                    continue
                try:
                    lig_pdb_idx = int(txt)
                except ValueError:
                    continue
                add_interaction(lig_pdb_idx, resnr, reschain, "halogen", dist)

    # --- Metal complexes ---
    for mc in root.findall(".//metal_complex"):
        resnr = mc.findtext("resnr")
        reschain = mc.findtext("reschain")
        if not (resnr and reschain):
            continue
        resid_text = ensure_resid(resnr, reschain)
        add_flag(resid_text, "metal")
        dist = _get_float_from_tags(mc, ["dist"])

        lig_list = mc.find("lig_idx_list")
        if lig_list is not None:
            for idx_elem in lig_list.findall("idx"):
                txt = idx_elem.text
                if not txt:
                    continue
                try:
                    lig_pdb_idx = int(txt)
                except ValueError:
                    continue
                add_interaction(lig_pdb_idx, resnr, reschain, "metal", dist)

    # --- Water bridges ---
    for wb in root.findall(".//water_bridge"):
        resnr = wb.findtext("resnr")
        reschain = wb.findtext("reschain")
        if not (resnr and reschain):
            continue
        resid_text = ensure_resid(resnr, reschain)
        add_flag(resid_text, "water_bridge")
        dist = _get_float_from_tags(wb, ["dist_a-w", "dist_d-w", "dist"])

        lig_list = wb.find("lig_idx_list")
        if lig_list is not None:
            for idx_elem in lig_list.findall("idx"):
                txt = idx_elem.text
                if not txt:
                    continue
                try:
                    lig_pdb_idx = int(txt)
                except ValueError:
                    continue
                add_interaction(lig_pdb_idx, resnr, reschain, "water_bridge", dist)

    return residues, interactions


# -----------------------------
# Main
# -----------------------------

def main():
    print("Loading Excel dataset...")
    df = pd.read_excel(EXCEL_FILE)

    if COMPOUND_COL not in df.columns or SMILES_COL not in df.columns or INHIB_COL not in df.columns:
        raise ValueError(
            f"Expected columns '{COMPOUND_COL}', '{SMILES_COL}', and '{INHIB_COL}' "
            f"in {EXCEL_FILE}. Found: {list(df.columns)}"
        )

    compounds = df[COMPOUND_COL].astype(str).values
    smiles_list = df[SMILES_COL].astype(str).values
    inhib_pct = df[INHIB_COL].astype(float).values

    y_binary = (inhib_pct > INHIB_THRESHOLD).astype(int)

    print(f"Total compounds: {len(compounds)}")
    print(f"Positive class (> {INHIB_THRESHOLD}%): {int(y_binary.sum())}")
    print(f"Negative class: {len(y_binary) - int(y_binary.sum())}")

    print("\nIndexing PLIP XML files...")
    xml_paths = glob.glob(os.path.join(PLIP_DIR, "*.xml"))
    xml_map = {}
    for xp in xml_paths:
        base = os.path.basename(xp)
        # Expect name like Ligand-1B1in-raw.xml or similar
        name_part = base.split("-1B1in")[0]
        xml_map[name_part] = xp

    print(f"Found {len(xml_paths)} XML files.")
    print(f"Indexed {len(xml_map)} base names.\n")

    graphs = []
    missing_xml = []
    bad_smiles = []
    no_residues = []
    no_atom_interactions = []

    for compound, smi, y in zip(compounds, smiles_list, y_binary):
        xml_path = xml_map.get(compound)
        if xml_path is None:
            missing_xml.append(compound)
            continue

        # Parse SMILES
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad_smiles.append(compound)
            continue

        # Atom nodes
        atom_nodes = [atom_feature_vector(atom) for atom in mol.GetAtoms()]
        atom_nodes = np.stack(atom_nodes, axis=0)
        n_atoms = atom_nodes.shape[0]

        # Residues + interactions from PLIP
        residues, interactions = parse_plip_xml(xml_path)
        if not residues:
            no_residues.append(compound)
            continue

        if not interactions:
            no_atom_interactions.append(compound)

        # Residue nodes
        res_nodes = []
        resid_list = sorted(residues.keys())
        resid_to_idx = {}
        for idx, resid_text in enumerate(resid_list):
            resid_to_idx[resid_text] = n_atoms + idx
            info = residues[resid_text]
            res_nodes.append(residue_feature_vector(info["aa"], info["flags"]))

        res_nodes = np.stack(res_nodes, axis=0)
        n_res = res_nodes.shape[0]

        # Stack all nodes
        x = np.vstack([atom_nodes, res_nodes])

        # Build edges and edge attributes
        edge_list = []
        edge_attr_list = []

        # 1) Ligand–ligand bonds
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Bond type mapping
            bt = bond.GetBondType()
            if bt == rdchem.BondType.SINGLE:
                btype = "single"
            elif bt == rdchem.BondType.DOUBLE:
                btype = "double"
            elif bt == rdchem.BondType.TRIPLE:
                btype = "triple"
            elif bt == rdchem.BondType.AROMATIC:
                btype = "aromatic"
            else:
                btype = None

            is_conj = bond.GetIsConjugated()
            is_ring = bond.IsInRing()

            attr = make_edge_attr(
                is_lig_lig=True,
                bond_type=btype,
                is_conjugated=is_conj,
                is_in_ring=is_ring,
                interaction_type=None,
                distance=None,   # no geometry here (could be added later via RDKit 3D)
            )

            # undirected (i->j and j->i)
            edge_list.append((i, j))
            edge_attr_list.append(attr)
            edge_list.append((j, i))
            edge_attr_list.append(attr)

        # 2) Ligand–residue edges from PLIP interactions (with distance)
        for lig_idx, resid_text, int_type, dist in interactions:
            if resid_text not in resid_to_idx:
                continue
            res_node_idx = resid_to_idx[resid_text]

            attr = make_edge_attr(
                is_lig_lig=False,
                bond_type=None,
                is_conjugated=False,
                is_in_ring=False,
                interaction_type=int_type,
                distance=dist,
            )

            edge_list.append((lig_idx, res_node_idx))
            edge_attr_list.append(attr)
            edge_list.append((res_node_idx, lig_idx))
            edge_attr_list.append(attr)

        if not edge_list:
            # extremely unlikely; fallback to empty edge_index/edge_attr
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, EDGE_DIM), dtype=np.float32)
        else:
            edge_index = np.array(edge_list, dtype=np.int64).T
            edge_attr = np.stack(edge_attr_list, axis=0).astype(np.float32)

        graph = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": int(y),
            "name": compound,
        }
        graphs.append(graph)

    print("\nSummary:")
    print(f"  Graphs built: {len(graphs)} / {len(compounds)}")

    if missing_xml:
        print(f"  Missing XML for {len(missing_xml)} compounds, e.g.: {missing_xml[:10]}")
    if bad_smiles:
        print(f"  Invalid SMILES for {len(bad_smiles)} compounds, e.g.: {bad_smiles[:10]}")
    if no_residues:
        print(f"  No residues parsed for {len(no_residues)} compounds, e.g.: {no_residues[:10]}")
    if no_atom_interactions:
        print(f"  No PLIP atom-level interactions for {len(no_atom_interactions)} compounds, e.g.: {no_atom_interactions[:10]}")

    print(f"\nSaving graphs to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(
            {
                "graphs": graphs,
                "feature_dim": FEAT_DIM,
                "atom_dim": ATOM_DIM,
                "res_dim": RES_DIM,
                "compounds_total": len(compounds),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("✔ Done.")


if __name__ == "__main__":
    main()
