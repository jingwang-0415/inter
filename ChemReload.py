
"""

重写了rdkit中的方法

"""
import os
import re
from rdkit import RDConfig
from rdkit import Chem
from rdkit.Chem import AllChem
from data_process.datautil import BOND_FLOAT_TO_TYPE
defaultPatternFileName = os.path.join(RDConfig.RDDataDir, 'FragmentDescriptors.csv')
BOND_LABEL_TO_FLOAT = {
    5: 0.0,
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 1.5,
}

patterns={}
with open(defaultPatternFileName, 'r') as inF:
  for line in inF.readlines():
    if len(line) and line[0] != '#':
      splitL = line.split('\t')
      if len(splitL) >= 3:
        name = splitL[0]
        descr = splitL[1]
        sma = splitL[2]
        patt = Chem.MolFromSmarts(sma)
        patterns[name]=patt
def LoadPatterns(pattern,mol,countUnique=True):
  if pattern in patterns.keys():

    return len(mol.GetSubstructMatches(patterns[pattern], uniquify=countUnique)),mol.GetSubstructMatches(patterns[pattern], uniquify=countUnique)
  else:
    return 0,0
def parse_smilesnomap(smiles_list):
  parse_lists=[]
  for smi in smiles_list:
    mol_no_maps = AllChem.MolFromSmiles(re.sub(r":[-+]?[0-9]+", "", smi))
    AllChem.SanitizeMol(mol_no_maps)
    parse_lists.append(AllChem.MolToSmiles(mol_no_maps, canonical=True))
  return parse_lists


def GetAtomByid(mol,id):
    return mol.GetAtomWithIdx(int(id))
def GetNerghbors(atom):
    return atom.GetNeighbors()
def GetAtomMapNum(atom):
    return atom.GetAtomMapNum()
def GetAtomSymbol(atom):
    return atom.GetSymbol()
def GetMolAtoms(mol):
    return mol.GetAtoms()
def GetAtomId(atom):
    return atom.GetIdx()
def SetAtomMapnum(atom,num):
    return atom.SetAtomMapNum(num)
def GetBond(mol,begin,end):
    return mol.GetBondBetweenAtoms(int(begin),int(end))
def GetBondType(bond):
    return bond.GetBondType()
def GetBondTypeAsDouble(bond):
    return bond.GetBondTypeAsDouble()

def  mol_edit(new_mol,edits):
    # Keep track of aromatic nitrogens, might cause explicit hydrogen issues
    aromatic_nitrogen_idx = set()
    aromatic_carbonyl_adj_to_aromatic_nH = {}
    aromatic_carbondeg3_adj_to_aromatic_nH0 = {}
    for a in new_mol.GetAtoms():
        if a.GetIsAromatic() and a.GetSymbol() == 'N':
            aromatic_nitrogen_idx.add(a.GetIdx())
            for nbr in a.GetNeighbors():
                nbr_is_carbon = (nbr.GetSymbol() == 'C')
                nbr_is_aromatic = nbr.GetIsAromatic()
                nbr_has_double_bond = any(b.GetBondTypeAsDouble() == 2 for b in nbr.GetBonds())
                nbr_has_3_bonds = (len(nbr.GetBonds()) == 3)

                if (a.GetNumExplicitHs() ==1 and nbr_is_carbon and nbr_is_aromatic
                    and nbr_has_double_bond):
                    aromatic_carbonyl_adj_to_aromatic_nH[nbr.GetIdx()] = a.GetIdx()
                elif (a.GetNumExplicitHs() == 0 and nbr_is_carbon and nbr_is_aromatic
                    and nbr_has_3_bonds):
                    aromatic_carbondeg3_adj_to_aromatic_nH0[nbr.GetIdx()] = a.GetIdx()
        else:
            a.SetNumExplicitHs(0)
    new_mol.UpdatePropertyCache()

    # Apply the edits as predicted
    for edit in edits:
        x, y,  new_bo = edit


        if y == 0:
            continue
        new_bo = float(new_bo)
        bond = new_mol.GetBondBetweenAtoms(x,y)
        a1 = new_mol.GetAtomWithIdx(x)
        a2 = new_mol.GetAtomWithIdx(y)

        if bond is not None:
            new_mol.RemoveBond(x,y)

            # Are we losing a bond on an aromatic nitrogen?
            if bond.GetBondTypeAsDouble() == 1.0:
                if x in aromatic_nitrogen_idx:
                    if a1.GetTotalNumHs() == 0:
                        a1.SetNumExplicitHs(1)
                    elif a1.GetFormalCharge() == 1:
                        a1.SetFormalCharge(0)
                elif y in aromatic_nitrogen_idx:
                    if a2.GetTotalNumHs() == 0:
                        a2.SetNumExplicitHs(1)
                    elif a2.GetFormalCharge() == 1:
                        a2.SetFormalCharge(0)

            # Are we losing a c=O bond on an aromatic ring? If so, remove H from adjacent nH if appropriate
            if bond.GetBondTypeAsDouble() == 2.0:
                if x in aromatic_carbonyl_adj_to_aromatic_nH:
                    new_mol.GetAtomWithIdx(aromatic_carbonyl_adj_to_aromatic_nH[x]).SetNumExplicitHs(0)
                elif y in aromatic_carbonyl_adj_to_aromatic_nH:
                    new_mol.GetAtomWithIdx(aromatic_carbonyl_adj_to_aromatic_nH[y]).SetNumExplicitHs(0)

        if new_bo > 0:
            new_mol.AddBond(x,y ,BOND_FLOAT_TO_TYPE[new_bo])

            # Special alkylation case?
            if new_bo == 1:
                if x in aromatic_nitrogen_idx:
                    if a1.GetTotalNumHs() == 1:
                        a1.SetNumExplicitHs(0)
                    else:
                        a1.SetFormalCharge(1)
                elif y in aromatic_nitrogen_idx:
                    if a2.GetTotalNumHs() == 1:
                        a2.SetNumExplicitHs(0)
                    else:
                        a2.SetFormalCharge(1)

            # Are we getting a c=O bond on an aromatic ring? If so, add H to adjacent nH0 if appropriate
            if new_bo == 2:
                if x in aromatic_carbondeg3_adj_to_aromatic_nH0:
                    new_mol.GetAtomWithIdx(aromatic_carbondeg3_adj_to_aromatic_nH0[x]).SetNumExplicitHs(1)
                elif y in aromatic_carbondeg3_adj_to_aromatic_nH0:
                    new_mol.GetAtomWithIdx(aromatic_carbondeg3_adj_to_aromatic_nH0[y]).SetNumExplicitHs(1)

    pred_mol = new_mol.GetMol()

    # Clear formal charges to make molecules valid
    # Note: because S and P (among others) can change valence, be more flexible
    for atom in pred_mol.GetAtoms():
        if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 1: # exclude negatively-charged azide
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals <= 3:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'N' and atom.GetFormalCharge() == -1: # handle negatively-charged azide addition
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals == 3 and any([nbr.GetSymbol() == 'N' for nbr in atom.GetNeighbors()]):
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'N':
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals == 4 and not atom.GetIsAromatic(): # and atom.IsInRingSize(5)):
                atom.SetFormalCharge(1)
        elif atom.GetSymbol() == 'C' and atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'O' and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]) + atom.GetNumExplicitHs()
            if bond_vals == 2:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() in ['Cl', 'Br', 'I', 'F'] and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals == 1:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'S' and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals in [2, 4, 6]:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'P': # quartenary phosphorous should be pos. charge with 0 H
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 4 and len(bond_vals) == 4:
                atom.SetFormalCharge(1)
                atom.SetNumExplicitHs(0)
            elif sum(bond_vals) == 3 and len(bond_vals) == 3: # make sure neutral
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'B': # quartenary boron should be neg. charge with 0 H
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 4 and len(bond_vals) == 4:
                atom.SetFormalCharge(-1)
                atom.SetNumExplicitHs(0)
        elif atom.GetSymbol() in ['Mg', 'Zn']:
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 1 and len(bond_vals) == 1:
                atom.SetFormalCharge(1)
        elif atom.GetSymbol() == 'Si':
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == len(bond_vals):
                atom.SetNumExplicitHs(max(0, 4 - len(bond_vals)))
    return pred_mol