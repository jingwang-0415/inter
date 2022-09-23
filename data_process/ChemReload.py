
"""

重写了rdkit中的方法

"""
import os
import re
from rdkit import RDConfig
from rdkit import Chem
from rdkit.Chem import AllChem
defaultPatternFileName = os.path.join(RDConfig.RDDataDir, 'FragmentDescriptors.csv')


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