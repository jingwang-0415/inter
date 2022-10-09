import re
from rdkit.Chem import AllChem
from rdkit import Chem
import pandas as pd
from rdkit.Chem import Draw
from tqdm import tqdm
BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, \
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],
}
BOND_FLOAT_TO_LABEL = {
    0.0: 5,
    1.0: 1,
    2.0: 2,
    3.0: 3,
    1.5: 4,
}
# file_dir="numhequal.txt"
# same =open(file_dir,'a')
# file_dir2="numhdiffer.txt"
# differ =open(file_dir2,'a')
# file_dir3 = 'beyond.txt'
# newsmiles = open(file_dir3,'a')
def parse_reaction_roles(rxn_smiles, as_what="smiles"):
    """ Convert a reaction SMILES string to lists of reactants, reagents and products in various data formats. """

    # Split the reaction SMILES string by the '>' symbol to obtain the reactants and products.
    rxn_roles = rxn_smiles.split(">")

    # NOTE: In some cases, there can be additional characters on the product side separated by a whitespace. For this
    # reason the product side string is always additionally split by the whitespace and the only the first element is
    # considered.

    # Parse the original SMILES strings from the reaction string including the reaction atom mappings.
    if as_what == "smiles":
        return [x for x in rxn_roles[0].split(".") if x != ""],\
               [x for x in rxn_roles[1].split(".") if x != ""],\
               [x for x in rxn_roles[2].split(" ")[0].split(".") if x != ""]

    # Parse the original SMILES strings from the reaction string excluding the reaction atom mappings.
    elif as_what == "smiles_no_maps":
        return [re.sub(r":[-+]?[0-9]+", "", x) for x in rxn_roles[0].split(".") if x != ""],\
               [re.sub(r":[-+]?[0-9]+", "", x) for x in rxn_roles[1].split(".") if x != ""],\
               [re.sub(r":[-+]?[0-9]+", "", x) for x in rxn_roles[2].split(" ")[0].split(".") if x != ""]

    # Parse the lists of atom map numbers from the reactions SMILES string.
    elif as_what == "atom_maps":
        return [[int(x[1:]) for x in re.findall(r":[-+]?[0-9]+", y)] for y in rxn_roles[0].split(".") if y != ""],\
               [[int(x[1:]) for x in re.findall(r":[-+]?[0-9]+", y)] for y in rxn_roles[1].split(".") if y != ""],\
               [[int(x[1:]) for x in re.findall(r":[-+]?[0-9]+", y)]
                for y in rxn_roles[2].split(" ")[0].split(".") if y != ""]

    # Parse the lists of number of mapped atoms per reactant and per product. Reagents usually do not contain any
    # mapping, but they are included here for the sake of consistency.
    elif as_what == "mapping_numbers":
        return [len([el for el in rxn_roles[0].split(".") if el != ""]),
                len([el for el in rxn_roles[1].split(".") if el != ""]),
                len([el for el in rxn_roles[2].split(" ")[0].split(".") if el != ""])]

    # Parsings that include initial conversion to RDKit Mol objects and need to include sanitization: mol, mol_no_maps,
    # canonical_smiles and canonical_smiles_no_maps.
    elif as_what in ["mol", "mol_no_maps", "canonical_smiles", "canonical_smiles_no_maps"]:
        reactants, reagents, products = [], [], []

        # Iterate through all of the reactants.
        for reactant in rxn_roles[0].split("."):
            if reactant != "" and as_what in ["mol", "canonical_smiles"]:
                mol_maps = AllChem.MolFromSmiles(reactant)
                AllChem.SanitizeMol(mol_maps)#检查是否是正确的smiles表达式

                if as_what == "mol":
                    reactants.append(mol_maps)
                    continue

                if as_what == "canonical_smiles":
                    reactants.append(AllChem.MolToSmiles(mol_maps, canonical=True))
                    continue

            elif reactant != "" and as_what in ["mol_no_maps", "canonical_smiles_no_maps"]:
                mol_no_maps = AllChem.MolFromSmiles(re.sub(r":[-+]?[0-9]+", "", reactant))
                AllChem.SanitizeMol(mol_no_maps)

                if as_what == "mol_no_maps":
                    reactants.append(mol_no_maps)
                    continue

                if as_what == "canonical_smiles_no_maps":
                    reactants.append(AllChem.MolToSmiles(mol_no_maps, canonical=True))
                    continue

        # Iterate through all of the reagents.
        for reagent in rxn_roles[1].split("."):
            if reagent != "" and as_what in ["mol", "canonical_smiles"]:
                mol_maps = AllChem.MolFromSmiles(reagent)
                AllChem.SanitizeMol(mol_maps)

                if as_what == "mol":
                    reagents.append(mol_maps)
                    continue

                if as_what == "canonical_smiles":
                    reagents.append(AllChem.MolToSmiles(mol_maps, canonical=True))
                    continue

            elif reagent != "" and as_what in ["mol_no_maps", "canonical_smiles_no_maps"]:
                mol_no_maps = AllChem.MolFromSmiles(re.sub(r":[-+]?[0-9]+", "", reagent))
                AllChem.SanitizeMol(mol_no_maps)

                if as_what == "mol_no_maps":
                    reagents.append(mol_no_maps)
                    continue

                if as_what == "canonical_smiles_no_maps":
                    reagents.append(AllChem.MolToSmiles(mol_no_maps, canonical=True))
                    continue

        # Iterate through all of the reactants.
        for product in rxn_roles[2].split(" ")[0].split("."):
            if product != "" and as_what in ["mol", "canonical_smiles"]:
                mol_maps = AllChem.MolFromSmiles(product)
                AllChem.SanitizeMol(mol_maps)

                if as_what == "mol":
                    products.append(mol_maps)
                    continue

                if as_what == "canonical_smiles":
                    products.append(AllChem.MolToSmiles(mol_maps, canonical=True))
                    continue

            elif product != "" and as_what in ["mol_no_maps", "canonical_smiles_no_maps"]:
                mol_no_maps = AllChem.MolFromSmiles(re.sub(r":[-+]?[0-9]+", "", product))
                AllChem.SanitizeMol(mol_no_maps)

                if as_what == "mol_no_maps":
                    products.append(mol_no_maps)
                    continue

                if as_what == "canonical_smiles_no_maps":
                    products.append(AllChem.MolToSmiles(mol_no_maps, canonical=True))
                    continue

        return reactants, reagents, products

    # Raise exception for any other keyword.
    else:
        raise Exception("Unknown parsing type. Select one of the following: "
                        "'smiles', 'smiles_no_maps', 'atom_maps', 'mapping_numbers', 'mol', 'mol_no_maps', "
                        "'canonical_smiles', 'canonical_smiles_no_maps'.")
def molecule_is_mapped(mol):
    """ Checks if a molecule created from a RDKit Mol object or a SMILES string contains at least one mapped atom."""

    # If it is a RDKit Mol object, check if any atom map number has a value other than zero.
    if not isinstance(mol, str):
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() != 0:
                return True
        return False

    # If it is a SMILES string, check if the string contains the symbol ":" used for mapping.
    else:
        return ":" in mol
def same_neighbourhood_size(atom_index_1, molecule_1, atom_index_2, molecule_2):
    """ Checks whether the same atoms in two different molecules (e.g., reactant and product molecules) have the same
        neighbourhood size. """

    if len(molecule_1.GetAtomWithIdx(atom_index_1).GetNeighbors()) != \
            len(molecule_2.GetAtomWithIdx(atom_index_2).GetNeighbors()):
        return False
    return True


def same_neighbour_atoms(atom_index_1, molecule_1, atom_index_2, molecule_2):
    """ Checks whether the same atoms in two different molecules (e.g., reactant and product molecules) have retained
        the same types of chemical elements in their immediate neighbourhood according to reaction mapping numbers. """

    neighbourhood_1, neighbourhood_2 = [], []

    for i in molecule_1.GetAtomWithIdx(atom_index_1).GetNeighbors():
        neighbourhood_1.append((i.GetAtomMapNum(), i.GetSymbol(), i.GetFormalCharge(),
                                i.GetNumRadicalElectrons(), i.GetTotalValence()))
    for j in molecule_2.GetAtomWithIdx(atom_index_2).GetNeighbors():
        neighbourhood_2.append((j.GetAtomMapNum(), j.GetSymbol(), j.GetFormalCharge(),
                                j.GetNumRadicalElectrons(), j.GetTotalValence()))

    return sorted(neighbourhood_1) == sorted(neighbourhood_2)


def same_neighbour_bonds(atom_index_1, molecule_1, atom_index_2, molecule_2):
    """ Checks whether the same atoms in two different molecules (e.g., reactant and product molecules) have retained
        the same types of chemical bonds amongst each other in their immediate neighbourhood. """

    neighbourhood_1, neighbourhood_2 = [], []

    for i in molecule_1.GetAtomWithIdx(atom_index_1).GetNeighbors():
        neighbourhood_1.append((i.GetAtomMapNum(),
                                str(molecule_1.GetBondBetweenAtoms(atom_index_1, i.GetIdx()).GetBondType())))
    for j in molecule_2.GetAtomWithIdx(atom_index_2).GetNeighbors():
        neighbourhood_2.append((j.GetAtomMapNum(),
                                str(molecule_2.GetBondBetweenAtoms(atom_index_2, j.GetIdx()).GetBondType())))
    #b=molecule_1.GetBondBetweenAtoms(atom_index_1,i.GetIdx()).GetIdx()
    return sorted(neighbourhood_1) == sorted(neighbourhood_2)
#new version
#
def align_kekule_pairs(reac_mol, prod_mol):

    max_amap = max([atom.GetAtomMapNum() for atom in reac_mol.GetAtoms()])
    for atom in reac_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap = max_amap + 1


    prod_prev = get_bond_info(prod_mol)
    Chem.Kekulize(prod_mol)
    prod_new = get_bond_info(prod_mol)

    reac_prev = get_bond_info(reac_mol)
    Chem.Kekulize(reac_mol)
    reac_new = get_bond_info(reac_mol)

    for bond in prod_new:
        if bond in reac_new and (prod_prev[bond][0] == reac_prev[bond][0]):
            reac_new[bond][0] = prod_new[bond][0]

    reac_mol = Chem.RWMol(reac_mol)
    amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reac_mol.GetAtoms()}

    for bond in reac_new:
        idx1, idx2 = amap_idx[bond[0]], amap_idx[bond[1]]
        bo = reac_new[bond][0]
        reac_mol.RemoveBond(idx1, idx2)
        reac_mol.AddBond(idx1, idx2, BOND_FLOAT_TO_TYPE[bo])

    return reac_mol.GetMol(), prod_mol
def get_difference_bond(reacs_mol, prod_mol,usekekule=False):
    #molecule_1 pro molecule_2 rec

    bond_results = {}
    atom_results = {}
    add_results = {}
    if usekekule:
        reacs_mol , prod_mol = align_kekule_pairs(reac_mol=reacs_mol,prod_mol=prod_mol)
    prod_bonds = get_bond_info(prod_mol)
    p_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in prod_mol.GetAtoms()}

    max_amap = max([atom.GetAtomMapNum() for atom in reacs_mol.GetAtoms()])
    for atom in reacs_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap += 1

    reac_bonds = get_bond_info(reacs_mol)
    reac_amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reacs_mol.GetAtoms()}
    for bond in prod_bonds:
        if bond in reac_bonds and prod_bonds[bond][0] != reac_bonds[bond][0]:
            a_start, a_end = bond
            reac_bo = reac_bonds[bond][0]
            bond_results[(p_amap_idx[a_start],p_amap_idx[a_end])] = BOND_FLOAT_TO_LABEL[reac_bo]
            atom_results[p_amap_idx[a_start]] = 1
            atom_results[p_amap_idx[a_end]] = 1
        if bond not in reac_bonds:
            a_start, a_end = bond
            reac_bo = 0.0
            bond_results[(p_amap_idx[a_start],p_amap_idx[a_end])] = BOND_FLOAT_TO_LABEL[reac_bo]
            atom_results[p_amap_idx[a_start]] = 1
            atom_results[p_amap_idx[a_end]] = 1
    #产生新的化学键 在产物内部
    for bond in reac_bonds:
        if bond not in prod_bonds:
            amap1, amap2 = bond
            if (amap1 in p_amap_idx) and (amap2 in p_amap_idx):
                a_start = p_amap_idx[amap1]
                a_end = p_amap_idx[amap2]
                atom_results[a_start] = 1
                atom_results[a_end] = 1
                add_results[(a_start,a_end)] = BOND_FLOAT_TO_LABEL[ reac_bonds[bond][0]]

    return bond_results, atom_results, add_results

def get_bond_info(mol: Chem.Mol):
    """Get information on bonds in the molecule.

    Parameters
    ----------
    mol: Chem.Mol
        Molecule
    """
    if mol is None:
        return {}

    bond_info = {}
    for bond in mol.GetBonds():
        a_start = bond.GetBeginAtom().GetAtomMapNum()
        a_end = bond.GetEndAtom().GetAtomMapNum()

        key_pair = sorted([a_start, a_end])
        bond_info[tuple(key_pair)] = [bond.GetBondTypeAsDouble(), bond.GetIdx()]

    return bond_info

def get_reaction_core_atoms(reactants,product,usekekule=False):

    bond_edits, atom_edits,add_edits = get_difference_bond(reacs_mol=reactants,prod_mol=product,usekekule=usekekule)

    if len(atom_edits) == 0 :
        for p_atom in product.GetAtoms():
            assert p_atom.GetAtomMapNum()!= 0
            for r_atom in reactants.GetAtoms():
                max_amap = max([atom.GetAtomMapNum() for atom in reactants.GetAtoms()])
                if r_atom.GetAtomMapNum() <= 0:
                    r_atom.SetAtomMapNum(max_amap + 1) #对recant中的没有mapnum的原子分配
                    max_amap += 1
                    continue
                if p_atom.GetAtomMapNum() == r_atom.GetAtomMapNum():
                    if not same_neighbourhood_size(p_atom.GetIdx(), product, r_atom.GetIdx(), reactants) or \
                            not same_neighbour_atoms(p_atom.GetIdx(), product, r_atom.GetIdx(), reactants) or \
                            not same_neighbour_bonds(p_atom.GetIdx(), product, r_atom.GetIdx(), reactants):
                        atom_edits[p_atom.GetIdx()] = 1
    return bond_edits,atom_edits,add_edits
# def get_reaction_core_atoms(rsmiles):
#     reactants, _, products = parse_reaction_roles(rsmiles, as_what="mol")
#     # draw = Draw.MolsToImage(reactants)
#     # draw.save('reac1.png')
#     # draw = Draw.MolsToImage(products)
#     # draw.save('pro1.png')
#     # draw = Draw.MolsToImage(reactants)
#     # draw.save('regant.png')
#     reactants_final = [set() for _ in range(len(reactants))]
#     products_final = [set() for _ in range(len(products))]
#     # reactants_final = [set() ]
#     # products_final = [set()]
#     bond_edits = {}
#     atom_edits = {}
#     count = 0
#     for p_ind, product in enumerate(products):
#
#         for r_ind, reactant in enumerate(reactants):
#             for p_atom in product.GetAtoms():
#                 if p_atom.GetAtomMapNum() <= 0:
#                     products_final[p_ind].add(p_atom.GetIdx())
#                     continue
#                 for r_atom in reactant.GetAtoms():
#                     max_amap = max([atom.GetAtomMapNum() for atom in reactant.GetAtoms()])
#
#                     if molecule_is_mapped(reactant) and r_atom.GetAtomMapNum() <= 0:
#                         reactants_final[r_ind].add(r_atom.GetIdx())
#                         r_atom.SetAtomMapNum(max_amap + 1) #对recant中的没有mapnum的原子分配
#                         max_amap += 1
#                         continue
#                     if p_atom.GetAtomMapNum() == r_atom.GetAtomMapNum():
#                         if not same_neighbourhood_size(p_atom.GetIdx(), product, r_atom.GetIdx(), reactant) or \
#                                 not same_neighbour_atoms(p_atom.GetIdx(), product, r_atom.GetIdx(), reactant) or \
#                                 not same_neighbour_bonds(p_atom.GetIdx(), product, r_atom.GetIdx(), reactant):
#                             reactants_final[r_ind].add(r_atom.GetIdx())
#                             products_final[p_ind].add(p_atom.GetIdx())
#                             bond_edit,atom_edit =get_difference_bond(p_atom.GetIdx(),product,r_atom.GetIdx(),reactant)
#                             bond_edits.update(bond_edit)
#                             atom_edits.update(atom_edit)
#     # if 2 in bond_edits.values() and 1 not in bond_edits.values():
#     #         r,p =rsmiles.split('>>')
#     #         reac_mol = Chem.MolFromSmiles(r)
#     #         prod_mol = Chem.MolFromSmiles(p)
#     #         reac_amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in
#     #                      reac_mol.GetAtoms()}
#     #         for atom in prod_mol.GetAtoms():
#     #             amap_num = atom.GetAtomMapNum()
#     #             numHs_prod = atom.GetTotalNumHs()
#     #             numHs_reac = reac_mol.GetAtomWithIdx(reac_amap[amap_num]).GetTotalNumHs()
#     #             if numHs_prod != numHs_reac:
#     #                 return
#     #         newsmiles.write(rsmiles + "\n")
#
#
#         #                     #对端氢原子是否改变
#         #                     if 2 in  bond_edit.values() and count ==0:
#         #                         count += 1
#         #                         numHs_prod = p_atom.GetTotalNumHs()
#         #                         numHs_reac = r_atom.GetTotalNumHs()
#         #                         if numHs_prod == numHs_reac :
#         #                             same.write(rsmiles+"\n")
#         #                         else:
#         #                             differ.write(rsmiles+"\n")
#         # #对端原子发生化学变化，但分子总体氢原子总数没变
#         # if 2 in bond_edit.values():
#         #     r,p =rsmiles.split('>>')
#         #     reac_mol = Chem.MolFromSmiles(r)
#         #     prod_mol = Chem.MolFromSmiles(p)
#         #     reac_amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in
#         #                  reac_mol.GetAtoms()}
#         #     for atom in prod_mol.GetAtoms():
#         #         amap_num = atom.GetAtomMapNum()
#         #         numHs_prod = atom.GetTotalNumHs()
#         #         numHs_reac = reac_mol.GetAtomWithIdx(reac_amap[amap_num]).GetTotalNumHs()
#         #         if numHs_prod == numHs_reac:
#         #             newsmiles.write(rsmiles + "\n")
#
#
#         # for atoms in products_final:
#         #     for i in atoms:
#         #         if i not in atom_edits.keys():
#         #             atom_edits[i] = 1
#     return reactants_final, products_final,bond_edits,atom_edits






#化学键类型的改变 1 化学键类型不变另外一端原子变 2 都不变 0 可以有反应中心而没有化学键的变化 比如说新增加两个分子间的连边
#原子连接化学键数量未变为0 数量改变为1
# def get_difference_bond(atom_index_1, molecule_1, atom_index_2, molecule_2):
#     #molecule_1 pro molecule_2 rec
#     neighbourhood_1_bondtype, neighbourhood_2_bondtype = [], []
#     neighbourhood_1_atomtype, neighbourhood_2_atomtype = [], []
#     molmaptoid = {}
#     a=molecule_1.GetAtomWithIdx(atom_index_1).GetAtomMapNum()
#     for i in molecule_1.GetAtomWithIdx(atom_index_1).GetNeighbors():
#         neighbourhood_1_bondtype.append((i.GetAtomMapNum(),
#                                 str(molecule_1.GetBondBetweenAtoms(atom_index_1, i.GetIdx()).GetBondType())))
#         neighbourhood_1_atomtype.append((i.GetAtomMapNum(), i.GetSymbol(), i.GetFormalCharge(),
#                                 i.GetNumRadicalElectrons(), i.GetTotalValence()))
#         molmaptoid[i.GetAtomMapNum()] = i.GetIdx()
#     for j in molecule_2.GetAtomWithIdx(atom_index_2).GetNeighbors():
#         neighbourhood_2_bondtype.append((j.GetAtomMapNum(),
#                                 str(molecule_2.GetBondBetweenAtoms(atom_index_2, j.GetIdx()).GetBondType())))
#         neighbourhood_2_atomtype.append((j.GetAtomMapNum(), j.GetSymbol(), j.GetFormalCharge(),
#                                 j.GetNumRadicalElectrons(), j.GetTotalValence()))
#     # neighbourhood_1_bondtype = sorted(neighbourhood_1_bondtype)
#     # neighbourhood_1_atomtype = sorted(neighbourhood_1_atomtype)
#     # neighbourhood_2_bondtype = sorted(neighbourhood_2_bondtype)
#     # neighbourhood_2_atomtype = sorted(neighbourhood_2_atomtype)
#     id_list = []
#     bond_results = {}
#     atom_results = {}
#     atom_results[atom_index_1] = 1
#     for id, pro in enumerate(neighbourhood_1_bondtype):
#         if pro not in  neighbourhood_2_bondtype:
#             id_list.append(id)
#     if len(id_list) >0:
#         for id in id_list:
#             atom_mapnum , _ = neighbourhood_1_bondtype[id]
#             atomid = molmaptoid[atom_mapnum]
#             #bond_results[molecule_1.GetBondBetweenAtoms(atom_index_1,atomid).GetIdx()] = 1
#             bond_results[(atom_index_1,atomid)] = 1
#     id_list = []
#     for id, pro in enumerate(neighbourhood_1_atomtype):
#         if pro not in neighbourhood_2_atomtype:
#             id_list.append(id)
#     if len(id_list) >0:
#         for id in id_list:
#             atom_mapnum , _,_,_,_= neighbourhood_1_atomtype[id]
#             atomid = molmaptoid[atom_mapnum]
#             if (atom_index_1,atomid) not in bond_results.keys():
#                 bond_results[(atom_index_1,atomid)] = 2
#
#
#                 #files.write(str(neighbourhood_1_atomtype)+"\t"+str(neighbourhood_2_atomtype)+"\n")
#             #bond_results[molecule_1.GetBondBetweenAtoms(atom_index_1,atomid).GetIdx()] = 2
#
#
#     return bond_results, atom_results
if __name__ == '__main__':
    # print(get_reaction_core_atoms('[CH3:1][c:2]1[cH:3][cH:4][c:5]([C:6]([CH3:7])([CH3:8])[CH3:9])[cH:10][c:11]1[N+:12](=[O:13])[O-:14]>>[CH3:1][c:2]1[cH:3][cH:4][c:5]([C:6]([CH3:7])([CH3:8])[CH3:9])[cH:10][c:11]1[NH2:12]'))
    data_dir = '../../datasets/uspto-50k/canonicalized_train.csv'
    pf=pd.read_csv(data_dir)
    smiles = pf["reactants>reagents>production"]
    for smile in tqdm(smiles):
        get_reaction_core_atoms(smile)