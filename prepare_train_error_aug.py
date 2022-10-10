import argparse
import re
import os
import pickle
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from collections import Counter
from ChemReload import *
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    default='USPTO50K',
                    help='dataset: USPTO50K')
parser.add_argument('--typed',
                    action='store_true',
                    default=False,
                    help='if given reaction types')
args = parser.parse_args()

assert args.dataset in ['USPTO50K', 'USPTO-full']
if args.typed:
    args.typed = 'typed'
    args.output_suffix = '-aug-typed'
else:
    args.typed = 'untyped'
    args.output_suffix = '-aug-untyped'
print(args)



# Get the mapping numbers in a SMILES.
def get_idx(smarts):
    item = re.findall('(?<=:)\d+', smarts)
    item = list(map(int, item))
    return item

def smi_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|\|)"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

# Convert smarts to smiles by remove mapping numbers
def smarts2smiles(smarts, canonical=True):
    t = re.sub(':\d*', '', smarts)
    mol = Chem.MolFromSmiles(t, sanitize=False)
    return Chem.MolToSmiles(mol, canonical=canonical)

def get_smarts_pieces(mol, src_adj, target_adj, add_bond=False):
    m, n = src_adj.shape
    emol = Chem.EditableMol(mol)
    for j in range(m):
        for k in range(j + 1, n):
            if target_adj[j][k] == src_adj[j][k]:
                continue
            if 0 == target_adj[j][k]:
                emol.RemoveBond(j, k)
            elif add_bond:
                emol.AddBond(j, k, Chem.rdchem.BondType.SINGLE)
    synthon_smiles = Chem.MolToSmiles(emol.GetMol(), isomericSmiles=True)
    return synthon_smiles
def getid2map(mol,atom_label):
    aids = np.where(atom_label > 0)[0]
    mapnums = []
    for i in aids:
        atom = GetAtomByid(mol, i)
        neighbors = GetNerghbors(atom)
        local_mapnums = [GetAtomMapNum(x) for x in neighbors]
        message = GetAtomMapNum(atom)
        mapnums.append(message)
        mapnums.extend(local_mapnums)
    return mapnums
def getmap2id(smiles,mapnums):
    #跳过对成键信息的验证
    mol = Chem.MolFromSmiles(smiles,sanitize=False)
    mapnum2id = []
    for atom in GetMolAtoms(mol):
        map = GetAtomMapNum(atom)
        if map in mapnums:
            mapnum2id.append(GetAtomId(atom))
    if len(mapnum2id) == 0:
        mapnum2id.append(-1)
    return mapnum2id
add_token =    {'[OH2]':'O',
     '[ClH]':'Cl',
     '[cH3]':'c',
     '[NH3]':'N',
     '[IH]':'I',
     '[cH2]':'c',
     '[cH4]':'c', '[BrH]':'Br', '[sH2]':'s','[Cl]':'Cl',
     '[SH2]':'S', '[oH2]':'o', '[nH2]':'n', '[Br]':'Br', '[CH4]':'C', '[PH3]':'P', '[BH]':'B'}
# def smiles_edit(smiles, ids,canonical=True):
#     mol = Chem.MolFromSmiles(smiles,sanitize=False)
#     maps = []
#     for id in ids :
#         if id == -1:
#             return smiles
#         atom = GetAtomByid(mol,id)
#         SetAtomMapnum(atom,id+1)
#
#         maps.append(':'+str(id+1)+']')
#     smiles = Chem.MolToSmiles(mol,canonical=canonical)
#     loc = [smiles.find(x) for x in maps]
#
#     loc.sort()
#     begin = loc[0]
#     end = loc[-1]
#     for i in range(begin, -1, -1):
#         if smiles[i] == '[':
#             while smiles[i - 1] == '(' or smiles[i - 1] == '=' or smiles[i - 1] == '#':
#
#                 if smiles[i - 1] == '=' or smiles[i - 1] == '#':
#                     i -= 2
#                 else:
#                     i -= 1
#                 if i > 0 and smiles[i] == ']':
#                     while i > 0:
#                         if smiles[i - 1] == '[':
#                             i -= 1
#                             break
#                         i -= 1
#
#             new_smiles = smiles[:i] + '|' + smiles[i:]
#             break
#
#     for i in range(end + 1, len(new_smiles)):
#         if new_smiles[i] == ']':
#             while i < len(new_smiles) - 1 and (
#                     new_smiles[i + 1].isdigit() or new_smiles[i + 1] == ')' or new_smiles[i + 1] == '=' or new_smiles[
#                 i + 1] == '#'):
#                 if new_smiles[i + 1] == '=' or new_smiles[i + 1] == '#':
#                     i += 2
#                 else:
#                     i += 1
#                 if new_smiles[i] == '[':
#                     while i < len(new_smiles) - 1:
#                         if new_smiles[i + 1] == ']':
#                             i += 1
#                             break
#                         i += 1
#             new_smiles = new_smiles[:i + 1] + '|' + new_smiles[i + 1:]
#             break
#     new_smiles = re.sub(':\d*', '', new_smiles)
#     abs = re.findall('\|\[[^\]]]', new_smiles)
#     assert len(abs) <= 1
#     for i in abs:
#         a = re.split('\[|\]', i)
#         item = a[0] + a[1]
#         new_smiles = re.sub('\|\[[^\]]]', item, new_smiles)
#     abs = re.findall('\[[^\]]]\||\[[^\]]]\d\|', new_smiles)
#
#     assert len(abs) <= 1
#     for i in abs:
#         a = re.split('\[|\]', i)
#         item = a[1] + a[2]
#         new_smiles = re.sub('\[[^\]]]\||\[[^\]]]\d\|', item, new_smiles)
#     abs = re.findall('\[[^\]]]', new_smiles)
#
#     for i in abs:
#         a = re.split('\[|\]', i)
#         item = a[1]
#         new_smiles = re.sub('\[[^\]]]', item, new_smiles, count=1)
#
#     for key in add_token.keys():
#         if key in new_smiles:
#             new_smiles = new_smiles.replace(key, add_token[key])
#     return new_smiles
#

atom_results = 'log/train_result_mol_atom_{}_{}.txt'.format(args.dataset, args.typed)
pred_results = 'log/train_result_mol_{}_{}.txt'.format(args.dataset, args.typed)
with open(pred_results) as f:
    pred_results = f.readlines()

bond_pred_results = 'log/train_disconnection_{}_{}.txt'.format(args.dataset, args.typed)
with open(bond_pred_results) as f:
    bond_pred_results = f.readlines()
with open(atom_results) as f:
    atom_results = f.readlines()

dataset_dir = 'data/{}/train'.format(args.dataset)
reaction_data_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pkl')]
reaction_data_files.sort()

product_adjs = []
product_mols = []
product_smiles = []
atom_labels = []
for data_file in tqdm(reaction_data_files):
    with open(os.path.join(dataset_dir, data_file), 'rb') as f:
        reaction_data = pickle.load(f)
    product_adjs.append(reaction_data['product_adj'])
    product_mols.append(reaction_data['product_mol'])
    product_smiles.append(Chem.MolToSmiles(reaction_data['product_mol'], canonical=False))
    atom_labels.append(reaction_data['atom_label'])


assert len(product_smiles) == len(bond_pred_results)



cnt = 0
guided_pred_results = []
bond_disconnection = []
bond_disconnection_gt = []
pred_bond_results = []
for i in range(len(bond_pred_results)):
    bond_pred_items = bond_pred_results[i].strip().split()
    bond_change_num = int(bond_pred_items[1]) * 2
    bond_change_num_gt = int(bond_pred_items[0]) * 2
    pred_resulst = pred_results[3*i].strip().split(' ')
    gt_adj_list = pred_results[3 * i + 1].strip().split(' ')
    gt_adj_list = np.array([int(k) for k in gt_adj_list])
    gt_adj_index = np.argsort(-gt_adj_list)
    gt_adj_index = gt_adj_index[:bond_change_num_gt]

    pred_adj_list = pred_results[3 * i + 2].strip().split(' ')
    pred_bond_results.append(pred_adj_list)
    pred_adj_list = np.array([float(k) for k in pred_adj_list])
    pred_adj_index = np.argsort(-pred_adj_list)
    pred_adj_index = pred_adj_index[:bond_change_num]

    bond_disconnection.append(pred_adj_index)
    bond_disconnection_gt.append(gt_adj_index)
    res = pred_resulst[1]
    if res == 'True':
        res = 1
    else :
        res = 0
    guided_pred_results.append(int(res))
    cnt += res

guided_atom_result = []
pred_atom_label_lists = []
for i in range(len(bond_pred_results)):
    true_atom_label_list = atom_results[3 * i + 1].strip().split(' ')
    pred_atom_label_list = atom_results[3 * i + 2].strip().split(' ')
    pred_atom_label_lists.append(np.array(pred_atom_label_list,dtype=np.int))
    res = true_atom_label_list == pred_atom_label_list
    guided_atom_result.append(int(res))
print('guided bond_disconnection prediction cnt and acc:', cnt, cnt / len(bond_pred_results))
print('bond_disconnection len:', len(bond_disconnection))

with open('opennmt_data/src-train.txt') as f:
    srcs = f.readlines()
with open('opennmt_data/tgt-train.txt') as f:
    tgts = f.readlines()


# Generate synthons from bond disconnection prediction
sources = []
targets = []
for i, prod_mol in enumerate(product_mols):
    if guided_pred_results[i] == 1:
        continue
    edits = []
    ids = bond_disconnection[i]
    x_adj = np.array(product_adjs[i])
    # find 1 index
    idxes = np.argwhere(x_adj > 0)
    pred_adj = product_adjs[i].copy()
    for k in ids:
        idx = idxes[k]
        edits.append((int(idx[0]), int(idx[1]), BOND_LABEL_TO_FLOAT[float(pred_bond_results[i][k])]))
    if len(edits) == 0:
        atom_label = pred_atom_label_lists[i]
        atom_label = np.array(atom_label, dtype=int)
        idx = np.where(atom_label > 0)[0]
        for id in idx:
            edits.append((int(id), 0, None))
    new_mol = Chem.RWMol(prod_mol)
    aidtmap2 = {atom.GetIdx(): atom.GetAtomMapNum() for atom in prod_mol.GetAtoms()}
    aidtmap = {atom.GetIdx(): atom.GetAtomMapNum() for atom in new_mol.GetAtoms()}
    assert aidtmap == aidtmap2
        # for k in bond_disconnection[i]:
        #     idx = idxes[k]
        #     assert pred_adj[idx[0], idx[1]] == 1
        #     pred_adj[idx[0], idx[1]] = 0

    pred_synthon = mol_edit(new_mol, edits)
    pred_synthon = Chem.MolToSmiles(pred_synthon, isomericSmiles=True)
        # map_num = getid2map(product_mols[i],pred_atom_label_lists[i])
        # map_nums.append(map_num)

    reactants = tgts[i].split('.')
    reactants = [r.strip() for r in reactants]

    syn_idx_list = [get_idx(s) for s in pred_synthon.split('.')]
    # if guided_atom_result[i] == 1:
    #
    #     mapnum_list = [getid2map(product_mols[i],atom_labels[i])]
    # else:
    #     mapnum_list = [getid2map(product_mols[i],pred_atom_label_lists[i])]

    #id_list_s = [getmap2id(s,mapnum_list) for s in pred_synthon.split('.')]

    react_idx_list = [get_idx(r) for r in reactants]
    react_max_common_synthon_index = []
    for react_idx in react_idx_list:
        react_common_idx_cnt = []
        for syn_idx in syn_idx_list:
            common_idx = list(set(syn_idx) & set(react_idx))
            react_common_idx_cnt.append(len(common_idx))
        max_cnt = max(react_common_idx_cnt)
        react_max_common_index = react_common_idx_cnt.index(max_cnt)
        react_max_common_synthon_index.append(react_max_common_index)
    react_synthon_index = np.argsort(react_max_common_synthon_index).tolist()
    reactants = [reactants[k] for k in react_synthon_index]
    #id_list_r = [getmap2id(s,mapnum_list) for s in reactants]

    # remove mapping number
    syns = pred_synthon.split('.')
    syns = [smarts2smiles(s, canonical=False) for s in syns]
    #syns = [smiles_edit(s,id,canonical=False) for s,id in zip(syns,id_list_s)]
    syns = [smi_tokenizer(s) for s in syns]
    src_items = srcs[i].strip().split(' ')
    src_items[2] = smi_tokenizer(smarts2smiles(src_items[2]))
    if args.typed == 'untyped':
        src_items[1] = '[RXN_0]'
    src_line = ' '.join(src_items[1:4]) + ' ' + ' . '.join(syns) + '\n'
    #reactants = [smiles_edit(r,id) for r,id in zip(reactants,id_list_r)]
    reactants = [smi_tokenizer(smarts2smiles(r)) for r in reactants]
    tgt_line = ' . '.join(reactants) + '\n'

    sources.append(src_line)
    targets.append(tgt_line)

print('augmentation data size:', len(sources))


savedir = 'OpenNMT-py/data/{}{}'.format(args.dataset, args.output_suffix)

with open(os.path.join(savedir, 'src-train-aug.txt')) as f:
    srcs = f.readlines()
with open(os.path.join(savedir, 'tgt-train-aug.txt')) as f:
    tgts = f.readlines()

src_train_aug_err = os.path.join(savedir, 'src-train-aug-err.txt')
print('save src_train_aug_err:', src_train_aug_err)
with open(src_train_aug_err, 'w') as f:
    f.writelines(srcs + sources)


tgt_train_aug_err = os.path.join(savedir, 'tgt-train-aug-err.txt')
print('save tgt_train_aug_err:', tgt_train_aug_err)
with open(tgt_train_aug_err, 'w') as f:
    f.writelines(tgts + targets)

