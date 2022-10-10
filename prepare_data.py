import argparse
import re
import os
import pickle
import numpy as np
from ChemReload import *
from tqdm import tqdm
from rdkit import Chem
from collections import Counter

#通过移除原子映射改smart为smiles
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
    args.output_suffix = '-aug-typed'
else:
    args.output_suffix = '-aug-untyped'
print(args)


savedir = 'OpenNMT-py/data/{}{}'.format(args.dataset, args.output_suffix)
if not os.path.exists(savedir):
    os.mkdir(savedir)


src = {
    'train': 'src-train-aug.txt',
    'test': 'src-test.txt',
    'valid': 'src-valid.txt',
}
tgt = {
    'train': 'tgt-train-aug.txt',
    'test': 'tgt-test.txt',
    'valid': 'tgt-valid.txt',
}
# sys = {
#     'train': 'sysmap2id-train-aug.txt',
#     'test': 'sysmap2id-test.txt',
#     'valid': 'sysmap2id-valid.txt',
# }
# tm = {
#     'train': 'tgtmap2id-train-aug.txt',
#     'test': 'tgtmap2id-test.txt',
#     'valid': 'tgtmap2id-valid.txt',
# }

# Get the mapping numbers in a SMILES.
# def get_idx(smarts):
#     item = re.findall('[a-z]|[A-Z]', smarts)
#     while 'H' in item:
#         item.remove('H')
#     #item = list(map(int, item))
#     return item

def smi_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|\|)"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)
add_token =    {'[OH2]':'O',
     '[ClH]':'Cl',
     '[cH3]':'c',
     '[NH3]':'N',
     '[IH]':'I',
     '[cH2]':'c',
     '[cH4]':'c', '[BrH]':'Br', '[sH2]':'s','[Cl]':'Cl',
     '[SH2]':'S', '[oH2]':'o', '[nH2]':'n', '[Br]':'Br', '[CH4]':'C', '[PH3]':'P', '[BH]':'B'}

# Convert smarts to smiles by remove mapping numbers
def smarts2smiles(smarts, canonical=True):
    t = re.sub(':\d*', '', smarts)
    mol = Chem.MolFromSmiles(t, sanitize=False)
    return Chem.MolToSmiles(mol, canonical=canonical)
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
#
#     loc = [smiles.find(x) for x in maps]
#
#     loc.sort()
#     begin = loc[0]
#     end = loc[-1]
#
#     for i in range(begin,-1,-1):
#         if smiles[i] == '[':
#             while smiles[i-1] =='(' or smiles[i-1] == '=' or smiles[i-1] == '#'  :
#
#                 if smiles[i-1] == '=' or smiles[i-1] == '#':
#                     i -= 2
#                 else:
#                     i -= 1
#                 if i>0 and smiles[i] == ']':
#                     while i > 0:
#                         if smiles[i-1] == '[':
#                             i -= 1
#                             break
#                         i -= 1
#
#             new_smiles = smiles[:i] + '|' +smiles[i:]
#             break
#
#     for i in range(end+1,len(new_smiles)):
#         if new_smiles[i] == ']':
#             while i<len(new_smiles)-1 and (new_smiles[i+1].isdigit() or new_smiles[i+1] ==')' or new_smiles[i+1] == '=' or new_smiles[i+1] == '#'):
#                 if new_smiles[i + 1] == '=' or new_smiles[i + 1] == '#':
#                     i += 2
#                 else:
#                     i += 1
#                 if new_smiles[i] == '[':
#                     while i<len(new_smiles)-1:
#                         if new_smiles[i+1] == ']':
#                             i += 1
#                             break
#                         i += 1
#             new_smiles = new_smiles[:i + 1] + '|' + new_smiles[i + 1:]
#             break
#     new_smiles = re.sub(':\d*','',new_smiles)
#     abs = re.findall('\|\[[^\]]]',new_smiles)
#     assert len(abs) <= 1
#     for i in abs:
#         a = re.split('\[|\]',i)
#         item = a[0]+a[1]
#         new_smiles = re.sub('\|\[[^\]]]',item,new_smiles)
#     abs = re.findall('\[[^\]]]\||\[[^\]]]\d\|',new_smiles)
#
#     assert len(abs) <= 1
#     for i in abs:
#         a = re.split('\[|\]', i)
#         item = a[1] + a[2]
#         new_smiles = re.sub('\[[^\]]]\||\[[^\]]]\d\|', item, new_smiles)
#     abs = re.findall('\[[^\]]]',new_smiles)
#
#     for i in abs:
#         a = re.split('\[|\]', i)
#         item = a[1]
#         new_smiles = re.sub('\[[^\]]]', item, new_smiles,count=1)
#
#
#     for key in add_token.keys():
#         if key in new_smiles:
#             new_smiles = new_smiles.replace(key,add_token[key])
#
#     return new_smiles

def read_mapnums(ids_list):
    ids = ids_list.split('\t')
    new_ids = []
    for id in ids:
        if id == '\n':
            continue
        id = id.replace(']','')
        id = id.replace('[','')
        id_list = [int(x) for x in id.split(',')]
        new_ids.append(id_list)
    return new_ids


tokens = Counter()
reaction_atoms = {}
reaction_atoms_list = []
for data_set in ['valid', 'train', 'test']:
    with open(os.path.join('opennmt_data', src[data_set])) as f:
        srcs = f.readlines()
    with open(os.path.join('opennmt_data', tgt[data_set])) as f:
        tgts = f.readlines()
    # with open(os.path.join('opennmt_data', sys[data_set])) as f:
    #     sys_ = f.readlines()
    # with open(os.path.join('opennmt_data', tm[data_set])) as f:
    #     tm_ = f.readlines()

    src_lines = []
    tgt_lines = []
    sys_lines = []
    reaction_atoms_lists = []
    unknown = set()
    for s, t in tqdm(list(zip(srcs, tgts))):
        tgt_items = t.strip().split()
        src_items = s.strip().split()
        src_items[2] = smi_tokenizer(smarts2smiles(src_items[2]))
        tokens.update(src_items[2].split(' '))
        # sid_list = read_mapnums(maps)
        # tid_list = read_mapnums(tmaps)
        #count = 4
        for idx in range(4, len(src_items)):
            if src_items[idx] == '.':
            #     count += 1
                continue
            smiles = smarts2smiles(src_items[idx], canonical=False)
            # if idx-count>=len(sid_list):
            #     a=1
            # smiles = smiles_edit(smiles,sid_list[idx-count],canonical=False)
            src_items[idx] = smi_tokenizer(smiles)
            tokens.update(src_items[idx].split(' '))
        #count = 0
        for idx in range(len(tgt_items)):
            if tgt_items[idx] == '.':
                #count += 1
                continue
            smiles = smarts2smiles(tgt_items[idx])
            # smiles = smiles_edit(smiles,tid_list[idx-count])
            tgt_items[idx] = smi_tokenizer(smiles)
            tokens.update(tgt_items[idx].split(' '))

        if not args.typed:
            src_items[1] = '[RXN_0]'

        src_line = ' '.join(src_items[1:])
        tgt_line = ' '.join(tgt_items)
        src_lines.append(src_line + '\n')
        tgt_lines.append(tgt_line + '\n')

    src_file = os.path.join(savedir, src[data_set])
    print('src_file:', src_file)
    with open(src_file, 'w') as f:
        f.writelines(src_lines)

    tgt_file = os.path.join(savedir, tgt[data_set])
    print('tgt_file:', tgt_file)
    with open(tgt_file, 'w') as f:
        f.writelines(tgt_lines)
