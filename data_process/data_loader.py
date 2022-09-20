import os.path

import numpy as np
import rdkit
from rdkit.Chem import Draw
import networkx as nx
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
import pickle
from rdkit import Chem
class GenData(object):
    def __init__(self, g_list, node_labels, graph_labels):
        self.g_list = g_list
        self.node_labels = node_labels
        self.graph_labels = graph_labels


class FileLoader(object):
    def __init__(self, args,weight = []):
        self.args = args
        self.weight = weight
    def mol_messages(self,mol  ,bond_label ,atom_label):
        adj =  Chem.rdmolops.GetAdjacencyMatrix(mol)
        edges = []
        edge_labels = []
        atom_symbol = {}
        have_visted = {}
        indices = np.where(adj == 1)
        indices = list(zip(indices[0],indices[1]))
        for indice in indices:
            begin , end = indice
            if (begin,end) not in have_visted and (end,begin) not in have_visted:
                have_visted[(begin,end)] = 1
                # beginsymbol= mol.GetAtomWithIdx(int(begin)).GetSymbol()
                # endsymbol = mol.GetAtomWithIdx(int(end)).GetSymbol()
                edges.append(((begin ,end)))
                # edge_labels.append(label[begin][end])
                atom_symbol[begin] = atom_label[begin]
                atom_symbol[end] = atom_label[end]
        graph_label = np.where(bond_label>0)[0]
        if graph_label.size == 0:
            graph_label=0
        else :
            graph_label = graph_label.size/2
        #可以提出来symbol组合了 但是 里边还有字符串和元组 需要 修改
        # indices = np.where(bond_label > 0)
        # atom_symbol = []
        # if len(indices[0]) == 0:
        #     atom_indices = np.where(atom_label>0)
        #     for atom_indice in  atom_indices[0] :
        #         neighbor_symbol =  mol.GetAtomWithIdx(int(atom_indice)).GetSymbol()
        #
        #         atoms = mol.GetAtomWithIdx(int(atom_indice)).GetNeighbors()
        #         for atom in atoms:
        #             neighbor_symbol += atom.GetSymbol()
        #         if neighbor_symbol not in atom_symbol:
        #             atom_symbol.append(tupple(neighbor_symbol))
        #
        # else:
        #     indices = list(zip(indices[0], indices[1]))
        #     bond_2 = np.where(bond_label==2)
        #     if len(bond_2[0])==0:
        #             for indice in indices :
        #                 begin,end = mol.GetAtomWithIdx(int(indice[0])).GetSymbol(),mol.GetAtomWithIdx(int(indice[1])).GetSymbol()
        #
        #
        #                 # beginmap,endmap = mol.GetAtomWithIdx(int(indice[0])).GetAtomMapNum(),mol.GetAtomWithIdx(int(indice[1])).GetAtomMapNum()
        #                 # Draw.MolToImage(mol).save('123.png')
        #                 if (begin,end) not in atom_symbol and (end,begin) not in atom_symbol:
        #                     atom_symbol.append((begin,end))
        #     else:
        #         atom_indices = np.where(atom_label > 0)
        #         for atom_indice in atom_indices[0]:
        #             for indice in indices:
        #                 if atom_indice in indice:
        #                     begin, end = mol.GetAtomWithIdx(int(indice[0])).GetSymbol(), mol.GetAtomWithIdx(
        #                         int(indice[1])).GetSymbol()
        #                     if (begin, end) not in atom_symbol and (end, begin) not in atom_symbol:
        #                         atom_symbol.append((begin, end))
        #     if len(atom_symbol)>1 :
        #         neighbos = []
        #         atoms = []
        #         new_symbol = []
        #         atom_indices = np.where(atom_label>0)
        #         for i  in atom_indices[0]:
        #             i = int(i)
        #
        #             new_symbol.append(mol.GetAtomWithIdx(i).GetSymbol())
        #             neigh = mol.GetAtomWithIdx(i).GetNeighbors()
        #             for nei in neigh:
        #                 neighbos.append(nei.GetIdx())
        #         flag = True
        #         for atom in atom_indices[0]:
        #             if atom not in neighbos:
        #                 flag = False
        #             else:
        #                 continue
        #         if flag == True:
        #             atom_symbol.clear()
        #             atom_symbol .append(tuple(new_symbol))
        #



        return edges,graph_label,atom_symbol
    def load_data(self):
        dataname = self.args.data
        dir = '../data/%s/' % (dataname)
        all_edges = []
        all_graph_labels= []
        all_edge_weights = []
        all_atom_symbols = []
        for tag in ['train','valid','test']:
            data_dir = os.path.join(dir, tag)
            data_files = [
                f for f in os.listdir(data_dir) if f.endswith('.pkl')
            ]
            data_files.sort()
            # with open(os.path.join(data_dir, 'rxn_data_10009.pkl'), 'rb') as f:
            #     data = pickle.load(f)
            #     mol = data['product_mol']
            #     mol2 = data['reactant_mol']
            #     Draw.MolToImage(mol).save('123.png')
            #     Draw.MolToImage(mol2).save('456.png')
            count = 0
            for file in tqdm(data_files,desc='read pkl'):
                with open(os.path.join(data_dir,file),'rb') as f :
                    data = pickle.load(f)
                mol = data['product_mol']
                bond_labels = data['bond_label']
                atom_labels = data['atom_label']
                count = count+1

                edges,graph_label,atom_symbols = self.mol_messages(mol,bond_labels,atom_labels)
                if len(edges) == 0 :
                    print(Chem.MolToSmiles(mol))
                    continue
                if self.args.edge_weight:
                    weight = []
                    for i in len(range(len(edges))):
                        weight.append(self.weight[i])
                else:
                    weight = []
                    for i in range(len(edges)):
                        weight.append(1)
                all_edge_weights.append(weight)
                all_edges.append(edges)
                all_graph_labels.append(graph_label)
                all_atom_symbols.append(atom_symbols)
                break

                #以下是针对symbol组合的
                # if len(atom_symbols)>1:
                #     c=atom_symbols
                #     mol = data['product_mol']
                #     smilse = Chem.MolToSmiles(mol)
                #     mol2 = data['reactant_mol']
                #     Draw.MolToImage(mol).save('123.png')
                #     Draw.MolToImage(mol2).save('456.png')
                # for i in atom_symbols:
                #     b = self.shift_right(i)
                #     if i not in all_atom_symbols :
                #         if b not in all_atom_symbols:
                #             all_atom_symbols[i] = 1
                #         else:
                #             all_atom_symbols[b] += 1
                #     else:
                #         all_atom_symbols[i] += 1


        g_list = []
        for i in tqdm(range(len(all_graph_labels)),desc='create graphs'):

            g_list.append(self.gen_graph(all_edges[i],all_edge_weights[i]))

        return GenData(g_list,all_atom_symbols,all_graph_labels)

    def gen_graph(self, data, weights):
        edges = data
        weights = weights
        g1 = []
        for i in range(len(edges)):
            l = list(edges[i])
            l.append(weights[i])
            g1.append(tuple(l))

        g = nx.Graph()
        g.add_weighted_edges_from(g1)
        return g
    def shift_right(self,l):
        if type(l) == tuple:
            l = list(l)
            return tuple([l[-1]] + l[:-1])
        elif type(l) == str:
            l = l.split(',')

            res = tuple([l[-1]] + l[:-1])
            return res
        else:
            print('ERROR!')
class GINDataLoader():
    def __init__(self,
                 dataset,
                 batch_size,
                 device,
                 collate_fn=None,
                 seed=0,):

        self.seed = seed
        self.kwargs = {'pin_memory': True} if 'cuda' in device.type else {}
        print(len(dataset))
        labels = [l for _, l in dataset]
        idx = []
        for i in range(len(labels)):
            idx.append(i)

        sampler = SubsetRandomSampler(idx)

        self.train_loader = GraphDataLoader(
            dataset, sampler=sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs)

    def train_valid_loader(self):
        return self.train_loader

