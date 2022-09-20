import argparse
import random
from model.gin import TwoGIN

import numpy as np
from sklearn.metrics import classification_report
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from data_newlabel_undirect import *
from model.interGAT import *
from util.misc import CSVLogger
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                    type=int,
                    default=2,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=80,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--in_dim',
                    type=int,
                    default=47 +657+7+2+9+8,
                    help='dim of atom feature')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--seed',
                    type=int,
                    default=123,
                    help='random seed (default: 123)')
parser.add_argument('--logdir', type=str, default='logs', help='logdir name')
parser.add_argument('--dataset', type=str, default='USPTO50K', help='dataset')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim')

parser.add_argument('--heads', type=int, default=4, help='number of heads')
parser.add_argument('--gat_layers',
                    type=int,
                    default=3,
                    help='number of gat layers')
parser.add_argument('--valid_only',
                    action='store_true',
                    default=False,
                    help='valid_only')
parser.add_argument('--test_only',
                    action='store_true',
                    default=False,
                    help='test_only')
parser.add_argument('--test_on_train',
                    action='store_true',
                    default=False,
                    help='run testing on training data')
parser.add_argument('--typed',
                    action='store_true',
                    default=False,
                    help='if given reaction types')
parser.add_argument('--use_cpu',
                    action='store_true',
                    default=False,
                    help='use gpu or cpu')
parser.add_argument('--load',
                    action='store_true',
                    default=False,
                    help='load model checkpoint.')


parser.add_argument('-h_dim', type=int, default=16, help='hidden dim')
parser.add_argument('-drop_n', type=float, default=0.2, help='drop net')
parser.add_argument('-drop_c', type=float, default=0.2, help='drop output')
parser.add_argument(
    '-learn_eps', action="store_true",
    help='learn the epsilon weighting')
parser.add_argument('-l_num', type=int, default=4, help='layer num')
parser.add_argument('-w_d', type=float, default=0.0005, help='weight decay')

args = parser.parse_args()


def collate(data):
    return map(list, zip(*data))


def test(model, test_dataloader,gat_dataloader, data_split='test', save_pred=False,bestacc=0,files=None):
    model.eval()
    correct = 0.
    acorrect =0.
    total = 0.
    epoch_loss = 0.
    true_bond_label = []
    pre_bond_label = []
    true_atom_label = []
    pre_atom_label = []
    # Bond disconnection probability
    pred_true_list = []
    pred_logits_mol_list = []
    # Bond disconnection number gt and prediction
    bond_change_gt_list = []
    bond_change_pred_list = []
    progress_bar = tqdm(test_dataloader, ncols=80)
    train_idx,valid_idx, test_idx = sep_data()
    offset = 0
    if data_split == 'valid':
        offset = len(train_idx)
    elif data_split == 'test':
        offset = len(train_idx) + len(valid_idx)
    for step, (input_nodes, seeds, blocks) in enumerate(progress_bar):

        # for i, data in enumerate(progress_bar):

        selected_idx = seeds
        IDs = []
        for block in blocks:
            IDs.append(block.edata[dgl.EID])
        rxn_class, x_pattern_feat, x_atom, x_adj, x_graph, atom_labels, bond_labels = [], [], [], [], [], [], []
        for i in selected_idx:
            rxn, pattern_feat, atom, adj, graph, _, _, atom_label, bond_label = gat_dataloader[i-offset]
            rxn_class.append(rxn)
            x_pattern_feat.append(pattern_feat)
            x_atom.append(atom)
            x_adj.append(adj)
            x_graph.append(graph)
            atom_labels.append(atom_label)
            bond_labels.append(bond_label)

        #for i, data in enumerate(tqdm(test_dataloader)):
        batch_inputs, batch_labels, batch_edge_weight = load_subtensor(node_features, labels, edge_weight, IDs,
                                                         seeds, input_nodes, device)
        #block means  g
        blocks = [block.int().to(device) for block in blocks]
        x_atom = list(map(lambda x: torch.from_numpy(x).float(), x_atom))
        x_pattern_feat = list(
            map(lambda x: torch.from_numpy(x).float(), x_pattern_feat))
        x_atom = list(
           map(lambda x, y: torch.cat([x, y], dim=1), x_atom,
                x_pattern_feat))
        atomscope = []
        atom_labels = list(
            map(lambda x: torch.from_numpy(x).long(), atom_labels))
        for atom_label in atom_labels:
            atomscope.extend(atom_label.shape)
        x_adj = list(map(lambda x: torch.from_numpy(np.array(x)), x_adj))
        mask = list(map(lambda x: x.contiguous().view(-1, 1).bool(), x_adj))
        bond_labels = list(
            map(lambda x: torch.from_numpy(x).long(), bond_labels))
        # 以true或者false的形式将x的邻接矩阵展开成单列
        bond_labels_list = list(
            map(lambda x, y: torch.masked_select(x.contiguous().view(-1, 1), y), bond_labels,
                mask))  # 根据product的邻接矩阵情况对反应物的邻接矩阵进行处理 在product中连接的情况下 判断rect中的对应位置是否相连 产物连 反应物连不连
        #x_groups = list(map(lambda x: torch.from_numpy(x).float(), x_groups))
        #x_groups = torch.cat(x_groups, dim=1)

        if args.typed:
            rxn_class = list(
                map(lambda x: torch.from_numpy(x).float(), rxn_class))
            x_atom = list(
                map(lambda x, y: torch.cat([x, y], dim=1), x_atom, rxn_class))

        x_atom = torch.cat(x_atom, dim=0)
        atom_labels =torch.cat(atom_labels,dim=0)
        bond_labels =torch.cat(bond_labels_list,dim=0)
        true_bond_label.extend(bond_labels.numpy().tolist())
        true_atom_label.extend(atom_labels.numpy().tolist())
        #disconnection_num = torch.LongTensor(disconnection_num)
        if not args.use_cpu:
            x_atom = x_atom.cuda()
            #x_groups =x_groups.cuda()
            atom_labels = atom_labels.cuda()
            bond_labels  = bond_labels.cuda()
        g_dgl = dgl.batch(x_graph)

        atom_pred, e_pred = gin(blocks, batch_inputs, batch_edge_weight, g_dgl, x_atom)

        e_pred = e_pred.squeeze()
        loss_h = nn.CrossEntropyLoss(reduction='sum')(atom_pred,
                                                      atom_labels)
        loss_ce = nn.CrossEntropyLoss(reduction='sum')(e_pred, bond_labels)
        loss = loss_ce + loss_h
        epoch_loss += loss.item()

        e_pred = torch.argmax(e_pred, dim=1)
        atom_pred = torch.argmax(atom_pred, dim=1)
        bond_change_pred_list.extend(e_pred.cpu().tolist())
        pre_bond_label.extend(e_pred.cpu().numpy().tolist())
        pre_atom_label.extend(atom_pred.cpu().numpy().tolist())
        start = end = 0
        edge_lens = list(map(lambda x: x.shape[0], bond_labels_list))
        cur_batch_size = len(edge_lens)
        bond_labels = bond_labels.long()
        atom_labels =atom_labels.long()
        for j in range(cur_batch_size):
            start = end
            end += edge_lens[j]
            label_mol = bond_labels[start:end]
            pred_proab = e_pred[start:end]
            real_atom_label = atom_labels[start:end]

            if torch.equal(pred_proab, label_mol):
                correct += 1
                pred_true_list.append(True)
                pred_logits_mol_list.append([
                    True,
                    label_mol.tolist(),
                    pred_proab.tolist(),
                ])
            else:
                pred_true_list.append(False)
                pred_logits_mol_list.append([
                    False,
                    label_mol.tolist(),
                    pred_proab.tolist(),
                ])

            total += 1
    start = end = 0
    for j in range(len(atomscope)):
        start = end
        end += atomscope[j]
        if torch.equal(atom_pred[start:end], atom_labels[start:end]):
            acorrect += 1
    pred_lens_true_list = list(
        map(lambda x, y: x == y, bond_change_gt_list, bond_change_pred_list))
    bond_change_pred_list = list(
        map(lambda x, y: [x, y], bond_change_gt_list, bond_change_pred_list))
    if save_pred:
        print('pred_true_list size:', len(pred_true_list))
        np.savetxt('logs/{}_disconnection_{}.txt'.format(data_split, args.exp_name),
                   np.asarray(bond_change_pred_list),
                   fmt='%d')
        np.savetxt('logs/{}_result_{}.txt'.format(data_split, args.exp_name),
                   np.asarray(pred_true_list),
                   fmt='%d')
        with open('logs/{}_result_mol_{}.txt'.format(data_split, args.exp_name),
                  'w') as f:
            for idx, line in enumerate(pred_logits_mol_list):
                f.write('{} {}\n'.format(idx, line[0]))
                f.write(' '.join([str(i) for i in line[1]]) + '\n')
                f.write(' '.join([str(i) for i in line[2]]) + '\n')

    print('Bond disconnection number prediction acc: {:.6f}'.format(
        np.mean(pred_lens_true_list)))

    print('Loss: ', epoch_loss / total)
    acc = correct / total
    atomacc = acorrect/ total

    print("before best_acc is:{} ".format(bestacc))
    if bestacc<acc:
        if bestacc!=0:
            torch.save(model.state_dict(),
                   'checkpoints_3/{}best_checkpoint.pt'.format(args.exp_name))
    print('Bond disconnection acc (without auxiliary task): {:.6f}'.format(acc))
    print('Bond Disconnection TAcc: {:.5f}'.format(atomacc))
    sk_report = classification_report(true_bond_label, pre_bond_label)
    files.write("\n" +data_split + " bond result" + "\n")
    files.write(sk_report)
    files.flush()
    sk_report = classification_report(true_atom_label, pre_atom_label)
    files.write("atom result" + "\n")
    files.write(sk_report)
    files.flush()
    return acc,atomacc
def sep_data():
    train_dir = 'data/%s/train' % (args.dataset)
    valid_dir = 'data/%s/valid' % (args.dataset)
    test_dir = 'data/%s/test' % (args.dataset)
    train_length = len([
        f for f in os.listdir(train_dir) if f.endswith('.pkl')
    ])
    valid_length = len([
        f for f in os.listdir(valid_dir) if f.endswith('.pkl')
    ])
    test_length = len([
        f for f in os.listdir(test_dir) if f.endswith('.pkl')
    ])
    train_idx = [i for i in range(train_length)]
    valid_idx = [i for i in range(train_length,valid_length+train_length)]
    test_idx = [i for i in range(train_length+valid_length,test_length+valid_length+train_length)]
    if not args.test_on_train:
        random.shuffle(train_idx)
    return train_idx,valid_idx, test_idx
def load_motif(train_idx,valid_idx, test_idx):
    if args.use_cpu:
        device = 'cpu'
    else:

        device = 'cuda:0'
    if args.dataset == 'USPTO50K':
        number_of_graphs = 50016
    with open('data/' + args.dataset  +'/motif2', 'rb') as input_file:
        g = pickle.load(input_file)
    num_cliques = int(g.number_of_nodes()) - number_of_graphs
    print(num_cliques)
    labels = g.ndata['labels']
    features = g.ndata['feat']
    in_feats = features.size()[1]

    edge_weight = g.edata['edge_weight'].to(device)

    g = g.to(device)
    node_features = features.to(device)
    labels.to(device)
    train_mask = [True if x in train_idx else False for x in range(int(g.num_nodes()))]
    train_mask = np.array(train_mask)
    valid_mask = [True if x in valid_idx else False for x in range(int(g.num_nodes()))]
    valid_mask = np.array(valid_mask)
    test_mask = [True if x in test_idx else False for x in range(int(g.num_nodes()))]
    test_mask = np.array(test_mask)
    g.ndata['train_mask'] = torch.from_numpy(train_mask).to(device)
    g.ndata['val_mask'] = torch.from_numpy(valid_mask).to(device)
    g.ndata['test_mask'] = torch.from_numpy(test_mask).to(device)

    train_mask = g.ndata['train_mask'].to(device)
    valid_mask = g.ndata['val_mask'].to(device)
    test_mask = g.ndata['test_mask'].to(device)
    train_nid = torch.nonzero(train_mask, as_tuple=True)[0].to(device)
    val_nid = torch.nonzero(valid_mask, as_tuple=True)[0].to(device)
    test_nid = torch.nonzero(test_mask, as_tuple=True)[0].to(device)

    g = g.to(device)

    return num_cliques,in_feats,edge_weight,g,node_features,labels,train_nid,val_nid,test_nid
def load_subtensor(nfeat, labels, edge_weight, EID, seeds, input_nodes, device):
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    batch_edge_weight = []
    for i in EID:
        batch_edge_weight.append(edge_weight[i])
    return batch_inputs, batch_labels, batch_edge_weight
if __name__ == '__main__':
    print(torch.backends.cudnn.version())
    train_idx,valid_idx, test_idx = sep_data()
    num_cliques, in_feats, edge_weight, g, node_features, labels, train_nid, val_nid, test_nid = load_motif(train_idx,valid_idx, test_idx)

    local_acc=0
    batch_size = args.batch_size
    epochs = args.epochs
    data_root = os.path.join('data', args.dataset)
    args.exp_name = args.dataset
    if args.typed:
        args.in_dim += 10
        args.exp_name += '_typed'
    else:
        args.exp_name += '_untyped'
    args.exp_name+="_gatandmotif_"

    print(args)
    test_id = '{}'.format(args.logdir)
    filename = 'logs/' + test_id+args.exp_name + '.csv'
    #sys.stdout = Logger('./out/'+args.exp_name+".log", sys.stdout)

    sk_filename = 'sk_logs/' + test_id+args.exp_name + '.txt'
    file = open(sk_filename,'a')
    # GAT_model = GATNet(
    #     in_dim=args.in_dim,
    #     num_layers=args.gat_layers,
    #     hidden_dim=args.hidden_dim,
    #     heads=args.heads,
    #     use_gpu=(args.use_cpu == False),
    # )
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(123)
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    if args.use_cpu:
        device = 'cpu'
    else:
        # GAT_model = GAT_model.cuda()

        device = 'cuda:0'

    if args.load:
        # GAT_model.load_state_dict(
        #     torch.load('checkpoints_3/{}best_checkpoint.pt'.format(args.exp_name),
        #                map_location=torch.device(device)), )
        # args.lr *= 0.2
        milestones = []
    else:
        milestones = [20, 40, 60, 80]

    # gat_optimizer = torch.optim.Adam([{
    #     'params': GAT_model.parameters()
    # }],
    #                              lr=args.lr)
    # scheduler = MultiStepLR(gat_optimizer, milestones=milestones, gamma=0.2)

    gin = TwoGIN(args.l_num, 2, in_feats, args.h_dim, 2, args.drop_n, args.learn_eps, 'sum',args.in_dim,args.hidden_dim,args.gat_layers,args.heads, use_gpu=(args.use_cpu == False),).to(device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gin.parameters(), lr=args.lr, weight_decay=args.w_d)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    motif_train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    motif_val_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        val_nid,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    if args.test_only:
        motif_test_dataloader = dgl.dataloading.NodeDataLoader(
            g,
            test_nid,
            sampler,
            device=device,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        test_data = RetroCenterDatasets(root=data_root, data_split='test')
        # test_dataloader = DataLoader(test_data,
        #                              batch_size=1* batch_size,
        #                              shuffle=False,
        #                              num_workers=0,
        #                              collate_fn=collate)
        test(gin, motif_test_dataloader,test_data, data_split='test', save_pred=True, files=file)
        exit(0)
    gat_valid_data = RetroCenterDatasets(root=data_root, data_split='valid')
    # valid_dataloader = DataLoader(valid_data,
    #                               batch_size=1 * batch_size,
    #                               shuffle=False,
    #                               num_workers=0,
    #                               collate_fn=collate)
    if args.valid_only:
        #test(GAT_model, valid_dataloader)
        exit(0)

    gat_train_data = RetroCenterDatasets(root=data_root, data_split='train')
    # train_dataloader = DataLoader(train_data,
    #                               batch_size=batch_size,
    #                               shuffle=True,
    #                               num_workers=0,
    #                               collate_fn=collate)
    if args.test_on_train:

        test(gin, motif_train_dataloader,gat_train_data, data_split='train', save_pred=True)
        exit(0)
    csv_logger = CSVLogger(
        args=args,
        fieldnames=['epoch', 'train_acc','valid_acc','valid_atomacc', 'train_loss'],
        filename=filename,
    )
    # Record epoch start time
    for epoch in range(1, 1 + epochs):
        newedgecorrect = 0.
        newatomcorrect = 0.
        total = 0.
        correct = 0.
        acorrect =0.
        epoch_loss = 0.
        epoch_loss_ce = 0.
        epoch_loss_h = 0.
        gin.train()
        pre_bond_label= []
        true_bond_label= []
        pre_atom_label = []
        true_atom_label = []
        progress_bar = tqdm(motif_train_dataloader,ncols=80)
        for step, (input_nodes, seeds, blocks) in enumerate(progress_bar):

        #for i, data in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            selected_idx = seeds
            IDs = []
            for block in blocks:
                IDs.append(block.edata[dgl.EID])
            rxn_class, x_pattern_feat, x_atom, x_adj, x_graph, atom_labels, bond_labels = [],[],[],[],[],[],[]
            for i in selected_idx:
                rxn, pattern_feat, atom, adj, graph, _, _, atom_label, bond_label = gat_train_data[i]
                rxn_class.append(rxn)
                x_pattern_feat.append(pattern_feat)
                x_atom.append(atom)
                x_adj.append(adj)
                x_graph.append(graph)
                atom_labels.append(atom_label)
                bond_labels.append(bond_label)

            batch_inputs, batch_labels, batch_edge_weight = load_subtensor(node_features, labels, edge_weight, IDs,
                                                             seeds, input_nodes, device)
            #block means  g
            blocks = [block.int().to(device) for block in blocks]
            x_atom = list(map(lambda x: torch.from_numpy(x).float(), x_atom))
            x_pattern_feat = list(
                map(lambda x: torch.from_numpy(x).float(), x_pattern_feat))
            x_atom = list(
                map(lambda x, y: torch.cat([x, y], dim=1), x_atom,
                    x_pattern_feat))
            if args.typed:
                rxn_class = list(
                    map(lambda x: torch.from_numpy(x).float(), rxn_class))
                x_atom = list(
                    map(lambda x, y: torch.cat([x, y], dim=1), x_atom,
                        rxn_class))
            atomscope = []
            atom_labels = list(
                map(lambda x: torch.from_numpy(x).long(), atom_labels))
            for atom_label in atom_labels :
                atomscope.extend(atom_label.shape)

            bond_labels = list(
                map(lambda x: torch.from_numpy(x).long(), bond_labels))
            x_adj = list(map(lambda x: torch.from_numpy(np.array(x)), x_adj))
            mask = list(map(lambda x: x.contiguous().view(-1, 1).bool(), x_adj))
            bond_labels_list = list(
                map(lambda x, y: torch.masked_select(x.contiguous().view(-1, 1), y), bond_labels,
                    mask))
            zero = torch.tensor(0)

            x_atom = torch.cat(x_atom, dim=0)
            atom_labels = torch.cat(atom_labels, dim=0)
            bond_labels = torch.cat(bond_labels_list, dim=0)
            true_bond_label.extend(bond_labels.numpy().tolist())
            label_index = torch.arange(bond_labels.shape[0])

            true_atom_label.extend(atom_labels.numpy().tolist())
            if not args.use_cpu:
                x_atom = x_atom.cuda()
                label_index = label_index.cuda()
                atom_labels = atom_labels.cuda()
                bond_labels  = bond_labels.cuda()
                zero = zero.cuda()

            g_dgl = dgl.batch(x_graph)

            a = g_dgl.adj(scipy_fmt='csr').toarray()
            x,y = np.where(a==1)
            location = list(zip(x,y))
            gin.zero_grad()
            # batch graph
            atom_pred,e_pred = gin(blocks, batch_inputs, batch_edge_weight, g_dgl, x_atom)
            #atom_pred, e_pred = GAT_model(g_dgl, x_atom)
            e_pred = e_pred.squeeze()
            loss_h = nn.CrossEntropyLoss(reduction='sum')(atom_pred,
                                                          atom_labels)
            loss_ce = nn.CrossEntropyLoss(reduction='sum')(e_pred,
                                                            bond_labels)




            start = end = 0
            e_p , e_label = torch.max(e_pred,dim=1)
            a_p , a_label = torch.max(atom_pred,dim=1)
            e_pred = torch.argmax(e_pred, dim=1)
            a_pred = torch.argmax(atom_pred,dim=1)
          #尝试定位损失
            # real_edtoatom = []
            # pre_edtoatom = []
            # real_edtoatom.append(zero)
            # pre_edtoatom.append(zero)
            # e_label =torch.where(e_label<1,e_label,label_index)
            # for i  in e_label.data:
            #     if i != 0 :
            #         start , _ = location[i]
            #
            #         pre_edtoatom.append(a_p[start])
            #         real_edtoatom.append(atom_labels[start])
            #
            # pre_edtoatom = torch.stack(pre_edtoatom,dim=0)
            # real_edtoatom = torch.stack(real_edtoatom,dim=0)
            # # la = e_label[1]
            # # if torch.cuda.is_available():
            # #     e_label = e_label.cpu()
            # connect_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')(pre_edtoatom,real_edtoatom.float())

            edge_lens = list(map(lambda x: x.shape[0], bond_labels_list))
            cur_batch_size = len(edge_lens)
            edge_global_losses = 0.
            atom_global_losses = 0.
            #尝试替换label
            one = torch.ones_like(bond_labels)
            new_bond_label = torch.where(bond_labels<1,bond_labels,one)
            one = torch.ones_like(atom_labels)
            new_atom_label = torch.where(atom_labels < 1, atom_labels ,one)
            one = torch.ones_like(e_pred)
            new_bond_pred = torch.where(e_pred < 1, e_pred,one)
            one = torch.ones_like(a_pred)
            new_atom_pred = torch.where(a_pred < 1, a_pred, one)
            loss = loss_ce + loss_h
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_loss_ce += loss_ce.item()
            epoch_loss_h += loss_h.item()
            bond_labels = bond_labels.long()
            atom_labels = atom_labels.long()

            start = end = 0

            for j in range(cur_batch_size):
                start = end
                end += edge_lens[j]
                if torch.equal(e_pred[start:end], bond_labels[start:end]):
                    correct += 1
                if torch.equal(new_bond_pred[start:end],new_bond_label[start:end]):
                    newedgecorrect += 1
            assert end == len(e_pred)

            start = end = 0
            for j in range(len(atomscope)):
                start = end
                end += atomscope[j]
                if torch.equal(a_pred[start:end], atom_labels[start:end]):
                    acorrect += 1
                if torch.equal(new_atom_pred[start:end],new_atom_label[start:end]):
                    newatomcorrect += 1
            assert end == len(atom_pred)
            pre_bond_label.extend(e_pred.cpu().numpy().tolist())
            pre_atom_label.extend(a_pred.cpu().numpy().tolist())
            total += cur_batch_size
            progress_bar.set_postfix(
                loss='%.5f' % (epoch_loss / total),
                acc='%.5f' % (correct / total),
                newedgecorrect = '%.5f' % (newedgecorrect/total),
                newatomcorrect = '%.5f' % (newatomcorrect/total),
                loss_ce='%.5f' % (epoch_loss_ce / total),
                loss_h='%.5f' % (epoch_loss_h / total),
            )
        scheduler.step(epoch)


        train_acc = correct / total

        train_loss = epoch_loss / total
        new_edge_acc = newedgecorrect /total
        print('Train Loss: {:.5f}'.format(train_loss))
        print('Train Bond Disconnection Acc: {:.5f}'.format(train_acc))
        print('Train newedgecorrect: {:.5f}'.format(new_edge_acc))
        sk_report = classification_report(true_bond_label,pre_bond_label)
        file.write("\n"+str(epoch)+"train bond result"+"\n")
        file.write(sk_report)
        file.flush()
        sk_report = classification_report(true_atom_label,pre_atom_label)
        file.write("atom result"+"\n")
        file.write(sk_report)
        file.flush()


        if epoch % 5 == 0:


            valid_acc,atomacc= test(gin, motif_val_dataloader,gat_valid_data,bestacc=local_acc,files=file,data_split='valid')
            if valid_acc>local_acc:
                local_acc=valid_acc
            row = {
                'epoch': str(epoch),
                'train_acc': str(train_acc),
                'valid_acc': str(valid_acc),
                'valid_atomacc': str(atomacc),
                'train_loss': str(train_loss),
            }
            csv_logger.writerow(row)


    csv_logger.close()
    torch.save(gin.state_dict(),
               'checkpoints_3/{}_checkpoint.pt'.format(args.exp_name))
