import argparse
from sklearn.metrics import classification_report
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from data_newlabel_undirect import *
from model.simpleinterGAT2_newlabel import *
from util.misc import CSVLogger
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                    type=int,
                    default=1,
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
                    default=True,
                    help='use gpu or cpu')
parser.add_argument('--load',
                    action='store_true',
                    default=False,
                    help='load model checkpoint.')

args = parser.parse_args()


def collate(data):
    return map(list, zip(*data))


def test(GAT_model, test_dataloader, data_split='test', save_pred=False,bestacc=0,files=None):
    GAT_model.eval()
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
    for i, data in enumerate(tqdm(test_dataloader)):
        rxn_class, x_pattern_feat, x_atom, x_adj, x_graph, _, x_groups,atom_labels,bond_labels = data

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

        atom_pred, e_pred = GAT_model(g_dgl, x_atom)
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
            torch.save(GAT_model.state_dict(),
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


if __name__ == '__main__':

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
    args.exp_name+="_simplegat(0.3)_newlabelundirect"

    print(args)
    test_id = '{}'.format(args.logdir)
    filename = 'logs/' + test_id+args.exp_name + '.csv'
    #sys.stdout = Logger('./out/'+args.exp_name+".log", sys.stdout)

    sk_filename = 'sk_logs/' + test_id+args.exp_name + '.txt'
    file = open(sk_filename,'a')
    GAT_model = GATNet(
        in_dim=args.in_dim,
        num_layers=args.gat_layers,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        use_gpu=(args.use_cpu == False),
    )

    if args.use_cpu:
        device = 'cpu'
    else:
        GAT_model = GAT_model.cuda()

        device = 'cuda:0'

    if args.load:
        GAT_model.load_state_dict(
            torch.load('checkpoints_3/{}best_checkpoint.pt'.format(args.exp_name),
                       map_location=torch.device(device)), )
        args.lr *= 0.2
        milestones = []
    else:
        milestones = [20, 40, 60, 80]

    optimizer = torch.optim.Adam([{
        'params': GAT_model.parameters()
    }],
                                 lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

    if args.test_only:
        test_data = RetroCenterDatasets(root=data_root, data_split='test')
        test_dataloader = DataLoader(test_data,
                                     batch_size=1* batch_size,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=collate)
        test(GAT_model, test_dataloader, data_split='test', save_pred=True,files=file)
        exit(0)

    valid_data = RetroCenterDatasets(root=data_root, data_split='valid')
    valid_dataloader = DataLoader(valid_data,
                                  batch_size=1 * batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  collate_fn=collate)
    if args.valid_only:
        test(GAT_model, valid_dataloader)
        exit(0)

    train_data = RetroCenterDatasets(root=data_root, data_split='train')
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=collate)
    if args.test_on_train:
        test_train_dataloader = DataLoader(train_data,
                                           batch_size=1 * batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           collate_fn=collate)
        test(GAT_model, test_train_dataloader, data_split='train', save_pred=True)
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
        GAT_model.train()
        pre_bond_label= []
        true_bond_label= []
        pre_atom_label = []
        true_atom_label = []
        progress_bar = tqdm(train_dataloader,ncols=80)
        for i, data in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))
            rxn_class, x_pattern_feat, x_atom, x_adj, x_graph, _, x_groups, atom_labels, bond_labels = data
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

            x_atom = torch.cat(x_atom, dim=0)
            atom_labels = torch.cat(atom_labels, dim=0)
            bond_labels = torch.cat(bond_labels_list, dim=0)
            true_bond_label.extend(bond_labels.numpy().tolist())
            true_atom_label.extend(atom_labels.numpy().tolist())
            if not args.use_cpu:
                x_atom = x_atom.cuda()

                atom_labels = atom_labels.cuda()
                bond_labels  = bond_labels.cuda()


            g_dgl = dgl.batch(x_graph)

            a = g_dgl.adj(scipy_fmt='csr').toarray()
            x,y = np.where(a==1)
            location = list(zip(x,y))
            GAT_model.zero_grad()
            # batch graph

            atom_pred, e_pred = GAT_model(g_dgl, x_atom)
            e_pred = e_pred.squeeze()
            loss_h = nn.CrossEntropyLoss(reduction='sum')(atom_pred,
                                                          atom_labels)
            loss_ce = nn.CrossEntropyLoss(reduction='sum')(e_pred,
                                                            bond_labels)




            start = end = 0
            e_label , _ = torch.max(e_pred,dim=1)
            a_label , _ = torch.max(atom_pred,dim=1)
            e_pred = torch.argmax(e_pred, dim=1)
            a_pred = torch.argmax(atom_pred,dim=1)

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
            loss = loss_ce + loss_h + edge_global_losses+atom_global_losses
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


            valid_acc,atomacc= test(GAT_model, valid_dataloader,bestacc=local_acc,files=file)
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
    torch.save(GAT_model.state_dict(),
               'checkpoints_3/{}_checkpoint.pt'.format(args.exp_name))
