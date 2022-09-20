import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

#改了边的计算方式 不再是聚集临边
class interGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, e_in_dim,e_out_dim):
        super(interGATLayer, self).__init__()
        self.embed_node = nn.Linear(in_dim, out_dim, bias=False)
        self.node_sf_attten=nn.Linear(out_dim*2,1)
        self.attn_fc = nn.Linear(2 * out_dim + e_in_dim, 1, bias=False)
        self.to_node_fc = nn.Linear(out_dim+e_in_dim+out_dim, out_dim, bias=False)
        self.edge_nor=nn.BatchNorm1d(num_features=e_in_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.e_in_dim=e_in_dim
        self.aggre_embed_edge=nn.Linear(out_dim*2+2*e_in_dim,e_out_dim,bias=False)#修改了 没有残差
        self.edge_drop=nn.Dropout(p=0.0)
        self.node_drop=nn.Dropout(p=0.0)
        self.attention_drop=nn.Dropout(p=0.3)
        self.fc_self = nn.Linear(in_dim, out_dim, bias=False)
        self.embed_edge = nn.Linear(e_in_dim, e_in_dim, bias=False)
        self.edge_sf_atten=nn.Linear(e_in_dim,1,bias=False)
        self.edge_linear=nn.Linear(e_in_dim*3,e_in_dim)
        self.concentrate_h = nn.Linear(out_dim*2+e_in_dim,out_dim)#修改了 没有残差
        self.reset_parameters()
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.embed_node.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.embed_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_self.weight, gain=gain)
        nn.init.xavier_normal_(self.aggre_embed_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.node_sf_attten.weight, gain=gain)
        nn.init.xavier_normal_(self.to_node_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.edge_linear.weight, gain=gain)
        nn.init.xavier_normal_(self.concentrate_h.weight, gain=gain)

    def self_attention(self,edges):
        z2 = torch.cat([edges.src['h'], edges.dst['h']],
                       dim=1)
        node_self_attentin=self.node_sf_attten(z2)
        edge_embed=self.embed_edge(edges.data['w'])
        edge_self_attention=self.edge_sf_atten(edge_embed)
        edge_self_attention=F.leaky_relu(edge_self_attention, negative_slope=0.1)
        edge_self_attention=F.softmax(edge_self_attention,dim=1)
        edge_embed=edge_self_attention*edge_embed
        return {'nsw':F.leaky_relu(node_self_attentin,negative_slope=0.1),'w':edge_embed}

    def inter_attention(self, edges):

        z2 = torch.cat([edges.src['h'], edges.dst['h'],edges.data['w']],
                       dim=1)
        a = self.attn_fc(z2)
        a=self.attention_drop(a)
        edge_weight=F.leaky_relu(a, negative_slope=0.1)
        edge_weight=F.softmax(edge_weight,dim=1)
        edge_embed=edge_weight*edges.data['w']
        return {'inw': F.leaky_relu(a, negative_slope=0.1),'w':edge_embed}

    def message_func(self, edges):
        return {
            'h': edges.src['h'],
            'nsw': edges.data['nsw'],
            'w':edges.data['w'],
            #'hw':edges.src['hw']
        }
    def message_func2(self,edges):
        return {'sh':edges.src['h'],'dh':edges.dst['h'],'w':edges.data['w'],'inw':edges.data['inw']}

    def reduce_func(self, nodes):
        alpha =F.softmax(nodes.mailbox['nsw'], dim=1)
        #t = torch.cat([nodes.mailbox['h'], nodes.mailbox['w']], dim=-1)
        #t=nodes.mailbox['h']
        t=nodes.mailbox['h']
        #t = self.to_node_fc(t)
        h = torch.sum(alpha * t, dim=1)


        return {'h':h}
    # def node_attr(self,nodes):
    #     #z=torch.cat(nodes.data['h'],dim=1)
    #     hw=self.attr_node(nodes.data['h'])
    #     return {'hw':F.leaky_relu(hw, negative_slope=0.1)}
    def reduce_func2(self,nodes):
        alpha =F.softmax(nodes.mailbox['inw'], dim=1)
        t=torch.cat([nodes.mailbox['sh'], nodes.mailbox['w']], dim=-1)
        h = torch.sum(alpha * t, dim=1)
        # w=torch.sum(alpha* nodes.mailbox['w'],dim=1)
        t = torch.cat([h,nodes.data['oh']] ,dim=-1)
        h = self.concentrate_h(t)
        h=self.node_drop(h)
        return {'h': h}

    def edge_calc(self, edges):
        w=self.edge_nor(edges.data['w'])
        z2 = torch.cat([edges.src['h'], edges.dst['h'],w,edges.data['ow']],
                       dim=1)
        # z2 = torch.cat([edges.src['h'], edges.dst['h'],w],
        #                dim=1)
        w = self.aggre_embed_edge(z2)
        w=self.edge_drop(w)
        return {'w': w}
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = self.embed_node(h)
            g.ndata['oh']=g.ndata['h']#残差
            g.edata['ow']=g.edata['w']#残差
            g.apply_edges(self.self_attention)
            g.update_all(self.message_func, self.reduce_func)
            g.apply_edges(self.inter_attention)
            g.update_all(self.message_func2, self.reduce_func2)

            # g.apply_nodes(self.node_attr)
            # #g.apply_edges(self.edge_calc)
            # g.update_all(self.mess2)
            g.apply_edges(self.edge_calc)
            # h_readout = dgl.mean_nodes(g, 'h')
            # gh = dgl.broadcast_nodes(g, h_readout)
            # return torch.cat((g.ndata['h'], gh), dim=1), g.edata['w']
            return g.ndata['h'], g.edata['w']



class MultiHeadGATLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 e_in_dim,
                 e_out_dim,
                 num_heads,
                 use_gpu=True):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        self.use_gpu = use_gpu
        for i in range(num_heads):
            self.heads.append(interGATLayer(in_dim, out_dim, e_in_dim, e_out_dim))

    def forward(self, g, h, merge):

        if self.use_gpu:
            g=g.to(torch.device('cuda'))
        outs = list(map(lambda x: x(g, h), self.heads))
        outs = list(map(list, zip(*outs)))
        head_outs = outs[0]
        edge_outs = outs[1]
        if merge == 'flatten':
            head_outs = torch.cat(head_outs, dim=1)
            edge_outs = torch.cat(edge_outs, dim=1)
        elif merge == 'mean':
            head_outs = torch.mean(torch.stack(head_outs), dim=0)
            edge_outs = torch.mean(torch.stack(edge_outs), dim=0)
        g.edata['w'] = edge_outs
        return head_outs, edge_outs



class GATNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, heads, use_gpu=True):
        super(GATNet, self).__init__()
        self.num_layers = num_layers
        self.gat = nn.ModuleList()
        self.h_em= nn.Linear(128,128)
        self.g_em=nn.Linear(128,128)
        self.gat.append(
            MultiHeadGATLayer(in_dim, hidden_dim, 13,128, heads, use_gpu))
        for l in range(1, num_layers):
            self.gat.append(
                MultiHeadGATLayer(
                    hidden_dim * heads,
                    hidden_dim,
                    128*heads,
                    128,
                    heads,
                    use_gpu,
                ))

        # self.linear_e = nn.Sequential(
        #     nn.Linear(128 * 2, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(128, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 1),
        # )
        self.linear_e = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
        )
        self.linear_atom = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )
        # self.linear_h = nn.Sequential(
        #     nn.Linear(128, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(32, 3),
        # )
        # self.linear_groups=nn.Sequential(
        #     nn.Linear(9, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(128, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        # )

    def forward(self, g:dgl.DGLGraph, h):
        if torch.cuda.is_available():
            g=g.to(torch.device('cuda'))
        for l in range(self.num_layers-1):
            h, e = self.gat[l](g, h, merge='flatten')
            h = F.elu(h)
            g.edata['w']=e
        h, e = self.gat[-1](g, h, merge='mean')

        # Graph level prediction
        g.ndata['h'] = h
        #atom_pred = self.linear_atom(h)
        #
        h_readout = dgl.mean_nodes(g, 'h')
        #h_pred = self.linear_h(h_readout)
        # Edge prediction
        eh = dgl.broadcast_edges(g, h_readout)
        e_fused = torch.cat((eh, e), dim=1)
        #e_pred = self.linear_e(e_fused)

        return g,h, e_fused

