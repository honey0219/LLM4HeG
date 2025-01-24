import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn


class FALayer(nn.Module):
    def __init__(self, g, in_dim, dropout):
        super(FALayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)
        self.yes_weight = None
        self.no_weight = None

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)

        g = torch.tanh(self.gate(h2)).squeeze()

        yes_no = torch.tanh((edges.data['yes_no'] * self.yes_weight) + ((1 - edges.data['yes_no']) * self.no_weight))

        e = ((g + yes_no)/2) * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)

        return {'e': e, 'm': g}

    def forward(self, h, yes_weight, no_weight):
        self.g.ndata['h'] = h
        self.yes_weight = yes_weight
        self.no_weight = no_weight
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']


class FAGCN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):
        super(FAGCN, self).__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(self.g, hidden_dim, dropout))

        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()
        self.no_weight = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))
        self.yes_weight = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h, self.yes_weight, self.no_weight)
            h = self.eps * raw + h
        h = self.t2(h)
        return F.log_softmax(h, 1)

