
from dgl.nn import GraphConv
from dgl.nn import EdgeGATConv
from dgl.nn import SAGEConv

import torch.nn as nn


class GCN(nn.Module):
    def __init__(self,args, in_dim, dropout_rate=0.2):
        super(GCN, self).__init__()
        self.in_dim = in_dim
        self.gcn1 = GraphConv(in_dim, in_dim, weight=True)
        self.gcn3= GraphConv(in_dim, in_dim, weight=True)
        self.bn1 = nn.BatchNorm1d(in_dim) 
        self.gcn2 = GraphConv(in_dim, in_dim, weight=True)
        self.activate = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)  # 드롭아웃 추가
        self.attn = nn.Linear(args.dim,1,bias=True)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, GraphConv) and m.weight is not None:
                nn.init.normal_(m.weight.data,mean=0.0,std=0.1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0)
    def forward(self, graph, node_weight,edge_weight):
        h = self.gcn1(graph, node_weight, edge_weight=edge_weight)
        h = self.dropout(self.bn1(h))
        h = self.activate(h)
        
        h = self.gcn2(graph,h, edge_weight=edge_weight)
        h = self.dropout(self.bn1(h))
        h = self.activate(h)

        h = self.gcn3(graph,h, edge_weight=edge_weight)
        h = self.bn1(h)

        return h
    

class GAT(nn.Module):
    def __init__(self, in_dim, dropout_rate):
        super(GAT,self).__init__()
        self.in_dim  = in_dim
        self.gat1 = EdgeGATConv(in_dim, in_dim,in_dim,1)
        self.bn1 = nn.BatchNorm1d(in_dim) 
        self.gat2 = EdgeGATConv(in_dim, in_dim,in_dim,1)
        self.gat3 = EdgeGATConv(in_dim, in_dim,in_dim,1)
        self.activate = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)  # 드롭아웃 추가
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, EdgeGATConv):
                nn.init.kaiming_uniform_(m.fc.weight, mode='fan_out')

    def forward(self, graph, node_weight,edge_weight):
        h = self.gat1(graph, node_weight,edge_weight).squeeze()
        h = self.bn1(self.dropout(h))
        h = self.activate(h)
        
        h = self.gat2(graph,h, edge_weight).squeeze()
        h = self.bn1(self.dropout(h))
        h = self.activate(h)

        h = self.gat3(graph,h, edge_weight).squeeze()
        h = self.bn1(self.dropout(h))
        return h
    

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, dropout_rate=0.5):
        super(GraphSAGE, self).__init__()
        self.num_layers = 3
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
 
        for i in range(self.num_layers):

            self.convs.append(SAGEConv(in_dim, in_dim, 'pool'))
            self.bns.append(nn.BatchNorm1d(in_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
        self.activate = nn.ELU()

    def forward(self, graph, node_weight,edge_weight):
        x = node_weight
        for i in range(self.num_layers):
            x = self.convs[i](graph, x,edge_weight = edge_weight)
            x = self.bns[i](x)
            x = self.activate(x)
            x = self.dropouts[i](x)
        return x


