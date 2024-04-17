import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from embedder import embedder
import numpy as np
import torch.nn.init as init
from dgl.nn import GraphConv
from datetime import datetime 
import random
from evaluate import *

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)

def load_data_to_gpu(batch):
    return [tensor.long().cuda() for tensor in batch]
def currentTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class NC(embedder):
    def __init__(self, args,train_dataset,val_dataset,test_dataset,labels,G,adj_matrix):
        embedder.__init__(self, args)
        self.clip_max = torch.FloatTensor([1.0]).cuda()
        self.train_dataset = train_dataset.tensors
        self.val_dataset = val_dataset.tensors
        self.test_dataset = test_dataset.tensors
        self.labels = labels
        self.g =G
        self.user_node = args.user_node
        self.adj_matrix =adj_matrix

    def train_DW(self,args):
   
        model_DW = modeler_warm(args,self.adj_matrix).to(self.device)
        parameters = filter(lambda p: p.requires_grad, model_DW.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr)
        loss_fcn = nn.CrossEntropyLoss()
        best =0
        print("[{}] Start warm-up".format(currentTime()))
        for epoch in range(0, 150):
            self.batch_loss = 0
 
            uids = load_data_to_gpu(self.train_dataset)
            optimizer.zero_grad()
            z, logits = model_DW()

            loss = loss_fcn(logits[uids],self.labels[uids])
            loss.backward()
            optimizer.step()

            model_DW.center_embedding.weight.data.div_(torch.max(torch.norm(model_DW.center_embedding.weight.data, 2, 1, True), self.clip_max).expand_as(model_DW.center_embedding.weight.data))
            test_loss, val_acc, val_micro_f1, val_macro_f1,val_auc, z = evaluate(model_DW, self.labels, load_data_to_gpu(self.val_dataset), loss_fcn)
            test_loss, test_acc, test_micro_f1, test_macro_f1,test_auc, z = evaluate(model_DW, self.labels, load_data_to_gpu(self.test_dataset), loss_fcn)
            nmi,ari = evaluate_cluster(z.cpu().detach().numpy() ,self.labels.cpu().detach().numpy() , self.test_dataset[0])
            print('Epoch {:d} | Train Loss {:.4f} |Val ACC {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f} | Val AUC f1 {:.4f} | NMI {:.4f} | ARI {:.4f}'.format(
                epoch + 1, loss.item(), test_acc, test_micro_f1, test_macro_f1,test_auc,nmi,ari))
            if val_macro_f1 > best:
                best = val_macro_f1
                cnt_wait = 0

            else:
                cnt_wait += 1
            if cnt_wait == args.patience and epoch > 50 or cnt_wait == args.patience+50 :
                print('Early stopping!')
                break
        return model_DW.get_embeds()


    def training(self,args):
        pretrained_embed = self.train_DW(args) if self.isInit else None

        result = []    
        self.args = args

        model_Nc = modeler_Nc(self.args,self.g,pretrained_embed).to(self.device)
        training_time = []
        print('#Parameters:', sum(p.numel() for p in model_Nc.parameters()))
        parameters = filter(lambda p: p.requires_grad, model_Nc.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr,weight_decay=args.reg_coef)
        loss_fcn = nn.CrossEntropyLoss()
        best =0  
        print("[{}] Start training Nc".format(currentTime()))
        self.batch_loss = 0
        for epoch in range(0, self.iter_max):
            t0 = datetime.now()

            uids = load_data_to_gpu(self.train_dataset)
            optimizer.zero_grad()
            z, logits = model_Nc()

            loss = loss_fcn(logits[uids], self.labels[uids]) 
            loss.backward()
            optimizer.step()
            training_time.append((datetime.now() - t0))

            _,_, _, aa,_, z = evaluate(model_Nc, self.labels, load_data_to_gpu(self.val_dataset), loss_fcn)
            val_loss,val_acc, val_micro_f1, val_macro_f1,val_auc, z = evaluate(model_Nc, self.labels, load_data_to_gpu(self.test_dataset), loss_fcn)
            nmi,ari = evaluate_cluster(z.cpu().detach().numpy() ,self.labels.cpu().detach().numpy(),self.test_dataset[0] )


            if aa > best:
                best = aa
                cnt_wait = 0
                result = [val_loss,val_acc, val_micro_f1, val_macro_f1,val_auc,nmi,ari]
                best_z = z
            else:
                cnt_wait += 1
            if cnt_wait == args.patience and epoch > 50 or cnt_wait == args.patience+50 :
                print('Early stopping!')
                break

            print('Epoch {:d} | Test Loss {:.4f} | Test ACC {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test AUC f1 {:.4f} | NMI {:.4f} | ARI {:.4f}'.format(
            epoch + 1,val_loss,val_acc, val_micro_f1, val_macro_f1,val_auc,nmi,ari))
        print("Total time: ", np.sum(training_time))    
        print('Best model Loss {} |  Test ACC {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test AUC f1 {:.4f}| NMI {:.4f} | ARI {:.4f}'.format(*result))
        #nmi,ari = evaluate_cluster(best_z.cpu().detach().numpy() ,self.labels.cpu().detach().numpy(),torch.LongTensor(range(args.user_node)) ,'print')
class modeler_warm(nn.Module):
    def __init__(self, args,adjacency_matrix):
        super(modeler_warm, self).__init__()
        self.center_embedding = nn.Embedding(args.num_nodes, args.dim)
        self.gcn1 = GraphConv( args.dim,args.dim)
        self.gcn2 = GraphConv( args.dim,args.dim)
        self.gcn3 = GraphConv(args.dim,args.dim)
        self.bn = nn.BatchNorm1d(args.dim) 
        self.adjacency_matrix = adjacency_matrix
        self.init_weights()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(args.dim, args.num_labels)
    def forward(self):
        x = self.center_embedding.weight.data
        x = self.gcn1(self.adjacency_matrix, x)
        x = F.elu(self.bn(self.dropout(x)))
        x = self.gcn2(self.adjacency_matrix,x)
        x = F.elu(self.bn(self.dropout(x)))
        x = self.gcn3(self.adjacency_matrix,x)

        return x,self.linear(x)

    def init_weights(self):
        nn.init.normal_(self.center_embedding.weight, mean=0.0, std=0.1)
    def get_embeds(self):
        with torch.no_grad():
            return self.center_embedding.weight.data.cpu()

class modeler_Nc(nn.Module):
    def __init__(self, args, g,pretrained_embed=None):
        super(modeler_Nc, self).__init__()
        self.num_aspects = args.num_aspects
        self.num_nodes = args.num_nodes
        self.dim = args.dim
        self.args = args
        self.aspect_embedding = nn.Embedding(self.num_nodes * self.num_aspects, self.dim)
        self.center_embedding = nn.Embedding(args.num_nodes, args.dim)
        
        self.pooling = args.pooling
        self.isInit = args.isInit
        self.g = g

        if self.isInit:
            self.init_weights(pretrained_embed=pretrained_embed)
        else:
            self.init_weights()
        self.dropout_rate = args.dropout
        self.GCN = GCN( self.dim,self.dropout_rate) 
        self.lstm = nn.LSTM(input_size= self.num_aspects*self.dim, hidden_size= self.num_aspects*self.dim, batch_first=True)
        self.linear12= nn.Linear(self.num_aspects*self.dim,self.num_aspects*self.dim,bias=True)
        self.attn = nn.Linear(self.dim,1,bias=True)
        self.gnn = args.gnn
        self.tau_gumbel = args.tau_gumbel
        self.linear = nn.Linear(self.dim,args.num_labels,bias=True)
    
        self.GAT = GAT( self.dim,self.dropout_rate) 
        self.GraphSAGE = GraphSAGE( self.dim,self.dropout_rate) 
    def init_weights(self, pretrained_embed=None):
        if pretrained_embed is not None:
            self.center_embedding.weight  =  torch.nn.Parameter(pretrained_embed)
            with torch.no_grad():
                for k in range(self.num_aspects):
                    self.aspect_embedding.weight[k * self.num_nodes: (k + 1) * self.num_nodes] =  torch.nn.Parameter(pretrained_embed)
   
        else:
            nn.init.normal_(self.aspect_embedding.weight.data, mean=0.0, std=0.1)
            nn.init.normal_(self.center_embedding.weight.data, mean=0.0, std=0.1)
    def forward(self):
        if self.num_aspects == 1:
            edge_weight = torch.sum(torch.stack([self.aspect_embedding.weight[self.g.edata['w']]],2),1).view(-1,self.dim)
        else:    
            edge_weight = torch.sum(torch.stack([self.aspect_embedding.weight[k * self.num_nodes: (k + 1) * self.num_nodes][self.g.edata['w']] for k in range(self.num_aspects)],2),1).view(-1,self.num_aspects,self.dim)
            attn_weight = F.gumbel_softmax(self.attn(edge_weight).view(-1,self.num_aspects),tau=self.tau_gumbel, hard=self.args.Is_hard)
            edge_weight = torch.sum((edge_weight.view(-1,self.num_aspects,self.dim)*attn_weight.unsqueeze(-1)),1).view(-1, self.dim)
        node_weight = self.center_embedding.weight.data
        if self.gnn == 'GCN':
            self.h = self.GCN(self.g,node_weight,edge_weight).view(-1,self.dim)
        elif self.gnn == 'GAT':
            self.h = self.GAT(self.g,node_weight,edge_weight).view(-1,self.dim)
        elif self.gnn == 'GraphSAGE':
            self.h = self.GraphSAGE(self.g,node_weight,edge_weight).view(-1,self.dim)
        return self.h, self.linear(self.h)

class GCN(nn.Module):
    def __init__(self,in_dim, dropout_rate):
        super(GCN, self).__init__()
        self.in_dim = in_dim
        self.gcn1 = GraphConv(in_dim, in_dim, weight=True)
        self.gcn2 = GraphConv(in_dim, in_dim, weight=True)
        self.gcn3= GraphConv(in_dim, in_dim, weight=True)
        self.bn1 = nn.BatchNorm1d(in_dim) 

        self.activate = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)  # 드롭아웃 추가
        self.attn = nn.Linear(in_dim,1,bias=True)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, GraphConv) and m.weight is not None:
                init.normal_(m.weight.data,mean=0.0,std=0.1)
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
    


from dgl.nn import EdgeGATConv

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
        self.attn = nn.Linear(in_dim,1,bias=True)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, EdgeGATConv):
                init.kaiming_uniform_(m.fc.weight, mode='fan_out')

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
    
from dgl.nn import SAGEConv
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, dropout_rate=0.5):
        super(GraphSAGE, self).__init__()
        self.num_layers = 3
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.attn = nn.Linear(in_dim,1,bias=True)
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
