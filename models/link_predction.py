import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from embedder import embedder
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
import random
from evaluate import *
import torch.nn.init as init
from dgl.nn import GraphConv
from tqdm import tqdm
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

class LP(embedder):
    def __init__(self, args,train_dataset,val_dataset,test_dataset,G,adj_matrix):
        embedder.__init__(self, args)
        self.clip_max = torch.FloatTensor([1.0]).cuda()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        self.g =G
        self.user_node = args.user_node
        self.adj_matrix = adj_matrix
        self.args =args
    def train_DW(self,args):
        model_DW = modeler_warm(self.args,self.adj_matrix).to(self.device)
        parameters = filter(lambda p: p.requires_grad, model_DW.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr)

        # Start training
        print("[{}] Start warm-up".format(currentTime()))
        for epoch in tqdm(range(0, 150)):
            
            epoch +=1
            self.batch_loss = 0
            for idx, batch in enumerate(self.train_loader):

                uids, iids, nids = load_data_to_gpu(batch)
                optimizer.zero_grad()
                score_pos,score_neg = model_DW(uids, iids+self.user_node , nids+self.user_node )

                score = -F.logsigmoid( torch.clamp(score_pos, max=10, min=-10))
                neg_score = -F.logsigmoid(-torch.clamp(score_neg, max=10, min=-10))
                train_loss = torch.mean(score + neg_score)

                self.batch_loss += train_loss.item()
                train_loss.backward()
                optimizer.step()

            model_DW.center_embedding.weight.data.div_(torch.max(torch.norm(model_DW.center_embedding.weight.data, 2, 1, True), self.clip_max).expand_as(model_DW.center_embedding.weight.data))
            pos_proba_list= []
            neg_proba_list =[]
            self.test_batch_loss = 0
            with torch.no_grad():
                for tidx, batch in enumerate(self.test_loader):
                    uids, iids, nids = load_data_to_gpu(batch)
                    optimizer.zero_grad()
                    score_pos,score_neg  = model_DW(uids, iids+self.user_node , nids+self.user_node )

                    pos_proba_list.extend(F.sigmoid(model_DW.score_pos).squeeze())
                    neg_proba_list.extend(F.sigmoid(model_DW.score_neg).squeeze())
            pos_proba_list = torch.stack(pos_proba_list)
            neg_proba_list = torch.stack(neg_proba_list)

            auc, ap, acc, f1 = evaluate_lp(pos_proba_list,neg_proba_list)
           # print(f'{epoch+1} and Train Loss : {self.batch_loss/(idx+1):.4f} and Test Loss : {self.test_batch_loss/(tidx+1):.4f} and AP : {ap:.4f} and AUC : {auc:.4f} and ACC : {acc:.4f} and F1 : {f1:.4f}')
   
        center_emb_intermediate = model_DW.get_embeds()
        return center_emb_intermediate


    def training(self,args):
        pretrained_embed = self.train_DW(args) if self.isInit else None
        model_LP = modeler_LP(self.args,self.g,pretrained_embed).to(self.device)
        parameters = filter(lambda p: p.requires_grad, model_LP.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr,weight_decay=args.reg_coef)
        print('#Parameters:', sum(p.numel() for p in model_LP.parameters()))
        # Start training    
        print("[{}] Start training LP".format(currentTime()))
        best_auc = 0
        training_time = []
        for epoch in range(0, self.iter_max):
            t0 = datetime.now()
            uids, iids, nids = load_data_to_gpu(self.train_dataset.tensors)
            optimizer.zero_grad()
            pos_score,neg_score = model_LP(uids, iids+self.user_node , nids+self.user_node )
            train_loss = torch.mean(-F.logsigmoid(pos_score) + -F.logsigmoid(-neg_score))

            self.batch_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()
            training_time.append((datetime.now() - t0))  
        
            pos_proba_list= []
            neg_proba_list =[]
            self.test_batch_loss = 0
            with torch.no_grad():
                uids, iids, nids = load_data_to_gpu(self.val_dataset.tensors)
                optimizer.zero_grad()

                pos_score,neg_score = model_LP(uids, iids+self.user_node , nids+self.user_node )
                self.test_batch_loss += torch.mean(-F.logsigmoid(pos_score) + -F.logsigmoid(-neg_score))

                pos_proba_list.extend(F.sigmoid(pos_score).squeeze())
                neg_proba_list.extend(F.sigmoid(neg_score).squeeze())
            pos_proba_list = torch.stack(pos_proba_list)
            neg_proba_list = torch.stack(neg_proba_list)
            auc, ap, acc, val_f1 = evaluate_lp(pos_proba_list,neg_proba_list)

            pos_proba_list= []
            neg_proba_list =[]
            self.test_batch_loss = 0
            with torch.no_grad():
                uids, iids, nids = load_data_to_gpu(self.test_dataset.tensors)
                optimizer.zero_grad()

                pos_score,neg_score = model_LP(uids, iids+self.user_node , nids+self.user_node )
                self.test_batch_loss += torch.mean(-F.logsigmoid(pos_score) + -F.logsigmoid(-neg_score))

                pos_proba_list.extend(F.sigmoid(pos_score).squeeze())
                neg_proba_list.extend(F.sigmoid(neg_score).squeeze())
            pos_proba_list = torch.stack(pos_proba_list)
            neg_proba_list = torch.stack(neg_proba_list)
            auc, ap, acc, f1 = evaluate_lp(pos_proba_list,neg_proba_list)

            if val_f1 > best_auc:
                best_auc = val_f1
                result=[epoch, auc, ap, acc, f1]
                cnt_wait = 0    
            else:
                cnt_wait += 1

            if cnt_wait == args.patience and epoch>50 :
                print('Early stopping!')
                break 
    
            print(f'{epoch+1} and Train Loss: {self.batch_loss:.4f} and Test Loss: {self.test_batch_loss:.4f} and AP: {ap:.4f} and AUC: {auc:.4f} and ACC: {acc:.4f} and F1: {f1:.4f}')
        print("Total time: ",np.sum(training_time), "s")
        print('Best Epoch {} |  Test AP {:.4f} | Test AUC {:.4f} | Test ACC {:.4f} | F1 {:.4f}'.format(*result))

class modeler_warm(nn.Module):
    def __init__(self, args,adjacency_matrix):
        super(modeler_warm, self).__init__()
        self.center_embedding = nn.Embedding(args.num_nodes, args.dim)
        self.gcn1 = GraphConv( args.dim,args.dim)
        self.gcn2 = GraphConv( args.dim,args.dim)
        self.gcn3 = GraphConv(args.dim,args.dim)
        self.bn2= nn.BatchNorm1d(args.dim) 
        self.adjacency_matrix = adjacency_matrix
        self.init_weights()
        self.dropout = nn.Dropout(0.5)
    def forward(self, uids, iids, nids ):
        x = self.center_embedding.weight.data
        x = self.gcn1(self.adjacency_matrix, x)
        x = F.elu(self.bn2(self.dropout(x)))
        x = self.gcn2(self.adjacency_matrix,x)
        x = F.elu(self.bn2(self.dropout(x)))
        x = self.gcn3(self.adjacency_matrix,x)

        self.score_pos = torch.bmm(x[uids].unsqueeze(1), x[iids].unsqueeze(-1)).squeeze(-1)
        self.score_neg = torch.bmm(x[uids].unsqueeze(1),  x[nids].unsqueeze(-1)).squeeze(-1)

        return self.score_pos,self.score_neg

    def init_weights(self):
        nn.init.normal_(self.center_embedding.weight, mean=0.0, std=0.01)
    def get_embeds(self):
        with torch.no_grad():
            return self.center_embedding.weight.data.cpu()


class modeler_LP(nn.Module):
    def __init__(self, args, g,pretrained_embed=None):
        super(modeler_LP, self).__init__()
        self.num_aspects = args.num_aspects
        self.num_nodes = args.num_nodes
        self.dim = args.dim
        self.device = args.device
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
        self.GCN = GCN( args,self.dim,self.dropout_rate) 
        self.lstm = nn.LSTM(input_size= self.num_aspects*self.dim, hidden_size= self.num_aspects*self.dim, batch_first=True)
        self.feature_fusion= nn.Linear(self.num_aspects*self.dim,self.num_aspects*self.dim,bias=True)
        self.attn = nn.Linear(self.dim,1,bias=True)
        self.dropout_rate = args.dropout
        self.gnn = args.gnn
        self.tau_gumbel = args.tau_gumbel

        self.GAT = GAT( self.dim,self.dropout_rate) 
        self.GraphSAGE = GraphSAGE( self.dim,self.dropout_rate) 
    def init_weights(self, pretrained_embed=None):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        if pretrained_embed is not None:
            self.center_embedding.weight  =  torch.nn.Parameter(pretrained_embed)
            with torch.no_grad():
                for k in range(self.num_aspects):
                    self.aspect_embedding.weight[k * self.num_nodes: (k + 1) * self.num_nodes] =  torch.nn.Parameter(pretrained_embed)

        else:
            nn.init.normal_(self.aspect_embedding.weight.data, mean=0.0, std=0.1)
            nn.init.normal_(self.center_embedding.weight.data, mean=0.0, std=0.1)

    def forward(self, uids, iids, nids):
        if self.num_aspects == 1:
            edge_weight = torch.sum(torch.stack([self.center_embedding.weight[self.g.edata['w']]],2),1).view(-1,self.dim)
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

        self.node_user = self.h[uids].view(-1,self.dim)
        self.node_item = self.h[iids].view(-1,self.dim)
        self.node_n_item =self.h[nids].view(-1,self.dim)

        self.score_pos = torch.bmm(self.node_user.unsqueeze(1),self.node_item.unsqueeze(-1)).squeeze(1)
        self.score_neg= torch.bmm(self.node_user.unsqueeze(1),self.node_n_item.unsqueeze(-1)).squeeze(1)

        return self.score_pos,self.score_neg



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
        self.dropout = nn.Dropout(dropout_rate)
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


