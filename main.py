import torch
import random
import numpy as np
import argparse
import pickle
import dgl    
from scipy.sparse import csr_matrix
from datetime import datetime
from utils import *
from models.link_predction import LP
from models.node_classification import NC


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)


def printConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))


    
def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='MF2Vec')

    # Essential parameters
    parser.add_argument('--embedder', nargs='?', default='MF2Vec')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=0.003, help="Learning rate")
    parser.add_argument('--dim', type=int, default=64, help="Dimension size.")
    parser.add_argument('--num_aspects', type=int, default=5, help="Number of aspects")
    parser.add_argument('--isInit', action='store_true', default=True , help="Warm-up")
    parser.add_argument('--reg_coef', type=float, default=0.00001)

    # Default parameters
    parser.add_argument('--batch_size', type=int, default=100000)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--dropout', type=int, default=0.4)
    parser.add_argument('--iter_max', type=int, default=1000)
    parser.add_argument('--tau_gumbel', type=float, default=0.5)
    parser.add_argument('--Is_hard', action='store_true', default=True )
    parser.add_argument('--gnn', type=str, default='GCN')
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--data', type=str, default='imdb', choices=['movielens', 'yelp', 'amazon', 'dblp', 'acm', 'imdb','freebase'])

    args, unknown = parser.parse_known_args()

    # Set defaults based on the data argument
    if args.data == 'movielens':
        num_nodes_default = 2672
        user_node_default = 943
        num_labels_default = 0
    elif args.data == 'yelp':
        num_nodes_default = 31081
        user_node_default = 16239
        num_labels_default = 0
    elif args.data == 'amazon':
        num_nodes_default = 13136
        user_node_default = 6170
        num_labels_default = 0
    elif args.data == 'dblp':
        num_nodes_default = 26128
        user_node_default = 4057
        num_labels_default = 4
    elif args.data == 'acm':
        num_nodes_default = 11246
        user_node_default = 4019
        num_labels_default = 3
    elif args.data == 'imdb':
        num_nodes_default = 11519
        user_node_default = 4181
        num_labels_default = 3
    elif args.data == 'freebase':
        num_nodes_default = 43854
        user_node_default = 3492
        num_labels_default = 3
    # Add other arguments with dynamic defaults
    parser.add_argument('--num_nodes', type=int, default=num_nodes_default)
    parser.add_argument('--user_node', type=int, default=user_node_default)
    parser.add_argument('--num_labels', type=int, default=num_labels_default)
    parser.add_argument('--graph', type=str, default='graph_0')
    return parser.parse_known_args()


def main():
    args, unknown = parse_args()

    print("Start Learning [{}]".format( datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    graph = dgl.load_graphs(f'../data/{args.data}/{args.graph}.bin')[0][0].to(args.device)
    adj_matrix = dgl.from_scipy(csr_matrix(torch.load(f"../data/{args.data}/all_adj_matrix.pt"), dtype=np.float32)).to(args.device)
    
    train = torch.load( f"../data/{args.data}/train_dataset_{args.ratio}.pt")
    val = torch.load( f"../data/{args.data}/val_dataset_{args.ratio}.pt")
    test = torch.load( f"../data/{args.data}/test_dataset_{args.ratio}.pt")

    if args.data =='dblp' or args.data =='acm' or args.data =='imdb' or args.data == 'freebase':
        labels = torch.tensor(np.load(f'../data/{args.data}/labels.npy')).to(args.device).long()
        embedder = NC(args,train,val,test,labels,graph,adj_matrix)
    elif args.data =='movielens' or args.data =='yelp':
        embedder = LP(args,train,val,test,graph,adj_matrix)

    embedder.training(args)
if __name__ == '__main__':
    main()
