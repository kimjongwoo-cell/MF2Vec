import numpy as np
import pickle
from datetime import datetime
import random
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import torch
import dgl
from scipy import sparse
import itertools
random.seed(41)

def currentTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def random_walks(G, start_node, end_node, max_length, neighbors_cache,max_attempts=1000):
    path = [start_node]
    for _ in range(max_length-1):
        current_node = path[-1]
        neighbors = neighbors_cache[current_node]
        # if start_node in neighbors:
        #     neighbors.remove(start_node)
        if not neighbors:
            return None 
        else:
            current_node = random.choice(neighbors)
            path.append(current_node)
    if current_node  in end_node :

        return path 
    else:
        return None
def random_walk_path(g,dataset,user_lenth,item_lenth,min_distance,max_attempts,node_range= None):
    paths = []
    neighbors_cache = {node: list(g.neighbors(node)) for node in g.nodes()}
    for node in tqdm(g.nodes()):
        if node >= user_lenth:
            end_node_range = range(user_lenth, item_lenth+user_lenth)
        else:
            end_node_range = range(0, user_lenth)
        if node <= user_lenth +item_lenth :
            for _ in range(max_attempts):
                path  = random_walks(g, node, end_node_range, min_distance,neighbors_cache,max_attempts)
                if  path is not None:
      
                    if node_range is not None:
                        if path is not None:
                            paths.append(path)   
                    else: 
                        paths.append(path)            
        paths.append([node]*min_distance)
    paths = list(item for item, _ in itertools.groupby(paths))
    pairs = set()

    unique_list = []
    for inner_list in paths:
        if len(inner_list) >= 2:
            pair = (inner_list[0], inner_list[-1])
            if pair not in pairs and pair[::-1] not in pairs:
                pairs.add(pair)
                unique_list.append(inner_list)

    src_nodes = [inner_list[0] for inner_list in unique_list]
    dst_nodes = [inner_list[-1] for inner_list in unique_list]
    path_nodes = [inner_list[1:-1] for inner_list in unique_list]
    
    new_g = dgl.graph((src_nodes, dst_nodes))
    edges = new_g.edge_ids(src_nodes, dst_nodes)
    new_g.edata['w'] = torch.zeros(new_g.number_of_edges(), min_distance-2, dtype=torch.int64)
    new_g.edata['w'][edges] = torch.tensor(path_nodes, dtype=torch.int64) 
    dgl.save_graphs(f'data/{dataset}/graph_{max_attempts}.bin', new_g)
    return unique_list


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
def save_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def get_course_dict_lp(filepath):
    with open(filepath, "r", encoding="utf8") as f:
        return [[int(tc_rel.strip().split("\t")[0])-1, int(tc_rel.strip().split("\t")[1])-1] for tc_rel in f.readlines()]

def get_course_dict_nc(filepath):
    with open(filepath, "r", encoding="utf8") as f:
        return [[int(tc_rel.strip().split("\t")[0]), int(tc_rel.strip().split("\t")[1])] for tc_rel in f.readlines()]

def create_one_hot_matrix(tc_rel_list):
    max_user_id = max(tc_rel[0] for tc_rel in tc_rel_list)
    max_item_id = max(tc_rel[1] for tc_rel in tc_rel_list)
    one_hot_matrix = np.zeros((max_user_id + 1, max_item_id + 1))
    for user_id, item_id in tc_rel_list:
        one_hot_matrix[user_id, item_id] = 1
    return one_hot_matrix


def node_classification_split(data,user_lenth):
    path =f"data/{data}/"
    uids_tensor = torch.tensor(range(0,user_lenth))
    labels = np.load(f"data/{data}/labels.npy")
    if data == 'dblp':
        num_label = 4
    else:
        num_label = 3
    for train_ratio in [0.8,0.6,0.4,0.2]:

        user_0 = [uids_tensor[i] for i in range(len(uids_tensor)) if labels[i] == 0]
        user_1 = [uids_tensor[i] for i in range(len(uids_tensor)) if labels[i] == 1]
        user_2 = [uids_tensor[i] for i in range(len(uids_tensor)) if labels[i] == 2]
        if num_label == 4:
            user_3 = [uids_tensor[i] for i in range(len(uids_tensor)) if labels[i] == 3]
        train_0, test_0 = train_test_split(user_0, test_size=(1-train_ratio),random_state=42)
        test_0, val_0 = train_test_split(test_0, test_size=0.5,random_state=42)

        train_1, test_1 = train_test_split(user_1, test_size=(1-train_ratio),random_state=42)
        test_1, val_1 = train_test_split(test_1, test_size=0.5,random_state=42)

        train_2, test_2 = train_test_split(user_2, test_size=(1-train_ratio),random_state=42)
        test_2, val_2 = train_test_split(test_2, test_size=0.5,random_state=42)

        if num_label == 4:
            train_3, test_3 = train_test_split(user_3, test_size=(1-train_ratio),random_state=42)
            test_3, val_3 = train_test_split(test_3, test_size=0.5,random_state=42)
        if num_label == 4:
            train_data = train_0 + train_1 + train_2+ train_3
            val_data = val_0 + val_1 + val_2 +val_3
            test_data = test_0 + test_1 + test_2 +test_3
        else:
            train_data = train_0 + train_1 + train_2
            val_data = val_0 + val_1 + val_2
            test_data = test_0 + test_1 + test_2

        
        torch.save(TensorDataset(torch.tensor(train_data)), path + f"train_dataset_{train_ratio}.pt")
        torch.save(TensorDataset(torch.tensor(val_data)), path + f"val_dataset_{train_ratio}.pt")
        torch.save(TensorDataset(torch.tensor(test_data)), path + f"test_dataset_{train_ratio}.pt")


        

def link_prediction_split(data,matrix):
    training_matrix = sparse.csr_matrix(matrix)
    uids, iids = training_matrix.nonzero()

    uids_tensor = torch.tensor(uids, dtype=torch.long)
    iids_tensor = torch.tensor(iids, dtype=torch.long)

    tensor_matrix = torch.tensor(training_matrix.toarray())
    nids_tensor = torch.tensor([random.choice(torch.nonzero(tensor_matrix[uid] == 0, as_tuple=False).squeeze().tolist()) for uid in uids_tensor], dtype=torch.long)

    for train_ratio in [0.8,0.6,0.4,0.2]:
        test_ratio = (1-train_ratio)/2
        data_size = len(uids_tensor)
        train_size, val_size = int(data_size * train_ratio), int(data_size * test_ratio)

        index = torch.randperm(data_size)
        uids_tensor, iids_tensor, nids_tensor = uids_tensor[index], iids_tensor[index], nids_tensor[index]
        split_tensors = lambda tensor: (tensor[:train_size], tensor[train_size:train_size+val_size], tensor[train_size+val_size:])

        uids_train_tensor, uids_val_tensor, uids_test_tensor = split_tensors(uids_tensor)
        iids_train_tensor, iids_val_tensor, iids_test_tensor = split_tensors(iids_tensor)
        nids_train_tensor, nids_val_tensor, nids_test_tensor = split_tensors(nids_tensor)

        train_dataset = TensorDataset(uids_train_tensor, iids_train_tensor, nids_train_tensor)
        val_dataset = TensorDataset(uids_val_tensor, iids_val_tensor, nids_val_tensor)
        test_dataset = TensorDataset(uids_test_tensor, iids_test_tensor, nids_test_tensor)

        torch.save(train_dataset, f"data/{data}]/train_dataset_{train_ratio}.pt")
        torch.save(val_dataset, f"data/{data}]/val_dataset_{train_ratio}.pt")
        torch.save(test_dataset, f"data/{data}]/test_dataset_{train_ratio}.pt")

