from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score,silhouette_score, f1_score,roc_auc_score,adjusted_rand_score,average_precision_score,accuracy_score
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch.nn.functional as F

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    

    return accuracy, micro_f1,macro_f1
def evaluate_cluster(embedding_dict, label, idx,printplt=None):
        
        if printplt == 'print':
 
            embedding_dict = embedding_dict[idx]
            num_clusters = max(label)+1
            Y_pred = KMeans(num_clusters, random_state=0).fit(np.array(embedding_dict)).predict(embedding_dict)
            nmi =  normalized_mutual_info_score(np.array(label), Y_pred)
            ari = adjusted_rand_score(np.array(label), Y_pred)
            tsne = TSNE(n_components=2)
            embedding_2d = tsne.fit_transform(embedding_dict)
            colors = ['b', 'g', 'r', 'y']  

            plt.figure(figsize=(10, 8))
            for i in range(num_clusters):  
                cluster_points = embedding_2d[Y_pred == i]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i}')

            plt.legend()
            plt.title('2D Embedding')
            plt.show()
        else:   
            embedding_dict = embedding_dict[idx]
            num_clusters = max(label)+1
            Y_pred = KMeans(num_clusters, random_state=0).fit(np.array(embedding_dict)).predict(embedding_dict)
            nmi =  normalized_mutual_info_score(np.array(label[idx]), Y_pred)
            ari = adjusted_rand_score(np.array(label[idx]), Y_pred)
        return nmi, ari

def evaluate(model, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        z, logits = model()
    
    prob = F.softmax(logits[mask],1)
    prob = prob.cpu().detach().numpy()
    accuracy, micro_f1,macro_f1 = score(logits[mask], labels[mask])
    labels_cpu = labels[mask].cpu()
    auc = roc_auc_score(labels_cpu, prob,multi_class='ovr')
    return loss_func(logits[mask],labels[mask]), accuracy, micro_f1,macro_f1, auc, z



def evaluate_lp(pos_proba_list, neg_proba_list):

    y_proba_test = torch.cat([pos_proba_list,neg_proba_list],0)
    y_proba_test = y_proba_test.cpu().numpy()
    y_true_test = [1] * len(pos_proba_list) + [0] * len(neg_proba_list)
    auc = roc_auc_score(y_true_test, y_proba_test)
    ap = average_precision_score(y_true_test, y_proba_test)
    acc = accuracy_score(y_true_test,np.round(y_proba_test))   
    f1 = f1_score(y_true_test,np.round(y_proba_test))  
    return auc, ap, acc, f1