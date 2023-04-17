from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import os
import torch
import torch.optim as optim
import scipy
import scipy.io
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import pandas as pd
import networkx as nx
from utils import *
from model_sub_Mean import *
from early_stop import EarlyStopping
from model_sub_Att import ADN_SpGAT

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', type=int, default=1, help='CUDA training.')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,help='Number of epoch to train.')
parser.add_argument('--patient', type=int, default=10,help='Number of epoch for patience on early_stop.')
parser.add_argument('--lr', type=float, default=0.001,help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0,help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', default='BlogCatalog_anomaly')
parser.add_argument('--model', default='model_sub_Mean')
parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('cuda running...')
dataset = args.dataset


def load_data(data_source, hidden_dim):
    data = scipy.io.loadmat("./all_new_datasets/{}.mat".format(data_source))
    gnds = data["gnd"]
    attributes_sprse = sp.csr_matrix(data["Attributes"])
    adj_csr_matrix = sp.csr_matrix(data["Network"])
    Graph = nx.from_scipy_sparse_matrix(adj_csr_matrix)
    attributes = attributes_sprse.todense()
    adj_maitrx = adj_csr_matrix.todense()
    hidden_dimension = hidden_dim
    degree = adj_maitrx.sum(axis=1).reshape(-1, 1)
    degree_divition = 1 / degree
    degree_matrix = np.tile(degree_divition, hidden_dimension)

    return adj_maitrx, attributes, gnds, Graph, degree_matrix


def random_nodes_selection(return_node_num, total_node, seed):
    randomlist = [i for i in range(total_node)]
    random.seed(seed)
    random.shuffle(randomlist)
    return randomlist[:return_node_num], randomlist[return_node_num: total_node]


def train_test_splitting(total_node, seed, split_ratio):
    randomlist = [i for i in range(total_node)]
    random.seed(seed)
    random.shuffle(randomlist)

    break_point = int(total_node * split_ratio)
    return randomlist[:break_point], randomlist[break_point: total_node]


def anomaly_score(node_embedding, c):
    return torch.sum((node_embedding - c) ** 2)


def nor_loss(node_embedding_list, c):
    s = 0
    num_node = node_embedding_list.size()[0]
    for i in range(num_node):
        s = s + anomaly_score(node_embedding_list[i], c)
    return s/num_node


def objecttive_loss_valid(normal_node_emb, c):
    Nloss = nor_loss(normal_node_emb, c)
    return Nloss


adj_dense, features, gnds, G, degree_mat = load_data(args.dataset, args.hidden)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_label = torch.from_numpy(gnds).to(device)
features = torch.FloatTensor(np.array(features)).to(device)
degree_mat = torch.FloatTensor(np.array(degree_mat)).to(device)
adj_dense = torch.FloatTensor(np.array(adj_dense)).to(device)
GDN_model = GDN_sub_mean(nfeat=features.shape[1],
                             nhid=args.hidden,
                             dropout=args.dropout,
                             Graph_networkx=G)
if args.model == 'Atten_Aggregate':
    GDN_model = ADN_SpGAT(nfeat=features.shape[1],
                nhid=args.hidden,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)

optimizer = optim.Adam(GDN_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
GDN_model.to(device)
GDN_model.eval()
polluted_train_emb = GDN_model(features, adj_dense, degree_mat).detach().cpu().numpy()
polluted_train_embed = torch.FloatTensor(polluted_train_emb).to(device)
center = torch.mean(polluted_train_embed, 0).to(device)
sps_outlier_list = []
for i in range(polluted_train_emb.shape[0]):
    sps_outlier_list.append(anomaly_score(polluted_train_embed[i, :], center).item())
sps_scores = np.array(sps_outlier_list)
sorted_sps_indices = np.argsort(-sps_scores, axis=0)
total_nodes_numb = features.shape[0]
train_valid_num = int(0.5 * total_nodes_numb)
total_test_indx = sorted_sps_indices[0: -train_valid_num]
train_valid_total_indx = sorted_sps_indices[-train_valid_num:]
train_vlid_number = train_valid_total_indx.shape[0]
train_seq, valid_seq = train_test_splitting(train_vlid_number, args.seed, split_ratio=(0.3/0.5))
train_final_indx = train_valid_total_indx[np.array(train_seq)]
valid_final_indx = train_valid_total_indx[np.array(valid_seq)]
gnd_1_indices = np.where(gnds[total_test_indx] == 1)[0]
test_outliers = total_test_indx[gnd_1_indices]
test_outliers_num = test_outliers.shape[0]
early_stopping = EarlyStopping(patience=args.patient, verbose=True)
t_total = time.time()
train_losses = []
val_losses = []
for epoch in range(args.epochs):
    t = time.time()
    GDN_model.train()
    optimizer.zero_grad()
    output = GDN_model(features, adj_dense, degree_mat)
    loss_train = objecttive_loss_valid(output[train_final_indx], center)
    loss_train.backward()
    optimizer.step()
    GDN_model.eval()
    output_val = GDN_model(features, adj_dense, degree_mat)
    loss_val = objecttive_loss_valid(output_val[valid_final_indx], center)
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.6f}'.format(loss_train.item()),
          'time: {:.4f}s'.format(time.time() - t))
    early_stopping(loss_val, GDN_model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
GDN_model.load_state_dict(torch.load('checkpoint.pt'))
GDN_model.eval()
test_pred_emb = GDN_model(features, adj_dense, degree_mat)
y_true = np.array([label[0] for label in gnds])
tmp_list = []
for i in range(test_pred_emb.shape[0]):
    tmp_list.append(anomaly_score(test_pred_emb[i, :], center).item())
anomaly_score = np.array(tmp_list)
roc_auc = roc_auc_score(gnds[total_test_indx], anomaly_score[total_test_indx])
roc_pr_area = average_precision_score(gnds[total_test_indx], anomaly_score[total_test_indx])
print('---auc: %.4f' % roc_auc)
print('---aupr: %.4f' % roc_pr_area)
all_results = []
result_folder = './result/' + args.dataset
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
all_results.append(roc_auc)
all_results.append(roc_pr_area)
df = pd.DataFrame(np.array(all_results).reshape(-1, 2), columns=['AUC', 'AUPR'])
df.to_csv(result_folder + '/' + args.dataset + '_results_'+str(args.seed)+'.csv')
