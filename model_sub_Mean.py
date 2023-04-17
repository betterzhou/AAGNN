import torch.nn as nn
import torch.nn.functional as F
from layers_sub_Mean import GraphConvolution


class GDN_sub_mean(nn.Module):
    def __init__(self, nfeat, nhid, dropout, Graph_networkx):
        super(GDN_sub_mean, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, Graph_networkx)
        self.dropout = dropout

    def forward(self, x, adj_matrix, degree_norm):
        embeddings = F.relu(self.gc1(x, adj_matrix, degree_norm))
        return embeddings
