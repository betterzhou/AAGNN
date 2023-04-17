import torch
import torch.nn as nn
import torch.nn.functional as F
from layers_sub_Att import SpGraphAttentionLayer


class ADN_SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(ADN_SpGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=False) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj, no_need_param):
        neighborhood_emb = torch.cat([encoder(x, adj)[0] for encoder in self.attentions], dim=1)
        assert len(self.attentions) == 1
        encoder_1 = self.attentions[0]
        map_fun_W = encoder_1(x, adj)[1]
        mapped_fea = torch.mm(x, map_fun_W)
        new_emb_mat = F.elu(mapped_fea - neighborhood_emb)
        return new_emb_mat

