import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from utils import normt_spm, spm_to_tensor


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        xavier_uniform_(self.w)

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj):
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        outputs = torch.mm(adj, torch.mm(inputs, self.w)) + self.b

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


class GCN(nn.Module):

    def __init__(self, n, edges, in_channels, out_channels, hidden_layers, norm_method='in'):
        super().__init__()

        edges = np.array(edges)
        adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                            shape=(n, n), dtype='float32')
        adj = normt_spm(adj, method=norm_method)
        adj = spm_to_tensor(adj)
        self.adj = adj.cuda()

        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False

        i = 0
        layers = []
        last_c = in_channels
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)

            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module('conv-last', conv)
        layers.append(conv)

        self.layers = layers

    def forward(self, x):
        for conv in self.layers:
            x = conv(x, self.adj)
        return F.normalize(x)

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()
        self.fudge = 1e-7


    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = (self.sigmoid(torch.mm(z, z.t())) + self.fudge) * (1 - 2 * self.fudge)
        return adj


class GAE(nn.Module):
    """Non-probabilistic graph auto-encoder (GAE) model"""
    # out_channels: 500
    def __init__(self, n, edges, in_channels, out_channels, hidden_layers, norm_method='in', decoder='nn'):
        super(GAE, self).__init__()
        self.encoder = GCN(n, edges, in_channels, out_channels, hidden_layers, norm_method=norm_method)
        edges = np.array(edges)
        adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                            shape=(n, n), dtype='float32')
        adj = spm_to_tensor(adj)
        self.adj = adj
        N = len(adj)
        n_edges = torch.sparse.sum(adj)
        self.pos_weight = (N*N - n_edges)/n_edges
        self.norm = N*N / float((N*N - n_edges) * 2)
        # wt_mat = adj.mul(weight)
        # wt_mat[wt_mat==0] = 1
        # self.wt_mat = wt_mat

        self.decoderA = InnerProductDecoder(0.3)
        if decoder == 'gcn':
            self.decoderX = GCN(n, edges, out_channels, in_channels, hidden_layers, norm_method=norm_method)
        else:
            self.decoderX = nn.Sequential(
                nn.Linear(out_channels,out_channels),
                nn.ReLU(),
                nn.Linear(out_channels,in_channels)
                )

    def getLatentEmbedding(self,x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        A_pred = self.decoderA(z)
        x_pred = self.decoderX(z)
        return A_pred, x_pred


class GAECrit(object):
    """docstring for GAECrit"""
    def __init__(self, arg):
        super(GAECrit, self).__init__()

    def weighted_cross_entropy(sigmout, targets, pos_weight):
        return torch.sum(targets * -logits.log() * pos_weight + 
                (1 - targets) * -(1 - logits).log())
    def BCELossOnA(self,A_pred,adj):
        # loss = (A_pred-adj)*wt_mat
        loss = self.weighted_cross_entropy(A_pred,adj.to_dense(),self.pos_weight)
        return loss

    def L2LossOnX(self,x_pred,x):
        return ((x_pred - x)**2).sum() / (len(x) * len(x[0]))
    
    def forward(self,A_pred,x_pred,adj,x):
        A_loss = self.BCELossOnA(A_pred,adj)
        x_loss = self.L2LossOnX(x_pred,x)
        return A_loss+x_loss

        
























        

