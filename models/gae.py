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

 
    def forward(self, z, rand_inds):  # 30000 inds
        # z = F.dropout(z, self.dropout, training=self.training)
        # return (self.sigmoid(torch.sum(z[rand_inds[0]]*z[rand_inds[1]],1))+ self.fudge) * (1 - 2 * self.fudge)
        adj_pred = (self.sigmoid(torch.mm(z[rand_inds], z[rand_inds].t())) + self.fudge) * (1 - 2 * self.fudge)
        return adj_pred


class GAE(nn.Module):
    """Non-probabilistic graph auto-encoder (GAE) model"""
    # out_channels: 500
    def __init__(self, n, edges, in_channels, out_channels, hidden_layers, inds2hops, norm_method='in', decoder='nn'):
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
        self.pos_indices_list = self.adj._indices().t().data.numpy()
        self.pos_indices = {}
        for ind in self.pos_indices_list:
            self.pos_indices[tuple(list(ind))] = 1

        # traverse 1000 known nodes, collect all 2-hops nodes
        self.nodes_2hops = self.get2hopnodes(inds2hops)
        self.targets = (adj.to_dense())[self.nodes_2hops][:,self.nodes_2hops]
        t_N = len(self.targets)
        t_n_edges = torch.sum(self.targets)
        self.pos_weight = (t_N*t_N - t_n_edges)/t_n_edges
        self.norm = t_N*t_N / float((t_N*t_N - t_n_edges) * 2)
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

    def getTargets(self):
        return self.targets

    def get2hopnodes(self, inds2hops):
        # posIndices = self.pos_indices_list.transpose()
        # def getNeighbs(ind):
        #     arginds = np.where(posIndices[0]==ind)
        #     alist = list(posIndices[1][arginds])
        #     return alist
        # nodeList = []
        # for i in range(1000):
        #     fstLst = []
        #     fstLst = getNeighbs(i)
        #     scdLst = []
        #     for ind in fstLst:
        #         scdLst += getNeighbs(ind)
        #     nodeList += fstLst
        #     nodeList += scdLst
        nodeList = list(set(inds2hops))
        # nodeList = [[[nodeList[i],nodeList[j]] for i in range(len(nodeList))] for j in range(len(nodeList))]
        # pos_inds = []
        # neg_inds = []
        # for i in range(len(nodeList)):
        #     indpair = nodeList[i]
        #     if tuple(indpair) in self.pos_indices:
        #         pos_inds.append(i)
        #     else:
        #         neg_inds.append(i)
        # nodeList = np.array(nodeList).transpose()
        return nodeList#, pos_inds, neg_inds

    def getLatentEmbedding(self,x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        # sample 3 pos and 29997 neg from adj

        # rand_inds=[]
        # pos_sample = np.random.choice(97412, 3)
        # zzz = self.pos_indices_list[pos_sample]
        # for i in range(3):
        #     rand_inds.append(zzz[i])
        # for i in range(29997):
        #     ind = np.random.choice(32324, 2)
        #     if tuple(list(ind)) not in self.pos_indices:
        #         rand_inds.append(ind)

        # self.rand_inds = np.array(rand_inds).transpose()
        A_pred = self.decoderA(z,self.nodes_2hops)
        x_pred = None#self.decoderX(z)
        return A_pred, x_pred


class GAECrit(nn.Module):
    """docstring for GAECrit"""
    def __init__(self,pos_weight,norm):
        super(GAECrit, self).__init__()
        self.pos_weight = pos_weight
        self.norm = norm

    def weighted_cross_entropy(self, logits, targets):
        # loss = (torch.sum(- sigmout[self.posi].log()) * self.pos_weight - torch.sum((1 - sigmout[self.negi]).log()))/len(sigmout)
        # for i in range(len(sigmout)):
        #     if i<3:
        #         loss = loss - sigmout[i].log() * self.pos_weight
        #     else:
        #         loss = loss - (1 - sigmout[i]).log()
        # return loss
        # targets = 
        zzz=torch.zeros_like(logits)
        zzz[logits>0.5]=1.0
        zzz[logits<=0.5]=0.
        error_rate = torch.sum(torch.abs(targets-zzz))/len(logits)**2
        return (torch.sum(targets * -logits.log() * self.pos_weight + (1 - targets) * -(1 - logits).log()))/(len(logits)**2), error_rate
    def BCELossOnA(self,A_pred,adj):
        # loss = (A_pred-adj)*wt_mat
        loss, error_rate = self.weighted_cross_entropy(A_pred,adj)
        loss = self.norm*loss
        return loss, error_rate

    def L2LossOnX(self,x_pred,x):
        return ((x_pred - x)**2).sum() / (len(x)*len(x[0]))
    
    def forward(self,A_pred,x_pred,adj,x):
        A_loss, error_rate = self.BCELossOnA(A_pred,adj)
        x_loss = 0#self.L2LossOnX(x_pred,x)
        return A_loss, x_loss, error_rate

        
























        

