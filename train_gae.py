import argparse
import json
import random
import os.path as osp

import torch
import torch.nn.functional as F

from utils import ensure_path, set_gpu, l2_loss
from models.gae import GAE, GAECrit
import time
import numpy as np
import scipy.sparse as sp

def save_checkpoint(name,gae):
    pred_obj = {
                    'wnids': wnids,
                    'pred': gae.getLatentEmbedding(word_vectors)#output_vectors
                }
    torch.save(gae.state_dict(), osp.join(save_path, name + '.pth'))
    torch.save(pred_obj, osp.join(save_path, name + '.pred'))


def mask_l2_loss(a, b, mask):
    return l2_loss(a[mask], b[mask])


def updateADJCoo(A_pred_0,A_pred_1,adj_coo,inds2hops,n):

    A_pred_0 = A_pred_0.cpu().detach().numpy() 
    A_pred_1 = A_pred_1.cpu().detach().numpy() 
    diff = A_pred_1 - A_pred_0
    booldiff = np.zeros_like(diff)
    booldiff[diff>0.095]=1
    numUpdates = np.sum(booldiff)
    print('Num of links need to update: '+str(numUpdates))
    indices = np.where(booldiff==1) # row: indices[0], col: indices[1]
    diff_values = diff[indices]

    # convert indices to real indices via inds2hops
    indices[0] = inds2hops[indices[0]]
    indices[1] = inds2hops[indices[1]]
    # update adj_coo. 
    row = adj_coo.row
    col = adj_coo.col
    val = adj_coo.data
    orig_inds = np.stack((row,col), axis=1) # (32544, 2)

    val_dict = {tuple(orig_inds[i]):val[i] for i in range(len(row))}

    indices = indices.transpose()

    for i in range(len(indices)):
        ind = indices[i]
        orig_val = val_dict.setdefault(tuple(ind),0.)
        val_dict[tuple(ind)] = min(diff_values[i]+orig_val,1.0)
    
    updated_inds = np.array(list(val_dict.keys())).transpose() # row: indices[0], col: indices[1]
    updated_vals = np.array(list(val_dict.values()))
    
    next_adj_coo = sp.coo_matrix((updated_vals, (updated_inds[0], updated_inds[1])),
                            shape=(n, n), dtype='float32')
    
    return next_adj_coo


def save_adj_coo(adj_coo,itr):
    scipy.sparse.save_npz(osp.join(save_path,'adj_coo_'+str(itr)+'.npz'), adj_coo)
    # sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epoch', type=int, default=5000)
    parser.add_argument('--trainval', default='10,0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--save-epoch', type=int, default=3000)
    parser.add_argument('--save-path', default='save/gae-basic')

    parser.add_argument('--gpu', default='0')

    parser.add_argument('--no-pred', action='store_true')
    parser.add_argument('--layers', default='d2048,d')
    parser.add_argument('--norm-method',default='in')
    args = parser.parse_args()

    set_gpu(args.gpu)

    save_path = args.save_path
    ensure_path(save_path)



    graph = json.load(open('materials/imagenet-induced-graph.json', 'r'))
    wnids = graph['wnids']
    n = len(wnids)
    edges = graph['edges']
    
    #################
    test_sets = json.load(open('materials/imagenet-testsets.json', 'r'))
    train_wnids = test_sets['train']
    test_wnids = test_sets['2-hops']
    
    def getInds(split,wnids):
        Inds = []
        for wnid in split:
            ind = wnids.index(wnid)
            Inds.append(ind)
        return Inds

    inds2hops = []
    inds2hops += getInds(train_wnids, wnids)
    inds2hops += getInds(test_wnids, wnids)
    ###############


    edges = edges + [(v, u) for (u, v) in edges]
    edges = edges + [(u, u) for u in range(n)]

    word_vectors = torch.tensor(graph['vectors']).cuda()
    word_vectors = F.normalize(word_vectors)

    fcfile = json.load(open('materials/fc-weights.json', 'r'))
    train_wnids = [x[0] for x in fcfile]
    fc_vectors = [x[1] for x in fcfile]
    assert train_wnids == wnids[:len(train_wnids)]
    fc_vectors = torch.tensor(fc_vectors).cuda()
    fc_vectors = F.normalize(fc_vectors)

    hidden_layers = args.layers #'d2048,d' #'2048,2048,1024,1024,d512,d'
    gae = GAE(n, edges, word_vectors.shape[1], 512,fc_vectors.shape[1], hidden_layers, inds2hops, args.norm_method).cuda()  #fc_vectors.shape[1]
    crit = GAECrit(gae.pos_weight, gae.norm).cuda()
    targets = gae.getTargets().cuda()
    print('{} nodes, {} edges'.format(n, len(edges)))
    print('word vectors:', word_vectors.shape)
    print('fc vectors:', fc_vectors.shape)
    print('hidden layers:', hidden_layers)

    optimizer = torch.optim.Adam(gae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    v_train, v_val = map(float, args.trainval.split(','))
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))
    print('num train: {}, num val: {}'.format(n_train, n_trainval - n_train))
    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)

    min_loss = 1e18

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['min_loss'] = 0

    def train_gae(involveXC=False, numiters=400):
        for epoch in range(1, numiters + 1):
            start = time.time()
            gae.train()
            A_pred, x_pred, c_pred = gae(word_vectors)
            lossA, lossX, lossC, error_rateA = crit(A_pred,x_pred,c_pred,targets,word_vectors,fc_vectors)
            
            loss = lossA# + lossX + lossC
            if involveXC:
                loss += lossX + lossC

            # loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end = time.time() - start
            print('epoch {}, A_loss={:.4f}, A_error_rate={:.4f}, X_loss:{:.4f}, C_loss:{:.4f}, iter_time:{:.2f}s'
                  .format(epoch, lossA.data, error_rateA.data, lossX.data, lossC.data, end))

        return A_pred

    # outer for loop
    outIterNum = 2
    for outiter in range(outIterNum):
        # train A
        # save Z
        A_pred_0 = train_gae(involveXC=False,numiters=args.save_epoch)
        torch.save(A_pred_0,osp.join(save_path,'A_pred_0.pt'))

        # train A+C
        # save Z
        A_pred_1 = train_gae(involveXC=True,numiters=args.save_epoch)
        torch.save(A_pred_1,osp.join(save_path,'A_pred_1.pt'))
        save_checkpoint('epoch-{}'.format(args.save_epoch),gae)

        # update A.
        # save A_pred
        next_adj_coo = updateADJCoo(A_pred_0,A_pred_1,gae.adj_coo,inds2hops,n)
        gae.updateADJInfo(next_adj_coo)
        save_adj_coo(next_adj_coo,outiter)

















    
    

