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

def save_checkpoint(name):
    torch.save(gae.state_dict(), osp.join(save_path, name + '.pth'))
    torch.save(pred_obj, osp.join(save_path, name + '.pred'))


def mask_l2_loss(a, b, mask):
    return l2_loss(a[mask], b[mask])


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
    gae = GAE(n, edges, word_vectors.shape[1], fc_vectors.shape[1], hidden_layers, inds2hops, args.norm_method).cuda()
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

    for epoch in range(1, args.max_epoch + 1):
        start = time.time()
        gae.train()
        A_pred, x_pred = gae(word_vectors)
        lossA, lossX = crit(A_pred,x_pred,targets,word_vectors)
        loss = lossA + lossX

        # loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # gae.eval()
        # output_vectors = gae(word_vectors)
        # train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train]).item()
        # A_pred, x_pred = gae(word_vectors)
        # lossA, lossX = crit(A_pred,x_pred,targets,word_vectors)
        # if v_val > 0:
        #     val_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[n_train:]).item()
        #     loss = val_loss
        # else:
        #     val_loss = 0
        #     loss = train_loss
        end = time.time() - start
        print('epoch {}, A_loss={:.4f}, X_loss:{:.4f}, iter_time:{:.2f}s'
              .format(epoch, lossA.data, lossX.data, end))

        # trlog['train_loss'].append(lossA.data+lossX.data)
        # trlog['val_loss'].append(0)
        # trlog['min_loss'] = min_loss
        # torch.save(trlog, osp.join(save_path, 'trlog'))

        if (epoch == args.save_epoch):
            if args.no_pred:
                pred_obj = None
            else:
                pred_obj = {
                    'wnids': wnids,
                    'pred': gae.getLatentEmbedding(word_vectors)#output_vectors
                }

        if epoch == args.save_epoch:
            save_checkpoint('epoch-{}'.format(epoch))
            break
        
        pred_obj = None

