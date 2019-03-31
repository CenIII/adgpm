import argparse
import json
import random
import os.path as osp

import torch
import torch.nn.functional as F

from utils import ensure_path, set_gpu, l2_loss
from models.gcn import GCN
from datasets.imagenet_train import ImageNetFeatsTrain
from torch.utils.data import DataLoader
import tqdm
import numpy as np

def save_checkpoint(name):
    torch.save(gcn.state_dict(), osp.join(save_path, name + '.pth'))
    torch.save(pred_obj, osp.join(save_path, name + '.pred'))


def mask_l2_loss(a, b, mask):
    return l2_loss(a[mask], b[mask])

def stach_l2_loss(output, data, label, wnid2index):
    inds = torch.tensor([wnid2index[x] for x in label]).cuda()
    preds = torch.index_select(output, 0, inds)
    loss = ((preds - data)**2).sum()/(len(inds)*2)
    return loss 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epoch', type=int, default=5000)
    parser.add_argument('--trainval', default='10,0')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--save-epoch', type=int, default=3000)
    parser.add_argument('--save-path', default='save/gcn-basic')

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
    gcn = GCN(n, edges, word_vectors.shape[1], fc_vectors.shape[1], hidden_layers, args.norm_method).cuda()

    print('{} nodes, {} edges'.format(n, len(edges)))
    print('word vectors:', word_vectors.shape)
    print('fc vectors:', fc_vectors.shape)
    print('hidden layers:', hidden_layers)

    optimizer = torch.optim.Adam(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    v_train, v_val = map(float, args.trainval.split(','))
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))
    print('num train: {}, num val: {}'.format(n_train, n_trainval - n_train))
    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)

    min_loss = 1e18

    wnid2index = {}

    for i in range(len(wnids)):
        wnid2index[wnids[i]] = i

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['min_loss'] = 0

    dataset = ImageNetFeatsTrain('./materials/datasets/imagenet_feats/')
    loader = DataLoader(dataset=dataset, batch_size=2048,
                        shuffle=False, num_workers=2)

    logger = open(osp.join(save_path,'loss_history'),'w')

    for epoch in range(1, args.max_epoch + 1):
        qdar = tqdm.tqdm(enumerate(loader, 1),
                                    total= len(loader),
                                    ascii=True)
        ep_loss = 0
        for batch_id, batch in qdar:
            data, label = batch 
            data = data.cuda()
            gcn.train()
            output_vectors = gcn(word_vectors)

            # 改变loss
            loss = stach_l2_loss(output_vectors, data, label, wnid2index)
            # loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_data = loss.data.cpu().numpy()
            qdar.set_postfix(loss=str(np.round(loss_data,3)))
            ep_loss += loss_data
        # gcn.eval()
        # output_vectors = gcn(word_vectors)
        # train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train]).item()
        # if v_val > 0:
        #     val_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[n_train:]).item()
        #     loss = val_loss
        # else:
        #     val_loss = 0
        #     loss = train_loss
        # print('epoch {}, train_loss={:.4f}'
        #       .format(epoch, loss.data.cpu().numpy()))

        # trlog['train_loss'].append(train_loss)
        # trlog['val_loss'].append(val_loss)
        # trlog['min_loss'] = min_loss
        # torch.save(trlog, osp.join(save_path, 'trlog'))

        # if (epoch == args.save_epoch):
        ep_loss = ep_loss/len(loader)
        print('Epoch average loss: '+str(ep_loss))
        logger.write(str(np.round(ep_loss,3))+'\n')
        if args.no_pred:
            pred_obj = None
        else:
            pred_obj = {
                'wnids': wnids,
                'pred': output_vectors
            }

        # if epoch == args.save_epoch:
        save_checkpoint('epoch-{}'.format(epoch))
        
        pred_obj = None

