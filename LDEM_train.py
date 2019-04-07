import argparse
import json
import random
import os.path as osp

import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import ensure_path, set_gpu, l2_loss
# from models.gcn import GCN
from datasets.imagenet_train import ImageNetFeatsTrain
from torch.utils.data import DataLoader
import tqdm
import numpy as np

def save_checkpoint(name):
    torch.save(ldem.state_dict(), osp.join(save_path, name + '.pth'))
    torch.save(pred_obj, osp.join(save_path, name + '.pred'))


# def mask_l2_loss(a, b, mask):
#     return l2_loss(a[mask], b[mask])


# todo: 先select inputs，再batch output
def stach_l2_loss(word_vectors, ldem, data, label, wnid2index):
    inds = torch.tensor([wnid2index[x] for x in label]).cuda()
    slct_wvs = torch.index_select(word_vectors, 0, inds)
    preds = ldem(slct_wvs)
    loss = ((preds - data)**2).sum()/(len(inds)*2)
    return loss 


class LDEM(nn.Module):
    """docstring for LDEM"""
    def __init__(self, inpdim, outdim):
        super(LDEM, self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(inpdim,700),
                    nn.ReLU(),
                    nn.Linear(700,outdim),
                    nn.ReLU()
                    )
        self.init_weights()

    def init_weights(self):
        self.net[0].weight.data.normal_(0.2)
        self.net[0].bias.data.fill_(0)
        self.net[2].weight.data.normal_(0.2)
        self.net[2].bias.data.fill_(0)
        return None

    def forward(self,x):
        output = self.net(x)
        return output
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epoch', type=int, default=5000)
    parser.add_argument('--trainval', default='10,0')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--save-epoch', type=int, default=3000)
    parser.add_argument('--save-path', default='save/gcn-basic')
    parser.add_argument('--semantic-embs', default='save/gae_C/epoch-3010.pred')
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
    # n = len(wnids)
    # edges = graph['edges']
    
    # edges = edges + [(v, u) for (u, v) in edges]
    # edges = edges + [(u, u) for u in range(n)]

    # word_vectors = torch.tensor(graph['vectors']).cuda()
    # word_vectors = F.normalize(word_vectors)

    # todo: 将word_vector换为512维learned embeddings
    pred_file = torch.load(args.semantic_embs)
    pred_wnids = pred_file['wnids']
    word_vectors = pred_file['pred']
    word_vectors.requires_grad = False
    word_vectors = word_vectors.cuda()
    word_vectors = F.normalize(word_vectors)


    fcfile = json.load(open('materials/fc-weights.json', 'r'))
    train_wnids = [x[0] for x in fcfile]
    fc_vectors = [x[1] for x in fcfile]
    assert train_wnids == wnids[:len(train_wnids)]
    fc_vectors = torch.tensor(fc_vectors).cuda()
    fc_vectors = F.normalize(fc_vectors)

    # hidden_layers = args.layers #'d2048,d' #'2048,2048,1024,1024,d512,d'

    # todo: 换为2layer linear+relu
    ldem = LDEM(word_vectors.shape[1], fc_vectors.shape[1]).cuda()

    # print('{} nodes, {} edges'.format(n, len(edges)))
    print('word vectors:', word_vectors.shape)
    print('fc vectors:', fc_vectors.shape)
    # print('hidden layers:', hidden_layers)

    optimizer = torch.optim.Adam(ldem.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

    dataset = ImageNetFeatsTrain('./materials/datasets/imagenet_feats/', train_wnids)
    loader = DataLoader(dataset=dataset, batch_size=100,
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
            ldem.train()

            # output_vectors = ldem(word_vectors)

            loss = stach_l2_loss(word_vectors, ldem, data, label, wnid2index)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ldem.parameters(),1.)
            optimizer.step()
            loss_data = loss.data.cpu().numpy()
            qdar.set_postfix(loss=str(np.round(loss_data,3)))
            ep_loss += loss_data
        
        ep_loss = ep_loss/len(loader)
        print('Epoch average loss: '+str(ep_loss))
        logger.write(str(np.round(ep_loss,3))+'\n')
        if args.no_pred:
            pred_obj = None
        else:
            output_vectors = ldem(word_vectors)
            pred_obj = {
                'wnids': wnids,
                'pred': output_vectors
            }

        if epoch>0 and epoch % args.save_epoch==0:
            save_checkpoint('epoch-{}'.format(epoch))
        
        pred_obj = None

