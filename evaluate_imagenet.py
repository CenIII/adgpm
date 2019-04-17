import argparse
import json
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.resnet import make_resnet50_base
from datasets.imagenet import ImageNetFeats
from utils import set_gpu, pick_vectors

from models.SimilarityLoss import SimilarityLoss
from models.LSTMEncoder import EncoderRNN
import pickle
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def getLSTMOuts(test_wnids,lstmEnc):
    with open('./materials/desc_enc.pkl','rb') as f:
        descEnc = pickle.load(f)

    desc_wnid2ind = descEnc['wnid2ind']
    encoded_desc = descEnc['encoded_desc']
    lengths = descEnc['lengths']

    _, maxLen = encoded_desc.shape
    desc_encoded = np.zeros([len(test_wnids),maxLen])
    desc_lengths = np.zeros(len(test_wnids))
    for i in range(len(test_wnids)):
        wnid = test_wnids[i]
        desc_encoded[i] = encoded_desc[desc_wnid2ind[wnid]]
        desc_lengths[i] = lengths[desc_wnid2ind[wnid]]
    desc_encoded = torch.LongTensor(desc_encoded)
    desc_lengths = torch.LongTensor(desc_lengths).squeeze()
    inds = torch.argsort(-desc_lengths)
    desc_encoded = desc_encoded[inds].to(device)
    desc_lengths = desc_lengths[inds].to(device)
    lstmEnc.eval()
    outs = []
    for i in range(int(len(test_wnids)/32)+1):
        outs.append(lstmEnc(desc_encoded[32*i:32*(i+1)],input_lengths=desc_lengths[32*i:32*(i+1)]))
    outs = torch.cat(outs,dim=0)
    return outs, desc_lengths

def reloadModel(model_path,lstmEnc):
	pt = torch.load(model_path)

	def subload(model,pt_dict):
		model_dict = model.state_dict()
		pretrained_dict = {}
		for k, v in pt_dict.items():
			if(k in model_dict):
				pretrained_dict[k] = v
		# 2. overwrite entries in the existing state dict
		model_dict.update(pretrained_dict)
		# 3. load the new state dict
		model.load_state_dict(model_dict)
		return model

	lstmEnc = subload(lstmEnc,pt)
	
	return lstmEnc

def test_on_subset(dataset, cnn, n, pred_vectors, all_label,
                   consider_trains,lstmOuts=None,lstmLens=None,crit=None,rerankNum=10):
    top = [1, 2, 5, 10, 20]
    hits = torch.zeros(len(top)).to(device)
    tot = 0

    # loader = DataLoader(dataset=dataset, batch_size=32,
    #                     shuffle=False, num_workers=2)

    # for batch_id, batch in enumerate(loader, 1):
        # data, label = batch 
    data = dataset.to(device)

    feat = data #cnn(data) # (batch_size, d)
    # feat = torch.cat([feat, torch.ones(len(feat)).view(-1, 1).to(device)], dim=1)

    fcs = pred_vectors.t()

    table = torch.matmul(feat, fcs)
    if not consider_trains:
        table[:, :n] = -1e18


    # foor loop 每行，每行取前10个score的inds
    for i in range(len(table)):
        # 提取10个inds的lstm hiddens
        scores = table[i]
        topkInds = torch.topk(scores,rerankNum)[1]
        # 计算新的10个分数，+1e18
        topkScrs = []
        for ind in topkInds:
            t_hiddens = lstmOuts[ind]
            t_lens = lstmLens[ind]
            table[i][ind] = crit.generate_similarity_matrix(feat,t_hiddens,t_lens)


    gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
    rks = (table >= gth_score).sum(dim=1)

    assert (table[:, all_label] == gth_score[:, all_label]).min() == 1

    for i, k in enumerate(top):
        hits[i] += (rks <= k).sum().item()
    tot += len(data)

    return hits, tot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn')
    parser.add_argument('--pred')

    parser.add_argument('--test-set')

    parser.add_argument('--output', default=None)

    parser.add_argument('--gpu', default='0')

    parser.add_argument('--keep-ratio', type=float, default=1)
    parser.add_argument('--consider-trains', action='store_true')
    parser.add_argument('--test-train', action='store_true')
    parser.add_argument('-p','--model_path',
        default='./lstmEnc.pt')

    args = parser.parse_args()

    set_gpu(args.gpu)

    test_sets = json.load(open('materials/imagenet-testsets.json', 'r'))
    train_wnids = test_sets['train']
    test_wnids = test_sets[args.test_set]

    print('test set: {}, {} classes, ratio={}'
          .format(args.test_set, len(test_wnids), args.keep_ratio))
    print('consider train classifiers: {}'.format(args.consider_trains))

    pred_file = torch.load(args.pred)
    pred_wnids = pred_file['wnids']
    pred_vectors = pred_file['pred']
    pred_dic = dict(zip(pred_wnids, pred_vectors))
    pred_vectors = pick_vectors(pred_dic, train_wnids + test_wnids, is_tensor=True).to(device)

    pred_vectors = pred_vectors.to(device)

    n = len(train_wnids)
    m = len(test_wnids)
    
    cnn = make_resnet50_base()
    cnn.load_state_dict(torch.load(args.cnn))
    cnn = cnn.to(device)
    cnn.eval()

    TEST_TRAIN = args.test_train

    imagenet_path = 'materials/datasets/imagenet_feats'
    dataset = ImageNetFeats(imagenet_path)
    dataset.set_keep_ratio(args.keep_ratio)

    s_hits = torch.FloatTensor([0, 0, 0, 0, 0]).to(device) # top 1 2 5 10 20
    s_tot = 0

    results = {}

    if TEST_TRAIN:
        for i, wnid in enumerate(train_wnids, 1):
            subset = dataset.get_subset(wnid)  # return truncated matrix
            hits, tot = test_on_subset(subset, cnn, n, pred_vectors, i - 1,
                                       consider_trains=args.consider_trains)
            results[wnid] = (hits / tot).tolist()

            s_hits += hits
            s_tot += tot

            print('{}/{}, {}:'.format(i, len(train_wnids), wnid), end=' ')
            for i in range(len(hits)):
                print('{:.0f}%({:.2f}%)'
                      .format(hits[i] / tot * 100, s_hits[i] / s_tot * 100), end=' ')
            print('x{}({})'.format(tot, s_tot))
    else:
        count = 0
        with open('./materials/desc_vocabs.pkl','rb') as f:
            vocab = pickle.load(f)
        wordembs = vocab['word_embs']

        lstmEnc = EncoderRNN(len(wordembs), 82, 1024, 300,
                     input_dropout_p=0, dropout_p=0,
                     n_layers=1, bidirectional=False, rnn_cell='lstm', variable_lengths=True,
                     embedding_parameter=wordembs, update_embedding=False).to(device)
        lstmEnc = reloadModel(args.model_path, lstmEnc)

        lstmOuts, lstmLens = getLSTMOuts(test_wnids,lstmEnc)
        
        crit = SimilarityLoss(0.5,0.5,1).to(device)

        for i, wnid in enumerate(test_wnids, 1):
            subset = dataset.get_subset(wnid)
            hits, tot = test_on_subset(subset, cnn, n, pred_vectors, n + i - 1,
                                       consider_trains=args.consider_trains,lstmOuts=lstmOuts,lstmLens=lstmLens,crit=crit)
            results[wnid] = (hits / tot).tolist()

            s_hits += hits
            s_tot += tot

            print('{}/{}, {}:'.format(i, len(test_wnids), wnid), end=' ')
            for i in range(len(hits)):
                print('{:.0f}%({:.2f}%)'
                      .format(hits[i] / tot * 100, s_hits[i] / s_tot * 100), end=' ')
            print('x{}({})'.format(tot, s_tot))

    print('summary:', end=' ')
    for s_hit in s_hits:
        print('{:.2f}%'.format(s_hit / s_tot * 100), end=' ')
    print('total {}'.format(s_tot))

    if args.output is not None:
        json.dump(results, open(args.output, 'w'))

