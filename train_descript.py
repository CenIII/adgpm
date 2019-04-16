
from datasets.imagenet_train import ImageNetFeatsTrain
# import linear net
# from models.LinearModel import LinearModel
# import lstmEnc
from models.LSTMEncoder import EncoderRNN
# import similarity loss

from models.SimilarityLoss import SimilarityLoss
import numpy as np

import json
import torch
import pickle
import tqdm
from torch.utils.data import DataLoader
import os
import argparse


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def makeInp(*inps):
	ret = []
	for inp in inps:
		ret.append(inp[0].to(device))
	return ret
def saveStateDict(lstmEnc,savepath):
		# models = {}
		# models['linNet'] = linNet.state_dict()
		# models['lstmEnc'] = lstmEnc.state_dict()
		torch.save(lstmEnc.state_dict(),os.path.join(savepath ,'lstmEnc.pt'))
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

def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument('-e','--evaluate_mode',
		action='store_true',
	  	help='check similarity matrix.')
	parser.add_argument('-p','--model_path',
		default='./lstmEnc.pt')
	parser.add_argument('-s','--save_path',
		default='./save/default/')
	parser.add_argument('-b','--batch_imgs',
		default=4, type=int)
	parser.add_argument('-i','--ind',
		default=100, type=int)
	args = parser.parse_args()
	return args

def reverseWord2Ind(word2ind):
	ind2word = {}
	for k,v in word2ind.items():
		ind2word[v] = k
	return ind2word

def decodeText(indarray,ind2word):
	indarray = list(indarray.data.cpu().numpy()[0])
	tokens = []
	for ind in indarray:
		tokens.append(ind2word[ind])
	return tokens


if __name__ == '__main__':
	args = parseArgs()
	os.makedirs(args.save_path,exist_ok=True)
	# graph = json.load(open('materials/imagenet-induced-graph.json', 'r'))
	# wnids = graph['wnids']
	# n = len(wnids)
	# edges = graph['edges']
	
	#################
	test_sets = json.load(open('materials/imagenet-testsets.json', 'r'))
	train_wnids = test_sets['train']
	# test_wnids = test_sets['2-hops']
	

	# todo: load vocab data
	with open('./materials/desc_vocabs.pkl','rb') as f:
		vocab = pickle.load(f)
	wordembs = vocab['word_embs']
	word2ind = vocab['word_dict']
	ind2word = reverseWord2Ind(word2ind)

	# init models
	lstmEnc = EncoderRNN(len(wordembs), 82, 1024, 300,
	                 input_dropout_p=0, dropout_p=0,
	                 n_layers=1, bidirectional=False, rnn_cell='lstm', variable_lengths=True,
	                 embedding_parameter=wordembs, update_embedding=False).to(device)
	optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, lstmEnc.parameters())), 0.0005)
	# todo: crit
	crit = SimilarityLoss(0.5,0.5,1).to(device)
	# todo: loader
	loader = ImageNetFeatsTrain('./materials/datasets/imagenet_feats/', train_wnids)
	
	if not args.evaluate_mode:

		# loader = DataLoader(dataset=dataset, batch_size=1,
		# 					shuffle=True, num_workers=2)

		for epoch in range(1, 100):
			indsList = list(range(len(loader)))
			random.shuffle(indsList)
			# ld = iter(loader)
			qdar = tqdm.tqdm(range(len(loader)),total=len(loader),ascii=True)
			ep_loss = 0
			for batch_id in qdar:
				batch = loader.__getitem__(indsList[batch_id])
				feats, texts, lengths = makeInp(*batch)   # feats: (numClasses, imPerClass, 2048, 1, 1) , texts: (numClasses, maxLens), lengths
				
				out2 = lstmEnc(texts,input_lengths=lengths)
				loss = crit(feats,out2,lengths)

				optimizer.zero_grad()
				loss.backward()
				# torch.nn.utils.clip_grad_norm_(linNet.parameters(),1.)
				optimizer.step()

				loss_data = loss.data.cpu().numpy()
				qdar.set_postfix(loss=str(np.round(loss_data,3)),shuffleCnt=np.sum(loader.getShuffleCnt()),
									maxCnt=loader.getMaxWnidCnt())
				ep_loss += loss_data
				if(batch_id>0 and batch_id%500==0):
					saveStateDict(lstmEnc,args.save_path)
			
			ep_loss = ep_loss/len(loader)
			print('Epoch average loss: '+str(ep_loss))

			saveStateDict(lstmEnc,args.save_path)
	else:
		lstmEnc = reloadModel(args.model_path, lstmEnc)
		feat,text,length = makeInp(*dataset.getOnePair(args.ind))
		print(decodeText(text,ind2word))
		out2 = lstmEnc(text,input_lengths=length)
		loss = crit.output_att(feat,out2,length)


