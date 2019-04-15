
from datasets.imagenet_train import ImageNetFeatsTrain
# import linear net
# from models.LinearModel import LinearModel
# import lstmEnc
from models.LSTMEncoder import EncoderRNN
# import similarity loss

from models.SimilarityLoss import SimilarityLoss


import json

import pickle
import tqdm



if __name__ == '__main__':

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

	# init models
	lstmEnc = LSTMEncoder(len(wordembs), 82, 2048, 300,
	                 input_dropout_p=0, dropout_p=0,
	                 n_layers=1, bidirectional=False, rnn_cell='lstm', variable_lengths=True,
	                 embedding_parameter=wordembs, update_embedding=False)
	# todo: crit
	crit = SimilarityLoss(0.5,0.5,1)
	# todo: loader
	dataset = ImageNetFeatsTrain('./materials/datasets/imagenet_feats/', train_wnids)
	loader = DataLoader(dataset=dataset, batch_size=1,
						shuffle=True, num_workers=1)

	for epoch in range(1, 10):
		qdar = tqdm.tqdm(enumerate(loader, 1),
									total=6666,
									ascii=True)
		ep_loss = 0
		for batch_id, batch in qdar:
			feats, texts, lengths = batch   # feats: (numClasses, imPerClass, 2048, 1, 1) , texts: (numClasses, maxLens), lengths
			
			out2 = lstmEnc(texts,input_lengths=lengths)
			loss = crit(feats,out2,lengths)

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(linNet.parameters(),1.)
			optimizer.step()

			loss_data = loss.data.cpu().numpy()
			qdar.set_postfix(loss=str(np.round(loss_data,3)))
			ep_loss += loss_data
		
		ep_loss = ep_loss/len(loader)
		print('Epoch average loss: '+str(ep_loss))