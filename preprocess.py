import pickle
import argparse, os, json, string
from collections import Counter
import numpy as np
import gensim

with open('./materials/stopwords','r') as f:
	lines = f.readlines()
stopwords = []
for line in lines:
	stopwords.append(line[:-1])
print(stopwords)

def words_preprocess(phrase):
	""" preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
	replacements = {
		'½': 'half',
		'—' : '-',
		'™': '',
		'¢': 'cent',
		'ç': 'c',
		'û': 'u',
		'é': 'e',
		'°': ' degree',
		'è': 'e',
		'…': '',
		}
	for k, v in replacements.items():
		phrase = phrase.replace(k, v)
	return str(phrase).lower().translate(str.maketrans('','',string.punctuation)).split()

def loadWnidsDesc(filepath):
	with open(filepath,'r') as f:
		content = f.readlines()
	descDict = {}
	for line in content:
		wnid, desc = line.split('\t')
		descDict[wnid] = desc[:-1]
	return descDict

def preprocess(descDict):
	maxLen = 0
	for k,v in descDict.items():
		proc_v = words_preprocess(v)
		descDict[k] = proc_v
		maxLen = max(maxLen,len(proc_v))
	return descDict, maxLen

def build_word2inds(descDict,min_token_instances=8,verbose=True):
	token_counter = Counter()
	for k,v in descDict.items():
		token_counter.update(v)
	vocab = set()
	for token, count in token_counter.items():
		if count >= min_token_instances:
			vocab.add(token)
	if verbose:
		print(('Keeping %d / %d tokens with enough instances'
			  % (len(vocab), len(token_counter))))
	
	if len(vocab) < len(token_counter):
		vocab.add('<UNK>')
		if verbose:
			print('adding special <UNK> token.')
	else:
		if verbose: 
			print('no <UNK> token needed.')

	# build word2inds
	word2inds = {}
	next_idx = 1
	for token in vocab:
		word2inds[token] = next_idx
		next_idx = next_idx + 1
	return word2inds # start from 1, leave 0 to pad.

def encodeT(tokens,word2inds,maxLen,f):
	encoded = np.zeros(maxLen, dtype=np.int32)
	isallunk = 1
	tokens_for_observe = []
	for i, token in enumerate(tokens):
		if token in word2inds:
			if token not in stopwords:
				isallunk = 0
			encoded[i] = word2inds[token]
			tokens_for_observe.append(token)
		else:
			encoded[i] = word2inds['<UNK>']
			tokens_for_observe.append('<UNK>')
	f.write(' '.join(tokens_for_observe)+'\n')
	# if cnt==0:
	# 	print("all <unk> desc...")
	return encoded, isallunk

def encodeTexts(descDict, word2inds, maxLen):
	encoded_desc = np.zeros([len(descDict), maxLen])
	lengths = np.zeros(len(descDict))
	wnid2ind = {}
	cnt = 0
	allunk_cnt = 0
	f = open('./materials/tokens_for_observe','w')
	for k,v in descDict.items():
		wnid2ind[k] = cnt
		lengths[cnt] = len(v)
		encoded, allunk = encodeT(v,word2inds,maxLen,f)
		allunk_cnt += allunk
		encoded_desc[cnt] = encoded
		cnt += 1
	print('allunk_cnt: '+str(allunk_cnt))
	return wnid2ind,encoded_desc,lengths

# load wnids and text pairs
descDict = loadWnidsDesc('./materials/gloss.txt') 
# lower and filt
descDict, maxLen = preprocess(descDict)

# build vocab, preprocess low freq words
word2inds = build_word2inds(descDict)

# encode descDict to inds, return wnid2ind, wordinds matrix.
wnid2ind,encoded_desc,lengths = encodeTexts(descDict, word2inds, maxLen)


# save wnids and inds pairs
descEnc = {}
descEnc['wnid2ind'] = wnid2ind
descEnc['encoded_desc'] = encoded_desc
descEnc['lengths'] = lengths
with open('./materials/desc_enc.pkl','wb')as f:
	pickle.dump(descEnc,f)

# save inds2embeddings
print('start word2vec...')
model = gensim.models.KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

numwords = len(word2inds)

word2inds['<START>'] = numwords+1
word2inds['<END>'] = numwords+2
word2inds['<PAD>'] = 0

word2vec = np.zeros([numwords+3,300])

for word,idx in word2inds.items():
	if word in model.wv:
		word2vec[idx] = model.wv[word]
	else:
		word2vec[idx] = np.random.uniform(-1,1,300)

VocabData = {}
VocabData['word_dict'] = word2inds
VocabData['word_embs'] = word2vec

with open('./materials/desc_vocabs.pkl', 'wb') as f:
	pickle.dump(VocabData,f)

