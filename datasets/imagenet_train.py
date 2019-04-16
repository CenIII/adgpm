import json
import os
import os.path as osp
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image
from torchvision import get_image_backend
import numpy as np
import pickle

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

class ImageNetFeatsTrain(Dataset):

    def __init__(self, path, train_wnids): # imagenet_feats
        self.path = path
        self.wnid_list = train_wnids #os.listdir(self.path)
        # self.npyfile_list = []

        # with open('./save/npyfile_list.pkl','rb') as f:
        #     self.npyfile_list = pickle.load(f)
        
        self.wnid_feats_list = {i:[] for i in range(len(self.wnid_list))}
        # with open('./materials/npyfile_list.pkl','rb') as f:
        #     self.wnid_feats_list = pickle.load(f)

        
        for j in range(len(self.wnid_list)):
            wnid = self.wnid_list[j]
            wnid_path = os.path.join(self.path,wnid)
            npy_list = os.listdir(wnid_path)
            if 'feats.npy' in npy_list:
                npy_list.remove('feats.npy')
            for i in range(len(npy_list)):
                npy_list[i] = os.path.join(wnid_path,npy_list[i])
            self.wnid_feats_list[j] = npy_list  
        # presList = self.removeEmptyInds(self.wnid_feats_list)
        self.wnid_cnt = [0 for i in range(len(self.wnid_list))]
        self.shuffleCnt = [0 for i in range(len(self.wnid_list))]

        # print('dump done.')
        # load A_pred_0
        A_pred = torch.load('./materials/A_pred_0.pt').cpu().detach().numpy()[:1000,:1000]
        # A_pred = A_pred[presList,:][:,presList]
        np.fill_diagonal(A_pred, 0.)
        self.probMat = softmax(A_pred,theta=2.0,axis=1)
        with open('./materials/desc_enc.pkl','rb') as f:
            descEnc = pickle.load(f)

        desc_wnid2ind = descEnc['wnid2ind']
        encoded_desc = descEnc['encoded_desc']
        lengths = descEnc['lengths']

        _, maxLen = encoded_desc.shape
        self.maxLen = maxLen
        self.desc_encoded = np.zeros([len(self.wnid_list),maxLen])
        self.desc_lengths = np.zeros(len(self.wnid_list))
        for i in range(len(self.wnid_list)):
            wnid = self.wnid_list[i]
            self.desc_encoded[i] = encoded_desc[desc_wnid2ind[wnid]]
            self.desc_lengths[i] = lengths[desc_wnid2ind[wnid]]

    def getShuffleCnt(self):
        return self.shuffleCnt

    def getMaxWnidCnt(self):
        return max(self.wnid_cnt)

    def removeEmptyInds(self,wnid_feats_list):
        emptyWnidList = []
        preserveList = []
        for k,v in wnid_feats_list.items():
            if len(v)==0:
                emptyWnidList.append(self.wnid_list[k])
            else:
                preserveList.append(k)
        for wnid in emptyWnidList:
            print('remove empty: '+str(wnid))
            self.wnid_list.remove(wnid)
        newFeatDict = {}
        for i in range(len(preserveList)):
            newFeatDict[i] = wnid_feats_list[preserveList[i]]
        self.wnid_feats_list = newFeatDict
        return preserveList
    # def resetNShuffle(self):
    #     self.wnid_cnt = {self.wnid_list[i]:0 for i in range(len(self.wnid_list))}
    #     for k,v in self.wnid_feats_list.item():
    #         random.shuffle(self.wnid_feats_list[k])
    

    # def get_subset(self, wnid):
    #     if wnid not in self.wnid_list:
    #         print("Subset "+str(wnid)+" not exist.")
    #         return None

    #     path = osp.join(osp.join(self.path, wnid),'feats.npy')
    #     feats = np.load(path)
    #     keeplen = max(int(len(feats)*self.keep_ratio),1)
    #     feats = feats[:keeplen]
    #     feats = torch.tensor(feats)
    #     return feats #ImageNetFeatsSubset(path, wnid, keep_ratio=self.keep_ratio)
    def getOnePair(self,ind):
        npylist = self.wnid_feats_list[ind]
        npy = np.zeros([1,1,2049,1,1])
        cnt = int(np.random.uniform(0,len(npylist),1))
        npy[0,0,:,0,0] = np.load(npylist[cnt])
        label = np.zeros([1,self.maxLen])
        length = np.zeros(1)
        label[0] = self.desc_encoded[ind]
        length[0] = self.desc_lengths[ind]
        npy = torch.FloatTensor(npy)
        label = torch.LongTensor(label)
        length = torch.LongTensor(length)
        return [npy], [label], [length]

    def sampleClasses(self,ind):
        probvec = self.probMat[ind]
        sampled = np.random.choice(len(self.wnid_list),30,p=probvec,replace=False)
        return sampled

    def sampleFeatsforOneWnid(self,ind):
        cnt = self.wnid_cnt[ind]
        npylist = self.wnid_feats_list[ind]
        npy = np.zeros([8,2049,1,1])
        for i in range(8):
            if cnt >= len(npylist):
                print('cnt: '+str(cnt)+' len npylist: '+str(len(npylist))+' wnid: '+str(self.wnid_list[ind]))
            npy[i,:,0,0] = np.load(npylist[cnt])
            cnt = cnt+1
            if cnt >= len(npylist):
                random.shuffle(self.wnid_feats_list[ind])
                self.shuffleCnt[ind] += 1
                cnt = 0
        self.wnid_cnt[ind] = cnt
        return npy

    def __len__(self):
        return len(self.wnid_list)

    def __getitem__(self, idx):
        # for idx class (1~1000), find by A_pred_0 probability 30 classes, draw 5 feats from each class, reshape it to (30,5,2048,1,1)
        # sample 30 classes by self.A_pred
        npy = np.zeros([30,8,2049,1,1])
        labels = np.zeros([30,self.maxLen])
        lengths = np.zeros(30)
        classids = self.sampleClasses(idx)
        for i in range(30):
            ind = classids[i]
            npy[i] = self.sampleFeatsforOneWnid(ind)
            labels[i] = self.desc_encoded[ind]
            lengths[i] = self.desc_lengths[ind]

        npy = torch.FloatTensor(npy)
        labels = torch.LongTensor(labels)
        lengths = torch.LongTensor(lengths).squeeze()
        inds = torch.argsort(-lengths)
        npy = npy[inds]
        labels = labels[inds]
        lengths = lengths[inds]
        return npy, labels, lengths


# class ImageNetFeatsSubset(Dataset):

#     def __init__(self, path, wnid, keep_ratio=1.0):
#         self.wnid = wnid

#         def default_loader(path):
#             try:
#                 img = np.load(f) #Image.open(f)
#             except OSError:
#                 return None
#             return img#.convert('RGB')

#         # get file list
#         try:
#             all_files = os.listdir(path)
#         except:
#             print(path+" does not exist.")
#             self.data=[]
#             return
#         files = []
#         for f in all_files:
#             if f.endswith('.npy'):
#                 files.append(f)
#         random.shuffle(files)
#         files = files[:max(1, round(len(files) * keep_ratio))]

#         # read images
#         data = []
#         for filename in files:
#             image = default_loader(osp.join(path, filename))
#             if image is None:
#                 continue
#             # pytorch model-zoo pre-process
#             # preprocess = transforms.Compose([
#             #     transforms.Resize(256),
#             #     transforms.CenterCrop(224),
#             #     transforms.ToTensor(),
#             #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#             #                          std=[0.229, 0.224, 0.225])
#             # ])
#             # data.append(preprocess(image))
#             data.append(image)
#         if data != []:
#             self.data = torch.stack(data) 
#         else:
#             self.data = []

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx], self.wnid

