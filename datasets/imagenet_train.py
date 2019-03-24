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


class ImageNetFeatsTrain(Dataset):

    def __init__(self, path): # imagenet_feats
        self.path = path
        self.wnid_list = os.listdir(self.path)
        self.npyfile_list = []
        for wnid in self.wnid_list:
            wnid_path = os.path.join(self.path,wnid)
            npy_list = os.listdir(wnid_path)
            if 'feats.npy' in npy_list:
                npy_list.remove('feats.npy')
            for i in range(len(npy_list)):
                npy_list[i] = (os.path.join(wnid_path,npy_list[i]),wnid)
            self.npyfile_list.append(npy_list)  # path, wnid
    
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


    def __len__(self):
        return len(self.npyfile_list)

    def __getitem__(self, idx):
        npyfile, wnid = self.npyfile_list[idx]
        npy = np.load(npyfile)
        return npy, wnid


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

