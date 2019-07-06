from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
import os

class TLDataset(Dataset):
    def __init__(self,dataDir):
        super(TLDataset,self).__init__()
        imgs=[]

        dirNames=os.listdir(dataDir)
        for ele in dirNames:
            if(os.path.isdir(dataDir+ele)):
                dirImgs=[(dataDir+ele+'/'+x,ele) for x in os.listdir(dataDir+ele)]
                imgs+=dirImgs
        self.imgs=imgs

    def getLabel(self,cate):
        return {
            'Red Circle':      0,
            'Green Circle':    1,
            'Red Left':        2,
            'Green Left':      3,
            'Red Up':          4,
            'Green Up':        5,
            'Red Right':       6,
            'Green Right':     7,
            'Red Negative':    8,
            'Green Negative':  9
        }.get(cate,'error')

    def __getitem__(self, index):
        imgName,category=self.imgs[index]
        img=Image.open(imgName).convert('RGB')

        label=self.getLabel(category)
        if(label=='error'):
            print("************error label")
            exit(0)
        img=img.resize((32, 32))
        img=torch.from_numpy(np.array(img,dtype=float))
        img=img.type(torch.FloatTensor)
        return img,label

    def __len__(self):
        return len(self.imgs)

