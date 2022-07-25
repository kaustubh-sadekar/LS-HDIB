from __future__ import print_function
from matplotlib import pyplot as plt
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset,DataLoader
import cv2
import numpy as np
import glob
from torch.autograd import Variable
from tqdm import tqdm
import natsort

class DatasetLSHDIB(Dataset):

    def __init__(self,input_path,gt_path,use_fraction=1,transform=None):
        self.input_path = input_path
        self.gt_path = gt_path
        self.ids=[]
        self.transform = transform
        self.len = len(glob.glob(self.input_path+"/*"))
        print("Number of images = ",self.len)
        self.ids = range(1, self.len+1)
        self.data = np.array(self.ids)
        np.random.shuffle(self.data)

    def __len__(self):
        return self.len
    
    def __getitem__(self,idx):
        x = cv2.imread(self.input_path+"/input_%d.jpg"%self.data[idx])
        y = cv2.imread(self.gt_path+"/mask_%d.jpg"%self.data[idx],0)
        x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x,y
