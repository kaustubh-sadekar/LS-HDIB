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


def gray_mse(out_hat, out):
    loss = torch.mean((out_hat - out)**2)
    return loss

rgb_bcel = nn.BCELoss()