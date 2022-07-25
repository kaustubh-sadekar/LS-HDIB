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
import segmentation_models_pytorch as smp
from datasets import DatasetLSHDIB
from losses import gray_mse, rgb_bcel
from tqdm import tqdm



def train(input_path, gt_path, output_path, weigths_path, start_epoch, start_iter=0):

    batch_size = 8
    learning_rate = 0.0001
    epochs = 30

    train_data = DatasetLSHDIB(input_path+'train',gt_path+'train',transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=False)

    val_data = DatasetLSHDIB(input_path+'val',gt_path+'val',transform=transforms.ToTensor())
    val_loader = DataLoader(val_data, batch_size=batch_size,shuffle=True)
    
    model = smp.Unet(in_channels=3, classes=1, activation="sigmoid")

    if start_iter!=0:
        model.load_state_dict(torch.load(weigths_path+'iter%d_epoch%d.pth'%(start_iter,start_epoch)))
    if start_iter==0 and start_epoch!=0:
        wp_ = glob.glob(weigths_path+'*_epoch%d.pth'%start_epoch)
        wp_.sort()
        model.load_state_dict(torch.load(wp_[-1]))
    
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = []
    val_loss = []

    for e in range(epochs):

        temp_loss = []

        if e==start_epoch and start_iter!=0:
            ary = np.load(output_path+'epoch%d_loss_data.npy'%e)
            temp_loss = ary.tolist()
        

        model.train()

        if e >= start_epoch:
            model.train()
            for i, data in enumerate(tqdm(train_loader)):
                if e == start_epoch and i<start_iter:
                    continue
                x,y = data

                x = Variable(x).to(device)
                y = Variable(y).to(device)

                y_hat = model(x)
                # print(y_hat.shape)
                # print(y.shape)
                loss = rgb_bcel(y_hat, y)
                # loss = gray_mse(y_hat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("Epoch %d Batch %d Loss : "%(e,i),loss.item())
                temp_loss.append(loss.item())

                if i%100==0:
                    print("Saving an image")
                    print("Epoch ",e)
                    print("Loss ",loss.item())
                    out_img = y_hat.cpu().data
                    # vutils.save_image(y,output_path+'GT_%d_epoch_%d.jpg'%(i,e),normalize=False)
                    # vutils.save_image(x,output_path+'Input_%d_epoch_%d.jpg'%(i,e),normalize=False)
                    vutils.save_image(y,output_path+'Train_GT_%d.jpg'%i,normalize=False)
                    vutils.save_image(x,output_path+'Train_Input_%d.jpg'%i,normalize=False)
                    vutils.save_image(out_img,output_path+'Train_Pred_%d_epoch_%d.jpg'%(i,e),normalize=False)
                    torch.save(model.state_dict(), weigths_path+'iter%d_epoch%d.pth'%(i,e))
                    # title = 'PageNet Epoch%d loss per iteration'%e
                    # plt.figure()
                    # plt.plot(temp_loss,label="training loss")
                    # plt.title(title)
                    # plt.legend()
                    # plt.savefig(output_path+'epoch%d_loss_data.png'%e)
                    np.save(output_path+'epoch%d_loss_data.npy'%e, temp_loss)
            if i >400:
                exit(-1)
            train_loss.append(np.sum(temp_loss)/len(temp_loss))
        
            temp_loss = []
            model.eval()
            for i,data in enumerate(val_loader):
                x, y = data

                x = Variable(x).to(device)
                y = Variable(y).to(device)
                
                y_hat = model(x)
                loss = rgb_bcel(y_hat, y)*100

                print("Epoch %d Batch %d Validation Loss : "%(e,i),loss.item())
                temp_loss.append(loss.item())
                if i%50 == 0:
                    out_img = y_hat.cpu().data
                    vutils.save_image(y,output_path+'Val_GT_%d.jpg'%i,normalize=False)
                    vutils.save_image(x,output_path+'Val_Input_%d.jpg'%i,normalize=False)
                    vutils.save_image(out_img,output_path+'Val_Pred_%d_epoch_%d.jpg'%(i,e),normalize=False)
                # plt.plot(temp_loss,label="validation loss")
                # plt.title(title)
                # plt.legend()
                # plt.savefig(output_path+'lossData.png')
            val_loss.append(np.sum(temp_loss)/len(temp_loss))


if __name__ == "__main__":

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("Device used : ",device)

    cudnn.benchmark = True
    manualSeed = 43
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Modify your paths as per requirement
    input_path = 'DataSet1KHDIB/Input_'
    gt_path = 'DataSet1KHDIB/GT_'
    output_path = 'OutputUnet/' 
    weigths_path = 'WeightsUnet/'

    try:
        os.mkdir(output_path)
    except:
        print(output_path, " folder already exists")
    
    try:
        os.mkdir(weigths_path)
    except:
        print(weigths_path, " folder already exists")

    start_epoch = 0 #int(input("Enter the start_epoch value : "))
    start_iter = 0 #int(input("Enter the start iteration value : "))
    train(input_path, gt_path, output_path, weigths_path,start_epoch, start_iter=start_iter)
