import os
import gc
import torch
import argparse
import librosa
import matplotlib
import numpy as np
from collections import Counter
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from c3d import *
from dataset import *
#from utils import progress_bar

import matplotlib.pyplot as plt
import matplotlib

parser = argparse.ArgumentParser(description='PyTorch Accident Detection')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') 
parser.add_argument('--batch_size', default=1, type=int) 
parser.add_argument('--epochs', '-e', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--preparedata', type=int, default=1)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.BCELoss()

print('==> Creating networks..')
model = C3D().to(device)
model.load_state_dict(torch.load('./checkpoints_orig/networkc3d_train_batch_500.ckpt'))
params = model.parameters()
optimizer = optim.Adam(params, lr=args.lr)  #, weight_decay=1e-4)

print('==> Loading data..')
trainset = AccidentDatasetC3D()

def train_accident(currepoch, epoch):
    dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)
    print('\n=> Epoch: %d' % currepoch)
    
    train_loss, dot_product, total, xnor = 0, 0, 0, 0
    best = 0

    for batch_idx in range(len(dataloader)):
        inputs, targets = next(dataloader)
        inputs = inputs / 255.0
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        y_pred = model(inputs)
        loss = criterion(y_pred, targets)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)

        y_pred_thresh = Variable(torch.Tensor([0.75]))
        y_pred_bin = (y_pred > y_pred_thresh.to(device)).float() * 1.0
        xnor += torch.sum((y_pred_bin == targets).float() * 1.0) / (targets.size(0) * targets.size(1))
        #dot_product += torch.sum(torch.mm(y_pred_bin, torch.transpose(targets, 0, 1))) / torch.sum(targets)

        with open("./logs/accidentc3d_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))
        with open("./logs/accidentc3d_train_xnor.log", "a+") as lfile:
            lfile.write("{}\n".format(xnor / total))

        if(batch_idx % 100 == 0):
            print(y_pred_bin[0])
            print(targets[0])

        del inputs
        del targets
        gc.collect()
        torch.cuda.empty_cache()
        # torch.save(model.state_dict(), './weights/networkc3d_train.ckpt')

        if(batch_idx % 100 == 0):
            torch.save(model.state_dict(), './checkpoints/networkc3d_train_batch_{}.ckpt'.format(batch_idx))

        print('Batch: [%d/%d], Loss: %.3f, Train Loss: %.3f, XNOR: %.6f ' % (batch_idx, len(dataloader), loss.item(), train_loss/(batch_idx+1), xnor / total))

    print('\n=> C3D Network : Epoch [{}/{}], Loss:{:.4f}'.format(currepoch+1, epoch, train_loss / len(dataloader)))

print('==> Training starts..')
for epoch in range(args.epochs):
    train_accident(epoch, args.epochs)
   
