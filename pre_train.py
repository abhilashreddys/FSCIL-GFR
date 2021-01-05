from __future__ import absolute_import, print_function
import argparse
import getpass
import os
import sys
import time
import random
import torch.utils.data
import pdb
import torch
from torch import nn
import torchvision 
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch.utils.data	
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.autograd as autograd
import scipy.io as sio

import models
from models.resnet import Generator, Discriminator, ClassifierMLP
from utils import mkdir_if_missing, logging, display
# from evaluations import extract_features, pairwise_distance
from dataload import miniimagenet
def load_my_state_dict(model, state_dict):
    count = 0
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        else:
        # if isinstance(param, torch.nn.parameter.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
            own_state[name].copy_(param)
            count += 1
    print(count)

model = models.create('resnet18_imagenet', pretrained=False, feat_dim=512,embed_dim=100,hidden_dim=256,norm=True)
resnet18 = torchvision.models.resnet18(pretrained=False)
PATH1 = 'resnet18-5c106cde.pth'
resnet18.load_state_dict(torch.load(PATH1))
#loading and freezing
load_my_state_dict(model, resnet18.state_dict())
#ct = 0
#lt = 8
#for name,child in model.named_children():
#    if ct < lt:
#        for param in child.parameters():
#            param.requires_grad = False
#        ct += 1

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #filter(lambda p: p.requires_grad, model.parameters())
scheduler = StepLR(optimizer, step_size=20, gamma=10)


# optimizer.step()
# scheduler.step()
global_step = 0
np.random.seed(2001)
num_classes = 100
nb_cl_fg = 60
random_perm = list(range(num_classes))
traindir = os.path.join('home/abhilash/trial/','miniimagenet')
trainfolder = miniimagenet('miniimagenet', mode='train', resize=84, cls_index=random_perm)#[:nb_cl_fg]
log_dir = 'checkpoints'
sys.stdout = logging.Logger(os.path.join(log_dir, 'pre_train.txt'))
tb_writer = SummaryWriter(log_dir)
batchsize = 16
train_loader = torch.utils.data.DataLoader(
                trainfolder, batch_size=batchsize,
                shuffle=True,
                drop_last=True, num_workers=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
epochs = 100
print("Training Base Classes")
print("Total No. of classes: ",num_classes)
print("Number of Base Classes: ",nb_cl_fg)
print("Batch Size: ",batchsize)
print("No. of Epochs: ",epochs)
writer = SummaryWriter()
model = model.to(device)
resnet18 = resnet18.to(device)
for epoch in range(epochs):
    # correct = 0
    # global global_step
    start = time.time()
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    for step,(x,y) in enumerate(train_loader):
        global_step += 1
        x,y = x.to(device),y.to(device)
        with autograd.detect_anomaly():
            #embed_feat = model(x)
            #soft_feat = model.embed(embed_feat)
            #loss = torch.nn.CrossEntropyLoss()(soft_feat, y)
            #print(loss.item())
            out = resnet18(x)
            loss = torch.nn.CrossEntropyLoss()(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        pred = F.softmax(out, dim=1).argmax(dim=1)
        correct = torch.eq(pred, y).sum().item()
        #num = x.size[0]
        loss_ = loss.item()
        accuracy = correct/batchsize
        loss_meter.update(loss_, batchsize)
        accuracy_meter.update(accuracy, batchsize)
        writer.add_scalar('Train/RunningLoss', loss_, global_step)
        writer.add_scalar('Train/RunningAccuracy', accuracy, global_step)
    print('Epoch {}'
        'Loss Avg ({:.4f}) '
        'Accuracy Avg ({:.4f})'.format(
        epoch,
        len(train_loader),
        loss_meter.avg,
        accuracy_meter.avg))
    elapsed = time.time() - start
    writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/Accuracy', accuracy_meter.avg, epoch)
    writer.add_scalar('Train/Time', elapsed, epoch)

torch.save(model.state_dict(), 'pre_train.pth')