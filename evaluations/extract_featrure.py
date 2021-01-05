from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
from copy import deepcopy

import torch
from utils import to_numpy
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F

# from .evaluation_metrics import cmc, mean_ap
from utils.meters import AverageMeter
from .cnn import extract_cnn_feature, extract_cnn_feature_classification
import pdb

def extract_features(net, data_loader,update_step_test=5, print_freq=1, update_lr=0.01, metric=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deepcopy(net)
    model = model.to(device)
    model.eval()
    labels = []
    features = []
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(data_loader, 0):
        BatchSize, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        for i in range(BatchSize):
            for k in range(1, update_step_test):
                embed_feat = model(x_spt[i])
                soft_feat = model.embed(embed_feat)
                loss = F.cross_entropy(soft_feat, y_spt[i])
                grad = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
                for e,param in enumerate(model.parameters(),0):
                    param.data -= update_lr * grad[e]
            embed_feat_q = model(x_qry[i])
            embed_feat_q = embed_feat_q.cpu().detach().numpy()
            if features == []:
                features = embed_feat_q
                labels = y_qry[i].cpu().numpy()
            else:
                features = np.vstack((features, embed_feat_q))
                labels = np.hstack((labels, y_qry[i].cpu().numpy()))
    return features, labels


def extract_features_classification(net, data_loader,update_step_test=5, print_freq=1, update_lr=0.01, metric=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deepcopy(net)
    model = model.to(device)
    model.eval()
    labels = []
    features = []
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(data_loader, 0):
        BatchSize, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        for i in range(BatchSize):
            for k in range(1, update_step_test):
                embed_feat = model(x_spt[i])
                soft_feat = model.embed(embed_feat)
                loss = F.cross_entropy(soft_feat, y_spt[i])
                grad = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
                for e,param in enumerate(model.parameters(),0):
                    param.data -= update_lr * grad[e]
            embed_feat_q = model(x_qry[i])
            soft_feat_q = model.embed(embed_feat_q)
            soft_feat_q = soft_feat_q.cpu().detach().numpy()
            # pred_q = F.softmax(soft_feat_q, dim=1).argmax(dim=1)
            # pred_q = pred_q.cpu().detach().numpy()
            if features == []:
                features = soft_feat_q
                labels = y_qry[i].cpu().numpy()
            else:
                features = np.vstack((features, soft_feat_q))
                labels = np.hstack((labels, y_qry[i].cpu().numpy()))
    return features, labels


def pairwise_distance(features, metric=None):
    n = len(features)
    x = torch.cat(features)
    x = x.view(n, -1)
    # print(4*'\n', x.size())
    if metric is not None:
        x = metric.transform(x)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True)
    # print(dist.size())
    dist = dist.expand(n, n)
    dist = dist + dist.t()
    dist = dist - 2 * torch.mm(x, x.t()) + 1e5 * torch.eye(n)
    dist = torch.sqrt(dist)
    return dist


def pairwise_similarity(features):
    n = len(features)
    x = torch.cat(features)
    x = x.view(n, -1)
    # print(4*'\n', x.size())
    similarity = torch.mm(x, x.t()) - 1e5 * torch.eye(n)
    return similarity

#
# features = torch.round(2*torch.rand(4, 2))
# print(features)
# distmat = pairwise_similarity(features)
# distmat = to_numpy(distmat)
# indices = np.argsort(distmat, axis=1)
# print(distmat)
# print(indices)
