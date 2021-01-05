# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import getpass
import os
import sys
import torch.utils.data
import pdb
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
from utils import RandomIdentitySampler, mkdir_if_missing, logging, display,truncated_z_sample
from evaluations import extract_features, pairwise_distance
from MiniImagenet import *

cudnn.benchmark = True
from copy import deepcopy

def to_binary(labels,args):
	# Y_onehot is used to generate one-hot encoding
    y_onehot = torch.FloatTensor(len(labels), args.num_class)
    y_onehot.zero_()
    y_onehot.scatter_(1, labels.cpu()[:,None], 1)
    code_binary = y_onehot.to(device)
    return code_binary

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return model

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def compute_gradient_penalty(D, real_samples, fake_samples, syn_label):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.cuda.FloatTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates, syn_label)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = \
        autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True,
                      retain_graph=True,
                      only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty

def clip_grad_by_norm_(grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

def compute_prototype(model, data_loader,number_samples=200,batch_size):
    model.eval()
    count = 0
    embeddings = []
    embeddings_labels = []
    terminate_flag = min(len(data_loader),number_samples)
    with torch.no_grad():
        for i, (x_spt, y_spt, x_qry, y_qry) in enumerate(data_loader):
            if i>terminate_flag:
                break
            count += 1
            for k in range(batch_size):
                inputs, labels = x_spt[k], y_spt[k]
                # wrap them in Variable
                inputs = Variable(inputs.to(device))
                embed_feat = model(inputs)
                embeddings_labels.append(labels.numpy())
                embeddings.append(embed_feat.cpu().numpy())

    embeddings = np.asarray(embeddings)
    embeddings = np.reshape(embeddings, (embeddings.shape[0] * embeddings.shape[1], embeddings.shape[2]))
    embeddings_labels = np.asarray(embeddings_labels)
    embeddings_labels = np.reshape(embeddings_labels, embeddings_labels.shape[0] * embeddings_labels.shape[1])
    labels_set = np.unique(embeddings_labels)
    class_mean = []
    class_std = []
    class_label = []
    for i in labels_set:
        ind_cl = np.where(i == embeddings_labels)[0]
        embeddings_tmp = embeddings[ind_cl]
        class_label.append(i)
        class_mean.append(np.mean(embeddings_tmp, axis=0))
        class_std.append(np.std(embeddings_tmp, axis=0))
    prototype = {'class_mean': class_mean, 'class_std': class_std,'class_label': class_label}

    return prototype
def fast_weights(grad,state_dict,update_lr):
    i = 0
    for key,value in state_dict.items():
        value -= update_lr * grad[i]
        i += 1
    return state_dict


def train_task(args, train_loader, current_task, prototype={}, pre_index=0):
    num_class_per_task = (args.num_class-args.nb_cl_fg) // args.num_task
    task_range = list(range(args.nb_cl_fg + (current_task - 1) * num_class_per_task, args.nb_cl_fg + current_task * num_class_per_task))
    if num_class_per_task==0:
        pass  # JT
    else:
        old_task_factor = args.nb_cl_fg // num_class_per_task + current_task - 1
    log_dir = os.path.join(args.ckpt_dir, args.log_dir)
    mkdir_if_missing(log_dir)

    sys.stdout = logging.Logger(os.path.join(log_dir, 'log_task{}.txt'.format(current_task)))
    tb_writer = SummaryWriter(log_dir)
    display(args)

    if 'miniimagenet' in args.data:
        model = models.create('resnet18_imagenet', pretrained=False, feat_dim=args.feat_dim,embed_dim=args.num_class)
    elif 'cifar' in args.data:
        model = models.create('resnet18_cifar', pretrained=False, feat_dim=args.feat_dim,embed_dim=args.num_class)

    # mlp = ClassifierMLP()

    if current_task > 0:
        model = torch.load(os.path.join(log_dir, 'task_' + str(current_task - 1).zfill(2) + '_%d_model.pkl' % int(args.epochs - 1)))
        model_old = deepcopy(model)
        model_old.eval()
        model_old = freeze_model(model_old)
        # mlp_old = deepcopy(mlp)
        # mlp_old.eval()
        # mlp_old = freeze_model(mlp_old)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.cuda()
    model = model.to(device)
    # mlp = mlp.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    # optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler_mlp = StepLR(optimizer_mlp, step_size=args.lr_decay_step, gamma=args.lr_decay)

    loss_mse = torch.nn.MSELoss(reduction='sum')

    # # Loss weight for gradient penalty used in W-GAN
    lambda_gp = args.lambda_gp
    lambda_lwf = args.gan_tradeoff
    # Initialize generator and discriminator
    if current_task == 0:
        generator = Generator(feat_dim=args.feat_dim,latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, class_dim=args.num_class)
        discriminator = Discriminator(feat_dim=args.feat_dim,hidden_dim=args.hidden_dim, class_dim=args.num_class)
    else:
        generator = torch.load(os.path.join(log_dir, 'task_' + str(current_task - 1).zfill(2) + '_%d_model_generator.pkl' % int(args.epochs_gan - 1)))
        discriminator = torch.load(os.path.join(log_dir, 'task_' + str(current_task - 1).zfill(2) + '_%d_model_discriminator.pkl' % int(args.epochs_gan - 1)))
        generator_old = deepcopy(generator)
        generator_old.eval()
        generator_old = freeze_model(generator_old)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor 

    # if args.learn_inner_lr:
    #         learned_lrs = []
    #         for i in range(args.update_steps):
    #             gen_lrs =[Variable(FloatTensor(1).fill_(args.update_lr), requires_grad=True)]*len(generator.parameters())
    #             # nway_lrs = [Variable(self.FloatTensor(1).fill_(self.update_lr), requires_grad=True)]*len(self.nway_net.parameters())
    #             discrim_lrs = [Variable(FloatTensor(1).fill_(args.update_lr), requires_grad=True)]*len(discriminator.parameters())

    #             learned_lrs.append((discrim_lrs, gen_lrs))

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    # optimizer_lr = torch.optim.Adam(learned_lrs, lr=args.gan_lr, betas=(0.5, 0.999))
    # scheduler_G = StepLR(optimizer_G, step_size=200, gamma=0.3)
    # scheduler_D = StepLR(optimizer_D, step_size=200, gamma=0.3)

    y_onehot = torch.FloatTensor(args.meta_batch_size, args.num_class)

    for p in generator.parameters():  # set requires_grad to False
        p.requires_grad = False

    if current_task>0:
        model = model.eval()

    for epoch in range(args.epochs):

        loss_log = {'C/loss': 0.0,
                    'C/loss_aug': 0.0,
                    'C/loss_cls': 0.0,
                    'C/loss_cls_q':0.0}
        scheduler.step()


##### MAML on feature extraction
		# db = DataLoader(mini, args.meta_batch_size, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(train_loader):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
			
			loss = torch.zeros(1).to(device)
            loss_cls = torch.zeros(1).to(device)
            loss_aug = torch.zeros(1).to(device)
            loss_tmp = torch.zeros(1).to(device)

			meta_batch_size, setsz, c_, h, w = x_spt.size()
            querysz = x_qry.size(1)

       	    losses_q = [0 for _ in range(args.update_step + 1)]  # losses_q[i] is the loss on step i
            corrects = [0 for _ in range(args.update_step + 1)]


            for i in range(args.meta_batch_size):
            	# 1. run the i-th task and compute loss for k=0
            	embed_feat = model(x_spt[i])

            	# $$$$$$$$$$$$$$$$
            	if current_task == 0:
            	    soft_feat = model.embed(embed_feat)
            	    # y_pred = mlp(soft_feat)
            	    loss_cls = torch.nn.CrossEntropyLoss()(soft_feat, y_spt[i])
            	    loss += loss_cls
            	else:                       
            	    embed_feat_old = model_old(x_spt[i]) 

            	### Feature Extractor Loss
            	if current_task > 0:                                    
            	    loss_aug = torch.dist(embed_feat, embed_feat_old , 2)  
            	    # loss_tmp += args.tradeoff * loss_aug * old_task_factor                  
            	    loss += args.tradeoff * loss_aug * old_task_factor
            	
            	### Replay and Classification Loss
            	if current_task > 0: 
            	    embed_sythesis = []
            	    embed_label_sythesis = []
            	    ind = list(range(len(pre_index)))

            	    if args.mean_replay:
            	        for _ in range(setsz):                        
            	            np.random.shuffle(ind)
            	            tmp = prototype['class_mean'][ind[0]]+np.random.normal()*prototype['class_std'][ind[0]]
            	            embed_sythesis.append(tmp)
            	            embed_label_sythesis.append(prototype['class_label'][ind[0]])
            	        embed_sythesis = np.asarray(embed_sythesis)
            	        embed_label_sythesis=np.asarray(embed_label_sythesis)
            	        embed_sythesis = torch.from_numpy(embed_sythesis).to(device)
            	        embed_label_sythesis = torch.from_numpy(embed_label_sythesis)
            	    else:
            	        for _ in range(setsz):
            	            np.random.shuffle(ind)
            	            embed_label_sythesis.append(pre_index[ind[0]])
            	        embed_label_sythesis = np.asarray(embed_label_sythesis)
            	        embed_label_sythesis = torch.from_numpy(embed_label_sythesis)
            	        y_onehot.zero_()
            	        y_onehot.scatter_(1, embed_label_sythesis[:, None], 1)
            	        syn_label_pre = y_onehot.to(device)

            	        z = torch.Tensor(np.random.normal(0, 1, (setsz, args.latent_dim))).to(device)
            	        
            	        embed_sythesis = generator(z, syn_label_pre)

            	    embed_sythesis = torch.cat((embed_feat,embed_sythesis))
            	    embed_label_sythesis = torch.cat((y_spt[i],embed_label_sythesis.to(device)))
            	    soft_feat_syt = model.embed(embed_sythesis)
            	    
            	    batch_size1 = inputs1.shape[0]
            	    batch_size2 = embed_feat.shape[0]
                    
                    # soft_feat_syt = mlp(soft_feat_syt)

            	    loss_cls = torch.nn.CrossEntropyLoss()(soft_feat_syt[:batch_size1], embed_label_sythesis[:batch_size1])

            	    loss_cls_old = torch.nn.CrossEntropyLoss()(soft_feat_syt[batch_size2:], embed_label_sythesis[batch_size2:])
            	    
            	    loss_cls += loss_cls_old * old_task_factor
            	    loss_cls /= args.nb_cl_fg // num_class_per_task + current_task
            	    loss += loss_cls
            	# $$$$$$$$$$$$$$$$
            	# loss = F.cross_entropy(embed_feat, y_spt[i])
            	grad = torch.autograd.grad(loss, model.parameters(),create_graph=True, retain_graph=True)
            	# fast_weights = list(map(lambda p: p[1] - args.update_lr * p[0], zip(grad, model.parameters())))
            	fast_weights_dict = fast_weights(grad,model.state_dict(),args.update_lr)
            	# this is the loss and accuracy before first update
            	with torch.no_grad():
            	    # [setsz, nway]
            	    embed_feat_q = model(x_qry[i])
            	    soft_feat_q = model.embed(embed_feat_q)
            	    # y_pred_q = mlp(soft_feat_q)
            	    loss_q = torch.nn.CrossEntropyLoss()(soft_feat_q, y_qry[i])
            	    # loss_q = F.cross_entropy(embed_feat_q, y_qry[i])
            	    # loss_q = torch.nn.CrossEntropyLoss()(soft_feat_q, y_qry[i])
            	    losses_q[0] += loss_q

            	    pred_q = F.softmax(soft_feat_q, dim=1).argmax(dim=1)
            	    correct = torch.eq(pred_q, y_qry[i]).sum().item()
            	    corrects[0] = corrects[0] + correct
            	# this is the loss and accuracy after the first update
            	with torch.no_grad():
            	    # [setsz, nway]
            	    model.load_state_dict(fast_weights_dict)
            	    embed_feat_q = model(x_qry[i])
            	    soft_feat_q = model.embed(embed_feat_q)
            	    # y_pred_q = mlp(soft_feat_q)
            	    loss_q = torch.nn.CrossEntropyLoss()(soft_feat_q, y_qry[i])
            	    # loss_q = torch.nn.cross_entropy(soft_feat_q, y_qry[i])
            	    losses_q[1] += loss_q
            	    # [setsz]
            	    pred_q = F.softmax(soft_feat_q, dim=1).argmax(dim=1)
            	    correct = torch.eq(pred_q, y_qry[i]).sum().item()
            	    corrects[1] = corrects[1] + correct

            	for k in range(1, args.update_step):
            	    # 1. run the i-th task and compute loss for k=1~K-1
            	    model.load_state_dict(fast_weights_dict)
            	    embed_feat = model(x_spt[i])
            	    # loss = torch.nn.cross_entropy(embed_feat, y_spt[i])
            	    loss = torch.zeros(1).to(device)
            	    if current_task>0:
            	    	embed_feat_old = model_old(x_spt[i])
            	    	loss_aug = torch.dist(embed_feat, embed_feat_old , 2)                    
            	    	loss += args.tradeoff * loss_aug * old_task_factor
            	    	soft_feat_syt = model.embed(embed_sythesis)
            	        batch_size1 = inputs1.shape[0]
            	        batch_size2 = embed_feat.shape[0]

            	        # soft_feat_syt = mlp(soft_feat_syt)

            	        loss_cls = torch.nn.CrossEntropyLoss()(soft_feat_syt[:batch_size1], embed_label_sythesis[:batch_size1])

            	        loss_cls_old = torch.nn.CrossEntropyLoss()(soft_feat_syt[batch_size2:], embed_label_sythesis[batch_size2:])
            	        
            	        loss_cls += loss_cls_old * old_task_factor
            	        loss_cls /= args.nb_cl_fg // num_class_per_task + current_task
            	        loss += loss_cls
            	    else:
            	    	soft_feat = model.embed(embed_feat)
            	    	# y_pred = mlp(soft_feat)
            	    	# loss_cls = torch.nn.CrossEntropyLoss()(y_pred, y_spt[i])
            	    	loss_cls = torch.nn.CrossEntropyLoss()(soft_feat, y_spt[i])
            	    	loss += loss_cls
            	    # 2. compute grad on theta_pi
            	    grad = torch.autograd.grad(loss, model.parameters(),create_graph=True, retain_graph=True)
            	    # 3. theta_pi = theta_pi - train_lr * grad
            	    # fast_weights = list(map(lambda p: p[1] - args.update_lr * p[0], zip(grad, fast_weights)))
            	    fast_weights_dict = fast_weights(grad,model.state_dict(),args.update_lr)
            	    model.load_state_dict(fast_weights_dict)
            	    embed_feat_q = model(x_qry[i])
            	    soft_feat_q = model.embed(embed_feat_q)
            	    # loss_q will be overwritten and just keep the loss_q on last update step.
            	    # soft_feat_q = mlp(soft_feat_q)
            	    loss_q = torch.nn.cross_entropy(soft_feat_q, y_qry[i])
            	    losses_q[k + 1] += loss_q

            	    with torch.no_grad():
            	        pred_q = F.softmax(soft_feat_q, dim=1).argmax(dim=1)
            	        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
            	        corrects[k + 1] = corrects[k + 1] + correct



            # end of all tasks
            # sum over all losses on query set across all tasks
            loss_q = losses_q[-1] / meta_batch_size
            # loss += loss_q
            # optimize theta parameters
            optimizer.zero_grad()
            # optimizer_mlp.zero_grad()
            # loss.backward()
            loss_q.backward()
            # print('meta update')
            # for p in self.net.parameters()[:5]:
            # 	print(torch.norm(p).item())
            optimizer.step()
            # optimizer_mlp.step()
            accs = np.array(corrects) / (querysz * meta_batch_size)
            loss_log['C/loss'] += loss.item()
            loss_log['C/loss_cls'] += loss_cls.item()
            loss_log['C/loss_aug'] += args.tradeoff*loss_aug.item() if args.tradeoff != 0 else 0
            loss_log['C/loss_cls_q'] += loss_q.item()
            del loss_cls
            del loss_q
            if epoch == 0 and i == 0:
                print(50 * '#')

        print('[Metric Epoch %05d]\t Total Loss: %.3f \t LwF Loss: %.3f \t Query Loss: %.3f \t Query Accuracy (FeatureX): %.3f \t'
                % (epoch + 1, loss_log['C/loss'], loss_log['C/loss_aug'], loss_log['C/loss_cls_q'], accs))
        for k, v in loss_log.items():
            if v != 0:
                tb_writer.add_scalar('Task {} - Classifier/{}'.format(current_task, k), v, epoch + 1)
        
        tb_writer.add_scalar('Task {}'.format(current_task), accs, epoch + 1)
        if epoch == args.epochs-1:
            torch.save(model, os.path.join(log_dir, 'task_' + str(
                current_task).zfill(2) + '_%d_model.pkl' % epoch))
            

################# feature extraction training end ########################

############################################## GAN Training ####################################################
    model = model.eval()
    # mlp = mlp.eval()
    for p in model.parameters():  # set requires_grad to False
        p.requires_grad = False
    for p in generator.parameters():  # set requires_grad to True
        p.requires_grad = True
    criterion_softmax = torch.nn.CrossEntropyLoss().to(device)
    if current_task != args.num_task:
        for epoch in range(args.epochs_gan):
            loss_log = {'D/loss': 0.0,
                        'D/new_rf': 0.0,
                        'D/new_lbls': 0.0,
                        'D/new_gp': 0.0,
                        'D/prev_rf': 0.0,
                        'D/prev_lbls': 0.0,
                        'D/prev_gp': 0.0,
                        'D/loss_q':0.0,
                        'D/new_rf_q':0.0,
                        'D/new_lbls_q':0.0,
                        'D/new_gp_q':0.0,
                        'G/loss': 0.0,
                        'G/new_rf': 0.0,
                        'G/new_lbls': 0.0,
                        'G/prev_rf': 0.0,
                        'G/prev_mse': 0.0,
                        'G/new_classifier':0.0,
                        'G/loss_q':0.0,
                        'G/new_rf_q':0.0,
                        'G/new_lbls_q':0.0,
                        'G/new_gp_q':0.0,
                        'E/kld': 0.0,
                        'E/mse': 0.0,
                        'E/loss': 0.0}
            # scheduler_D.step()
            # scheduler_G.step()
            for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(train_loader, 0):
                for p in discriminator.parameters():
                    p.requires_grad = True
                x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
                # inputs, labels = data
                # d_loss_total = 0.0
                # g_loss_total = 0.0
                meta_batch_size, setsz, c_, h, w = x_spt.size()
                querysz = x_qry.size(1)
                for i in range(args.meta_batch_size): # This is inner loop not task
                    inputs = Variable(x_spt[i])
                    labels = y_spt[i]
                    ############################# Train Disciminator###########################
             
                    real_feat = model(inputs)
                    z = torch.Tensor(np.random.normal(0, 1, (setsz, args.latent_dim))).to(device)             
                                    

                    y_onehot.zero_()
                    y_onehot.scatter_(1, labels[:, None], 1)
                    syn_label = y_onehot.to(device)
                    fake_feat = generator(z, syn_label)

                    real_feat_q = model(x_qry[i])
                    z_q = torch.Tensor(np.random.normal(0, 1, (querysz, args.latent_dim))).to(device)             
                    y_onehot_q.zero_()
                    y_onehot_q.scatter_(1, y_qry[:, None], 1)
                    syn_label_q = y_onehot_q.to(device)
                    fake_feat_q = generator(z_q, syn_label_q)

                    d_losses_q = [0 for _ in range(args.update_step)]

                    for k in range(args.update_step):
                        fake_validity, _               = discriminator(fake_feat, syn_label)
                        real_validity, disc_real_acgan = discriminator(real_feat, syn_label)

                        # Adversarial loss
                        d_loss_rf = -torch.mean(real_validity) + torch.mean(fake_validity)
                        gradient_penalty = compute_gradient_penalty(discriminator, real_feat, fake_feat, syn_label).mean()
                        d_loss_lbls = criterion_softmax(disc_real_acgan, labels.to(device))
                        d_loss = d_loss_rf + lambda_gp * gradient_penalty 

                        grad = torch.autograd.grad(d_loss, discriminator.parameters(),create_graph=True, retain_graph=True)
                        fast_weights_dict = fast_weights(grad,discriminator.state_dict(),args.update_lr)
                        discriminator.load_state_dict(fast_weights_dict)
                        
                        fake_validity_q, _               = discriminator(fake_feat_q, syn_label_q)
                        real_validity_q, disc_real_acgan_q = discriminator(real_feat_q, syn_label_q)

                        # Adversarial loss query
                        d_loss_rf_q = -torch.mean(real_validity_q) + torch.mean(fake_validity_q)
                        gradient_penalty_q = compute_gradient_penalty(discriminator, real_feat_q, fake_feat_q, syn_label_q).mean()
                        d_loss_lbls_q = criterion_softmax(disc_real_acgan_q, y_qry.to(device))
                        d_loss_q = d_loss_rf_q + lambda_gp * gradient_penalty_q

                        d_losses_q[k] += d_loss_q
                
                optimizer_D.zero_grad()
                d_loss_q = d_losses_q[-1]/args.meta_batch_size 
                d_loss_q.backward()
                optimizer_D.step()
                loss_log['D/loss'] += d_loss.item()
                loss_log['D/new_rf'] += d_loss_rf.item()
                loss_log['D/new_lbls'] += d_loss_lbls.item() #!!!
                loss_log['D/new_gp'] += gradient_penalty.item() if lambda_gp != 0 else 0
                loss_log['D/loss_q'] += d_loss_q.item()
                loss_log['D/new_rf_q'] += d_loss_rf_q.item()
                loss_log['D/new_lbls_q'] += d_loss_lbls_q.item() #!!!
                loss_log['D/new_gp_q'] += gradient_penalty_q.item() if lambda_gp != 0 else 0
                del d_loss_rf, d_loss_lbls
                ############################# Train Generaator###########################
                # Train the generator every n_critic steps
                # if step % args.n_critic == 0:
                for p in discriminator.parameters():
                    p.requires_grad = False                   
                    ############################# Train GAN###########################
                    
                    # Generate a batch of images
                for i in range(args.meta_batch_size):
                	inputs = Variable(x_spt[i])
                	labels = y_spt[i]
                	meta_batch_size, setsz, c_, h, w = x_spt.size()
                	querysz = x_qry.size(1)
                	
                	real_feat = model(inputs)
                	z = torch.Tensor(np.random.normal(0, 1, (setsz, args.latent_dim))).to(device)             
                	                

                	y_onehot.zero_()
                	y_onehot.scatter_(1, labels[:, None], 1)
                	syn_label = y_onehot.to(device)

                	real_feat_q = model(x_qry[i])
                	z_q = torch.Tensor(np.random.normal(0, 1, (querysz, args.latent_dim))).to(device)             
                	y_onehot_q.zero_()
                	y_onehot_q.scatter_(1, y_qry[:, None], 1)
                	syn_label_q = y_onehot_q.to(device)

                	g_losses_q = [0 for _ in range(args.update_step)]

                	for k in range(args.update_step):
                		fake_feat = generator(z, syn_label)
                	    # Loss measures generator's ability to fool the discriminator
                	    # Train on fake images
                	    fake_validity, disc_fake_acgan = discriminator(fake_feat, syn_label)
                	    if current_task == 0:
                	        loss_aug = 0 * torch.sum(fake_validity)
                	    else:
                	        ind = list(range(len(pre_index)))
                	        embed_label_sythesis = []
                	        for _ in range(setsz):
                	            np.random.shuffle(ind)
                	            embed_label_sythesis.append(pre_index[ind[0]])


                	        embed_label_sythesis = np.asarray(embed_label_sythesis)
                	        embed_label_sythesis = torch.from_numpy(embed_label_sythesis)
                	        y_onehot.zero_()
                	        y_onehot.scatter_(1, embed_label_sythesis[:, None], 1)
                	        syn_label_pre = y_onehot.to(device)

                	        pre_feat = generator(z, syn_label_pre) 
                	        pre_feat_old = generator_old(z, syn_label_pre)
                	        loss_aug = loss_mse(pre_feat, pre_feat_old)
                	    g_loss_rf = -torch.mean(fake_validity)
                	    g_loss_lbls = criterion_softmax(disc_fake_acgan, labels.to(device))
                	    g_loss = g_loss_rf + lambda_lwf*old_task_factor * loss_aug + g_loss_lbls

                	    grad = torch.autograd.grad(g_loss, generator.parameters(),create_graph=True, retain_graph=True)
                	    fast_weights_dict = fast_weights(grad,generator.state_dict(),args.update_lr)
                	    generator.load_state_dict(fast_weights_dict)

                		fake_feat_q = generator(z_q, syn_label_q)
                	    fake_validity_q, disc_fake_acgan_q = discriminator(fake_feat_q, syn_label_q)
                	    # if current_task == 0:
                	    #     loss_aug_q = 0 * torch.sum(fake_validity_q)
                	    # else:
                	    #     ind_q = list(range(len(pre_index)))
                	    #     embed_label_sythesis_q = []
                	    #     for _ in range(querysz):
                	    #         np.random.shuffle(ind_q)
                	    #         embed_label_sythesis_q.append(pre_index[ind[0]])


                	    #     embed_label_sythesis_q = np.asarray(embed_label_sythesis_q)
                	    #     embed_label_sythesis_q = torch.from_numpy(embed_label_sythesis_q)
                	    #     y_onehot_q.zero_()
                	    #     y_onehot_q.scatter_(1, embed_label_sythesis_q[:, None], 1)
                	    #     syn_label_pre_q = y_onehot_q.to(device)

                	    #     pre_feat_q = generator(z_q, syn_label_pre_q) 
                	    #     pre_feat_old_q = generator_old(z, syn_label_pre_q)
                	    #     loss_aug_q = loss_mse(pre_feat_q, pre_feat_old_q)
                	    g_loss_rf_q = -torch.mean(fake_validity_q)
                	    g_loss_lbls_q = criterion_softmax(disc_fake_acgan_q, y_qry[i].to(device))
                	    g_loss_q = g_loss_rf_q + g_loss_lbls_q# + lambda_lwf*old_task_factor * loss_aug_q
                	    g_losses_q[k] += g_loss_q
                optimizer_G.zero_grad()
                g_loss_q = g_losses_q[-1]/args.meta_task_num
                g_loss_q.backward()
                optimizer_G.step()
                # g_loss_total += g_loss
                loss_log['G/loss'] += g_loss.item()
                loss_log['G/new_rf'] += g_loss_rf.item()
                loss_log['G/new_lbls'] += g_loss_lbls.item() #!
                loss_log['G/loss_q'] += g_loss_q.item()
                loss_log['G/new_rf_q'] += g_loss_rf_q.item()
                loss_log['G/new_lbls_q'] += g_loss_lbls_q.item() #!!!
                loss_log['G/new_classifier'] += 0 #!
                loss_log['G/prev_mse'] += loss_aug.item() if lambda_lwf != 0 else 0
                del g_loss_rf, g_loss_lbls
                # loss_log['D/loss_total']+= d_loss_total
                # loss_log['G/loss_total']+= g_loss_total
            print('[GAN Epoch %05d]\t D Total Loss: %.3f \t G Total Loss: %.3f \t LwF Loss: %.3f' % (
                epoch + 1, loss_log['D/loss'], loss_log['G/loss'], loss_log['G/prev_rf']))
            print('[GAN Epoch %05d]\t D Total Loss Query: %.3f \t G Total Loss Query: %.3f \t' % (
                epoch + 1, loss_log['D/loss_q'], loss_log['G/loss_q']))
            for k, v in loss_log.items():
                if v != 0:
                    tb_writer.add_scalar('Task {} - GAN/{}'.format(current_task, k), v, epoch + 1)

            if epoch ==args.epochs_gan - 1:
                torch.save(generator, os.path.join(log_dir, 'task_' + str(
                    current_task).zfill(2) + '_%d_model_generator.pkl' % epoch))
                torch.save(discriminator, os.path.join(log_dir, 'task_' + str(
                    current_task).zfill(2) + '_%d_model_discriminator.pkl' % epoch))
    tb_writer.close()

    prototype = compute_prototype(model,train_loader,batch_size=meta_batch_size)  #!
    return prototype


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generative Feature Replay Training')
    # image sizes
    argparser.add_argument('--img_sz', type=int, help='img_sz', default=84)
    argparser.add_argument('--img_c', type=int, help='img_c', default=3)

    # n_way, k_shot setting
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    

    # task setting
    parser.add_argument('-data', default='miniimagenet', required=True, help='path to Data Set')
    parser.add_argument('-num_class', default=100, type=int, metavar='n', help='dimension of embedding space')
    parser.add_argument('-nb_cl_fg', type=int, default=50, help="Number of class, first group")
    parser.add_argument('-num_task', type=int, default=2, help="Number of Task after initial Task")

    # method parameters
    parser.add_argument('-mean_replay', type=bool, action = 'store_true', help='Mean Replay')
    parser.add_argument('-tradeoff', type=float, default=1.0, help="Feature Distillation Loss")

    # basic parameters
    parser.add_argument('-load_dir_aug', default='', help='Load first task')
    parser.add_argument('-ckpt_dir', default='checkpoints', help='checkpoints dir')
    parser.add_argument('-dir', default='/data/datasets/featureGeneration/', help='data dir')
    parser.add_argument('-log_dir', default=None, help='where the trained models save')
    parser.add_argument('-name', type=str, default='tmp', metavar='PATH')

    parser.add_argument("-gpu", type=str, default='0', help='which gpu to choose')
    parser.add_argument('-nThreads', '-j', default=4, type=int, metavar='N', help='number of data loading threads')

    # hyper-parameters
    parser.add_argument('-meta_batch_size', '-b', default=4, type=int, metavar='N', help='meta-batch size')
    parser.add_argument('-lr', type=float, default=1e-3, help="learning rate of new parameters")
    parser.add_argument('-lr_decay', type=float, default=0.1, help='Decay learning rate')
    parser.add_argument('-lr_decay_step', type=float, default=100, help='Decay learning rate every x steps')
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-weight-decay', type=float, default=2e-4)

    # hype-parameters for W-GAN
    parser.add_argument('-gan_tradeoff', type=float, default=1e-3, help="learning rate of new parameters")
    parser.add_argument('-gan_lr', type=float, default=1e-4, help="learning rate of new parameters")
    parser.add_argument('-lambda_gp', type=float, default=10.0, help="learning rate of new parameters")
    parser.add_argument('-n_critic', type=int, default=5, help="learning rate of new parameters")

    parser.add_argument('-latent_dim', type=int, default=200, help="learning rate of new parameters")
    parser.add_argument('-feat_dim', type=int, default=512, help="learning rate of new parameters")
    parser.add_argument('-hidden_dim', type=int, default=512, help="learning rate of new parameters")
    
    # training parameters
    parser.add_argument('-epochs', default=201, type=int, metavar='N', help='epochs for training process')
    parser.add_argument('-epochs_gan', default=1001, type=int, metavar='N', help='epochs for training process')
    parser.add_argument('-seed', default=2001, type=int, metavar='N', help='seeds for training process')
    parser.add_argument('-start', default=0, type=int, help='resume epoch')

    args = parser.parse_args()

    # Data
    print('==> Preparing data..')
    
    # if args.data == 'miniimagenet':
    #     mean_values = [0.485, 0.456, 0.406]
    #     std_values = [0.229, 0.224, 0.225]
    #     transform_train = transforms.Compose([
    #         #transforms.Resize(256),
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=mean_values,
    #                              std=std_values)
    #     ])
    #     traindir = os.path.join(args.dir, 'ILSVRC12_256', 'train')

    # if  args.data == 'cifar':
    #     transform_train = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])
    #     traindir = args.dir + '/cifar'

    num_classes = args.num_class 
    num_task = args.num_task
    num_class_per_task = (num_classes-args.nb_cl_fg) // num_task
    
    random_perm = list(range(num_classes))      # multihead fails if random permutaion here
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    prototype = {}

    if args.mean_replay:
        args.epochs_gan = 500
        
    for i in range(args.start, num_task+1):
        print ("-------------------Get started--------------- ")
        print ("Training on Task " + str(i))
        if i == 0:
            pre_index = 0
            class_index = random_perm[:args.nb_cl_fg]
        else:
            pre_index = random_perm[:args.nb_cl_fg + (i-1) * num_class_per_task]
            class_index = random_perm[args.nb_cl_fg + (i-1) * num_class_per_task:args.nb_cl_fg + (i) * num_class_per_task]

        if args.data == 'cifar100':
            np.random.seed(args.seed)
            target_transform = np.random.permutation(num_classes)
            trainset = CIFAR100(root=traindir, train=True, download=True, transform=transform_train, target_transform = target_transform, index = class_index)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.meta_batch_size, shuffle=True, num_workers=args.nThreads,drop_last=True)
        else:
            trainfolder = MiniImagenet('./data/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.img_sz)
            train_loader = torch.utils.data.DataLoader(
                trainfolder, batch_size=args.meta_batch_size,
                shuffle=True,
                drop_last=True, num_workers=args.nThreads)

        prototype_old = prototype
        prototype = train_task(args, train_loader, i, prototype=prototype, pre_index=pre_index)

        if args.start>0:
            pass
            # Currently it only works for our PrototypeLwF method
        else:
            if i >= 1:
                # Increase the prototype as increasing number of task
                for k in prototype.keys():
                    prototype[k] = np.concatenate((prototype[k], prototype_old[k]), axis=0)