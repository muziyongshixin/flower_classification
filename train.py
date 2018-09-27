#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

__author__ = 'liyz'

import os
import torch
import logging
import torch.optim as optim
from dataset.filelist_Dataset import filelist_DataSet
from utils.load_config import read_config
from eval import eval_on_model
from IPython import embed
import torchvision.transforms as transforms
from utils.functions import *
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

from torchvision.models import densenet121
from torchvision.models import densenet201

logger = logging.getLogger(__name__)


def train(config_path, experiment_info):
    logger.info('------------ flower classification Train --------------')
    logger.info('------------ loading config file ------------')
    global_config = read_config(config_path)
    logger.info(open(config_path).read())
    logger.info('------------   config file info above ------------')
    # set random seed
    seed = global_config['global']['random_seed']
    torch.manual_seed(seed)

    enable_cuda = global_config['train']['enable_cuda']
    device = torch.device("cuda" if enable_cuda else "cpu")
    if torch.cuda.is_available() and not enable_cuda:
        logger.warning("CUDA is avaliable, you can enable CUDA in config file")
    elif not torch.cuda.is_available() and enable_cuda:
        raise ValueError("CUDA is not abaliable, please unable CUDA in config file")

    cudnn.benchmark = True

    ############################### 获取数据集 ############################
    train_filelist_path = global_config['data']['dataset']['train_path']
    dev_filelist_path = global_config['data']['dataset']['dev_path']
    logger.info('constructing dataset...')
    train_dataset = filelist_DataSet(train_filelist_path,
                                     transform=transforms.Compose([
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]),
                                     ]))
    dev_dataset = filelist_DataSet(dev_filelist_path,
                                   transform=transforms.Compose([
                                       transforms.Resize([224,224]),
                                       # transforms.CenterCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]),
                                   ]))
    assert train_dataset, dev_dataset

    train_batch_size = global_config['train']['batch_size']
    valid_batch_size = global_config['train']['valid_batch_size']
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=4)

    logger.info('constructing model...')
    model_choose = global_config['global']['model']
    logger.info('Using model is: %s ' % model_choose)
    model = globals()[model_choose]()
    print_network(model)

    gpu_nums = torch.cuda.device_count()
    logger.info('dataParallel using %d GPU.....' % gpu_nums)
    if gpu_nums > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    task_criterion = CrossEntropyLoss().to(device)

    # optimizer
    optimizer_choose = global_config['train']['optimizer']
    optimizer_lr = global_config['train']['learning_rate']
    optimizer_eps = float(global_config['train']['eps'])
    optimizer_param = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_choose == 'adam':
        optimizer = optim.Adam(optimizer_param, lr=optimizer_lr, eps=optimizer_eps)
    elif optimizer_choose == 'sgd':
        optimizer = optim.SGD(optimizer_param, lr=optimizer_lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError('optimizer "%s" in config file not recoginized' % optimizer_choose)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)

    # check if exist model weight
    weight_path = global_config['train']['model_path']
    if os.path.exists(weight_path) and global_config['train']['continue']:
        logger.info('loading existing weight............')
        if enable_cuda:
            weight = torch.load(weight_path, map_location=lambda storage, loc: storage.cuda())
        else:
            weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(weight, strict=False)

    # save current code
    # save_cur_experiment_code_path = "savedcodes/" + experiment_info
    # save_current_codes(save_cur_experiment_code_path, global_config)
    # tensorboardX writer
    tensorboard_writer = SummaryWriter(log_dir=os.path.join('tensorboard_logdir', experiment_info))

    # training arguments
    logger.info('start training............................................')
    best_valid_acc = None
    # every epoch
    for epoch in range(global_config['train']['epoch']):
        # train
        model.train()  # set training = True, make sure right dropout
        train_avg_loss, train_avg_binary_acc = train_on_model(model=model,
                                                              criterion=task_criterion,
                                                              optimizer=optimizer,
                                                              batch_data=train_loader,
                                                              epoch=epoch,
                                                              device=device
                                                              )

        # evaluate
        with torch.no_grad():
            model.eval()  # let training = False, make sure right dropout
            val_loss, val_acc = eval_on_model(model=model,
                                              criterion=task_criterion,
                                              batch_data=dev_loader,
                                              epoch=epoch,
                                              device=device)

        # save model when best accuracy score
        if best_valid_acc is None or val_acc > best_valid_acc:
            cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
            epoch_info = "%s epoch=%d, cur best accuracy=%.4f" % (cur_time, epoch, val_acc)
            save_model(model,
                       epoch_info=epoch_info,
                       model_weight_path=global_config['train']['model_weight_dir'] + experiment_info + "_weight.pt",
                       save_ckpt_log_path=global_config['train']['ckpt_log_path'] + experiment_info + "_save.log")
            logger.info("=========  saving model weight on epoch=%d  =======" % epoch)
            best_valid_acc = val_acc

        tensorboard_writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
        tensorboard_writer.add_scalar("train/avg_loss", train_avg_loss, epoch)
        tensorboard_writer.add_scalar("train/binary_acc", train_avg_binary_acc, epoch)
        tensorboard_writer.add_scalar("val/avg_loss", val_loss, epoch)
        tensorboard_writer.add_scalar("val/avg_accuracy", val_acc, epoch)
        #  adjust learning rate
        scheduler.step(train_avg_loss)

    logger.info('finished.................................')
    tensorboard_writer.close()


def train_on_model(model, criterion, optimizer, batch_data, epoch, device):
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()
    batch_cnt = len(batch_data)
    for i, batch in enumerate(batch_data, 0):
        batch_start_time = time.time()
        optimizer.zero_grad()
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        # forward
        pred_labels = model.forward(images)
        # get task loss
        loss = criterion.forward(pred_labels, labels)
        loss.backward()
        optimizer.step()  # update parameters

        # logging
        batch_loss = loss.item()
        epoch_loss.update(batch_loss, len(labels))

        accuracy = compute_accuracy(pred_labels.data, labels.data)
        epoch_acc.update(accuracy.item(), len(labels))

        batch_time = time.time() - batch_start_time
        logger.info('epoch=%d, batch=%d/%d, time=%.2f  loss=%.5f binary_acc=%.4f ' % (
            epoch, i, batch_cnt, batch_time, batch_loss, accuracy))

    logger.info('===== epoch=%d, batch_count=%d, epoch_average_loss=%.5f, avg_binary_acc=%.4f ====' % (
        epoch, batch_cnt, epoch_loss.avg, epoch_acc.avg))
    return epoch_loss.avg, epoch_acc.avg
