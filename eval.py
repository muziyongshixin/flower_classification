#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import time
from utils.functions import *
import torch
from IPython import  embed

logger = logging.getLogger(__name__)

def eval_on_model(model, criterion, batch_data, epoch, device):
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()
    batch_cnt = len(batch_data)
    eval_start_time=time.time()
    for i, batch in enumerate(batch_data, 0):
        batch_start_time = time.time()

        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        # forward
        pred_labels = model.forward(images)
        # get task loss
        loss = criterion.forward(pred_labels, labels)

        # logging
        batch_loss = loss.item()
        epoch_loss.update(batch_loss, len(labels))

        accuracy = compute_accuracy(pred_labels.data, labels.data)
        epoch_acc.update(accuracy.item(), len(labels))

        batch_time = time.time() - batch_start_time
        logger.info('epoch=%d, batch=%d/%d, time=%.2f  loss=%.5f binary_acc=%.4f ' % (
            epoch, i, batch_cnt, batch_time, batch_loss, accuracy))

    eval_time=time.time()-eval_start_time
    logger.info('===== epoch=%d, eval_time=%.2f, eval_avg_loss=%.5f, eval_avg_accuracy=%.4f ====' % (
        epoch, eval_time, epoch_loss.avg, epoch_acc.avg))
    return epoch_loss.avg, epoch_acc.avg






