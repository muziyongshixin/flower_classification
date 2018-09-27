#!/usr/bin/env python
# -*- coding: utf-8 -*-
import socket
import time

__author__ = 'han'

import logging
import argparse
from test import test
from train import train
from utils.load_config import init_logging
import sys,traceback
from utils.mailSender import MailSender


parser = argparse.ArgumentParser(description="train/test the model")
parser.add_argument('mode', help='preprocess or train or test')
parser.add_argument('--mail', '-m',default=False, required=False, dest='mailsend')
parser.add_argument('--config', '-c', required=False, dest='config_path', default='config/global_config.yaml')
parser.add_argument('--output', '-o', required=False, dest='out_path')
parser.add_argument('--remark', required=False, dest='remark',default="")
args = parser.parse_args()


cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
host_name = socket.gethostname()
experiment_info = host_name + "_" + cur_time+"_"+args.remark

init_logging(experiment_info)
logger = logging.getLogger(__name__)
logger.info('========================  %s  ================================='%experiment_info)
try:
    if args.mode == 'train':
        train(args.config_path, experiment_info)
    elif args.mode == 'test':
        test(args.config_path, experiment_info)
    else:
        raise ValueError('Unrecognized mode selected.')
except Exception as e:
    exc_traceback = ''.join(traceback.format_exception(*sys.exc_info()))
    if args.mailsend:
        mailsender = MailSender()
        mailsender.send(exc_traceback)
    print(exc_traceback)
