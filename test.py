#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

__author__ = 'liyz'

from utils.load_config import read_config
import torchvision.transforms as transforms

import csv
import torch.backends.cudnn as cudnn
from dataset.filelist_Dataset import filelist_DataSet
from utils.functions import *
from torchvision.models import densenet121
from torchvision.models import densenet201

logger = logging.getLogger(__name__)


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logger.info(str(net))
    logger.info('Total number of parameters: %d' % num_params)


def test(config_path, experiment_info):
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
    torch.set_grad_enabled(False)  # make sure all tensors below have require_grad=False,

    logger.info('constructing dataset...')
    test_filelist_path = global_config['data']['dataset']['test_path']
    test_dataset = filelist_DataSet(test_filelist_path,
                                   transform=transforms.Compose([
                                       transforms.Resize([224,224]),
                                       # transforms.CenterCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]),
                                   ]))
    assert test_dataset
    test_batch_size = global_config['test']['test_batch_size']
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    logger.info('constructing model...')
    model_choose = global_config['global']['model']
    logger.info('Using model is: %s ' % model_choose)
    model = globals()[model_choose]()
    gpu_nums = torch.cuda.device_count()
    logger.info('dataParallel using %d GPU.....' % gpu_nums)
    if gpu_nums > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    model.eval()  # let training = False, make sure right dropout

    # load model weight
    weight_path = global_config['test']['model_path']
    if os.path.exists(weight_path):
        logger.info('loading existing weight............')
        if enable_cuda:
            weight = torch.load(weight_path, map_location=lambda storage, loc: storage.cuda())
        else:
            weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(weight, strict=False)
    else:
        raise ValueError("invalid weight path !!!")

    # forward
    logger.info('evaluate forwarding...')

    out_path=global_config['test']['output_file_path']+experiment_info+"_result.csv"
    # to just evaluate score or write answer to file
    if out_path is not None:
        predict_on_model(model=model,batch_data=test_loader,device=device,out_path=out_path)

    logging.info('finished.')


def predict_on_model(model, batch_data, device,out_path):

    epoch_acc = AverageMeter()
    batch_cnt = len(batch_data)
    eval_start_time = time.time()
    for i, batch in enumerate(batch_data, 0):
        batch_start_time = time.time()
        images, idx = batch
        images = images.to(device)
        # forward
        pred_labels = model.forward(images)
        save_test_result_to_csv( pred_labels,idx, out_path)
        batch_time=time.time()-batch_start_time
        logger.info("batch=%d time=%.2f"%(i,batch_time))

    test_time = time.time() - eval_start_time
    logger.info('===== test finished test_time=%.2f====' % (test_time))
    return 0


def save_test_result_to_csv(pred_labels,idx,csv_file_path):
    class_map = {"daisy": 0, "dandelion": 1, "rose": 2, "sunflower": 3, "tulip": 4}
    id2class_map={0:"daisy",1:"dandelion",2:"rose",3:"sunflower",4:"tulip"}
    problem_num = len(pred_labels)
    is_first_batch = True
    if os.path.exists(csv_file_path):
        is_first_batch = False
    out = open(csv_file_path, 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    if is_first_batch:
        head_info = ['Id', 'Expected']
        csv_write.writerow(head_info)

    pred_labels = torch.argmax(pred_labels, dim=1)  # 得到一个20*1的矩阵
    for case_count in range(problem_num):
        class_name=id2class_map[pred_labels[case_count].item()]
        cur_row_info = [int(idx[case_count].item()),class_name]
        csv_write.writerow(cur_row_info)
    out.flush()
    out.close()



