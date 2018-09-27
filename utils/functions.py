import torch
import logging
import os
import shutil
from torch.nn import init

logger = logging.getLogger(__name__)


def save_model(model, epoch_info, model_weight_path, save_ckpt_log_path):
    # save model weight
    model_weight = model.state_dict()

    torch.save(model_weight, model_weight_path)
    with open(save_ckpt_log_path, 'w') as checkpoint_f:
        checkpoint_f.write(epoch_info + "\n")


def compute_accuracy(pred_labels, real_labels):
    pred_labels = torch.argmax(pred_labels, dim=1)  # 得到一个16*1的矩阵，
    difference = torch.abs(pred_labels - real_labels)
    difference[difference != 0] = 1.0
    accuracy = 1.0 - torch.mean(difference.float())
    return accuracy

# 保存本次实验的部分代码
def save_current_codes(des_path, global_config):
    if not os.path.exists(des_path):
        os.mkdir(des_path)

    train_file_path = os.path.realpath(__file__)  # eg：/n/liyz/videosteganography/train.py
    cur_work_dir, trainfile = os.path.split(train_file_path)  # eg：/n/liyz/videosteganography/

    new_train_path = os.path.join(des_path, trainfile)
    shutil.copyfile(train_file_path, new_train_path)

    config_file_path = cur_work_dir + "/config/global_config.yaml"
    config_file_name = 'global_config.yaml'
    new_config_file_path = os.path.join(des_path, config_file_name)
    shutil.copyfile(config_file_path, new_config_file_path)

    model_choose = global_config['global']['model']
    model_file_name = model_choose + ".py"
    model_file_path = cur_work_dir + "/models/" + model_file_name
    new_model_file_path = os.path.join(des_path, model_file_name)
    shutil.copyfile(model_file_path, new_model_file_path)


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logger.info(str(net))
    logger.info('Total number of parameters: %d' % num_params)


# custom weights initialization called on model
def weights_init(m):
    for name, params in m.named_parameters():
        if name.find('decision_layer.4') != -1 or name.find('decision_layer.0') != -1:
            init.xavier_uniform_(params, gain=init.calculate_gain('relu'))
        elif name.find('conv') != -1:
            pass
        elif name.find('norm') != -1:
            pass


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
