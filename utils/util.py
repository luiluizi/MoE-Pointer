import torch
import os
import datetime
import logging
import sys
import numpy as np

def get_gard_norm(params):
    return torch.stack([param.norm() for param in params]).norm()

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_cosine_schedule(optimizer, epoch, total_epochs, initial_lr, min_lr=0):
    """余弦衰减"""
    progress = epoch / total_epochs
    lr = min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

logger_instance = None

def get_logger(is_form=True):
    is_form = False
    global logger_instance
    if logger_instance is not None:
        return logger_instance
    
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    
    # log_dir = './baseline/log'
    # log_dir = './no/log'
    if is_form:
        log_dir = '/mnt/jfs6/g-bairui/log'
    else:
        log_dir = '/mnt/jfs6/g-bairui/log'
    # if is_form:
    #     log_dir = './train_log/log'
    # else:
    #     log_dir = './test_log/log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = '{}.log'.format(cur)
    logfilepath = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger("drone")

    level = logging.INFO

    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    logger_instance = logger

    logger.info('Log directory: %s', log_dir)
    return logger