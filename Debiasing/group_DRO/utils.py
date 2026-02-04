import sys
import os
import torch
import numpy as np
import csv

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class CSVBatchLogger:
    def __init__(self, csv_path, n_groups, mode='w', wb=None, split=''):
        columns = ['epoch', 'batch']
        for idx in range(n_groups):
            columns.append(f"{split + '-' if split != '' else ''}avg_loss_group_{idx}")
            columns.append(f"{split + '-' if split != '' else ''}exp_avg_loss_group_{idx}")
            columns.append(f"{split + '-' if split != '' else ''}avg_acc_group_{idx}")
            columns.append(f"{split + '-' if split != '' else ''}processed_data_count_group_{idx}")
            columns.append(f"{split + '-' if split != '' else ''}update_data_count_group_{idx}")
            columns.append(f"{split + '-' if split != '' else ''}update_batch_count_group_{idx}")
        columns.append(f"{split + '-' if split != '' else ''}avg_actual_loss")
        columns.append(f"{split + '-' if split != '' else ''}avg_per_sample_loss")
        columns.append(f"{split + '-' if split != '' else ''}avg_acc")
        columns.append(f"{split + '-' if split != '' else ''}model_norm_sq")
        columns.append(f"{split + '-' if split != '' else ''}reg_loss")

        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        self.wb = wb
        if mode=='w':
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        stats_dict['epoch'] = epoch
        stats_dict['batch'] = batch
        self.writer.writerow(stats_dict)
        if self.wb is not None:
            self.wb.log_output(stats_dict)
            


    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def log_args(args, logger):
    for argname, argval in vars(args).items():
        logger.write(f'{argname.replace("_"," ").capitalize()}: {argval}\n')
    logger.write('\n')

