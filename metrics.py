"""
Metrics and generic training utilities.
"""
import math
import random

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
from torch import nn


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(config_optim, parameters):
    if config_optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config_optim.lr,
                          weight_decay=config_optim.weight_decay,
                          betas=(config_optim.beta1, 0.999),
                          amsgrad=config_optim.amsgrad,
                          eps=config_optim.eps)
    elif config_optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config_optim.lr,
                             weight_decay=config_optim.weight_decay)
    elif config_optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config_optim.lr,
                         weight_decay=1e-4, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config_optim.optimizer))


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup."""
    if epoch < config.training.warmup_epochs:
        lr = config.optim.lr * epoch / config.training.warmup_epochs
    else:
        lr = config.optim.min_lr + (config.optim.lr - config.optim.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - config.training.warmup_epochs) / (
                     config.training.n_epochs - config.training.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_dataset(args, config):
    if config.data.dataset == "MHIST":
        from data.loading import MHISTDataset
        train_dataset = MHISTDataset(data_list=config.data.traindata, train=True)
        test_dataset = MHISTDataset(data_list=config.data.testdata, train=False)
    elif config.data.dataset == "chaoyang":
        from data.loading import chaoyangDataset
        train_dataset = chaoyangDataset(data_list=config.data.traindata, train=True)
        test_dataset = chaoyangDataset(data_list=config.data.testdata, train=False)
    elif config.data.dataset == "HITFAH_GCML":
        from data.loading import HITFAH_GCMLDataset
        train_dataset = HITFAH_GCMLDataset(data_list=config.data.traindata, train=True)
        test_dataset = HITFAH_GCMLDataset(data_list=config.data.testdata, train=False)
    else:
        raise NotImplementedError(
            "Supported datasets: MHIST, chaoyang, HITFAH_GCML.")
    return None, train_dataset, test_dataset


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k.
    output: the prediction from the diffusion model (B x n_classes).
    target: label indices (B).
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def cohen_kappa(output, target, topk=(1,)):
    """
    output: the prediction from the diffusion model (B x n_classes).
    target: label indices (B).
    """
    maxk = min(max(topk), output.size()[1])
    _, pred = output.topk(maxk, 1, True, True)
    kappa = cohen_kappa_score(pred, target, weights='quadratic')
    return kappa


def cast_label_to_one_hot_and_prototype(y_labels_batch, config, return_prototype=True):
    """
    y_labels_batch: a vector of length batch_size.
    """
    y_one_hot_batch = nn.functional.one_hot(
        y_labels_batch, num_classes=config.data.num_classes).float()
    if return_prototype:
        label_min, label_max = config.data.label_min_max
        y_logits_batch = torch.logit(nn.functional.normalize(
            torch.clip(y_one_hot_batch, min=label_min, max=label_max), p=1.0, dim=1))
        return y_one_hot_batch, y_logits_batch
    else:
        return y_one_hot_batch


def compute_f1_score(gt, pred):
    gt_class = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    pred_class = np.argmax(pred_np, axis=1)
    return f1_score(gt_class, pred_class, average='macro')


def compute_confusion_matrix(gt, pred):
    pred_class = pred.argmax(dim=1).cpu()
    return confusion_matrix(gt.cpu(), pred_class)
