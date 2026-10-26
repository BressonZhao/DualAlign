# Shared training utilities for multiclass node prediction and binary domain prediction.
# Class predictions are logits [N, C]; domain predictions are sigmoid probabilities.
import argparse
import torch.optim as optim
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, confusion_matrix, fbeta_score, \
    precision_score, recall_score
import torch
import numpy as np
import math
import copy
import torch.nn.functional as F
import sys


# argparse helper: type=bool treats any non-empty string (including "False") as True.
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('boolean expected')


# Optional parser extension; the supplied entry points define their own optimizer flags.
def parse_optimizer(parser):
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
                            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
                            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
                            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
                            help='Number of epochs before decay', default=50)
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
                            help='Learning rate decay ratio', default=0.8)
    opt_parser.add_argument('--lr', dest='lr', type=float,
                            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
                            help='Gradient clipping.')
    opt_parser.add_argument('--weight_decay', type=float,
                            help='Optimizer weight decay.', default=0)


# Build (scheduler, optimizer), optimizing only parameters with requires_grad=True.
# The two active training entry points instead construct Adam and StepLR directly.
def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300, 500, 700, 900],
                                                   gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer



# Tee stdout to the current terminal and append to <dir>/log.dat.
# The caller creates the directory. Re-wrapping an existing Logger reuses its terminal
# so that per-seed loggers do not duplicate every line into earlier log files.
class Logger(object):
    def __init__(self, dir):
        if isinstance(sys.stdout, Logger):
            self.terminal = sys.stdout.terminal
            sys.stdout.log.close()
        else:
            self.terminal = sys.stdout
        self.log = open(f"{dir}/log.dat", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Multiclass cross-entropy expects raw logits and integer class IDs [N].
def CE_loss(pred, label):
    label = label.type(torch.int64)
    loss = nn.CrossEntropyLoss()
    return loss(pred, label)

# Binary domain cross-entropy expects probabilities, not raw logits; flatten both inputs.
def BCE_loss(pred, label):
    pred = pred.view(-1)
    label = label.view(-1).type(torch.float32)
    loss = nn.BCELoss()
    return loss(pred, label)

# Compute single-label accuracy from the largest class score on each node.
def get_acc_score(pred, label):
    pred_label = pred.detach().clone()
    pred_label = pred_label.argmax(dim=1)
    acc_score = accuracy_score(label.cpu().detach().numpy(), pred_label.cpu().detach().numpy())
    return acc_score

# Return micro-averaged F1 for single-label predictions.
def get_f1_scores(pred, label):
    # Detach the prediction and move to CPU if necessary
    pred_label = pred.detach().clone()
    pred_label = pred_label.argmax(dim=1)  # Select the class with the largest score.

    # Convert to numpy arrays for use with sklearn metrics
    pred_label = pred_label.cpu().detach().numpy()
    label = label.cpu().detach().numpy()

    # Calculate F1 score for binary or multiclass classification
    f1 = f1_score(label, pred_label, average='micro')  # Micro averaging is fixed here.
    return f1

# Softmax logits before binary AUC (class-1 score) or multiclass one-vs-rest AUC.
# ValueError is caught and reported as None, for example when a split lacks classes.
def get_auc_score(pred, label):
    try:
        pred = F.softmax(pred, dim=1)
        #print(torch.unique(label).size(0))
        if torch.unique(label).size(0) > 2:
            auc_score = roc_auc_score(label.cpu().detach().numpy(), pred.cpu().detach().numpy(), multi_class='ovr')
        else:
            auc_score = roc_auc_score(label.cpu().detach().numpy(), pred[:, 1].cpu().detach().numpy())
        return auc_score
    except ValueError:
        print("he")
        return None
        pass


# Return metrics in (accuracy, AUC) order, as expected by both training scripts.
def get_scores(pred, label):
    auc_score = get_auc_score(pred, label)
    acc_score = get_acc_score(pred, label)
    

    return acc_score, auc_score

# Return (accuracy, micro-F1); this helper is separate from the active AUC reports.
def get_micro_scores(pred, label):
    acc_score = get_acc_score(pred, label)
    f1_score = get_f1_scores(pred, label)
    

    return acc_score, f1_score


# Aggregate seed metrics with population standard deviation (NumPy ddof=0).
# Missing values (e.g. undefined AUC returned as None) are stored as NaN and skipped;
# genuine zeros such as 0.0 accuracy are kept.
def get_avg_std_report(reports):
    all_keys = {k: [] for k in reports[0]}
    for report in reports:
        for k in report:
            all_keys[k].append(report[k] if report[k] is not None else np.nan)

    avg_report = {k: float(np.nanmean(v)) if not np.all(np.isnan(v)) else float('nan')
                  for k, v in all_keys.items()}
    avg_std_report = {k: f'{np.nanmean(v):.5f} +/- {np.nanstd(v):.5f}' if not np.all(np.isnan(v)) else 'nan'
                      for k, v in all_keys.items()}
    return avg_report, avg_std_report
