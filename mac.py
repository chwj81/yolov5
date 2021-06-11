import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import argparse
import os
import logging
import sys
import itertools

import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

from IPython.display import clear_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5x', help='initial model type')
    parser.add_argument('--nc', type=int, default=2, help='initial class size')
    parser.add_argument('--size', type=int, default=320, help='initial image size')
    opt = parser.parse_args()

    model = './models/' + opt.model + ".yaml"
    nc = opt.nc
    size = opt.size
    net = Model(model, ch=3, nc=nc, anchors=None).to('cpu')
    macs, params = get_model_complexity_info(net, (3, size, size), as_strings=True, print_per_layer_stat=True, verbose=True)
    clear_output()
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
