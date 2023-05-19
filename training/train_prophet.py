import copy
from dataclasses import dataclass, field
import os
import time

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from transformers import HfArgumentParser
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm, trange
from numpy.lib.stride_tricks import sliding_window_view
import torch.multiprocessing as mp
import os
import torch.distributed as dist

from models import ProphetModule
from utils import TMLightning, ModelConfig

parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

model = ProphetModule()
lt_model = TMLightning(model, args.data, clip=args.clip, online=True)

train_set, test_set = lt_model.train_set, lt_model.test_set
mean = np.mean(train_set[train_set > 0])
std = np.std(train_set[train_set > 0])

train_set[train_set == 0] = np.nan
test_set[test_set == 0] = np.nan

train_set = (train_set - mean) / std
test_set = (test_set - mean) / std

train_set[np.isnan(train_set)] = 0
test_set[np.isnan(test_set)] = 0



import concurrent.futures
import os
import multiprocessing as mp
import random

import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

cpu_id = 0
num_cpus = 56

def pinned_core_initilizer():
    global cpu_id
    os.system(f"taskset -p -c {random.randint(0, num_cpus-1)} {os.getpid()} &> /dev/null")

outputs = []
targets = []

num_features = train_set.shape[-1]
futures = []
models = [ProphetModule() for _ in range(num_features)]
with concurrent.futures.ProcessPoolExecutor(
    max_workers=mp.cpu_count(), initializer=pinned_core_initilizer
) as executor:
    for k in range(num_features):
        futures.append(executor.submit(models[k].fit, train_set[..., k]))

    for future in tqdm(concurrent.futures.as_completed(futures)):
        future.result()

    futures = []

    for k in range(num_features):
        futures.append(executor.submit(executor.submit(model, test_set.shape[0])))
    
    for k in range(num_features):
        outputs.append(future.result().reshape(-1, 1))
        targets.append(test_set[..., k].reshape(-1, 1))

outputs = np.concatenate(outputs, axis=1)
targets = np.concatenate(targets, axis=1)
test_mae = np.mean(np.abs((outputs - targets) * std)[targets != 0])
print(f"Test MAE: {test_mae:.4f}")

np.save(f"training/outputs/{args.dataset}_outputs_prophet_clip[{args.clip}].npy", outputs, allow_pickle=False)
np.save(f"training/outputs/{args.dataset}_targets_prophet_clip[{args.clip}].npy", targets, allow_pickle=False)
