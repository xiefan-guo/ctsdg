import torch
import torch.nn as nn
from torch.utils import data


def data_sampler(dataset, shuffle, distributed):
    
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


