import os
import time

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn import metrics

from parse import HTNp_parse
from CodeXGLUEutils import MyDataset, file_parse_noNorm, is_best, getscore

if __name__ == '__main__':
    logfile = open('MyModelLog.txt', 'w')
    for i in range(100):
        print("{}\n".format(i+100))
        print("{}\n".format(i+100), file=logfile)
    logfile.close()
