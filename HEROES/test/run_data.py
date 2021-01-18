from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
sys.path.append("../")
from utils.data_loader import DataLoader

basedir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(basedir, "../data")

def running_dataloader(dataname, datasave, seqlen):
    print("===== START LOADING TRAINING DATA =====")
    trainloader = DataLoader(datadir, dataname, datasave, seqlen, data_type="train")
    _ = trainloader.load_data()
    print("===== FINISH LOADING TRAINING DATA =====")
    print("===== START LOADING TESTING DATA =====")
    testloader = DataLoader(datadir, dataname, datasave, seqlen, data_type="test")
    _ = testloader.load_data()
    print("===== FINISH LOADING TESTING DATA =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--dataname", type=str, help="taobao, ali, criteo", default="taobao")
    parser.add_argument("-s", "--datasave", type=bool, help="save or not", default=True)
    parser.add_argument("-l", "--seqlen", type=int, help="max seq len", default=32)
    args = parser.parse_args()
    running_dataloader(args.dataname, args.datasave, args.seqlen)
