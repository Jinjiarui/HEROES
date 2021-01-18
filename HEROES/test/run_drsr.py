from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import sys
import os
import pickle as pkl
sys.path.append("../")
from algo.drsr import DRSR

base_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(base_dir, "log")
data_dir = os.path.join(base_dir, "../data/")

def running_drsr(running_round, l2_weight, lr, grad_clip, batch_size, device, keep_prob, 
                    signal_weight, observation_weight, feature_dim, hidden_dim,
                    position_dim, signal_type, data_name, is_binary, print_interval):
    # load dataset
    print("===== START LOADING DATA =====")
    if data_name == "taobao":
        train_file = open(data_dir+"taobao_train.pkl", 'rb')
        test_file = open(data_dir+"taobao_test.pkl", 'rb')
        feature_num = 638072
        seq_len = 160
    elif data_name == "criteo":
        train_file = open(data_dir+"criteo_train.pkl", 'rb')
        test_file = open(data_dir+"criteo_test.pkl", 'rb')
        feature_num = 5897
        seq_len = 10
    elif data_name == "ali":
        raise NotImplementedError
    else:
        raise NameError
    train_data = pkl.load(train_file)
    test_data = pkl.load(test_file)
    train_file.close()
    test_file.close()
    print("===== FINISH LOADING DATA =====")
    # initialize
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model = DRSR(sess, l2_weight, lr, grad_clip, batch_size, device, keep_prob, signal_weight, 
                    observation_weight, feature_dim, hidden_dim, position_dim, seq_len, feature_num,
                    signal_type, data_name, is_binary)
    summary = SummaryObj(log_dir=logdir, log_name=drsr, n_group=1, sess=sess)
    summary.register(["train_acc, train_auc, train_f1, train_logloss, train_ndcg3, train_ndcg5, train_ndcg10, train_map3, train_map5, train_map10"+"test_acc, test_auc, test_f1, test_logloss, test_ndcg3, test_ndcg5, test_ndcg10, test_map3, test_map5, test_map10"])
    train_acc, train_auc, train_f1, train_logloss = [], [], [], []
    train_ndcg3, train_ndcg5, train_ndcg10, train_map3, train_map5, train_map10 = [], [], [], [], [], [] 
    for iteration in range(running_round):
        _acc, _auc, _f1, _logloss, _ndcg3, _ndcg5, _ndcg10, _map3, _map5, _map10 = model.train(traindata, print_interval)
        train_acc.append(_acc)
        train_auc.append(_auc)
        train_f1.append(_f1)
        train_logloss.append(_logloss)
        train_ndcg3.append(_ndcg3)
        train_ndcg5.append(_ndcg5)
        train_ndcg10.append(_ndcg10)
        train_map3.appned(_map3)
        train_map5.append(_map5)
        train_map10.append(_map10)
        print("===== TRAIN: ACC: {0:<.4f}, AUC: {0:<.4f}, F1: {0:<.4f}, LOGLOSS: {0:<.4f}, NDCG3: {0:<.4f}, NDCG5: {0:<.4f}, NDCG10: {0:<.4f}, MAP3: {0:<.4f}, MAP5: {0:<.4f}, MAP10: {0:<.4f} =====")
    test_acc, test_auc, test_f1, test_logloss, test_ndcg3, test_ndcg5, test_ndcg10, test_map3, test_map5, test_map10 = model.test(testdata, print_interval)
    print("===== TEST: ACC: {0:<.4f}, AUC: {0:<.4f}, F1: {0:<.4f}, LOGLOSS: {0:<.4f}, NDCG3: {0:<.4f}, NDCG5: {0:<.4f}, NDCG10: {0:<.4f}, MAP3: {0:<.4f}, MAP5: {0:<.4f}, MAP10: {0:<.4f} =====")    
    summary.write({
        "train_acc": np.mean(train_acc),
        "train_auc": np.mean(train_auc),
        "train_f1": np.mean(train_f1),
        "train_logloss": np.mean(train_logloss),
        "train_ndcg3": np.mean(train_ndcg3),
        "train_ndcg5": np.mean(train_ndcg5),
        "train_ndcg10": np.mean(train_ndcg10),
        "train_map3": np.mean(train_map3),
        "train_map5": np.mean(train_map5),
        "train_map10": np.mean(train_map10),
        "test_acc": np.mean(test_acc),
        "test_auc": np.mean(test_auc),
        "test_f1": np.mean(test_f1),
        "test_logloss": np.mean(test_logloss),
        "test_ndcg3": np.mean(test_ndcg3),
        "test_ndcg5": np.mean(test_ndcg5),
        "test_ndcg10": np.mean(test_ndcg10),
        "test_map3": np.mean(test_map3),
        "test_map5": np.mean(test_map5),
        "test_map10": np.mean(test_map10),
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learningrate", type=float, help="learning rate", default=1e-4)
    parser.add_argument("-b", "--batchsize", type=int, help="batch size", default=2)
    parser.add_argument("-n", "--hiddendim", type=int, help="hidden size", default=64)
    parser.add_argument("-c", "--cuda", type=int, help="cuda number", default=0)
    parser.add_argument("-d", "--dataname", type=str, help="name of dataset", default="criteo")
    parser.add_argument("-p", "--printinterval", type=int, help="print interval of loss", default=50)
    parser.add_argument("-i", "--isbinary", type=bool, help="decide size of output", default=False)
    parser.add_argument("-r", "--traininground", type=int, help="training round", default=20)
    parser.add_argument("-w", "--l2weight", type=float, help="l2 weight", default=1e-5)
    parser.add_argument("-g", "--gradclip", type=float, help="gradient clip", default=5)
    parser.add_argument("-k", "--keepprob", type=float, help="keep probability of dropout", default=0.8)
    parser.add_argument("-s", "--signalweight", type=float, help="loss weight of click or conversion", default=1.0)
    parser.add_argument("-o", "--observationweight", type=float, help="loss weight of observation", default=0.0)
    parser.add_argument("-f", "--featuredim", type=int, help="dim of feature", default=128)
    parser.add_argument("-m", "--positiondim", type=int, help="dim of position", default=64)
    parser.add_argument("-t", "--signaltype", type=str, help="type of signal: ctr and cvr", default="ctr")
    args = parser.parse_args()
    # device = "cuda:" + str(args.cuda)
    device = "/cpu:*"
    running_drsr(args.traininground, args.l2weight, args.learningrate, args.gradclip, args.batchsize, device, 
                args.keepprob, args.signalweight, args.observationweight, args.featuredim, args.hiddendim, args.positiondim,
                args.signaltype, args.dataname, args.isbinary, args.printinterval)


