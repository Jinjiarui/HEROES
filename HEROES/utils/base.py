import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score, average_precision_score, ndcg_score

class BaseModel(object):
    def __init__(self, sess, name):
        self.globalscope = None
        self.name = name
        self.sess = sess

    @property
    def gpu_config(self):
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        return gpu_config
    
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.globalscope)

    def train(self, **kwargs):
        raise NotImplementedError
    
    def save(self, step, model_dir):
        assert self.sess is not None
        model_dir = os.path.join(model_dir, self.name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print("===== SAVING MODEL ======")
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.globalscope)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, os.path.join(model_dir, self.name), global_step=step)
        print("====== FINGISH SAVING =====")
    
    def load(self, step, model_dir):
        assert self.sess is not None
        save_path = os.path.join(model_dir, self.name, self.name+"_"+str(step))
        print("===== RESTORING MODEL ======")
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.globalscope)
        saver = tf.train.Saver(model_vars)
        saver.restore(self.sess, save_path)
        print("===== FINISH RESTORING =====")

def cal_auc(pred, label, isbinary):
    # pred, label: list
    if np.sum(label) == 0:
        padding = [1]
        label = np.vstack([label, padding])
        pred = np.vstack([pred, padding])
    elif np.sum(label) == len(label):
        padding = [0]
        label = np.vstack([label, padding])
        pred = np.vstack([pred, padding])
    return roc_auc_score(y_score=pred, y_true=label)

def cal_acc(pred, label, isbinary):
    # pred, label: list
    res = []
    if _value in pred:
        if _value >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return accuracy_score(y_pred=res, y_true=label)

def cal_f1(pred, label):
    res = []
    for _value in pred:
        if _value >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return f1_score(y_pred=res, y_true=label)

def cal_logloss(pred, label):
    if np.sum(label) == 0:
        padding = [1]
        label = np.vstack([label, padding])
        pred = np.vstack([pred, padding])
    elif np.sum(label) == len(label):
        padding = [0]
        label = np.vstack([label, padding])
        pred = np.vstack([pred, padding])
    return log_loss(y_true=label, y_pred=pred, eps=1e-7, normalize=True)

def cal_ndcg(k, pred, label, length, isbinary):
    pred = np.array_split(pred.flatten(), pred.shape[0]/length)
    label = np.array_split(label.flatten(), label.shape[0]/length)
    return ndcg_score(y_true=label, y_score=pred, k=k)

def cal_map(k, pred, label, batch, length, isbinary):
    pred = np.array_split(pred.flatten(), pred.shape[0]/length)
    label = np.array_split(label.flatten(), label.shape[0]/length)
    _map = []
    for _pred, _label in zip(pred, label):
        index = np.argsort(-_pred)
        cum, count = 0.0, 0.0
        for i in range(0, k):
            if _label[index[i]] == 1 and pred[index[i]] >= 0.5:
                cum += (count + 1.0) / (i + 1.0)
                count += 1.0
        _sum = sum(_label)
        if _sum == 0:
            _map.append(0)
        else:
            _map.append(float(cum/k))
    return np.mean(_map)