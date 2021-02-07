import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score


def glorot(shape, name=None, scale=1.):
    import torch
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[-1] + shape[-2])) * scale
    initial = np.random.uniform(-init_range, init_range, shape)
    return torch.Tensor(initial)


def evaluate_auc(pred, label):
    if np.sum(label) == 0:
        m = [1]
        label = np.vstack([label, m])
        pred = np.vstack([pred, m])
    if np.sum(label) == len(label):
        m = [0]
        label = np.vstack([label, m])
        pred = np.vstack([pred, m])
    res = roc_auc_score(y_score=pred, y_true=label)
    return res


def evaluate_acc(pred, label):
    res = []
    for _value in pred:
        if _value >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return accuracy_score(y_pred=res, y_true=label)


def evaluate_f1_score(pred, label):
    res = []
    for _value in pred:
        if _value >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return f1_score(y_pred=res, y_true=label)


def evaluate_logloss(pred, label):
    if np.sum(label) == 0:
        m = [1]
        label = np.vstack([label, m])
        pred = np.vstack([pred, m])
    if np.sum(label) == len(label):
        m = [0]
        label = np.vstack([label, m])
        pred = np.vstack([pred, m])
    res = log_loss(y_true=label, y_pred=pred, eps=1e-7, normalize=True)
    return res


def evaluate_ndcg(k, pred_list, label_list, list_length):
    preds = np.array_split(pred_list.flatten(), list_length)
    labels = np.array_split(label_list.flatten(), list_length)
    '''NDCG = ndcg_score(y_true=labels, y_score=preds, k=k)'''

    def get_dcg(p_l):
        pred, label = p_l
        idx = np.argsort(-pred)
        sorted_label = label[np.argsort(-label)]
        if k is None:
            idx_range = np.arange(0, len(idx))
        else:
            idx_range = np.arange(0, min(k, len(idx)))
        accumulation = np.sum(label[idx[idx_range]] / np.log2(idx_range + 2.0))
        normalization = np.sum(sorted_label[idx_range] / np.log2(idx_range + 2.0))
        if normalization == 0.0:
            return -1
        else:
            return accumulation / normalization

    p_l_zip = zip(preds, labels)
    ndcg = np.array(list(map(get_dcg, p_l_zip)))
    NDCG = np.mean(ndcg[ndcg >= 0])
    return NDCG

def evaluate_map(k, pred_list, label_list, list_length):
    preds = np.array_split(pred_list.flatten(), list_length)
    labels = np.array_split(label_list.flatten(), list_length)

    def get_map(p_l):
        pred, label = p_l
        idx = np.argsort(-pred)
        if k is None:
            idx_range = np.arange(0, len(idx))
        else:
            idx_range = np.arange(0, min(k, len(idx)))
        count = np.zeros_like(idx_range)
        count[label[idx[idx_range]] == 1] = 1
        count[pred[idx[idx_range]] >= 0.5] += 1
        count[count < 2] = 0
        count_zero = np.argwhere(count == 0)
        count[count > 0] = 1
        count = np.cumsum(count)
        count[count_zero] = 0
        accumulation = np.sum(count / (idx_range + 1.0))
        x = label.sum()
        if x == 0:
            return -1
        else:
            if k is None or k > len(idx):
                return float(accumulation / len(idx))
            else:
                return float(accumulation / k)

    p_l_zip = zip(preds, labels)
    Map = np.array(list(map(get_map, p_l_zip)))
    MAP = np.mean(Map[Map >= 0])
    return MAP
