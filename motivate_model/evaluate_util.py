import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score


def glorot(shape, scale=1.):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[-1] + shape[-2])) * scale
    initial = np.random.uniform(-init_range, init_range, shape)
    return tf.convert_to_tensor(initial)


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
    res = np.where(pred >= 0.5, np.ones_like(pred), np.zeros_like(pred))
    return accuracy_score(y_pred=res, y_true=label)


def evaluate_f1_score(pred, label):
    res = np.where(pred >= 0.5, np.ones_like(pred), np.zeros_like(pred))
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


def evaluate_ndcg(k, pred_label_list, len_cumsum):
    """len_cumsum = tf.cumsum(list_length)
    len_cumsum = tf.concat([[0], len_cumsum], axis=-1)"""
    # print(pred_label_list)
    # print(len_cumsum)
    if k is None:
        k = tf.shape(pred_label_list)[0]

    def get_dcg(i):
        pred, label = pred_label_list[len_cumsum[i]:len_cumsum[i + 1], 0], \
                      pred_label_list[len_cumsum[i]:len_cumsum[i + 1], 1]
        idx = tf.argsort(-pred)
        # sorted_label = label[tf.argsort(-label)]
        sorted_label = -tf.sort(-label)
        idx_range = tf.range(0, tf.minimum(k, tf.shape(idx)[0]))
        idx_log = tf.math.log(tf.cast(idx_range, tf.float32) + 2.0)
        accumulation = tf.reduce_sum(tf.gather(label, tf.gather(idx, idx_range)) / idx_log)
        normalization = tf.reduce_sum(tf.gather(sorted_label, idx_range) / idx_log)
        return tf.where(tf.equal(normalization, 0), -1.0, accumulation / normalization)

    ndcg = tf.map_fn(get_dcg, tf.range(tf.shape(len_cumsum)[0] - 1), dtype=tf.float32,
                     parallel_iterations=72)
    NDCG = tf.reduce_mean(tf.gather_nd(ndcg, tf.where(ndcg >= 0)))
    return NDCG


def evaluate_map(k, pred_label_list, len_cumsum):
    """len_cumsum = tf.cumsum(list_length)
    len_cumsum = tf.concat([[0], len_cumsum], axis=-1)"""
    # print(pred_label_list)
    # print(len_cumsum)
    if k is None:
        k = tf.shape(pred_label_list)[0]

    def get_map(i):
        pred, label = pred_label_list[len_cumsum[i]:len_cumsum[i + 1], 0], \
                      pred_label_list[len_cumsum[i]:len_cumsum[i + 1], 1]
        idx = tf.argsort(-pred)
        k_i = tf.minimum(k, tf.shape(idx)[0])
        idx_range = tf.range(0, k_i)
        cond1 = tf.gather(label, tf.gather(idx, idx_range)) > 0
        cond2 = tf.gather(pred, tf.gather(idx, idx_range)) >= 0.5
        count = tf.where(cond1 & cond2, tf.ones_like(idx_range), tf.zeros_like(idx_range))

        count_sum = tf.cast(tf.cumsum(count), tf.float32)
        count_sum = tf.where(tf.equal(count, 0), tf.zeros_like(count, dtype=count_sum.dtype), count_sum)
        accumulation = tf.reduce_sum(count_sum / (tf.cast(idx_range, tf.float32) + 1.0), axis=-1)
        return tf.where(tf.equal(tf.reduce_sum(label, axis=-1), 0.0),
                        -1.0, accumulation / tf.cast(k_i, accumulation.dtype))

    map = tf.map_fn(get_map, tf.range(tf.shape(len_cumsum)[0] - 1), dtype=tf.float32,
                    parallel_iterations=72)

    MAP = tf.reduce_mean(tf.gather_nd(map, tf.where(map >= 0)))
    return MAP


def copy_positive(pred, label, list_length):
    label_list = np.array_split(label.flatten(), list_length)
    pred_list = np.array_split(pred.flatten(), list_length)
    positive_loc = list(map(lambda i: np.argwhere(i > 0).flatten(), label_list))
    negative_loc = list(map(lambda i: np.argwhere(i < 1).flatten(), label_list))

    def pred_f(i):
        if np.sum(positive_loc[i]) == 0:
            return pred_list[i]
        weight = int(np.size(negative_loc[i]) / (np.size(positive_loc[i])))
        return np.append(pred_list[i][negative_loc[i]], pred_list[i][positive_loc[i]].repeat(weight))

    def label_f(i):
        if np.sum(positive_loc[i]) == 0:
            return label_list[i]
        weight = int(np.size(negative_loc[i]) / (np.size(positive_loc[i])))
        return np.append(label_list[i][negative_loc[i]], label_list[i][positive_loc[i]].repeat(weight))

    label_positive_copy = list(map(label_f, range(len(label_list))))
    pred_positive_copy = list(map(pred_f, range(len(label_list))))
    new_list_length = np.cumsum(list(map(lambda i: np.size(i), label_positive_copy)))
    return np.concatenate(pred_positive_copy), np.concatenate(label_positive_copy), new_list_length


if __name__ == '__main__':
    import os
    import time

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    pred =[0.6, 0.5, 0.8, 0.1]
    label = [1, 0, 0, 0]
    len_list = [4]
    len_list = np.cumsum(len_list)
    len_list = np.append([0], len_list)
    print(len_list)
    pred_label = np.concatenate((np.expand_dims(pred, axis=-1), np.expand_dims(label, axis=-1)), axis=-1)
    print(pred_label)
    pred_label, len_list = tf.convert_to_tensor(pred_label, dtype=tf.float32), tf.convert_to_tensor(len_list)
    print(pred_label, len_list)
    pred_label_holder = tf.placeholder(tf.float32, shape=[len(label), 2], name='seqlen')
    NDCG = evaluate_ndcg(None, pred_label, len_list)
    MAP = evaluate_map(None, pred_label, len_list)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        t1 = time.perf_counter()
        print(sess.run([NDCG]))
        t2 = time.perf_counter()
        print(t2 - t1)
        t1 = time.perf_counter()
        print(sess.run([MAP]))
        t2 = time.perf_counter()
        print(t2 - t1)
