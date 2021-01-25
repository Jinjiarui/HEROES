import glob
import os
import pickle
import shutil
import sys
from datetime import timedelta, date

import numpy as np
import tensorflow as tf

sys.path.append("../")
from utils import utils
from load_alicpp import loadAliBatch

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("num_threads", 64, "Number of threads")
tf.flags.DEFINE_integer("max_features", 638072, "Number of max_features")
tf.flags.DEFINE_integer("embedding_size", 96, "Embedding size")
tf.flags.DEFINE_integer("seq_max_len", 160, "seq_max_len")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of epochs")
tf.flags.DEFINE_integer("batch_size", 50, "Number of batch size")
tf.flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
tf.flags.DEFINE_float("l2_reg", 0.001, "L2 regularization")
tf.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.flags.DEFINE_float("ctr_task_wgt", 0.5, "loss weight of ctr task")
tf.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.flags.DEFINE_integer("n_hidden", 256, "deep layers")
tf.flags.DEFINE_integer("n_classes", 1, "deep layers")
tf.flags.DEFINE_string("deep_layers", '196,128,64', "deep layers")
tf.flags.DEFINE_string("keep_prob", '0.5,0.5,0.5', "dropout rate")
tf.flags.DEFINE_string("gpus", '0', "list of gpus")
tf.flags.DEFINE_boolean("batch_norm", True, "perform batch normaization (True or False)")
tf.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.flags.DEFINE_string("data_dir", './../alicpp', "data dir")
tf.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.flags.DEFINE_string("model_dir", './../alicpp/model_alicpp_single_Heroes', "model check point dir")
tf.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval}")
tf.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus


# 转化一个序列列表为稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices[:, 1].max() + 1], dtype=np.int64)
    return indices, values, shape


input_id = tf.sparse_placeholder(tf.int32, shape=[None, None], name='id')
input_value = tf.sparse_placeholder(tf.float32, shape=[None, None], name='value')
seq_len = tf.placeholder(tf.int32, shape=[None], name='seqlen')
click_label = tf.placeholder(tf.float32, shape=[None, FLAGS.seq_max_len, FLAGS.n_classes])
conversion_label = tf.placeholder(tf.float32, shape=[None, FLAGS.seq_max_len, FLAGS.n_classes], name='labels')

embedding_size = FLAGS.embedding_size
l2_reg = FLAGS.l2_reg
learning_rate = FLAGS.learning_rate
max_features = FLAGS.max_features
seq_max_len = FLAGS.seq_max_len
ctr_task_wgt = FLAGS.ctr_task_wgt
layers = list(map(int, FLAGS.deep_layers.split(',')))
dropout = list(map(float, FLAGS.keep_prob.split(',')))

position_embedding = tf.Variable(tf.random_normal([seq_max_len, embedding_size], stddev=0.1))
position_copy = tf.tile(position_embedding, [tf.shape(click_label)[0], 1])  # (bs*seq,embed)
embedding_matrix = tf.Variable(tf.random_normal([max_features, embedding_size], stddev=0.1))
inputs = tf.nn.embedding_lookup_sparse(embedding_matrix, sp_ids=input_id, sp_weights=input_value)  # (bs*seq,embed)
inputs = tf.concat([inputs, position_copy], axis=-1)  # (bs*seq,2*embed)
inputs = tf.reshape(inputs, (-1, seq_max_len, inputs.shape[-1]))  # (bs,seq,2*embed)
inputs = tf.transpose(inputs, [1, 0, 2])  # (seq,bs,embed)


def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=True, reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


def fully_connected(x, basename, mode):
    for i in range(len(layers)):
        x = tf.layers.dense(inputs=x, units=layers[i], use_bias=True, reuse=tf.AUTO_REUSE, name='%s%d' % (basename, i))
        x = tf.nn.relu(x)
        if mode == 'train':
            train_phase = True
        else:
            train_phase = False
        if FLAGS.batch_norm:
            x = batch_norm_layer(x, train_phase=train_phase, scope_bn='%s_bn_%d' % (basename, i))
        if mode == 'train':
            x = tf.nn.dropout(x, keep_prob=dropout[i])
            # Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)
    return tf.sigmoid(tf.layers.dense(inputs=x, units=n_classes, reuse=tf.AUTO_REUSE, use_bias=True, name=basename))


with tf.name_scope('RNN'), tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
    n_hidden = FLAGS.n_hidden
    n_classes = FLAGS.n_classes
    H_c = tf.zeros(shape=(tf.shape(inputs)[1], n_hidden))  # (bs,hidden)
    H_v = tf.zeros(shape=(tf.shape(inputs)[1], n_hidden))  # (bs,hidden)
    s_c = tf.zeros(shape=(tf.shape(inputs)[1], n_hidden))  # (bs,hidden)
    s_v = tf.zeros(shape=(tf.shape(inputs)[1], n_hidden))  # (bs,hidden)
    prediction_c = []
    prediction_v = []
    mode = FLAGS.task_type
    '''g = tf.where(tf.sigmoid(tf.layers.dense(H_c, units=n_classes, name='H_c_p')) >= 0.5,
                 tf.ones(shape=(tf.shape(H_c)[0], n_classes)),
                 tf.zeros(shape=(tf.shape(H_c)[0], n_classes)))  # (bs,1)'''
    g = tf.sigmoid(tf.layers.dense(inputs=H_c, units=n_classes, reuse=tf.AUTO_REUSE, use_bias=True, name='H_c'))
    pc = tf.ones_like(g)  # The product of 1-H_c
    pv = tf.ones_like(g)  # The product of 1-H_v
    g = tf.tile(g, [1, n_hidden])  # (bs,hidden)
    for i in range(seq_max_len):
        f_c = tf.sigmoid(tf.layers.dense(inputs[i], units=n_hidden, use_bias=True, name='xfc')
                         + tf.layers.dense(H_c, units=n_hidden, use_bias=False, name='hfc'))
        i_c = tf.sigmoid(tf.layers.dense(inputs[i], units=n_hidden, use_bias=True, name='xic')
                         + tf.layers.dense(H_c, units=n_hidden, use_bias=False, name='hic'))
        o_c = tf.sigmoid(tf.layers.dense(inputs[i], units=n_hidden, use_bias=True, name='xoc')
                         + tf.layers.dense(H_c, units=n_hidden, use_bias=False, name='hoc'))
        g_c = tf.tanh(tf.layers.dense(inputs[i], units=n_hidden, use_bias=True, name='xgc')
                      + tf.layers.dense(H_c, units=n_hidden, use_bias=False, name='hgc'))
        s_c_hat = tf.tanh(tf.multiply(1 - g, tf.layers.dense(H_c, units=n_hidden, use_bias=False, name='s_c_hat_c')) \
                          + tf.multiply(g, tf.layers.dense(H_v, units=n_hidden, use_bias=False, name='s_c_hat_v')))
        s_c = s_c_hat + tf.multiply(i_c, g_c) + tf.multiply(1 - g, tf.multiply(f_c, s_c))
        H_c = tf.multiply(o_c, tf.tanh(s_c))
        H_c_p = tf.sigmoid(tf.layers.dense(inputs=H_c, units=n_classes, reuse=tf.AUTO_REUSE, use_bias=True, name='H_c'))
        prediction_c.append(H_c_p)
        # g = tf.where(prediction_c[-1] >= 0.5, tf.ones_like(prediction_c[-1]), tf.zeros_like(prediction_c[-1]))
        pc = tf.where(prediction_c[-1] >= 0.5, tf.ones_like(prediction_c[-1]), tf.multiply(1 - H_c_p, pc))
        g = tf.tile(H_c_p, [1, n_hidden])
        f_v = tf.sigmoid(tf.layers.dense(inputs[i], units=n_hidden, use_bias=True, name='xfv')
                         + tf.layers.dense(H_v, units=n_hidden, use_bias=False, name='hfv'))
        i_v = tf.sigmoid(tf.layers.dense(inputs[i], units=n_hidden, use_bias=True, name='xiv')
                         + tf.layers.dense(H_v, units=n_hidden, use_bias=False, name='hiv'))
        o_v = tf.sigmoid(tf.layers.dense(inputs[i], units=n_hidden, use_bias=True, name='xov')
                         + tf.layers.dense(H_v, units=n_hidden, use_bias=False, name='hov'))
        g_v = tf.tanh(tf.layers.dense(inputs[i], units=n_hidden, use_bias=True, name='xgv')
                      + tf.layers.dense(H_v, units=n_hidden, use_bias=False, name='hgv'))
        s_v_hat = tf.tanh(tf.layers.dense(H_v, units=n_hidden, use_bias=False, name='s_v_hat_v') \
                          + g * tf.layers.dense(H_c, units=n_hidden, use_bias=False, name='s_v_hat_c'))
        s_v = s_v_hat + tf.multiply(1 - g, s_v) + tf.multiply(g, tf.multiply(f_v, s_v) + tf.multiply(i_v, g_v))
        H_v = tf.multiply(1 - g, H_v) + tf.multiply(g, tf.multiply(o_v, tf.tanh(s_v)))

        H_v_p = tf.sigmoid(tf.layers.dense(inputs=H_v, units=n_classes, reuse=tf.AUTO_REUSE, use_bias=True, name='H_v'))
        prediction_v.append(tf.multiply(H_v_p, pv))
        pv = tf.where(prediction_c[-1] > 0.5, tf.ones_like(prediction_v[-1]), tf.multiply(1 - H_v_p, pv))

mask = tf.sequence_mask(seq_len, seq_max_len)
prediction_c = tf.boolean_mask(tf.transpose(tf.stack(prediction_c), [1, 0, 2]), mask)
prediction_v = tf.boolean_mask(tf.transpose(tf.stack(prediction_v), [1, 0, 2]), mask)
reshape_click_label = tf.boolean_mask(click_label, mask)
reshape_conversion_label = tf.boolean_mask(conversion_label, mask)
epsilon = 1e-7
click_num = tf.to_float(tf.count_nonzero(reshape_click_label))
conversion_num = tf.to_float(tf.count_nonzero(reshape_conversion_label))
click_weight = (click_num + 1) / (tf.to_float(tf.size(reshape_click_label)) + 2)
click_weight = tf.maximum(click_weight, 0.09)
conversion_weight = (conversion_num + 1) / (tf.to_float(tf.size(reshape_conversion_label)) + 2)
conversion_weight = tf.maximum(conversion_weight, 0.05)
'''click_loss = tf.reduce_mean(tf.losses.log_loss(labels=reshape_click_label, predictions=prediction_c))
conversion_loss = tf.reduce_mean(tf.losses.log_loss(labels=reshape_conversion_label, predictions=prediction_v))'''
click_loss = -(1 - click_weight) * reshape_click_label * tf.log(prediction_c + epsilon) - \
             click_weight * (1 - reshape_click_label) * tf.log(1 - prediction_c + epsilon)
click_loss = tf.reduce_mean(click_loss)
conversion_loss = -(1 - conversion_weight) * reshape_conversion_label * tf.log(prediction_v + epsilon) - \
                  conversion_weight * (1 - reshape_conversion_label) * tf.log(1 - prediction_v + epsilon)
conversion_loss = tf.reduce_mean(conversion_loss)
loss = ((1 - ctr_task_wgt) * click_loss + ctr_task_wgt * conversion_loss) * 100
for v in tf.trainable_variables():
    loss += l2_reg * tf.nn.l2_loss(v)
tf.summary.scalar('ctr_loss', click_loss)
tf.summary.scalar('ctcvr_loss', conversion_loss)

threshold = 0.5
one_click = tf.ones_like(reshape_click_label)
zero_click = tf.zeros_like(reshape_click_label)
one_cvr = tf.ones_like(reshape_conversion_label)
zero_cvr = tf.zeros_like(reshape_conversion_label)
eval_metric_ops = {
    "CTR_AUC": tf.metrics.auc(reshape_click_label, prediction_c),
    "CTR_ACC": tf.metrics.accuracy(reshape_click_label,
                                   tf.where(prediction_c >= threshold, one_click, zero_click)),
    "CTCVR_AUC": tf.metrics.auc(reshape_conversion_label, prediction_v),
    "CTCVR_ACC": tf.metrics.accuracy(reshape_conversion_label, tf.where(prediction_v >= threshold, one_cvr, zero_cvr))
}
global_step = tf.Variable(0, trainable=False)
cov_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 50000, 0.96)
optimizer = tf.train.AdamOptimizer(learning_rate=cov_learning_rate)
train_op = optimizer.minimize(loss, global_step=global_step)
saver = tf.train.Saver(max_to_keep=3)


def main(_):
    # ------check Arguments------
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir + "R"

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('seq_max_len', FLAGS.seq_max_len)
    print('max_features', FLAGS.max_features)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('n_hidden', FLAGS.n_hidden)
    print('n_classes', FLAGS.n_classes)
    print('keep_prob ', FLAGS.keep_prob)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('l2_reg ', FLAGS.l2_reg)
    print('ctr_task_wgt ', FLAGS.ctr_task_wgt)

    tr_files = glob.glob("%s/train/remap_sample/r*.txt" % FLAGS.data_dir)
    print("train_files:", tr_files)
    te_files = glob.glob("%s/test/remap_sample/r*.txt" % FLAGS.data_dir)
    print("test_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                            device_count={'CPU': FLAGS.num_threads})
    config.gpu_options.allow_growth = True
    if FLAGS.task_type == 'train':
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if not FLAGS.clear_existing_model:
                saver.restore(sess, os.path.join(FLAGS.model_dir, 'BestModel-100'))
            for tr in tr_files:
                tr_len = tr + '.pkl'
                with open(tr_len, 'rb') as len_f:
                    tr_len_list = list(pickle.load(len_f))
                for i in range(FLAGS.num_epochs):
                    step = 0
                    tr_infile = open(tr, 'r')
                    best_auc = 0
                    while True:
                        step += 1
                        total_data_id, total_data_value, total_click, total_label, total_seqlen = loadAliBatch(
                            FLAGS.seq_max_len, tr_infile,
                            tr_len_list[(step - 1) * FLAGS.batch_size:step * FLAGS.batch_size])
                        if not total_label:
                            break
                        feed_dict = {
                            input_id: sparse_tuple_from(total_data_id),
                            input_value: sparse_tuple_from(total_data_value, dtype=np.float32),
                            seq_len: total_seqlen,
                            click_label: total_click,
                            conversion_label: total_label
                        }
                        batch_c_label, batch_c, _, batch_loss, batch_cvr_loss, batch_click_loss, batch_eval = sess.run(
                            [reshape_click_label, prediction_c, train_op, loss, conversion_loss, click_loss,
                             eval_metric_ops],
                            feed_dict=feed_dict)
                        print(np.sum(batch_c_label), np.sum(batch_c >= 0.5))
                        print(utils.evaluate_acc(label=batch_c_label, pred=batch_c),
                              utils.evaluate_auc(label=batch_c_label, pred=batch_c),
                              utils.evaluate_acc(label=batch_c_label, pred=np.zeros_like(batch_c_label)))

                        print("Epoch:{}\tStep:{}".format(i, step))
                        print("click_AUC = " + "{}".format(batch_eval['CTR_AUC'][0]), end='\t')
                        print("conversion_AUC = " + "{}".format(batch_eval['CTCVR_AUC'][0]), end='\t')
                        print("click_ACC = " + "{}".format(batch_eval['CTR_ACC'][0]), end='\t')
                        print("conversion_ACC = " + "{}".format(batch_eval['CTCVR_ACC'][0]))
                        print("Loss = " + "{}".format(batch_loss), end='\t')
                        print("Clk Loss = " + "{}".format(batch_click_loss), end='\t')
                        print("Cov_Loss = " + "{}".format(batch_cvr_loss))
                        if step % 50 == 0:
                            saver.save(sess, os.path.join(FLAGS.model_dir, 'MyModel'), global_step=step)
                            print("Test----------------")
                            test_cvr_auc = 0.0
                            test_ctr_auc = 0.0
                            test_cvr_acc = 0.0
                            test_ctr_acc = 0.0
                            test_loss = 0
                            test_clk_loss = 0
                            test_cvr_loss = 0
                            test_step = 0
                            for te in te_files:
                                te_infile = open(te, 'r')
                                te_len = te + '.pkl'
                                with open(te_len, 'rb') as len_f:
                                    te_len_list = list(pickle.load(len_f))
                                for _ in range(10):
                                    test_step += 1
                                    total_data_id, total_data_value, total_click, total_label, total_seqlen = loadAliBatch(
                                        FLAGS.seq_max_len, te_infile,
                                        te_len_list[(test_step - 1) * FLAGS.batch_size:test_step * FLAGS.batch_size])
                                    if not total_seqlen:
                                        break
                                    feed_dict = {
                                        input_id: sparse_tuple_from(total_data_id),
                                        input_value: sparse_tuple_from(total_data_value, dtype=np.float32),
                                        seq_len: total_seqlen,
                                        click_label: total_click,
                                        conversion_label: total_label
                                    }
                                    batch_loss, batch_cvr_loss, batch_click_loss, batch_eval = sess.run(
                                        [loss, conversion_loss, click_loss, eval_metric_ops],
                                        feed_dict=feed_dict)
                                    test_ctr_auc += batch_eval['CTR_AUC'][0]
                                    test_cvr_auc += batch_eval['CTCVR_AUC'][0]
                                    test_ctr_acc += batch_eval['CTR_ACC'][0]
                                    test_cvr_acc += batch_eval['CTCVR_ACC'][0]
                                    test_loss += batch_loss
                                    test_cvr_loss += batch_cvr_loss
                                    test_clk_loss += batch_click_loss
                                    if len(total_label) != FLAGS.batch_size:
                                        break
                                te_infile.close()
                            test_ctr_auc /= test_step
                            test_cvr_auc /= test_step
                            test_cvr_acc /= test_step
                            test_ctr_acc /= test_step
                            test_loss /= test_step
                            test_clk_loss /= test_step
                            test_cvr_loss /= step

                            print("click_AUC = " + "{}".format(test_ctr_auc), end='\t')
                            print("conversion_AUC = " + "{}".format(test_cvr_auc), end='\t')
                            print("click_ACC = " + "{}".format(test_ctr_acc), end='\t')
                            print("conversion_ACC = " + "{}".format(test_cvr_acc))
                            print("Loss = " + "{}".format(test_loss), end='\t')
                            print("Clk Loss = " + "{}".format(test_clk_loss), end='\t')
                            print("Cov_Loss = " + "{}".format(test_cvr_loss))
                            if test_cvr_auc > best_auc:
                                print("Save----------------")
                                saver.save(sess, os.path.join(FLAGS.model_dir, 'BestModel'), global_step=step)
                    tr_infile.close()
    if FLAGS.task_type == 'eval':
        with tf.Session(config=config) as sess:
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, os.path.join(FLAGS.model_dir, 'BestModel-150'))
            for te in te_files:
                print(te)
                te_infile = open(te, 'r')
                te_len = te + '.pkl'
                with open(te_len, 'rb') as len_f:
                    te_len_list = list(pickle.load(len_f))
                step = 0
                while True:
                    step += 1
                    total_data_id, total_data_value, total_click, total_label, total_seqlen = loadAliBatch(
                        FLAGS.seq_max_len, te_infile,
                        te_len_list[(step - 1) * FLAGS.batch_size:step * FLAGS.batch_size])
                    if not total_seqlen:
                        break
                    # print(total_click)
                    feed_dict = {
                        input_id: sparse_tuple_from(total_data_id),
                        input_value: sparse_tuple_from(total_data_value, dtype=np.float32),
                        seq_len: total_seqlen,
                        click_label: total_click,
                        conversion_label: total_label
                    }
                    batch_loss, batch_cvr_loss, batch_click_loss, batch_eval = sess.run(
                        [loss, conversion_loss, click_loss, eval_metric_ops], feed_dict=feed_dict)
                    print("click_AUC = " + "{}".format(batch_eval['CTR_AUC'][0]), end='\t')
                    print("conversion_AUC = " + "{}".format(batch_eval['CTCVR_AUC'][0]), end='\t')
                    print("click_ACC = " + "{}".format(batch_eval['CTR_ACC'][0]), end='\t')
                    print("conversion_ACC = " + "{}".format(batch_eval['CTCVR_ACC'][0]))
                    print("Loss = " + "{}".format(batch_loss), end='\t')
                    print("Clk Loss = " + "{}".format(batch_click_loss), end='\t')
                    print("Cov_Loss = " + "{}".format(batch_cvr_loss))
                    if len(total_label) != FLAGS.batch_size:
                        break
    if FLAGS.task_type == 'infer':
        with tf.Session(config=config) as sess:
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, os.path.join(FLAGS.model_dir, 'BestModel-150'))
            for te in te_files:
                print(te)
                te_infile = open(te, 'r')
                te_len = te + '.pkl'
                with open(te_len, 'rb') as len_f:
                    te_len_list = list(pickle.load(len_f))
                step = 0
                pctr = np.array([])
                y = np.array([])
                pctcvr = np.array([])
                z = np.array([])
                while True:
                    step += 1
                    total_data_id, total_data_value, total_click, total_label, total_seqlen = loadAliBatch(
                        FLAGS.seq_max_len, te_infile,
                        te_len_list[(step - 1) * FLAGS.batch_size:step * FLAGS.batch_size])
                    print(step)
                    if not total_seqlen:
                        break
                    feed_dict = {
                        input_id: sparse_tuple_from(total_data_id),
                        input_value: sparse_tuple_from(total_data_value, dtype=np.float32),
                        seq_len: total_seqlen,
                        click_label: total_click,
                        conversion_label: total_label
                    }
                    p_click, l_click, p_conver, l_conver = sess.run([prediction_c, reshape_click_label, prediction_v,
                                                                     reshape_conversion_label], feed_dict=feed_dict)
                    pctr = np.append(pctr, p_click)
                    y = np.append(y, l_click)
                    pctcvr = np.append(pctcvr, p_conver)
                    z = np.append(z, l_conver)
                click_result = {'loss': 0, 'acc': 0, 'auc': 0, 'f1': 0, 'ndcg': 0, 'map': 0}
                conversion_result = {'loss': 0, 'acc': 0, 'auc': 0, 'f1': 0, 'ndcg': 0, 'map': 0}
                click_result['loss'] = utils.evaluate_logloss(pctr, y)
                click_result['acc'] = utils.evaluate_acc(pctr, y)
                click_result['auc'] = utils.evaluate_auc(pctr, y)
                click_result['f1'] = utils.evaluate_f1_score(pctr, y)
                click_result['ndcg'] = utils.evaluate_ndcg(None, pctr, y, len(pctr))
                click_result['ndcg1'] = utils.evaluate_ndcg(1, pctr, y, len(pctr))
                click_result['ndcg3'] = utils.evaluate_ndcg(3, pctr, y, len(pctr))
                click_result['ndcg5'] = utils.evaluate_ndcg(5, pctr, y, len(pctr))
                click_result['ndcg10'] = utils.evaluate_ndcg(10, pctr, y, len(pctr))
                click_result['map'] = utils.evaluate_map(len(pctr), pctr, y, len(pctr))
                click_result['map1'] = utils.evaluate_map(1, pctr, y, len(pctr))
                click_result['map3'] = utils.evaluate_map(3, pctr, y, len(pctr))
                click_result['map5'] = utils.evaluate_map(5, pctr, y, len(pctr))
                click_result['map10'] = utils.evaluate_map(10, pctr, y, len(pctr))

                conversion_result['loss'] = utils.evaluate_logloss(pctcvr, z)
                conversion_result['acc'] = utils.evaluate_acc(pctcvr, z)
                conversion_result['auc'] = utils.evaluate_auc(pctcvr, z)
                conversion_result['f1'] = utils.evaluate_f1_score(pctcvr, z)
                conversion_result['ndcg'] = utils.evaluate_ndcg(None, pctcvr, z, len(pctcvr))
                conversion_result['ndcg1'] = utils.evaluate_ndcg(1, pctcvr, z, len(pctcvr))
                conversion_result['ndcg3'] = utils.evaluate_ndcg(3, pctcvr, z, len(pctcvr))
                conversion_result['ndcg5'] = utils.evaluate_ndcg(5, pctcvr, z, len(pctcvr))
                conversion_result['ndcg10'] = utils.evaluate_ndcg(10, pctcvr, z, len(pctcvr))
                conversion_result['map'] = utils.evaluate_map(len(pctcvr), pctcvr, z, len(pctcvr))
                conversion_result['map1'] = utils.evaluate_map(1, pctcvr, z, len(pctcvr))
                conversion_result['map3'] = utils.evaluate_map(3, pctcvr, z, len(pctcvr))
                conversion_result['map5'] = utils.evaluate_map(5, pctcvr, z, len(pctcvr))
                conversion_result['map10'] = utils.evaluate_map(10, pctcvr, z, len(pctcvr))
                print("Click Result")
                for k, v in click_result.items():
                    print("{}:{}".format(k, v))
                print()
                print("Conversion Result")
                for k, v in conversion_result.items():
                    print("{}:{}".format(k, v))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
