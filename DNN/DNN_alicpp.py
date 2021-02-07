import glob
import os
import pickle
import shutil
import sys
from collections import defaultdict
from datetime import timedelta, date
from multiprocessing import Process, Queue

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
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
tf.flags.DEFINE_float("l2_reg", 0.01, "L2 regularization")
tf.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.flags.DEFINE_float("ctr_task_wgt", 0.5, "loss weight of ctr task")
tf.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.flags.DEFINE_integer("n_hidden", 256, "deep layers")
tf.flags.DEFINE_integer("n_classes", 1, "deep layers")
tf.flags.DEFINE_float("keep_prob", 0.5, "dropout rate")
tf.flags.DEFINE_string("gpus", '', "list of gpus")
tf.flags.DEFINE_boolean("batch_norm", True, "perform batch normaization (True or False)")
tf.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.flags.DEFINE_string("data_dir", './../alicpp', "data dir")
tf.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.flags.DEFINE_string("model_dir", './../alicpp/model_alicpp_dnn', "model check point dir")
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
click_label = tf.placeholder(tf.float32, shape=[None, FLAGS.seq_max_len, FLAGS.n_classes], name='click_labels')
conversion_label = tf.placeholder(tf.float32, shape=[None, FLAGS.seq_max_len, FLAGS.n_classes], name='labels')

embedding_size = FLAGS.embedding_size
l2_reg = FLAGS.l2_reg
learning_rate = FLAGS.learning_rate
max_features = FLAGS.max_features
seq_max_len = FLAGS.seq_max_len
ctr_task_wgt = FLAGS.ctr_task_wgt
n_hidden = FLAGS.n_hidden
n_classes = FLAGS.n_classes
embedding_matrix = tf.Variable(
    tf.random_normal([max_features, embedding_size], stddev=0.1))

x = tf.nn.embedding_lookup_sparse(embedding_matrix, sp_ids=input_id, sp_weights=input_value)  # (bs*seq,embed)
x = tf.reshape(x, (-1, seq_max_len, x.shape[-1]))  # (bs,seq,embed)

with tf.name_scope('P_o'), tf.variable_scope('P_o', reuse=tf.AUTO_REUSE):
    lstm_cell_o = tf.contrib.rnn.BasicLSTMCell(FLAGS.n_hidden, state_is_tuple=True)
    lstm_cell_o = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_o, input_keep_prob=FLAGS.keep_prob,
                                                output_keep_prob=FLAGS.keep_prob)
    states_o, last_o = tf.nn.dynamic_rnn(cell=lstm_cell_o, inputs=x, sequence_length=seq_len, dtype=tf.float32)

with tf.name_scope('P_r'), tf.variable_scope('P_r', reuse=tf.AUTO_REUSE):
    lstm_cell_r = tf.contrib.rnn.BasicLSTMCell(FLAGS.n_hidden, state_is_tuple=True)
    lstm_cell_r = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_r, input_keep_prob=FLAGS.keep_prob,
                                                output_keep_prob=FLAGS.keep_prob)
    states_r, last_r = tf.nn.dynamic_rnn(cell=lstm_cell_r, inputs=x, sequence_length=seq_len,
                                         dtype=tf.float32)  # (bs,seq,hidden)

with tf.name_scope('P_c'), tf.variable_scope('P_c', reuse=tf.AUTO_REUSE):
    x_c = tf.multiply(states_o, states_r)
    W_c = tf.get_variable('W_c', [n_hidden, n_classes], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_c = tf.get_variable('b_c', [n_classes], initializer=tf.truncated_normal_initializer(stddev=0.1))
    x_c = tf.nn.xw_plus_b(x_c, W_c, b_c)  # (bs,seq,n_classes)
    P_c = tf.nn.sigmoid(x_c)

with tf.name_scope('P_a'), tf.variable_scope('P_a', reuse=tf.AUTO_REUSE):
    lstm_cell_a = tf.contrib.rnn.BasicLSTMCell(FLAGS.n_hidden, state_is_tuple=True)
    lstm_cell_a = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_a, input_keep_prob=FLAGS.keep_prob,
                                                output_keep_prob=FLAGS.keep_prob)
    states_a, last_a = tf.nn.dynamic_rnn(cell=lstm_cell_a, inputs=states_r, sequence_length=seq_len,
                                         dtype=tf.float32)  # (bs,seq,hidden)
    W_a = tf.get_variable('W_a', [n_hidden, n_classes], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_a = tf.get_variable('b_a', [n_classes], initializer=tf.truncated_normal_initializer(stddev=0.1))
    x_a = tf.nn.xw_plus_b(states_a, W_a, b_a)  # (bs,seq,n_classes)

with tf.name_scope('P_v'):
    x_v = tf.multiply(x_c, x_a)
    P_v = tf.nn.sigmoid(x_v)

print(P_v.shape)

# ------bulid loss------
mask = tf.sequence_mask(seq_len, seq_max_len)

reshape_click_label = tf.boolean_mask(click_label, mask)
reshape_conversion_label = tf.boolean_mask(conversion_label, mask)
P_c = tf.boolean_mask(P_c, mask)
P_v = tf.boolean_mask(P_v, mask)
epsilon = 1e-7
click_weight = 0.08
conversion_weight = 0.04
click_loss = - (1 - click_weight) * reshape_click_label * tf.log(P_c + epsilon) - \
             click_weight * (1 - reshape_click_label) * tf.log(1 - P_c + epsilon)
click_loss = tf.reduce_mean(click_loss)
conversion_loss = - (1 - conversion_weight) * reshape_conversion_label * tf.log(P_v + epsilon) - \
                  conversion_weight * (1 - reshape_conversion_label) * tf.log(1 - P_v + epsilon)
conversion_loss = tf.reduce_mean(conversion_loss)
# click_loss = tf.reduce_mean(tf.losses.log_loss(labels=reshape_click_label, predictions=P_c))
# conversion_loss = tf.reduce_mean(tf.losses.log_loss(labels=reshape_conversion_label, predictions=P_v))
loss = ((1 - ctr_task_wgt) * click_loss + ctr_task_wgt * conversion_loss) * 1000
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
    "CTR_AUC": tf.metrics.auc(reshape_click_label, P_c),
    "CTR_ACC": tf.metrics.accuracy(reshape_click_label,
                                   tf.where(P_c >= threshold, one_click, zero_click)),
    "CTCVR_AUC": tf.metrics.auc(reshape_conversion_label, P_v),
    "CTCVR_ACC": tf.metrics.accuracy(reshape_conversion_label, tf.where(P_v >= threshold, one_cvr, zero_cvr))
}

global_step = tf.Variable(0, trainable=False)
cov_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 50000, 0.96)
optimizer = tf.train.AdamOptimizer(learning_rate=cov_learning_rate)
train_op = optimizer.minimize(loss, global_step=global_step)
saver = tf.train.Saver(max_to_keep=3)


def write(q, flag, file_name, num_epoch=1, buffer_size=10):
    print(file_name)

    file_len = file_name + '.pkl'
    with open(file_len, 'rb') as len_f:
        file_len_list = list(pickle.load(len_f))

    for _ in range(num_epoch):
        step = 0
        infile = open(file_name, 'r')
        while True:
            if q.qsize() <= buffer_size:
                print(step)
                step += 1
                total_data = loadAliBatch(
                    FLAGS.seq_max_len, infile,
                    file_len_list[(step - 1) * FLAGS.batch_size:step * FLAGS.batch_size])
                total_data_id, total_data_value, total_click, total_label, total_seqlen = total_data
                q.put(total_data)
                if not total_seqlen:
                    break
        infile.close()
    flag.get()


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
        def read_train(q, flag):
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                if not FLAGS.clear_existing_model:
                    saver.restore(sess, os.path.join(FLAGS.model_dir, 'BestModel-150'))
                epoch = 0
                step = 0
                best_auc = 0.0
                while not flag.empty() or not q.empty():
                    total_data_id, total_data_value, total_click, total_label, total_seqlen = q.get(True)
                    if not total_seqlen:
                        epoch += 1
                        step = 0
                        continue
                    step += 1
                    feed_dict = {
                        input_id: sparse_tuple_from(total_data_id),
                        input_value: sparse_tuple_from(total_data_value, dtype=np.float32),
                        seq_len: total_seqlen,
                        click_label: total_click,
                        conversion_label: total_label
                    }
                    _, batch_loss, batch_cvr_loss, batch_click_loss, batch_eval = sess.run(
                        [train_op, loss, conversion_loss, click_loss, eval_metric_ops], feed_dict=feed_dict)
                    print("Epoch:{}\tStep:{}".format(epoch, step))
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
                        print("click_AUC = " + "{}".format(batch_eval['CTR_AUC'][0]), end='\t')
                        print("conversion_AUC = " + "{}".format(batch_eval['CTCVR_AUC'][0]), end='\t')
                        print("click_ACC = " + "{}".format(batch_eval['CTR_ACC'][0]), end='\t')
                        print("conversion_ACC = " + "{}".format(batch_eval['CTCVR_ACC'][0]))
                        print("Loss = " + "{}".format(batch_loss), end='\t')
                        print("Clk Loss = " + "{}".format(batch_click_loss), end='\t')
                        print("Cov_Loss = " + "{}".format(batch_cvr_loss))
                        if test_cvr_auc > best_auc:
                            print("Save----------------")
                            saver.save(sess, os.path.join(FLAGS.model_dir, 'BestModel'), global_step=step)

        for tr in tr_files:
            q = Queue()
            flag = Queue()
            flag.put(True)
            Pw = Process(target=write, args=(q, flag, tr, FLAGS.num_epochs))
            Pr = Process(target=read_train, args=(q, flag))
            Pw.start()
            Pr.start()
            Pw.join()
            Pr.join()
    if FLAGS.task_type == 'eval':
        with tf.Session(config=config) as sess:
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, os.path.join(FLAGS.model_dir, 'BestModel'))
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
    if FLAGS.task_type == 'infer':
        def read_infer(q, flag):
            result = defaultdict(lambda: np.array([]))
            with tf.Session(config=config) as sess:
                sess.run(tf.local_variables_initializer())
                saver.restore(sess, os.path.join(FLAGS.model_dir, 'BestModel-100'))
                while not flag.empty() or not q.empty():
                    total_data_id, total_data_value, total_click, total_label, total_seqlen = q.get(True)
                    feed_dict = {
                        input_id: sparse_tuple_from(total_data_id),
                        input_value: sparse_tuple_from(total_data_value, dtype=np.float32),
                        seq_len: total_seqlen,
                        click_label: total_click,
                        conversion_label: total_label
                    }
                    p_click, l_click, p_conver, l_conver = sess.run(
                        [P_c, reshape_click_label, P_v,
                         reshape_conversion_label], feed_dict=feed_dict)
                    result['pctr'] = np.append(result['pctr'], p_click)
                    result['y'] = np.append(result['y'], l_click)
                    result['pctcvr'] = np.append(result['pctcvr'], p_conver)
                    result['z'] = np.append(result['z'], l_conver)
                    print(len(result['pctr']), len(result['y']), len(result['pctcvr']), len(result['z']))
                pctr = result['pctr']
                y = result['y']
                pctcvr = result['pctcvr']
                z = result['z']
                print(len(result['pctr']), len(result['y']), len(result['pctcvr']), len(result['z']))
                click_result = {'loss': 0, 'acc': 0, 'auc': 0, 'f1': 0, 'ndcg': 0, 'map': 0}
                conversion_result = {'loss': 0, 'acc': 0, 'auc': 0, 'f1': 0, 'ndcg': 0, 'map': 0}
                te_files_pkl = glob.glob("%s/test/remap_sample/r*txt.pkl" % FLAGS.data_dir)[0]
                with open(te_files_pkl, 'rb') as len_f:
                    te_len_list = np.array(pickle.load(len_f))
                te_len_cut = np.where(te_len_list >= seq_max_len, seq_max_len, te_len_list)
                print(sum(te_len_cut))
                indices = np.cumsum(te_len_cut)
                click_result['loss'] = utils.evaluate_logloss(pctr, y)
                click_result['acc'] = utils.evaluate_acc(pctr, y)
                click_result['auc'] = utils.evaluate_auc(pctr, y)
                click_result['f1'] = utils.evaluate_f1_score(pctr, y)
                click_result['ndcg'] = utils.evaluate_ndcg(None, pctr, y, indices)
                click_result['ndcg1'] = utils.evaluate_ndcg(1, pctr, y, indices)
                click_result['ndcg3'] = utils.evaluate_ndcg(3, pctr, y, indices)
                click_result['ndcg5'] = utils.evaluate_ndcg(5, pctr, y, indices)
                click_result['ndcg10'] = utils.evaluate_ndcg(10, pctr, y, indices)
                click_result['map'] = utils.evaluate_map(None, pctr, y, indices)
                click_result['map1'] = utils.evaluate_map(1, pctr, y, indices)
                click_result['map3'] = utils.evaluate_map(3, pctr, y, indices)
                click_result['map5'] = utils.evaluate_map(5, pctr, y, indices)
                click_result['map10'] = utils.evaluate_map(10, pctr, y, indices)

                conversion_result['loss'] = utils.evaluate_logloss(pctcvr, z)
                conversion_result['acc'] = utils.evaluate_acc(pctcvr, z)
                conversion_result['auc'] = utils.evaluate_auc(pctcvr, z)
                conversion_result['f1'] = utils.evaluate_f1_score(pctcvr, z)
                conversion_result['ndcg'] = utils.evaluate_ndcg(None, pctcvr, z, indices)
                conversion_result['ndcg1'] = utils.evaluate_ndcg(1, pctcvr, z, indices)
                conversion_result['ndcg3'] = utils.evaluate_ndcg(3, pctcvr, z, indices)
                conversion_result['ndcg5'] = utils.evaluate_ndcg(5, pctcvr, z, indices)
                conversion_result['ndcg10'] = utils.evaluate_ndcg(10, pctcvr, z, indices)
                conversion_result['map'] = utils.evaluate_map(None, pctcvr, z, indices)
                conversion_result['map1'] = utils.evaluate_map(1, pctcvr, z, indices)
                conversion_result['map3'] = utils.evaluate_map(3, pctcvr, z, indices)
                conversion_result['map5'] = utils.evaluate_map(5, pctcvr, z, indices)
                conversion_result['map10'] = utils.evaluate_map(10, pctcvr, z, indices)
                print("Click Result")
                for k, v in click_result.items():
                    print("{}:{}".format(k, v))
                print()
                print("Conversion Result")
                for k, v in conversion_result.items():
                    print("{}:{}".format(k, v))

        for te in te_files:
            q = Queue()
            flag = Queue()
            flag.put(True)
            Pw = Process(target=write, args=(q, flag, te))
            Pr = Process(target=read_infer, args=(q, flag))
            Pw.start()
            Pr.start()
            Pw.join()
            Pr.join()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
