import glob
import os
import shutil
import sys
from datetime import timedelta, date

import numpy as np
import tensorflow as tf

sys.path.append("../")
from utils import utils

from load_criteo import loadCriteoBatch

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("num_threads", 64, "Number of threads")
tf.flags.DEFINE_integer("max_features", 5897, "Number of max_features")
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
tf.flags.DEFINE_string("data_dir", './../Criteo', "data dir")
tf.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.flags.DEFINE_string("model_dir", './../Criteo/model_Criteo_dnn', "model check point dir")
tf.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval}")
tf.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

input = tf.placeholder(tf.float32,
                       shape=[None, FLAGS.seq_max_len, 11])
seq_len = tf.placeholder(tf.int32, shape=[None], name='seqlen')
click_label = tf.placeholder(tf.float32, shape=[None, FLAGS.seq_max_len, FLAGS.n_classes], name='clicks')
conversion_label = tf.placeholder(tf.float32, shape=[None, FLAGS.seq_max_len, FLAGS.n_classes], name='labels')
embedding_size = FLAGS.embedding_size
l2_reg = FLAGS.l2_reg
learning_rate = FLAGS.learning_rate
max_features = FLAGS.max_features
seq_max_len = FLAGS.seq_max_len
ctr_task_wgt = FLAGS.ctr_task_wgt
embedding_matrix = tf.Variable(
    tf.random_normal([max_features, embedding_size], stddev=0.1))
print(input.shape)
n_hidden = FLAGS.n_hidden
n_classes = FLAGS.n_classes
x1, x2 = tf.split(input, [1, 10], 2)
x2 = tf.to_int32(x2)
x2 = tf.nn.embedding_lookup(embedding_matrix, x2)
x2 = tf.reshape(x2, [-1, seq_max_len, 10 * embedding_size])
x = tf.concat((x1, x2), axis=2)
print(x.shape)
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

# ------bulid loss------
mask = tf.sequence_mask(seq_len, seq_max_len)
conversion_loss = tf.reduce_mean(
    tf.boolean_mask(tf.nn.sigmoid_cross_entropy_with_logits(labels=conversion_label, logits=x_v), mask))
click_loss = tf.reduce_mean(
    tf.boolean_mask(tf.nn.sigmoid_cross_entropy_with_logits(labels=click_label, logits=x_c), mask))
loss = (1 - ctr_task_wgt) * click_loss + ctr_task_wgt * conversion_loss

for v in tf.trainable_variables():
    loss += l2_reg * tf.nn.l2_loss(v)
threshold = 0.5
reshape_click_label = tf.squeeze(tf.boolean_mask(click_label, mask))
reshape_click_pred = tf.squeeze(tf.boolean_mask(P_c, mask))

click_zero = tf.where(reshape_click_label < 0.5)
click_one = tf.where(reshape_click_label > 0.5)

reshape_conversion_label = tf.squeeze(tf.boolean_mask(conversion_label, mask))
reshape_conversion_pred = tf.squeeze(tf.boolean_mask(P_v, mask))
conversion_zero = tf.where(reshape_conversion_label < 0.5)
conversion_one = tf.where(reshape_conversion_label > 0.5)

eval_metric_ops = {
    "CTR_AUC": tf.metrics.auc(reshape_click_label, reshape_click_pred),
    "CTR_positive_ACC": tf.metrics.accuracy(tf.gather(reshape_click_label, click_one),
                                            tf.where(tf.gather(reshape_click_pred, click_one) >= threshold,
                                                     tf.ones(shape=(tf.shape(click_one))),
                                                     tf.zeros(shape=tf.shape(click_one)))),
    "CTR_negative_ACC": tf.metrics.accuracy(tf.gather(reshape_click_label, click_zero),
                                            tf.where(tf.gather(reshape_click_pred, click_zero) >= threshold,
                                                     tf.ones(shape=tf.shape(click_zero)),
                                                     tf.zeros(shape=tf.shape(click_zero)))),
    "CVR_AUC": tf.metrics.auc(reshape_conversion_label, reshape_conversion_pred),
    "CVR_positive_ACC": tf.metrics.accuracy(tf.gather(reshape_conversion_label, conversion_one),
                                            tf.where(tf.gather(reshape_conversion_pred, conversion_one) >= threshold,
                                                     tf.ones(shape=tf.shape(conversion_one)),
                                                     tf.zeros(shape=tf.shape(conversion_one)))),
    "CVR_negative_ACC": tf.metrics.accuracy(tf.gather(reshape_conversion_label, conversion_zero),
                                            tf.where(tf.gather(reshape_conversion_pred, conversion_zero) >= threshold,
                                                     tf.ones(shape=tf.shape(conversion_zero)),
                                                     tf.zeros(shape=tf.shape(conversion_zero))))
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
    FLAGS.model_dir = FLAGS.model_dir + '20210111' + "R"

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

    tr_files = glob.glob("%s/train/*.txt" % FLAGS.data_dir)
    print("train_files:", tr_files)
    te_files = glob.glob("%s/test/*.txt" % FLAGS.data_dir)
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
                saver.restore(sess, os.path.join(FLAGS.model_dir, 'BestModel'))
            for tr in tr_files:
                for i in range(FLAGS.num_epochs):
                    step = 0
                    tr_infile = open(tr, 'r')
                    best_auc = 0
                    while True:
                        step += 1
                        total_data, total_click_label, total_label, total_seqlen = loadCriteoBatch(
                            FLAGS.batch_size, FLAGS.seq_max_len, tr_infile)
                        # print(total_data, total_click_label, total_label, total_seqlen)
                        feed_dict = {
                            input: total_data,
                            seq_len: total_seqlen,
                            click_label: total_click_label,
                            conversion_label: total_label
                        }
                        _, batch_loss, batch_cvr_loss, batch_click_loss, batch_eval = sess.run(
                            [train_op, loss, conversion_loss, click_loss, eval_metric_ops],
                            feed_dict=feed_dict)
                        print("Epoch:{}\tStep:{}".format(i, step))
                        print("CTR_AUC = " + "{}".format(batch_eval['CTR_AUC'][0]), end='\t')
                        print("CVR_AUC = " + "{}".format(batch_eval['CVR_AUC'][0]), end='\t')
                        print("CTR_positive_ACC = " + "{}".format(batch_eval['CTR_positive_ACC'][0]), end='\t')
                        print("CTR_negative_ACC = " + "{}".format(batch_eval['CTR_negative_ACC'][0]), end='\t')
                        print("CVR_positive_ACC = " + "{}".format(batch_eval['CVR_positive_ACC'][0]), end='\t')
                        print("CVR_negative_ACC = " + "{}".format(batch_eval['CVR_negative_ACC'][0]))
                        print("Loss = " + "{}".format(batch_loss), end='\t')
                        print("Clk_Loss = " + "{}".format(batch_click_loss), end='\t')
                        print("Cov_Loss = " + "{}".format(batch_cvr_loss))
                        if step % 50 == 0:
                            saver.save(sess, os.path.join(FLAGS.model_dir, 'MyModel'), global_step=step)
                            print("Test----------------")
                            test_cvr_auc = 0.0
                            test_ctr_auc = 0.0
                            test_cvr_negative_acc = 0.0
                            test_ctr_positive_acc = 0.0
                            test_ctr_negative_acc = 0.0
                            test_cvr_positive_acc = 0.0
                            test_loss = 0
                            test_clk_loss = 0
                            test_cvr_loss = 0
                            test_step = 0
                            for te in te_files:
                                te_infile = open(te, 'r')
                                for _ in range(10):
                                    test_step += 1
                                    test_total_data, test_total_click_label, test_total_label, test_total_seqlen = loadCriteoBatch(
                                        FLAGS.batch_size, FLAGS.seq_max_len, te_infile)
                                    print(test_total_seqlen)
                                    feed_dict = {
                                        input: test_total_data,
                                        seq_len: test_total_seqlen,
                                        click_label: test_total_click_label,
                                        conversion_label: test_total_label
                                    }
                                    batch_loss, batch_cvr_loss, batch_click_loss, batch_eval = sess.run(
                                        [loss, conversion_loss, click_loss, eval_metric_ops],
                                        feed_dict=feed_dict)
                                    test_ctr_auc += batch_eval['CTR_AUC'][0]
                                    test_cvr_auc += batch_eval['CVR_AUC'][0]
                                    test_ctr_positive_acc += batch_eval['CTR_positive_ACC'][0]
                                    test_ctr_negative_acc += batch_eval['CTR_negative_ACC'][0]
                                    test_cvr_negative_acc += batch_eval['CVR_negative_ACC'][0]
                                    test_cvr_positive_acc += batch_eval['CVR_positive_ACC'][0]
                                    test_loss += batch_loss
                                    test_cvr_loss += batch_cvr_loss
                                    test_clk_loss += batch_click_loss
                                    if len(total_label) != FLAGS.batch_size:
                                        break
                                te_infile.close()
                            test_ctr_auc /= test_step
                            test_cvr_auc /= test_step
                            test_cvr_positive_acc /= test_step
                            test_cvr_negative_acc /= test_step
                            test_ctr_positive_acc /= test_step
                            test_ctr_negative_acc /= test_step
                            test_loss /= test_step
                            test_clk_loss /= test_step
                            test_cvr_loss /= step

                            print("click_AUC = " + "{}".format(test_ctr_auc), end='\t')
                            print("conversion_AUC = " + "{}".format(test_cvr_auc), end='\t')
                            print("click_positive_ACC = " + "{}".format(test_ctr_positive_acc), end='\t')
                            print("conversion_positive_ACC = " + "{}".format(test_cvr_positive_acc), end='\t')
                            print("click_negative_ACC = " + "{}".format(test_ctr_negative_acc), end='\t')
                            print("conversion_negative_ACC = " + "{}".format(test_cvr_negative_acc))
                            print("Loss = " + "{}".format(test_loss), end='\t')
                            print("Clk Loss = " + "{}".format(test_clk_loss), end='\t')
                            print("Cov_Loss = " + "{}".format(test_cvr_loss))
                            if test_cvr_auc > best_auc:
                                print("Save----------------")
                                saver.save(sess, os.path.join(FLAGS.model_dir, 'BestModel'), global_step=step)
                        if len(total_label) != FLAGS.batch_size:
                            break
                    tr_infile.close()
    if FLAGS.task_type == 'eval':
        with tf.Session(config=config) as sess:
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, os.path.join(FLAGS.model_dir, 'BestModel'))
            for te in te_files:
                print(te)
                te_infile = open(te, 'r')
                step = 0
                while True:
                    step += 1
                    total_data, total_click_label, total_label, total_seqlen = loadCriteoBatch(
                        FLAGS.batch_size, FLAGS.seq_max_len, te_infile)
                    feed_dict = {
                        input: total_data,
                        seq_len: total_seqlen,
                        click_label: total_click_label,
                        conversion_label: total_label
                    }
                    batch_loss, batch_cvr_loss, batch_click_loss, batch_eval = sess.run(
                        [loss, conversion_loss, click_loss, eval_metric_ops], feed_dict=feed_dict)
                    print("CTR_AUC = " + "{}".format(batch_eval['CTR_AUC'][0]), end='\t')
                    print("CVR_AUC = " + "{}".format(batch_eval['CVR_AUC'][0]), end='\t')
                    print("CTR_positive_ACC = " + "{}".format(batch_eval['CTR_positive_ACC'][0]), end='\t')
                    print("CTR_negative_ACC = " + "{}".format(batch_eval['CTR_negative_ACC'][0]), end='\t')
                    print("CVR_positive_ACC = " + "{}".format(batch_eval['CVR_positive_ACC'][0]), end='\t')
                    print("CVR_negative_ACC = " + "{}".format(batch_eval['CVR_negative_ACC'][0]))
                    print("Loss = " + "{}".format(batch_loss), end='\t')
                    print("Clk Loss = " + "{}".format(batch_click_loss), end='\t')
                    print("Cov_Loss = " + "{}".format(batch_cvr_loss))
                    if len(total_label) != FLAGS.batch_size:
                        break
    if FLAGS.task_type == 'infer':
        with tf.Session(config=config) as sess:
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, os.path.join(FLAGS.model_dir, 'BestModel-9300'))
            for te in te_files:
                print(te)
                te_infile = open(te, 'r')
                step = 0
                pctr = np.array([])
                y = np.array([])
                pctcvr = np.array([])
                z = np.array([])
                while True:
                    step += 1
                    total_data, total_click_label, total_label, total_seqlen = loadCriteoBatch(
                        FLAGS.batch_size, FLAGS.seq_max_len, te_infile)
                    print(step)
                    feed_dict = {
                        input: total_data,
                        seq_len: total_seqlen,
                        click_label: total_click_label,
                        conversion_label: total_label
                    }
                    p_click, l_click, p_conver, l_conver = sess.run(
                        [reshape_click_pred, reshape_click_label, reshape_conversion_pred,
                         reshape_conversion_label], feed_dict=feed_dict)
                    pctr = np.append(pctr, p_click)
                    y = np.append(y, l_click)
                    pctcvr = np.append(pctcvr, p_conver)
                    z = np.append(z, l_conver)
                    if len(total_data) != FLAGS.batch_size:
                        break
                click_result = {'loss': 0, 'acc': 0, 'auc': 0, 'f1': 0, 'ndcg': 0, 'map': 0}
                conversion_result = {'loss': 0, 'acc': 0, 'auc': 0, 'f1': 0, 'ndcg': 0, 'map': 0}
                click_result['loss'] = utils.evaluate_logloss(pctr, y)
                click_result['acc'] = utils.evaluate_acc(pctr, y)
                click_result['auc'] = utils.evaluate_auc(pctr, y)
                click_result['f1'] = utils.evaluate_f1_score(pctr, y)
                click_result['ndcg'] = utils.evaluate_ndcg(None, pctr, y, len(pctr))
                click_result['map'] = utils.evaluate_map(len(pctr), pctr, y, len(pctr))

                conversion_result['loss'] = utils.evaluate_logloss(pctcvr, z)
                conversion_result['acc'] = utils.evaluate_acc(pctcvr, z)
                conversion_result['auc'] = utils.evaluate_auc(pctcvr, z)
                conversion_result['f1'] = utils.evaluate_f1_score(pctcvr, z)
                conversion_result['ndcg'] = utils.evaluate_ndcg(None, pctcvr, z, len(pctcvr))
                conversion_result['map'] = utils.evaluate_map(len(pctcvr), pctcvr, z, len(pctcvr))
                print("Click Result")
                for k, v in click_result.items():
                    print("{}:{}".format(k, v), end='\t')
                print()
                print("Conversion Result")
                for k, v in conversion_result.items():
                    print("{}:{}".format(k, v), end='\t')


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
