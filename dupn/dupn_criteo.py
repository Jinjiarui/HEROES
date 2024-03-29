import glob
import os
import shutil
import sys
from datetime import timedelta, date
from multiprocessing import Process, Queue

import numpy as np
import tensorflow as tf

sys.path.append("../")
from utils import utils
from load_criteo import loadCriteoBatch

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("num_threads", 64, "Number of threads")
tf.flags.DEFINE_integer("max_features", 638072, "Number of max_features")
tf.flags.DEFINE_integer("embedding_size", 96, "Embedding size")
tf.flags.DEFINE_integer("seq_max_len", 50, "seq_max_len")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of epochs")
tf.flags.DEFINE_integer("batch_size", 100, "Number of batch size")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
tf.flags.DEFINE_float("l2_reg", 0.1, "L2 regularization")
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
tf.flags.DEFINE_string("model_dir", './../Criteo/model_Criteo_dupn', "model check point dir")
tf.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval}")
tf.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

input = tf.placeholder(tf.float32, shape=[None, FLAGS.seq_max_len, 11])
seq_len = tf.placeholder(tf.int32, shape=[None], name='seqlen')
click_label = tf.placeholder(tf.float32, shape=[None, FLAGS.seq_max_len, FLAGS.n_classes], name='clicks')
conversion_label = tf.placeholder(tf.float32, shape=[None, FLAGS.seq_max_len, FLAGS.n_classes], name='labels')

embedding_size = FLAGS.embedding_size
l2_reg = FLAGS.l2_reg
learning_rate = FLAGS.learning_rate
max_features = FLAGS.max_features
seq_max_len = FLAGS.seq_max_len
ctr_task_wgt = FLAGS.ctr_task_wgt
embedding_matrix = tf.Variable(tf.random_normal([max_features, embedding_size], stddev=0.1))

print(input.shape)
n_hidden = FLAGS.n_hidden
n_classes = FLAGS.n_classes
x1, x2 = tf.split(input, [1, 10], 2)
x2 = tf.to_int32(x2)
x2 = tf.nn.embedding_lookup(embedding_matrix, x2)
x2 = tf.reshape(x2, [-1, seq_max_len, 10 * embedding_size])
x = tf.concat((x1, x2), axis=2)
print(x.shape)  # (bs,seq,embed)
n_input = int(x.shape[2])
index = tf.range(0, tf.shape(x)[0]) * FLAGS.seq_max_len + (seq_len - 1)
mask = tf.sequence_mask(seq_len, seq_max_len)
'''x_last = tf.gather(params=tf.reshape(x, [-1, n_input]), indices=index)  # (bs,embed)
print(x_last.shape)
x_last = tf.reshape(x_last, [tf.shape(x)[0], 1, n_input])  # (bs,1,embed)
print(x_last.shape)'''

with tf.name_scope('LstmNet'), tf.variable_scope("LstmNet", reuse=tf.AUTO_REUSE):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=FLAGS.keep_prob,
                                              output_keep_prob=FLAGS.keep_prob)
    states_h, last_h = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x, sequence_length=seq_len, dtype=tf.float32)
    # (bs, seq, n_hidden)

with tf.name_scope('attention'), tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
    Ux = tf.layers.dense(x, units=n_hidden, use_bias=True, name='Ux')  # (bs,seq,n_hidden)
    Uh = tf.layers.dense(states_h, units=n_hidden, use_bias=True, name='Uh')  # (bs, seq, n_hidden)
    e = tf.reduce_sum(tf.tanh(Ux + Uh), reduction_indices=2)  # (bs,seq)
    a = tf.map_fn(
        lambda i: tf.concat([tf.nn.softmax(e[i][:seq_len[i]]), tf.zeros(shape=(seq_max_len - seq_len[i]))], axis=0),
        tf.range(0, tf.shape(e)[0]), dtype=e.dtype)  # (bs,seq)

    c = tf.map_fn(lambda i: tf.reduce_sum(
        tf.multiply(tf.reshape(a[i][:seq_len[i]], shape=(-1, 1)), states_h[i][:seq_len[i]]), axis=0),
                  tf.range(0, tf.shape(e)[0]), dtype=a.dtype)  # (bs,n_hidden)

    c_seq_click = tf.layers.dense(c, units=states_h.shape[1], use_bias=True, name='Pc')  # (bs,seq)
    c_seq_conversion = tf.layers.dense(c, units=states_h.shape[1], use_bias=True, name='Pv')  # (bs,seq)

    prediction_c = tf.sigmoid(c_seq_click)  # (bs,seq)
    prediction_v = tf.sigmoid(c_seq_conversion)  # (bs,seq)
    prediction_v = tf.multiply(prediction_c, prediction_v)

prediction_c = tf.boolean_mask(prediction_c, mask)  # (real_seq)
prediction_v = tf.boolean_mask(prediction_v, mask)
reshape_click_label = tf.squeeze(tf.boolean_mask(click_label, mask))
reshape_conversion_label = tf.squeeze(tf.boolean_mask(conversion_label, mask))
epsilon = 1e-7
click_loss = tf.reduce_mean(tf.losses.log_loss(labels=reshape_click_label, predictions=prediction_c))
conversion_loss = tf.reduce_mean(tf.losses.log_loss(labels=reshape_conversion_label, predictions=prediction_v))
loss = ((1 - ctr_task_wgt) * click_loss + ctr_task_wgt * conversion_loss) * 100
for v in tf.trainable_variables():
    loss += l2_reg * tf.nn.l2_loss(v)
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


def write(q, flag, file_name, num_epoch=1, buffer_size=10):
    print(file_name)

    for _ in range(num_epoch):
        step = 0
        infile = open(file_name, 'r')
        while True:
            if q.qsize() <= buffer_size:
                print(step)
                step += 1
                total_zip_data = loadCriteoBatch(FLAGS.batch_size, FLAGS.seq_max_len, infile)
                total_data, total_click_label, total_label, total_seqlen = total_zip_data
                q.put(total_zip_data)
                if len(total_label) != FLAGS.batch_size:
                    break
        infile.close()
    flag.get()


def main(_):
    # ------check Arguments------
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir

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
        def read_train(q, flag):
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                if not FLAGS.clear_existing_model:
                    saver.restore(sess, os.path.join(FLAGS.model_dir, 'BestModel-2500'))
                epoch = 0
                step = 0
                best_auc = 0.0
                while not flag.empty() or not q.empty():
                    step += 1
                    total_data, total_click_label, total_label, total_seqlen = q.get(True)
                    feed_dict = {
                        input: total_data,
                        seq_len: total_seqlen,
                        click_label: total_click_label,
                        conversion_label: total_label
                    }
                    batch_pv, batch_conversion, batch_pc, batch_click, _, batch_loss, batch_cvr_loss, batch_click_loss, batch_eval = sess.run(
                        [prediction_v, reshape_conversion_label, prediction_c, reshape_click_label, train_op, loss,
                         conversion_loss, click_loss, eval_metric_ops],
                        feed_dict=feed_dict)
                    '''print(batch_pc)
                    print(batch_click)
                    print(batch_pv)
                    print(batch_conversion)'''
                    print("Epoch:{}\tStep:{}".format(epoch, step))
                    print("click_AUC = " + "{}".format(batch_eval['CTR_AUC'][0]), end='\t')
                    print("conversion_AUC = " + "{}".format(batch_eval['CTCVR_AUC'][0]), end='\t')
                    print("click_ACC = " + "{}".format(batch_eval['CTR_ACC'][0]), end='\t')
                    print("conversion_ACC = " + "{}".format(batch_eval['CTCVR_ACC'][0]))
                    print("Loss = " + "{}".format(batch_loss), end='\t')
                    print("Clk Loss = " + "{}".format(batch_click_loss), end='\t')
                    print("Cov_Loss = " + "{}".format(batch_cvr_loss))
                    if step % 500 == 0:
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
                            for _ in range(10):
                                test_step += 1
                                test_total_data, test_total_click_label, test_total_label, test_total_seqlen = loadCriteoBatch(
                                    FLAGS.batch_size, FLAGS.seq_max_len, te_infile)
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
                                test_cvr_auc += batch_eval['CTCVR_AUC'][0]
                                test_ctr_acc += batch_eval['CTR_ACC'][0]
                                test_cvr_acc += batch_eval['CTCVR_ACC'][0]
                                test_loss += batch_loss
                                test_cvr_loss += batch_cvr_loss
                                test_clk_loss += batch_click_loss
                                if len(test_total_label) != FLAGS.batch_size:
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
                    if len(total_label) != FLAGS.batch_size:
                        epoch += 1
                        step = 0
                        continue

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
        def read_infer(q, flag):
            with tf.Session(config=config) as sess:
                sess.run(tf.local_variables_initializer())
                saver.restore(sess, os.path.join(FLAGS.model_dir, 'BestModel-3500'))
                te_len_list = []
                pctr = np.array([])
                y = np.array([])
                pctcvr = np.array([])
                z = np.array([])
                while not flag.empty() or not q.empty():
                    total_data, total_click_label, total_label, total_seqlen = q.get(True)
                    te_len_list += total_seqlen
                    feed_dict = {
                        input: total_data,
                        seq_len: total_seqlen,
                        click_label: total_click_label,
                        conversion_label: total_label
                    }
                    p_click, l_click, p_conver, l_conver = sess.run(
                        [prediction_c, reshape_click_label, prediction_v,
                         reshape_conversion_label], feed_dict=feed_dict)
                    pctr = np.append(pctr, p_click)
                    y = np.append(y, l_click)
                    pctcvr = np.append(pctcvr, p_conver)
                    z = np.append(z, l_conver)
                print(pctr.shape)
                print(sum(te_len_list))
                click_result = {'loss': 0, 'acc': 0, 'auc': 0, 'f1': 0, 'ndcg': 0, 'map': 0}
                conversion_result = {'loss': 0, 'acc': 0, 'auc': 0, 'f1': 0, 'ndcg': 0, 'map': 0}
                indices = np.cumsum(te_len_list)
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
