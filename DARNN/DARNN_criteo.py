import glob
import os
import random
import shutil
from datetime import timedelta, date

import tensorflow as tf

from load_criteo import loaddualattention

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("num_threads", 16, "Number of threads")
tf.flags.DEFINE_integer("max_features", 5897, "Number of max_features")
tf.flags.DEFINE_integer("embedding_size", 96, "Embedding size")
tf.flags.DEFINE_integer("seq_max_len", 160, "seq_max_len")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of epochs")
tf.flags.DEFINE_integer("batch_size", 100, "Number of batch size")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.flags.DEFINE_float("l2_reg", 0.01, "L2 regularization")
tf.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.flags.DEFINE_float("ctr_task_wgt", 0.4, "loss weight of ctr task")
tf.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.flags.DEFINE_integer("n_hidden", 256, "deep layers")
tf.flags.DEFINE_integer("n_classes", 2, "deep layers")
tf.flags.DEFINE_float("keep_prob", 0.5, "dropout rate")
tf.flags.DEFINE_string("gpus", '', "list of gpus")
tf.flags.DEFINE_boolean("batch_norm", True, "perform batch normaization (True or False)")
tf.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.flags.DEFINE_string("data_dir", './../Criteo', "data dir")
tf.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.flags.DEFINE_string("model_dir", './../Criteo/model_Criteo_darnn', "model check point dir")
tf.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval}")
tf.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

input = tf.placeholder(tf.float32,
                       shape=[None, FLAGS.seq_max_len, 12])
seq_len = tf.placeholder(tf.int32, shape=[None], name='seqlen')
click_label = tf.placeholder(tf.float32, shape=[None, FLAGS.seq_max_len, FLAGS.n_classes])
conversion_label = tf.placeholder(tf.float32, shape=[None], name='labels')
embedding_size = FLAGS.embedding_size
l2_reg = FLAGS.l2_reg
learning_rate = FLAGS.learning_rate
max_features = FLAGS.max_features
seq_max_len = FLAGS.seq_max_len
ctr_task_wgt = FLAGS.ctr_task_wgt
embedding_matrix = tf.Variable(
    tf.random_normal([max_features, embedding_size], stddev=0.1))
print(input.shape)
x1, x2 = tf.split(input, [2, 10], 2)
x2 = tf.to_int32(x2)
x2 = tf.nn.embedding_lookup(embedding_matrix, x2)
x2 = tf.reshape(x2, [-1, seq_max_len, 10 * embedding_size])
x = tf.concat((x1, x2), axis=2)
print(x.shape, x.shape[2])
n_input = int(x.shape[2])
# Define Variables
W = tf.Variable(tf.random_normal([FLAGS.n_classes, FLAGS.n_hidden], stddev=0.1), name='W')
U = tf.Variable(tf.random_normal([FLAGS.n_hidden, FLAGS.n_hidden], stddev=0.1), name='U')
C = tf.Variable(tf.random_normal([FLAGS.n_hidden, FLAGS.n_hidden], stddev=0.1), name='C')
v_a = tf.Variable(tf.random_normal([FLAGS.n_hidden], stddev=0.1), name='v_a')
W_z = tf.Variable(tf.random_normal([FLAGS.n_classes, FLAGS.n_hidden], stddev=0.1), name='W_z')
U_z = tf.Variable(tf.random_normal([FLAGS.n_hidden, FLAGS.n_hidden], stddev=0.1), name='U_z')
C_z = tf.Variable(tf.random_normal([FLAGS.n_hidden, FLAGS.n_hidden], stddev=0.1), name='C_z')
W_r = tf.Variable(tf.random_normal([FLAGS.n_classes, FLAGS.n_hidden], stddev=0.1), name='W_r')
U_r = tf.Variable(tf.random_normal([FLAGS.n_hidden, FLAGS.n_hidden], stddev=0.1), name='U_r')
C_r = tf.Variable(tf.random_normal([FLAGS.n_hidden, FLAGS.n_hidden], stddev=0.1), name='C_r')
W_s = tf.Variable(tf.random_normal([FLAGS.n_hidden, FLAGS.n_hidden], stddev=0.1), name='W_s')
W_o = tf.Variable(tf.random_normal([FLAGS.n_hidden, FLAGS.n_classes], stddev=0.1), name='W_o')
b_o = tf.Variable(tf.random_normal([FLAGS.n_classes], stddev=0.1), name='b_o')
W_h = tf.Variable(tf.random_normal([FLAGS.n_hidden, FLAGS.n_hidden], stddev=0.1), name='W_h')
U_s = tf.Variable(tf.random_normal([FLAGS.n_hidden, FLAGS.n_hidden], stddev=0.1), name='U_s')
W_C = tf.Variable(tf.random_normal([FLAGS.n_hidden, FLAGS.n_hidden], stddev=0.1), name='W_C')
W_x1 = tf.Variable(tf.random_normal([n_input, FLAGS.n_hidden], stddev=0.1), name='W_x1')
W_x2 = tf.Variable(tf.random_normal([n_input, FLAGS.n_hidden], stddev=0.1), name='W_x2')
W_x3 = tf.Variable(tf.random_normal([n_input, FLAGS.n_hidden], stddev=0.1), name='W_x3')
v_a2 = tf.Variable(tf.random_normal([FLAGS.n_hidden], stddev=0.1), name='v_a2')
v_a3 = tf.Variable(tf.random_normal([FLAGS.n_hidden], stddev=0.1), name='v_a3')
W_c = tf.Variable(tf.random_normal([FLAGS.n_hidden], stddev=0.1), name='W_c')
b_c = tf.Variable(tf.random_normal([1]))

index = tf.range(0, tf.shape(x)[0]) * FLAGS.seq_max_len + (seq_len - 1)
x_last = tf.gather(params=tf.reshape(x, [-1, n_input]), indices=index)
# x = tf.transpose(x, [1, 0, 2])  # seqlen*batch*dim
mask = tf.sequence_mask(seq_len, seq_max_len)
y = tf.transpose(click_label, [1, 0, 2])
y = tf.reshape(y, [-1, FLAGS.n_classes])
y = tf.split(value=y, num_or_size_splits=seq_max_len, axis=0)
with tf.name_scope('LstmNet'):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.n_hidden, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=FLAGS.keep_prob,
                                              output_keep_prob=FLAGS.keep_prob)
    states_h, last_h = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x, sequence_length=seq_len, dtype=tf.float32)

with tf.name_scope('decoder'):
    last_state_h = tf.gather(params=tf.reshape(states_h, [-1, FLAGS.n_hidden]), indices=index)
    # decoder
    state_s = tf.tanh(tf.matmul(last_state_h, W_s))
    states_s = [state_s]
    outputs = []
    output = tf.zeros(shape=[tf.shape(input)[0], FLAGS.n_classes])
    last_output = tf.nn.softmax(output)
    c = last_state_h
    for i in range(seq_max_len):
        r = tf.sigmoid(tf.matmul(last_output, W_r) + tf.matmul(states_s[i], U_r) + tf.matmul(c, C_r))
        z = tf.sigmoid(tf.matmul(last_output, W_z) + tf.matmul(states_s[i], U_z) + tf.matmul(c, C_z))
        s_hat = tf.tanh(tf.matmul(last_output, W) + tf.matmul(tf.multiply(r, states_s[i]), U) + tf.matmul(c, C))
        state_s = tf.multiply(tf.subtract(1.0, z), states_s[i]) + tf.multiply(z, s_hat)
        states_s.append(state_s)
        state_s = tf.nn.dropout(state_s, FLAGS.keep_prob)
        output = tf.matmul(state_s, W_o) + b_o
        outputs.append(output)
        if FLAGS.task_type == 'train':
            last_output = y[i]
        else:
            last_output = tf.nn.softmax(output)

with tf.name_scope('dual_attention'):
    Ux = tf.matmul(x_last, W_x1)
    Ux2 = tf.matmul(x_last, W_x2)
    states_h = tf.transpose(states_h, [1, 0, 2])
    states_s = tf.stack(states_s[:-1])
    print("H", states_h.shape)
    print("S", states_s.shape)
    e2 = tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(states_h, W_h) + Ux), v_a2), reduction_indices=2)
    e3 = tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(states_s, U_s) + Ux2), v_a3), reduction_indices=2)
    e2 = tf.transpose(e2, [1, 0])
    e3 = tf.transpose(e3, [1, 0])
    print("E2", e2.shape)
    print("E3", e3.shape)

    a2 = tf.map_fn(
        lambda i: tf.concat([tf.nn.softmax(e2[i][:seq_len[i]]), tf.zeros(shape=(seq_max_len - seq_len[i]))], axis=0),
        tf.range(0, tf.shape(e2)[0]), dtype=e2.dtype)
    a3 = tf.map_fn(
        lambda i: tf.concat([tf.nn.softmax(e3[i][:seq_len[i]]), tf.zeros(shape=(seq_max_len - seq_len[i]))], axis=0),
        tf.range(0, tf.shape(e2)[0]), dtype=e2.dtype)
    a2 = tf.concat(a2, axis=0)
    a3 = tf.concat(a3, axis=0)
    a2 = tf.boolean_mask(a2, mask)
    a3 = tf.boolean_mask(a3, mask)
    print("A2", a2.shape)
    print("A3", a3.shape)
    states_h = tf.transpose(states_h, [1, 0, 2])
    states_h = tf.boolean_mask(states_h, mask)
    states_s = tf.transpose(states_s, [1, 0, 2])
    states_s = tf.boolean_mask(states_s, mask)


    def f(i):
        left = tf.reduce_sum(seq_len[:i])
        right = left + seq_len[i]
        return tf.reduce_sum(tf.multiply(tf.reshape(a2[left:right], shape=(-1, 1)), states_h[left:right]), axis=0)


    def f2(i):
        left = tf.reduce_sum(seq_len[:i])
        right = left + seq_len[i]
        return tf.reduce_sum(tf.multiply(tf.reshape(a3[left:right], shape=(-1, 1)), states_s[left:right]), axis=0)


    c2 = tf.map_fn(lambda i: f(i), tf.range(0, tf.shape(e2)[0]), dtype=a2.dtype)
    c3 = tf.map_fn(lambda i: f2(i), tf.range(0, tf.shape(e2)[0]), dtype=a3.dtype)
    print("C2,C3:", c2.shape, c3.shape)
    e4 = []
    Ux3 = tf.matmul(x_last, W_x3)
    e4.append(tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(c2, W_C) + Ux3), v_a), reduction_indices=1))
    e4.append(tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(c3, W_C) + Ux3), v_a), reduction_indices=1))
    e4 = tf.stack(e4)
    print("E4:", e4.shape)
    a4 = tf.split(tf.nn.softmax(e4, dim=0), 2, 0)
    print("A4：", a4[0].shape, a4[1].shape)
    C = tf.multiply(c2, tf.transpose(a4[0])) + tf.multiply(c3, tf.transpose(a4[1]))
    print(C.shape)

cvr = tf.reduce_sum(tf.multiply(C, W_c), axis=1) + b_c
# cvr = tf.nn.dropout(cvr, keep_prob=FLAGS.keep_prob)
pcvr = tf.nn.sigmoid(cvr)

outputs = tf.stack(outputs)
outputs = tf.transpose(outputs, [1, 0, 2])
click_pred = tf.nn.softmax(outputs)
predictions = {"pctr": click_pred, "pcvr": pcvr}
# ------bulid loss------
conversion_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=conversion_label, logits=cvr))
loss_click = tf.nn.softmax_cross_entropy_with_logits(labels=click_label, logits=outputs)
loss_click = tf.boolean_mask(loss_click, mask)  # return real seq
loss_click = tf.reduce_mean(loss_click)

loss = (1 - ctr_task_wgt) * loss_click + ctr_task_wgt * conversion_loss
for v in tf.trainable_variables():
    loss += l2_reg * tf.nn.l2_loss(v)
    '''loss_click += l2_reg * tf.nn.l2_loss(v)
    conversion_loss += l2_reg * tf.nn.l2_loss(v)'''
tf.summary.scalar('ctr_loss', loss_click)
tf.summary.scalar('ctcvr_loss', conversion_loss)

# Provide an estimator spec for `ModeKeys.EVAL`
threshold = 0.5
reshape_click_label = tf.reshape(tf.boolean_mask(click_label, mask), shape=(-1, 2))[:, 1]
reshape_click_pred = tf.reshape(tf.boolean_mask(click_pred, mask), shape=(-1, 2))[:, 1]

one_click = tf.ones_like(reshape_click_label)
zero_click = tf.zeros_like(reshape_click_label)
one_cvr = tf.ones_like(conversion_label)
zero_cvr = tf.zeros_like(conversion_label)
eval_metric_ops = {
    "CTR_AUC": tf.metrics.auc(reshape_click_label, reshape_click_pred),
    "CTR_ACC": tf.metrics.accuracy(reshape_click_label,
                                   tf.where(reshape_click_pred >= threshold, one_click, zero_click)),
    "CTCVR_AUC": tf.metrics.auc(conversion_label, pcvr),
    "CTCVR_ACC": tf.metrics.accuracy(conversion_label, tf.where(pcvr >= threshold, one_cvr, zero_cvr))
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
    FLAGS.model_dir = FLAGS.model_dir +  FLAGS.dt_dir

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
    random.shuffle(tr_files)
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
                        total_data, total_click_label, total_label, total_seqlen = loaddualattention(
                            FLAGS.batch_size, FLAGS.seq_max_len, 12, tr_infile)
                        feed_dict = {
                            input: total_data,
                            seq_len: total_seqlen,
                            click_label: total_click_label,
                            conversion_label: total_label
                        }
                        _, batch_loss, batch_cvr_loss, batch_click_loss, batch_eval, batch_pred = sess.run(
                            [train_op, loss, conversion_loss, loss_click, eval_metric_ops, pcvr], feed_dict=feed_dict)
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
                                for _ in range(200):
                                    test_step += 1
                                    total_data, total_click_label, total_label, total_seqlen = loaddualattention(
                                        FLAGS.batch_size, FLAGS.seq_max_len, 12, te_infile)
                                    feed_dict = {
                                        input: total_data,
                                        seq_len: total_seqlen,
                                        click_label: total_click_label,
                                        conversion_label: total_label
                                    }
                                    batch_loss, batch_cvr_loss, batch_click_loss, batch_eval = sess.run(
                                        [loss, conversion_loss, loss_click, eval_metric_ops],
                                        feed_dict=feed_dict)
                                    test_ctr_auc += batch_eval['CTR_AUC'][0]
                                    test_cvr_auc += batch_eval['CTCVR_AUC'][0]
                                    test_ctr_acc += batch_eval['CTR_ACC'][0]
                                    test_cvr_acc += batch_eval['CTCVR_ACC'][0]
                                    test_loss += batch_loss
                                    test_cvr_loss += batch_cvr_loss
                                    test_clk_loss += batch_click_loss
                                    if len(total_data) != FLAGS.batch_size:
                                        break
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
                                saver.save(sess, os.path.join(FLAGS.model_dir, 'BestModel'),global_step=step)
                        if len(total_data) != FLAGS.batch_size:
                            break
    if FLAGS.task_type == 'eval':
        with tf.Session(config=config) as sess:
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, os.path.join(FLAGS.model_dir, 'BestModel'))
            for te in te_files:
                te_infile = open(te, 'r')
                while True:
                    total_data, total_click_label, total_label, total_seqlen = loaddualattention(
                        FLAGS.batch_size, FLAGS.seq_max_len, 12, te_infile)
                    feed_dict = {
                        input: total_data,
                        seq_len: total_seqlen,
                        click_label: total_click_label,
                        conversion_label: total_label
                    }
                    batch_loss, batch_cvr_loss, batch_click_loss, batch_eval = sess.run(
                        [loss, conversion_loss, loss_click, eval_metric_ops], feed_dict=feed_dict)
                    print("click_AUC = " + "{}".format(batch_eval['CTR_AUC'][0]), end='\t')
                    print("conversion_AUC = " + "{}".format(batch_eval['CTCVR_AUC'][0]), end='\t')
                    print("click_ACC = " + "{}".format(batch_eval['CTR_ACC'][0]), end='\t')
                    print("conversion_ACC = " + "{}".format(batch_eval['CTCVR_ACC'][0]))
                    print("Loss = " + "{}".format(batch_loss), end='\t')
                    print("Clk Loss = " + "{}".format(batch_click_loss), end='\t')
                    print("Cov_Loss = " + "{}".format(batch_cvr_loss))
                    if len(total_data) != FLAGS.batch_size:
                        break


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
