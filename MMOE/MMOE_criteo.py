import glob
import os
import random
import shutil
import sys
from datetime import date, timedelta

import numpy as np

sys.path.append("../")
from utils import utils
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("num_threads", 16, "Number of threads")
tf.flags.DEFINE_integer("feature_size", 10, "Size of Features")
tf.flags.DEFINE_integer("feature_num", 5867, "Num of Features")
tf.flags.DEFINE_integer("embedding_size", 96, "Embedding size")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of epochs")
tf.flags.DEFINE_integer("batch_size", 8000, "Number of batch size")
tf.flags.DEFINE_integer("log_steps", 10, "save summary every steps")
tf.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.flags.DEFINE_float("l2_reg", 0.01, "L2 regularization")
tf.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.flags.DEFINE_float("ctr_task_wgt", 0.5, "loss weight of ctr task")
tf.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.flags.DEFINE_string("expert_layers", '256,128', "deep layers")
tf.flags.DEFINE_string("deep_layers", '96,64', "deep layers")
tf.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.flags.DEFINE_string("gpus", '', "list of gpus")
tf.flags.DEFINE_boolean("batch_norm", True, "perform batch normaization (True or False)")
tf.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.flags.DEFINE_string("data_dir", './../Criteo', "data dir")
tf.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.flags.DEFINE_string("model_dir", './../Criteo/model_Criteo_mmoe', "model check point dir")
tf.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval}")
tf.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus


def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=True, reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False, predict_mode=False):
    print('Parsing', filenames)

    def _parse_fn(record):
        features = {
            "y": tf.FixedLenFeature([], tf.float32),
            "z": tf.FixedLenFeature([], tf.float32),
            "features": tf.FixedLenFeature([FLAGS.feature_size], tf.int64)
        }
        parsed = tf.parse_single_example(record, features)
        if predict_mode:
            return parsed, {}
        y = parsed.pop('y')
        z = parsed.pop('z')
        return parsed, {"y": y, "z": z}

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TFRecordDataset(filenames).map(_parse_fn, num_parallel_calls=64).prefetch(
        1000000)  # multi-thread pre-process then prefetch
    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=5000)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)  # Batch size to use

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    # ------hyperparameters----
    features_num = params['feature_num']
    feature_size = params['feature_size']
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    # optimizer = params["optimizer"]
    layers = list(map(int, params["deep_layers"].split(',')))
    expert_layers = list(map(int, params["expert_layers"].split(',')))
    dropout = list(map(float, params["dropout"].split(',')))
    ctr_task_wgt = params["ctr_task_wgt"]

    Emb = tf.get_variable(name='Embeddings', shape=[features_num, embedding_size],
                          initializer=tf.glorot_normal_initializer())

    the_features = features['features']
    print(the_features.shape)
    with tf.variable_scope("Shared-Embedding-layer"):
        emb_feature = tf.nn.embedding_lookup(Emb, the_features)
        print(emb_feature.shape)
        emb_reshape = tf.reshape(emb_feature, shape=(-1, emb_feature.shape[1] * emb_feature.shape[2]))
        print(emb_reshape.shape)

    expert_num = 3
    with tf.variable_scope("Gate1"):
        gate1 = tf.contrib.layers.fully_connected(inputs=emb_reshape, num_outputs=expert_num,
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                  scope='gate1')
        gate1 = tf.nn.softmax(gate1)
    with tf.variable_scope("Gate2"):
        gate2 = tf.contrib.layers.fully_connected(inputs=emb_reshape, num_outputs=expert_num,
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                  scope='gate2')
        gate2 = tf.nn.softmax(gate2)  # (bs,3)

    expert_result = []
    for i in range(expert_num):
        with tf.name_scope("Expert{}".format(i)):
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
            else:
                train_phase = False
            expert = emb_reshape
            for j in range(len(expert_layers)):
                expert = tf.contrib.layers.fully_connected(inputs=expert, num_outputs=expert_layers[j],
                                                           weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                           scope='Expert{}{}'.format(i, j))
                expert = tf.nn.relu(expert)
                if FLAGS.batch_norm:
                    expert = batch_norm_layer(expert, train_phase=train_phase,
                                              scope_bn='exp_bn_{}{}'.format(i, j))
                if mode == tf.estimator.ModeKeys.TRAIN:
                    expert = tf.nn.dropout(expert, keep_prob=dropout[j])
        expert_result.append(expert)
    expert_result = tf.transpose(tf.stack(expert_result), (1, 0, 2))  # (bs,3,dim)
    with tf.name_scope("CVR_Task"):
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_phase = True
        else:
            train_phase = False
        x_cvr = tf.reduce_sum(tf.multiply(expert_result, tf.reshape(gate1, shape=(-1, 3, 1))), axis=1)
        for i in range(len(layers)):
            x_cvr = tf.contrib.layers.fully_connected(inputs=x_cvr, num_outputs=layers[i],
                                                      weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                      scope='cvr_mlp%d' % i)
            x_cvr = tf.nn.relu(x_cvr)
            if FLAGS.batch_norm:
                x_cvr = batch_norm_layer(x_cvr, train_phase=train_phase,
                                         scope_bn='cvr_bn_%d' % i)
            if mode == tf.estimator.ModeKeys.TRAIN:
                x_cvr = tf.nn.dropout(x_cvr, keep_prob=dropout[i])
                # Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)

        y_cvr = tf.contrib.layers.fully_connected(inputs=x_cvr, num_outputs=1, activation_fn=tf.identity,
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                  scope='cvr_out')
        y_cvr = tf.reshape(y_cvr, shape=[-1])

    with tf.name_scope("CTR_Task"):
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_phase = True
        else:
            train_phase = False
        x_ctr = tf.reduce_sum(tf.multiply(expert_result, tf.reshape(gate2, shape=(-1, 3, 1))), axis=1)
        for i in range(len(layers)):
            x_ctr = tf.contrib.layers.fully_connected(inputs=x_ctr, num_outputs=layers[i],
                                                      weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                      scope='ctr_mlp%d' % i)
            x_ctr = tf.nn.relu(x_ctr)
            if FLAGS.batch_norm:
                x_ctr = batch_norm_layer(x_ctr, train_phase=train_phase,
                                         scope_bn='ctr_bn_%d' % i)
            if mode == tf.estimator.ModeKeys.TRAIN:
                x_ctr = tf.nn.dropout(x_ctr, keep_prob=dropout[
                    i])  # Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)

        y_ctr = tf.contrib.layers.fully_connected(inputs=x_ctr, num_outputs=1, activation_fn=tf.identity,
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                  scope='ctr_out')
        y_ctr = tf.reshape(y_ctr, shape=[-1])
    with tf.variable_scope("MTL-Layer"):
        pctr = tf.sigmoid(y_ctr)
        pcvr = tf.sigmoid(y_cvr)
        pctcvr = pctr * pcvr
    predictions = {"pcvr": pcvr, "pctr": pctr, "pctcvr": pctcvr}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions["y"] = features["y"]
        predictions["z"] = features["z"]
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # ------bulid loss------
    y = labels['y']
    z = labels['z']

    '''epsilon = 1e-7
    conversion_num = tf.to_float(tf.count_nonzero(z))
    conversion_weight = (tf.to_float(tf.size(z)) - conversion_num) / (conversion_num + epsilon)'''

    ctr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_ctr, labels=y))
    ctcvr_loss = tf.reduce_mean(tf.losses.log_loss(predictions=pctcvr, labels=z))
    loss = ctr_task_wgt * ctr_loss + (1 - ctr_task_wgt) * ctcvr_loss + l2_reg * tf.nn.l2_loss(Emb)

    tf.summary.scalar('ctr_loss', ctr_loss)
    tf.summary.scalar('cvr_loss', ctcvr_loss)

    # Provide an estimator spec for `ModeKeys.EVAL`
    threshold = 0.5
    one = tf.ones_like(y)
    zero = tf.zeros_like(y)
    eval_metric_ops = {
        "CTR_AUC": tf.metrics.auc(y, pctr),
        "CTR_ACC": tf.metrics.accuracy(y, tf.where(pctr > threshold, one, zero)),
        "CVR_AUC": tf.metrics.auc(z, pcvr),
        "CVR_ACC": tf.metrics.accuracy(z, tf.where(pcvr > threshold, one, zero)),
        "CTCVR_AUC": tf.metrics.auc(z, pctcvr),
        "CTCVR_ACC": tf.metrics.accuracy(z, tf.where(pctcvr > threshold, one, zero))
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


def main(_):
    # ------check Arguments------
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir + "2"

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('feature_size ', FLAGS.feature_size)
    print('feature_num', FLAGS.feature_num)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print("expert_layers", FLAGS.expert_layers)
    print('dropout ', FLAGS.dropout)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('l2_reg ', FLAGS.l2_reg)
    print('ctr_task_wgt ', FLAGS.ctr_task_wgt)

    tr_files = glob.glob("%s/train/*tfrecord" % FLAGS.data_dir)
    random.shuffle(tr_files)
    print("train_files:", tr_files)
    te_files = glob.glob("%s/test/*tfrecord" % FLAGS.data_dir)
    print("test_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    model_params = {
        "feature_size": FLAGS.feature_size,
        "feature_num": FLAGS.feature_num,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout,
        "ctr_task_wgt": FLAGS.ctr_task_wgt,
        "expert_layers": FLAGS.expert_layers
    }
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                            device_count={'CPU': FLAGS.num_threads})
    config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig().replace(
        session_config=config,
        log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    Estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)
    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=200,
            start_delay_secs=120, throttle_secs=120)
        tf.estimator.train_and_evaluate(Estimator, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        Estimator.evaluate(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=None)
    elif FLAGS.task_type == 'infer':
        preds = Estimator.predict(
            input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size, predict_mode=True),
            predict_keys=None)
        click_result = {'loss': 0, 'acc': 0, 'auc': 0, 'f1': 0, 'ndcg': 0, 'map': 0}
        conversion_result = {'loss': 0, 'acc': 0, 'auc': 0, 'f1': 0, 'ndcg': 0, 'map': 0}
        pctr = []
        y = []
        pctcvr = []
        z = []
        for prob in preds:
            pctr.append(prob['pctr'])
            y.append(prob['y'])

            pctcvr.append(prob['pctcvr'])
            z.append(prob['z'])
        pctr = np.array(pctr)
        y = np.array(y)
        pctcvr = np.array(pctcvr)
        z = np.array(z)
        print(len(pctr))
        te_files_pkl = glob.glob("%s/test/*txt" % FLAGS.data_dir)[0]
        te_len_list = []
        with open(te_files_pkl) as fin:
            while True:
                try:
                    (seq_len, label) = [int(_) for _ in fin.readline().rstrip().split()]
                    te_len_list.append(seq_len)
                    for _ in range(seq_len):
                        fin.readline()
                except:
                    break
        te_len_list = np.array(te_len_list)
        print(sum(te_len_list))

        seq_max_len = 50

        def f(_):
            if _ > 2:
                if _ > seq_max_len:
                    return np.array([False] * (_ - seq_max_len) + [True] * seq_max_len)
                else:
                    return np.array([True] * _)
            else:
                return np.array([False] * _)

        te_len_cond = list(map(f, te_len_list))
        te_len_cond = np.concatenate(te_len_cond)
        print(te_len_cond[:1000])
        print(sum(te_len_cond))
        te_len_arg = np.argwhere(te_len_cond)
        pctr = pctcvr[te_len_arg]
        y = y[te_len_arg]
        pctcvr = pctcvr[te_len_arg]
        z = z[te_len_arg]
        print(len(z))
        te_len_cut = te_len_list[te_len_list > 2]
        print(sum(te_len_cut))
        te_len_cut = np.where(te_len_cut >= seq_max_len, seq_max_len, te_len_cut)
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


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
