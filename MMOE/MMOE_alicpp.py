import glob
import os
import pickle
import random
import shutil
import sys
from datetime import date, timedelta

import numpy as np

sys.path.append("../")
from utils import utils
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("num_threads", 64, "Number of threads")
tf.flags.DEFINE_integer("feature_size", 638095, "Size of other_size")
tf.flags.DEFINE_integer("embedding_size", 96, "Embedding size")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of epochs")
tf.flags.DEFINE_integer("field_size", 11, "Number of common fields")
tf.flags.DEFINE_integer("batch_size", 8000, "Number of batch size")
tf.flags.DEFINE_integer("log_steps", 10, "save summary every steps")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_float("l2_reg", 0.01, "L2 regularization")
tf.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.flags.DEFINE_float("ctr_task_wgt", 0.5, "loss weight of ctr task")
tf.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.flags.DEFINE_string("deep_layers", '512,256', "deep layers")
tf.flags.DEFINE_string("expert_layers", '64,32,16', "expert_layers")
tf.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.flags.DEFINE_string("gpus", '0', "list of gpus")
tf.flags.DEFINE_boolean("batch_norm", True, "perform batch normaization (True or False)")
tf.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.flags.DEFINE_string("data_dir", './../alicpp', "data dir")
tf.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.flags.DEFINE_string("model_dir", './../alicpp/model_alicpp_mmoe', "model check point dir")
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
            "click_y": tf.FixedLenFeature([], tf.float32),
            "conversion_y": tf.FixedLenFeature([], tf.float32),
            "feat_ids": tf.FixedLenFeature([FLAGS.field_size], tf.int64),
            # "feat_vals": tf.FixedLenFeature([None], tf.float32),
            "u_catids": tf.VarLenFeature(tf.int64),
            "u_catvals": tf.VarLenFeature(tf.float32),
            "u_shopids": tf.VarLenFeature(tf.int64),
            "u_shopvals": tf.VarLenFeature(tf.float32),
            "u_intids": tf.VarLenFeature(tf.int64),
            "u_intvals": tf.VarLenFeature(tf.float32),
            "u_brandids": tf.VarLenFeature(tf.int64),
            "u_brandvals": tf.VarLenFeature(tf.float32),
            "a_catids": tf.FixedLenFeature([], tf.int64),
            "a_shopids": tf.FixedLenFeature([], tf.int64),
            "a_brandids": tf.FixedLenFeature([], tf.int64),
            "a_intids": tf.VarLenFeature(tf.int64),
            "x_aids": tf.VarLenFeature(tf.int64),
            "x_avals": tf.VarLenFeature(tf.float32),
            "x_bids": tf.VarLenFeature(tf.int64),
            "x_bvals": tf.VarLenFeature(tf.float32),
            "x_cids": tf.VarLenFeature(tf.int64),
            "x_cvals": tf.VarLenFeature(tf.float32),
            "x_dids": tf.VarLenFeature(tf.int64),
            "x_dvals": tf.VarLenFeature(tf.float32),
        }
        parsed = tf.parse_single_example(record, features)
        if predict_mode:
            return parsed, {}
        y = parsed.pop('click_y')
        z = parsed.pop('conversion_y')
        return parsed, {"click_y": y, "conversion_y": z}

    # Extract lines from input files using the Dataset API, can pass one filename or filename list

    dataset = tf.data.TFRecordDataset(filenames).map(_parse_fn, num_parallel_calls=64).prefetch(
        5000000)  # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=20000)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)  # Batch size to use

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    # ------hyperparameters----
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    field_size = params["field_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    # optimizer = params["optimizer"]
    layers = list(map(int, params["deep_layers"].split(',')))
    expert_layers = list(map(int, params["expert_layers"].split(',')))
    dropout = list(map(float, params["dropout"].split(',')))
    ctr_task_wgt = params["ctr_task_wgt"]

    feat_emb = tf.get_variable(name='other_embeddings', shape=[feature_size, embedding_size],
                               initializer=tf.glorot_normal_initializer())
    common_dims = field_size * embedding_size
    feat_ids = features['feat_ids']
    # {User multi-hot}
    u_catids = features['u_catids']
    u_catvals = features['u_catvals']
    u_shopids = features['u_shopids']
    u_shopvals = features['u_shopvals']
    u_intids = features['u_intids']
    u_intvals = features['u_intvals']
    u_brandids = features['u_brandids']
    u_brandvals = features['u_brandvals']
    # {Ad}
    a_catids = features['a_catids']
    a_shopids = features['a_shopids']
    a_brandids = features['a_brandids']
    a_intids = features['a_intids']  # multi-hot
    # {X}
    x_aids = features['x_aids']
    x_avals = features['x_avals']
    x_bids = features['x_bids']
    x_bvals = features['x_bvals']
    x_cids = features['x_cids']
    x_cvals = features['x_cvals']
    x_dids = features['x_dids']
    x_dvals = features['x_dvals']

    with tf.variable_scope("Shared-Embedding-layer"):
        common_embs = tf.nn.embedding_lookup(feat_emb, feat_ids)  # None * F' * K
        u_cat_emb = tf.nn.embedding_lookup_sparse(feat_emb, sp_ids=u_catids, sp_weights=u_catvals)  # None * K
        u_shop_emb = tf.nn.embedding_lookup_sparse(feat_emb, sp_ids=u_shopids, sp_weights=u_shopvals)
        u_brand_emb = tf.nn.embedding_lookup_sparse(feat_emb, sp_ids=u_brandids, sp_weights=u_brandvals)
        u_int_emb = tf.nn.embedding_lookup_sparse(feat_emb, sp_ids=u_intids, sp_weights=u_intvals)
        a_int_emb = tf.nn.embedding_lookup_sparse(feat_emb, sp_ids=a_intids, sp_weights=None)
        a_cat_emb = tf.nn.embedding_lookup(feat_emb, a_catids)
        a_shop_emb = tf.nn.embedding_lookup(feat_emb, a_shopids)
        a_brand_emb = tf.nn.embedding_lookup(feat_emb, a_brandids)
        x_a_emb = tf.nn.embedding_lookup_sparse(feat_emb, sp_ids=x_aids, sp_weights=x_avals)
        x_b_emb = tf.nn.embedding_lookup_sparse(feat_emb, sp_ids=x_bids, sp_weights=x_bvals)
        x_c_emb = tf.nn.embedding_lookup_sparse(feat_emb, sp_ids=x_cids, sp_weights=x_cvals)
        x_d_emb = tf.nn.embedding_lookup_sparse(feat_emb, sp_ids=x_dids, sp_weights=x_dvals)
        x_concat = tf.concat(
            [tf.reshape(common_embs, shape=[-1, common_dims]), u_cat_emb, u_shop_emb, u_brand_emb, u_int_emb, a_cat_emb,
             a_shop_emb, a_brand_emb, a_int_emb, x_a_emb, x_b_emb, x_c_emb, x_d_emb], axis=1)
        print(x_concat.shape)
    expert_num = 3
    with tf.variable_scope("Gate1"):
        gate1 = tf.contrib.layers.fully_connected(inputs=x_concat, num_outputs=expert_num,
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                  scope='gate1')
        gate1 = tf.nn.softmax(gate1)
    with tf.variable_scope("Gate2"):
        gate2 = tf.contrib.layers.fully_connected(inputs=x_concat, num_outputs=expert_num,
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
            expert = x_concat
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
        predictions["click_y"] = features["click_y"]
        predictions["conversion_y"] = features["conversion_y"]
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # ------bulid loss------
    y = labels['click_y']
    z = labels['conversion_y']

    epsilon = 1e-7
    click_weight = 0.14
    conversion_weight = 0.023
    ctr_loss = - (1 - click_weight) / click_weight * y * tf.log(pctr + epsilon) - (1 - y) * tf.log(1 - pctr + epsilon)
    ctr_loss = tf.reduce_mean(ctr_loss)
    ctcvr_loss = - (1 - conversion_weight) / conversion_weight * z * tf.log(pctcvr + epsilon) - \
                 (1 - z) * tf.log(1 - pctcvr + epsilon)
    ctcvr_loss = tf.reduce_mean(ctcvr_loss)
    # ctr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_ctr, labels=click_y))
    # ctcvr_loss = tf.reduce_mean(tf.losses.log_loss(predictions=pctcvr, labels=conversion_y))
    loss = (ctr_task_wgt * ctr_loss + (1 - ctr_task_wgt) * ctcvr_loss) * 100 + l2_reg * tf.nn.l2_loss(feat_emb)
    tf.summary.scalar('ctr_loss', ctr_loss)
    tf.summary.scalar('cvr_loss', ctcvr_loss)

    # Provide an estimator spec for `ModeKeys.EVAL`
    threshold = 0.5
    one = tf.ones_like(y)
    zero = tf.zeros_like(y)
    eval_metric_ops = {
        "CTR_AUC": tf.metrics.auc(y, pctr),
        "CTR_ACC": tf.metrics.accuracy(y, tf.where(pctr >= threshold, one, zero)),
        "CVR_AUC": tf.metrics.auc(z, pcvr),
        "CVR_ACC": tf.metrics.accuracy(z, tf.where(pcvr >= threshold, one, zero)),
        "CTCVR_AUC": tf.metrics.auc(z, pctcvr),
        "CTCVR_ACC": tf.metrics.accuracy(z, tf.where(pctcvr >= threshold, one, zero))
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    gvs, v = zip(*optimizer.compute_gradients(loss))
    gvs, _ = tf.clip_by_global_norm(gvs, 5.0)
    gvs = zip(gvs, v)
    train_op = optimizer.apply_gradients(gvs, global_step=tf.train.get_global_step())

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
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('field_size ', FLAGS.field_size)
    print('feature_size', FLAGS.feature_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('dropout ', FLAGS.dropout)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('expert_layers', FLAGS.expert_layers)
    print('learning_rate ', FLAGS.learning_rate)
    print('l2_reg ', FLAGS.l2_reg)
    print('ctr_task_wgt ', FLAGS.ctr_task_wgt)

    tr_files = glob.glob("%s/train/r*txt.tfrecord" % FLAGS.data_dir)
    random.shuffle(tr_files)
    print("train_files:", tr_files)
    te_files = glob.glob("%s/test/r*txt.tfrecord" % FLAGS.data_dir)
    print("test_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        'expert_layers': FLAGS.expert_layers,
        "dropout": FLAGS.dropout,
        "ctr_task_wgt": FLAGS.ctr_task_wgt
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
            input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,
                                      perform_shuffle=True))
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=100,
            start_delay_secs=1000, throttle_secs=1200)
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
            y.append(prob['click_y'])

            pctcvr.append(prob['pctcvr'])
            z.append(prob['conversion_y'])
        pctr = np.array(pctr)
        y = np.array(y)
        pctcvr = np.array(pctcvr)
        z = np.array(z)
        print(len(pctr))
        te_files_pkl = glob.glob("%s/test/remap_sample/r*txt.pkl" % FLAGS.data_dir)[0]
        with open(te_files_pkl, 'rb') as len_f:
            te_len_list = np.array(pickle.load(len_f))
        seq_max_len = 160
        print(sum(te_len_list))

        def f(_):
            if _ > seq_max_len:
                return np.array([False] * (_ - seq_max_len) + [True] * seq_max_len)
            else:
                return np.array([True] * _)

        te_len_cond = list(map(f, te_len_list))
        te_len_cond = np.concatenate(te_len_cond)
        print(sum(te_len_cond))
        te_len_arg = np.argwhere(te_len_cond)
        pctr = pctcvr[te_len_arg]
        y = y[te_len_arg]
        pctcvr = pctcvr[te_len_arg]
        z = z[te_len_arg]
        print(len(z))
        te_len_cut = np.where(te_len_list >= seq_max_len, seq_max_len, te_len_list)
        print(sum(te_len_cut))
        indices = np.cumsum(te_len_cut)

        pctr_copy, y_copy, indices_click = utils.copy_positive(pctr, y, indices)
        pctcvr_copy, z_copy, indices_label = utils.copy_positive(pctcvr, z, indices)
        print(len(pctr_copy), len(y_copy), len(pctcvr_copy), len(z_copy))
        click_result['loss'] = utils.evaluate_logloss(pctr_copy, y_copy)
        click_result['acc'] = utils.evaluate_acc(pctr_copy, y_copy)
        click_result['auc'] = utils.evaluate_auc(pctr, y)
        click_result['f1'] = utils.evaluate_f1_score(pctr_copy, y_copy)
        click_result['ndcg'] = utils.evaluate_ndcg(None, pctr, y, indices)
        click_result['ndcg1'] = utils.evaluate_ndcg(1, pctr, y, indices)
        click_result['ndcg3'] = utils.evaluate_ndcg(3, pctr, y, indices)
        click_result['ndcg5'] = utils.evaluate_ndcg(5, pctr, y, indices)
        click_result['ndcg10'] = utils.evaluate_ndcg(10, pctr, y, indices)
        click_result['map'] = utils.evaluate_map(None, pctr_copy, y_copy, indices_click)
        click_result['map1'] = utils.evaluate_map(1, pctr_copy, y_copy, indices_click)
        click_result['map3'] = utils.evaluate_map(3, pctr_copy, y_copy, indices_click)
        click_result['map5'] = utils.evaluate_map(5, pctr_copy, y_copy, indices_click)
        click_result['map10'] = utils.evaluate_map(10, pctr_copy, y_copy, indices_click)

        conversion_result['loss'] = utils.evaluate_logloss(pctcvr_copy, z_copy)
        conversion_result['acc'] = utils.evaluate_acc(pctcvr_copy, z_copy)
        conversion_result['auc'] = utils.evaluate_auc(pctcvr, z)
        conversion_result['f1'] = utils.evaluate_f1_score(pctcvr_copy, z_copy)
        conversion_result['ndcg'] = utils.evaluate_ndcg(None, pctcvr, z, indices)
        conversion_result['ndcg1'] = utils.evaluate_ndcg(1, pctcvr, z, indices)
        conversion_result['ndcg3'] = utils.evaluate_ndcg(3, pctcvr, z, indices)
        conversion_result['ndcg5'] = utils.evaluate_ndcg(5, pctcvr, z, indices)
        conversion_result['ndcg10'] = utils.evaluate_ndcg(10, pctcvr, z, indices)
        conversion_result['map'] = utils.evaluate_map(None, pctcvr_copy, z_copy, indices_label)
        conversion_result['map1'] = utils.evaluate_map(1, pctcvr_copy, z_copy, indices_label)
        conversion_result['map3'] = utils.evaluate_map(3, pctcvr_copy, z_copy, indices_label)
        conversion_result['map5'] = utils.evaluate_map(5, pctcvr_copy, z_copy, indices_label)
        conversion_result['map10'] = utils.evaluate_map(10, pctcvr_copy, z_copy, indices_label)

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
