import glob
import os
import random
import shutil
from datetime import date, timedelta

import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("num_threads", 16, "Number of threads")
tf.flags.DEFINE_integer("user_size", 24262, "Number of Users")
tf.flags.DEFINE_integer("item_size", 24816, "Number of Items")
tf.flags.DEFINE_integer("other_size", 42, "Size of other_size")
tf.flags.DEFINE_integer("embedding_size", 64, "Embedding size")
tf.flags.DEFINE_integer("num_epochs", 1000, "Number of epochs")
tf.flags.DEFINE_integer("batch_size", 8000, "Number of batch size")
tf.flags.DEFINE_integer("log_steps", 10, "save summary every steps")
tf.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.flags.DEFINE_float("ctr_task_wgt", 0.5, "loss weight of ctr task")
tf.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
tf.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.flags.DEFINE_string("gpus", '', "list of gpus")
tf.flags.DEFINE_boolean("batch_norm", True, "perform batch normaization (True or False)")
tf.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.flags.DEFINE_string("data_dir", './../alicpp', "data dir")
tf.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.flags.DEFINE_string("model_dir", './../alicpp/model_alicpp', "model check point dir")
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


def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print('Parsing', filenames)

    def _parse_fn(record):
        features = {
            "y": tf.FixedLenFeature([], tf.float32),
            "z": tf.FixedLenFeature([], tf.float32),
            "user_id": tf.FixedLenFeature([], tf.int64),
            "item_id": tf.FixedLenFeature([], tf.int64),
            "other_feature_id": tf.VarLenFeature(tf.int64),
            "other_feature_val": tf.VarLenFeature(tf.float32),
        }
        parsed = tf.parse_single_example(record, features)
        y = parsed.pop('y')
        z = parsed.pop('z')
        return parsed, {"y": y, "z": z}

    # Extract lines from input files using the Dataset API, can pass one filename or filename list

    dataset = tf.data.TFRecordDataset(filenames).map(_parse_fn, num_parallel_calls=16).prefetch(
        5000000)  # multi-thread pre-process then prefetch

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
    user_size = params["user_size"]
    item_size = params['item_size']
    other_size = params['other_size']
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    # optimizer = params["optimizer"]
    layers = list(map(int, params["deep_layers"].split(',')))
    dropout = list(map(float, params["dropout"].split(',')))
    ctr_task_wgt = params["ctr_task_wgt"]

    User_Emb = tf.get_variable(name='user_embeddings', shape=[user_size, embedding_size],
                               initializer=tf.glorot_normal_initializer())
    Item_Emb = tf.get_variable(name='item_embeddings', shape=[item_size, embedding_size],
                               initializer=tf.glorot_normal_initializer())
    Other_Emb = tf.get_variable(name='other_embeddings', shape=[other_size, embedding_size],
                                initializer=tf.glorot_normal_initializer())

    user_id = features['user_id']
    item_id = features['item_id']
    other_feature_id = features['other_feature_id']
    other_feature_val = features['other_feature_val']

    with tf.variable_scope("Shared-Embedding-layer"):
        user_feature = tf.nn.embedding_lookup(User_Emb, user_id)
        item_feature = tf.nn.embedding_lookup(Item_Emb, item_id)
        other_feature = tf.nn.embedding_lookup_sparse(Other_Emb, sp_ids=other_feature_id, sp_weights=other_feature_val)
        x_concat = tf.concat([user_feature, item_feature, other_feature], axis=1)
        print(x_concat.shape)

    with tf.name_scope("CVR_Task"):
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_phase = True
        else:
            train_phase = False
        x_cvr = x_concat
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
        x_ctr = x_concat
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
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # ------bulid loss------
    y = labels['y']
    z = labels['z']
    ctr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_ctr, labels=y))
    ctcvr_loss = tf.reduce_mean(tf.losses.log_loss(predictions=pctcvr, labels=z))
    loss = ctr_task_wgt * ctr_loss + (1 - ctr_task_wgt) * ctcvr_loss + l2_reg * tf.nn.l2_loss(
        User_Emb) + l2_reg * tf.nn.l2_loss(Item_Emb) + l2_reg * tf.nn.l2_loss(Other_Emb)

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
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir + "sample"

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('user_size ', FLAGS.user_size)
    print('item_size', FLAGS.item_size)
    print('other_size', FLAGS.other_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('dropout ', FLAGS.dropout)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('l2_reg ', FLAGS.l2_reg)
    print('ctr_task_wgt ', FLAGS.ctr_task_wgt)

    tr_files = glob.glob("%s/train/*rejection.tfrecord" % FLAGS.data_dir)
    random.shuffle(tr_files)
    print("train_files:", tr_files)
    te_files = glob.glob("%s/test/*rejection.tfrecord" % FLAGS.data_dir)
    print("test_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    model_params = {
        "user_size": FLAGS.user_size,
        "item_size": FLAGS.item_size,
        "other_size": FLAGS.other_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
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
            input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=None,
            start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(Estimator, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        Estimator.evaluate(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=None)
    elif FLAGS.task_type == 'infer':
        preds = Estimator.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size),
                                  predict_keys=None)
        with open(FLAGS.data_dir + "/pred2.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\t%f\n" % (prob['pctr'], prob['pcvr']))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
