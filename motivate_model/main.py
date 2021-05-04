import glob
import os
import pickle
from multiprocessing import Process, Queue

import numpy as np
import tensorflow as tf

import load_data
import utils
import evaluate_util


def get_placeholders(args):
    placeholders = {'seq_len': tf.placeholder(tf.int32, shape=[None], name='seqlen'),
                    'click_label': tf.placeholder(tf.float32, shape=[None, args['seq_max_len'], args['n_classes']],
                                                  name='clicks'),
                    'conversion_label': tf.placeholder(tf.float32,
                                                       shape=[None, args['seq_max_len'], args['n_classes']],
                                                       name='labels')}
    if args['dataset'] == 'Criteo':
        placeholders.update({'input': tf.placeholder(tf.float32, shape=[None, args['seq_max_len'], 11])})
    else:
        placeholders.update({'input_id': tf.sparse_placeholder(tf.int32, shape=[None, None], name='id'),
                             'input_value': tf.sparse_placeholder(tf.float32, shape=[None, None], name='value')})
    return placeholders


def write(q, flag, file_list, dataset, batch_size, seq_max_len, buffer_size=10, valid=False):
    for file_name in file_list:
        file_len = file_name + '.pkl'
        if os.path.exists(file_len):
            with open(file_len, 'rb') as len_f:
                file_len_list = list(pickle.load(len_f))
        else:
            file_len_list = None
        step = 0
        infile = open(file_name, 'r')
        while True:
            if q.qsize() <= buffer_size:
                step += 1
                if file_len_list is not None:
                    total_data = load_data.load_fun(dataset, seq_max_len, infile,
                                                    file_len_list[(step - 1) * batch_size:step * batch_size])
                    total_data_id, total_data_value, total_click, total_label, total_seqlen = total_data
                    if total_data_id:
                        total_data_id = utils.sparse_tuple_from(total_data_id)
                        total_data_value = utils.sparse_tuple_from(total_data_value, dtype=np.float32)
                        total_data = [total_data_id, total_data_value, total_click, total_label, total_seqlen]
                else:
                    total_data = load_data.load_fun(dataset, seq_max_len, infile, batch_size)
                if not total_data[-1]:
                    break
                q.put(total_data)
                if valid and step >= buffer_size:
                    break
        infile.close()
    print("Load data Quit!")
    flag.get()


def get_feed(total_data, placeholders, dataset='Criteo'):
    feed_dict = {placeholders['click_label']: total_data[-3],
                 placeholders['conversion_label']: total_data[-2],
                 placeholders['seq_len']: total_data[-1]}
    if dataset == 'Criteo':
        feed_dict.update({placeholders['input']: total_data[0]})
    else:
        feed_dict.update({placeholders['input_id']: total_data[0],
                          placeholders['input_value']: total_data[1]})
    return feed_dict


def print_result(loss, conversion_loss, click_loss, eval_metric_ops):
    print("Loss:{}\nctr loss:{}\tacc:{}\tauc:{}\ncvr loss:{}\tacc:{}\tauc:{}".format(
        loss,
        click_loss, eval_metric_ops['CTR_ACC'][0], eval_metric_ops['CTR_AUC'][0],
        conversion_loss, eval_metric_ops['CTCVR_ACC'][0], eval_metric_ops['CTCVR_AUC'][0]
    ))
    return loss, click_loss, eval_metric_ops['CTR_ACC'][0], eval_metric_ops['CTR_AUC'][0], conversion_loss, \
           eval_metric_ops['CTCVR_ACC'][0], eval_metric_ops['CTCVR_AUC'][0]


def evaluate(pctr, click_y, pctcvr, conversion_y, seq_len, dataset):
    print(len(pctr), len(click_y), len(pctcvr), len(conversion_y), len(seq_len))
    print(click_y[:100])
    predict_label_click = np.concatenate([np.expand_dims(pctr, axis=-1), np.expand_dims(click_y, axis=-1)], axis=-1)
    predict_label_conversion = np.concatenate([np.expand_dims(pctcvr, axis=-1), np.expand_dims(conversion_y, axis=-1)],
                                              axis=-1)
    click_result = {'loss': 0, 'acc': 0, 'auc': 0, 'f1': 0, 'ndcg': 0, 'map': 0}
    conversion_result = {'loss': 0, 'acc': 0, 'auc': 0, 'f1': 0, 'ndcg': 0, 'map': 0}
    indices = np.cumsum(seq_len, dtype=np.int)
    indices = np.append([0], indices)
    print(seq_len)
    print(indices)
    if dataset == 'alicpp':
        pctr_copy, click_y_copy, indices_click = evaluate_util.copy_positive(pctr, click_y, indices)
        pctcvr_copy, conversion_y_copy, indices_conversion = evaluate_util.copy_positive(pctcvr, conversion_y, indices)
        predict_label_click_copy = np.concatenate(
            [np.expand_dims(pctr_copy, axis=-1), np.expand_dims(click_y_copy, axis=-1)], axis=-1)
        predict_label_conversion_copy = np.concatenate(
            [np.expand_dims(pctcvr_copy, axis=-1), np.expand_dims(conversion_y_copy, axis=-1)], axis=-1)

        print(len(pctr_copy), len(click_y_copy), len(pctcvr_copy), len(conversion_y_copy))
    else:
        pctr_copy, click_y_copy, indices_click = pctr, click_y, indices
        pctcvr_copy, conversion_y_copy, indices_conversion = pctcvr, conversion_y, indices
        predict_label_click_copy, predict_label_conversion_copy = predict_label_click, predict_label_conversion
    click_result['loss'] = evaluate_util.evaluate_logloss(pctr_copy, click_y_copy)
    click_result['acc'] = evaluate_util.evaluate_acc(pctr_copy, click_y_copy)
    click_result['auc'] = evaluate_util.evaluate_auc(pctr, click_y)
    click_result['f1'] = evaluate_util.evaluate_f1_score(pctr_copy, click_y_copy)

    predict_label_click = tf.convert_to_tensor(predict_label_click, dtype=tf.float32)
    indices = tf.convert_to_tensor(indices)
    print(predict_label_click, indices)
    click_result['ndcg'] = evaluate_util.evaluate_ndcg(None, predict_label_click, indices)
    click_result['ndcg1'] = evaluate_util.evaluate_ndcg(1, predict_label_click, indices)
    click_result['ndcg3'] = evaluate_util.evaluate_ndcg(3, predict_label_click, indices)
    click_result['ndcg5'] = evaluate_util.evaluate_ndcg(5, predict_label_click, indices)
    click_result['ndcg10'] = evaluate_util.evaluate_ndcg(10, predict_label_click, indices)

    predict_label_click_copy = tf.convert_to_tensor(predict_label_click_copy, dtype=tf.float32)
    indices_click = tf.convert_to_tensor(indices_click)

    click_result['map'] = evaluate_util.evaluate_map(None, predict_label_click_copy, indices_click)
    click_result['map1'] = evaluate_util.evaluate_map(1, predict_label_click_copy, indices_click)
    click_result['map3'] = evaluate_util.evaluate_map(3, predict_label_click_copy, indices_click)
    click_result['map5'] = evaluate_util.evaluate_map(5, predict_label_click_copy, indices_click)
    click_result['map10'] = evaluate_util.evaluate_map(10, predict_label_click_copy, indices_click)

    conversion_result['loss'] = evaluate_util.evaluate_logloss(pctcvr_copy, conversion_y_copy)
    conversion_result['acc'] = evaluate_util.evaluate_acc(pctcvr_copy, conversion_y_copy)
    conversion_result['auc'] = evaluate_util.evaluate_auc(pctcvr, conversion_y)
    conversion_result['f1'] = evaluate_util.evaluate_f1_score(pctcvr_copy, conversion_y_copy)

    predict_label_conversion = tf.convert_to_tensor(predict_label_conversion, dtype=tf.float32)
    conversion_result['ndcg'] = evaluate_util.evaluate_ndcg(None, predict_label_conversion, indices)
    conversion_result['ndcg1'] = evaluate_util.evaluate_ndcg(1, predict_label_conversion, indices)
    conversion_result['ndcg3'] = evaluate_util.evaluate_ndcg(3, predict_label_conversion, indices)
    conversion_result['ndcg5'] = evaluate_util.evaluate_ndcg(5, predict_label_conversion, indices)
    conversion_result['ndcg10'] = evaluate_util.evaluate_ndcg(10, predict_label_conversion, indices)

    predict_label_conversion_copy = tf.convert_to_tensor(predict_label_conversion_copy, dtype=tf.float32)
    indices_conversion = tf.convert_to_tensor(indices_conversion)
    conversion_result['map'] = evaluate_util.evaluate_map(None, predict_label_conversion_copy, indices_conversion)
    conversion_result['map1'] = evaluate_util.evaluate_map(1, predict_label_conversion_copy, indices_conversion)
    conversion_result['map3'] = evaluate_util.evaluate_map(3, predict_label_conversion_copy, indices_conversion)
    conversion_result['map5'] = evaluate_util.evaluate_map(5, predict_label_conversion_copy, indices_conversion)
    conversion_result['map10'] = evaluate_util.evaluate_map(10, predict_label_conversion_copy, indices_conversion)

    return click_result, conversion_result


def main(args):
    placeholders = get_placeholders(args)
    args['data_dir'] = args['data_dir'].format(args['dataset'])
    if args['dataset'] == 'Criteo':
        tr_files = glob.glob("%s/train/*.txt" % args['data_dir'])
        te_files = glob.glob("%s/test/*.txt" % args['data_dir'])
    else:
        tr_files = glob.glob("%s/train/remap_sample/r*.txt" % args['data_dir'])
        te_files = glob.glob("%s/test/remap_sample/r*.txt" % args['data_dir'])
    print("train_files:", tr_files)
    print("test_files:", te_files)
    model_folder = "./saved_models/"
    model_path = model_folder + str(args["exp_name"]) + "_" + str(args["postfix"])
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model = utils.load_model(placeholders, args)
    click_loss, conversion_loss, loss, eval_metric_ops, train_op = model.forward()
    saver = tf.train.Saver(max_to_keep=3)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    q = Queue()
    flag = Queue()
    flag.put(True)
    write(q, flag, te_files, args['dataset'], 5 * args['batch_size'], args['seq_max_len'], valid=True)
    valid_data = q.get()
    best_auc = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if args['load_model']:
            saver.restore(sess, model_path)
        print("Begin Train:")
        for i in range(args['epoch']):
            step = 0
            q = Queue()
            flag = Queue()
            flag.put(True)
            Pw = Process(target=write,
                         args=(q, flag, tr_files, args['dataset'], args['batch_size'], args['seq_max_len']))
            Pw.start()
            while not q.empty() or not flag.empty():
                total_data = q.get()
                feed_dict = get_feed(total_data, placeholders, args['dataset'])
                batch_loss, batch_cvr_loss, batch_click_loss, batch_eval, _ = sess.run(
                    [loss, conversion_loss, click_loss, eval_metric_ops, train_op], feed_dict=feed_dict)
                print("Epoch:{}\tStep:{}".format(i, step), end='\t')
                print_result(batch_loss, batch_cvr_loss, batch_click_loss, batch_eval)
                step += 1
            print()
            feed_dict = get_feed(valid_data, placeholders, args['dataset'])

            batch_loss, batch_cvr_loss, batch_click_loss, batch_eval = sess.run(
                [loss, conversion_loss, click_loss, eval_metric_ops], feed_dict=feed_dict)
            print("Valid Result:{}".format(i, step), end='\t')
            print_result(batch_loss, batch_cvr_loss, batch_click_loss, batch_eval)
            print()
            if batch_eval['CTCVR_AUC'][0] > best_auc:
                best_auc = batch_eval['CTCVR_AUC'][0]
                saver.save(sess, model_path)
    print("Test:")
    click_loss, conversion_loss, loss, eval_metric_ops, \
    prediction_c, reshape_click_label, prediction_v, reshape_conversion_label \
        = model.forward(training=False)
    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, model_path)
        step = 0
        q = Queue()
        flag = Queue()
        flag.put(True)
        Pw = Process(target=write,
                     args=(q, flag, te_files, args['dataset'], 5 * args['batch_size'], args['seq_max_len']))
        Pw.start()
        all_result = [0] * 7
        pctr, click_y, pctcvr, conversion_y, seq_len = np.array([]), np.array([]), np.array([]), np.array([]), np.array(
            [])
        while not q.empty() or not flag.empty():
            total_data = q.get(True)
            feed_dict = get_feed(total_data, placeholders, args['dataset'])
            batch_loss, batch_cvr_loss, batch_click_loss, batch_eval, p_click, l_click, p_conver, l_conver = sess.run(
                [loss, conversion_loss, click_loss, eval_metric_ops, prediction_c, reshape_click_label, prediction_v,
                 reshape_conversion_label], feed_dict=feed_dict)
            print("Step:{}".format(step), end='\t')
            eval_result = print_result(batch_loss, batch_cvr_loss, batch_click_loss, batch_eval)
            for i in range(len(all_result)):
                all_result[i] += eval_result[i]
            step += 1
            pctr = np.append(pctr, p_click)
            click_y = np.append(click_y, l_click)
            pctcvr = np.append(pctcvr, p_conver)
            conversion_y = np.append(conversion_y, l_conver)
            seq_len = np.append(seq_len, total_data[-1])

        print("Test Quit!")
        click_result, conversion_result = evaluate(pctr, click_y, pctcvr, conversion_y, seq_len,
                                                   dataset=args['dataset'])
        print("Click Result")
        for k, v in click_result.items():
            if k[:3] in ['map', 'ndc']:
                v = sess.run(v)
            print("{}:{}".format(k, v))
        print()
        print("Conversion Result")
        for k, v in conversion_result.items():
            if k[:3] in ['map', 'ndc']:
                v = sess.run(v)
            print("{}:{}".format(k, v))


if __name__ == "__main__":
    import argparse
    from config import get_exp_configure

    parser = argparse.ArgumentParser(description="Unbiased learning")
    parser.add_argument(
        "-m", "--model", type=str,
        choices=['motivate', "Heroes", 'motivate-single', 'RRN', 'LSTM', 'time_LSTM', 'STAMP', 'NARM', 'DUPN',
                 'Motivate-Heroes'],
        default="motivate-single",
        help="Model to use"
    )
    parser.add_argument('-d', '--dataset', type=str, choices=["Criteo", "alicpp"], default="Criteo",
                        help="Dataset to use")
    parser.add_argument('--data_dir', type=str, default='./../{}')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--load_model', action="store_true", default=False)
    parser.add_argument('--ctr_task_wgt', type=float, default=0.5,
                        help='The proportion of the two behavior loss during optimization')
    parser.add_argument("-c", "--cuda", type=str, default="0")
    parser.add_argument("--postfix", type=str, default="", help="a string appended to the file name of the saved model")
    parser.add_argument("--rand_seed", type=int, default=-1, help="random seed for torch and numpy")
    parser.add_argument("--residual", action="store_true")
    args = parser.parse_args().__dict__
    args["exp_name"] = "_".join([args["model"], args["dataset"]])
    args.update(get_exp_configure(args))
    if args["cuda"] == "none":
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args["cuda"]
    main(args)
