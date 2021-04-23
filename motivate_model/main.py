import glob
import os
import pickle
from multiprocessing import Process, Queue

import numpy as np
import tensorflow as tf

import load_data
import utils


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
                else:
                    total_data = load_data.load_fun(dataset, seq_max_len, infile, batch_size)
                if not total_data[-1]:
                    break
                q.put(total_data)
                if valid and step >= buffer_size:
                    break
        infile.close()
    flag.get()


def get_feed(total_data, placeholders, dataset='Criteo'):
    feed_dict = {placeholders['click_label']: total_data[-3],
                 placeholders['conversion_label']: total_data[-2],
                 placeholders['seq_len']: total_data[-1]}
    if dataset == 'Criteo':
        feed_dict.update({placeholders['input']: total_data[0]})
    else:
        feed_dict.update({placeholders['input_id']: utils.sparse_tuple_from(total_data[0]),
                          placeholders['input_value']: utils.sparse_tuple_from(total_data[1], dtype=np.float32)})
    return feed_dict


def print_result(loss, conversion_loss, click_loss, eval_metric_ops):
    print("Loss:{}\nctr loss:{}\tacc:{}\tauc:{}\ncvr loss:{}\tacc:{}\tauc:{}".format(
        loss,
        click_loss, eval_metric_ops['CTR_ACC'][0], eval_metric_ops['CTR_AUC'][0],
        conversion_loss, eval_metric_ops['CTCVR_ACC'][0], eval_metric_ops['CTCVR_AUC'][0]
    ))
    return loss, click_loss, eval_metric_ops['CTR_ACC'][0], eval_metric_ops['CTR_AUC'][0], conversion_loss, \
           eval_metric_ops['CTCVR_ACC'][0], eval_metric_ops['CTCVR_AUC'][0]


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
            Pw.join()
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
    click_loss, conversion_loss, loss, eval_metric_ops, _ = model.forward(training=False)
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
        while not q.empty() or not flag.empty():
            total_data = q.get(True)
            feed_dict = get_feed(total_data, placeholders, args['dataset'])
            batch_loss, batch_cvr_loss, batch_click_loss, batch_eval = sess.run(
                [loss, conversion_loss, click_loss, eval_metric_ops], feed_dict=feed_dict)
            print("Step:{}".format(step), end='\t')
            eval_result = print_result(batch_loss, batch_cvr_loss, batch_click_loss, batch_eval)
            for i in range(len(all_result)):
                all_result[i] += eval_result[i]
            step += 1
        Pw.join()
        all_result = [i / step for i in all_result]
        print(all_result)


if __name__ == "__main__":
    import argparse
    from config import get_exp_configure

    parser = argparse.ArgumentParser(description="Unbiased learning")
    parser.add_argument(
        "-m", "--model", type=str, choices=['motivate', "Heroes"], default="motivate", help="Model to use"
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
