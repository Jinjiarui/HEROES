import glob
import os
from multiprocessing import Pool as ThreadPool
from collections import defaultdict
import numpy as np
import tensorflow as tf
import random

flags = tf.flags
FLAGS = flags.FLAGS
LOG = tf.logging
tf.flags.DEFINE_string("input_dir_train", "./../alicpp/%s/remap_sample" % ('train'), "train input dir")
tf.flags.DEFINE_string("output_dir_train", "./../alicpp/%s" % ('train'), "train output dir")
tf.flags.DEFINE_string("input_dir_test", "./../alicpp/%s/remap_sample" % ('test'), "test input dir")
tf.flags.DEFINE_string("output_dir_test", "./../alicpp/%s" % ('test'), "test output dir")
tf.flags.DEFINE_integer("threads", 32, "threads num")
other_feat_num = 41


def do_write(fields, tfrecord_out):
    y, z = fields[2], fields[3]
    y, z = [float(y)], [float(z)]
    feature = {
        "y": tf.train.Feature(float_list=tf.train.FloatList(value=y)),
        "z": tf.train.Feature(float_list=tf.train.FloatList(value=z))
    }

    user_id, item_id = [int(fields[4])], [int(fields[5])]
    feature.update({
        "user_id": tf.train.Feature(int64_list=tf.train.Int64List(value=user_id)),
        "item_id": tf.train.Feature(int64_list=tf.train.Int64List(value=item_id))
    })
    if len(fields) >= 7:
        splits = [_.split(':') for _ in fields[6:]]
        ffv = np.reshape(splits, (-1, 2))
        feature.update(
            {"other_feature_id": tf.train.Feature(
                int64_list=tf.train.Int64List(value=ffv[:, 0].astype(np.int))),
                "other_feature_val": tf.train.Feature(
                    float_list=tf.train.FloatList(value=ffv[:, 1].astype(np.float)))})
    else:
        feature.update(
            {"other_feature_id": tf.train.Feature(
                int64_list=tf.train.Int64List(value=np.array([other_feat_num]).astype(np.int))),
                "other_feature_val": tf.train.Feature(
                    float_list=tf.train.FloatList(value=np.array([1.0]).astype(np.float)))})
    # serialized to Example
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()
    tfrecord_out.write(serialized)


def gen_tfrecords(in_file):
    print(in_file)
    basename_1 = os.path.basename(in_file[0]) + "_repeat.tfrecord"
    out_file_1 = os.path.join(FLAGS.output_dir_train, basename_1)
    tfrecord_out_1 = tf.python_io.TFRecordWriter(out_file_1)
    basename_2 = os.path.basename(in_file[1]) + "_repeat.tfrecord"
    out_file_2 = os.path.join(FLAGS.output_dir_test, basename_2)
    tfrecord_out_2 = tf.python_io.TFRecordWriter(out_file_2)
    num = 0
    conversion_list_1 = []
    non_conversion_list_1 = []
    conversion_list_2 = []
    non_conversion_list_2 = []
    with open(in_file[0]) as fi:
        for line in fi:
            num += 1
            if num % 100000 == 0:
                print(num)
                # break
            fields = line.rstrip().split()
            if fields[3] == '1':
                conversion_list_1.append(fields)
            else:
                non_conversion_list_1.append(fields)
    with open(in_file[1]) as fi:
        for line in fi:
            num += 1
            if num % 100000 == 0:
                print(num)
                # break
            fields = line.rstrip().split()
            if fields[3] == '1':
                conversion_list_2.append(fields)
            else:
                non_conversion_list_2.append(fields)
    list_1 = non_conversion_list_1 + conversion_list_1 * (len(non_conversion_list_1) // len(conversion_list_1))
    random.shuffle(list_1)
    list_2 = non_conversion_list_2 + conversion_list_2 * (len(non_conversion_list_2) // len(conversion_list_2))
    random.shuffle(list_2)
    for fields in list_1:
        do_write(fields, tfrecord_out_1)
    for fields in list_2:
        do_write(fields, tfrecord_out_2)
    random.shuffle(list_1)
    random.shuffle(list_2)
    tfrecord_out_1.close()
    tfrecord_out_2.close()


def main(_):
    if not os.path.exists(FLAGS.output_dir_train):
        os.mkdir(FLAGS.output_dir_train)
    file_list_train = glob.glob(os.path.join(FLAGS.input_dir_train, "*.txt"))
    if not os.path.exists(FLAGS.output_dir_test):
        os.mkdir(FLAGS.output_dir_test)
    file_list_test = glob.glob(os.path.join(FLAGS.input_dir_test, "*.txt"))
    print(file_list_train, file_list_test)
    print("total files: %d" % (len(file_list_train) + len(file_list_test)))

    pool = ThreadPool(FLAGS.threads)  # Sets the pool size
    pool.map(gen_tfrecords, [file_list_train + file_list_test])
    pool.close()
    pool.join()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
