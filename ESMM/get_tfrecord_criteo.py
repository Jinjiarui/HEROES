import glob
import os
from multiprocessing import Pool as ThreadPool

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
LOG = tf.logging
mod = 'test'
tf.flags.DEFINE_string("input_dir", "./../Criteo/%s" % (mod), "input dir")
tf.flags.DEFINE_string("output_dir", "./../Criteo/%s" % (mod), "output dir")
tf.flags.DEFINE_integer("threads", 64, "threads num")


def gen_tfrecords(in_file):
    basename = os.path.basename(in_file) + ".tfrecord"
    out_file = os.path.join(FLAGS.output_dir, basename)
    tfrecord_out = tf.python_io.TFRecordWriter(out_file)
    num = 0
    with open(in_file) as fin:
        line = fin.readline()
        while line:
            num += 1
            if num % 100000 == 0:
                print(num)
            (seq_len, label) = [int(_) for _ in line.split()]
            # 认为在最后一个数据才发生了转化
            for _ in range(seq_len - 1):
                tmpline = fin.readline().split()
                feature = {
                    "click_y": tf.train.Feature(float_list=tf.train.FloatList(value=[float(tmpline[1])])),
                    "conversion_y": tf.train.Feature(float_list=tf.train.FloatList(value=[float(0)]))
                }
                feature.update({
                    "features": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[int(t) for t in tmpline[2:]]))
                })
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                serialized = example.SerializeToString()
                tfrecord_out.write(serialized)

            tmpline = fin.readline().split()
            feature = {
                "click_y": tf.train.Feature(float_list=tf.train.FloatList(value=[float(tmpline[1])])),
                "conversion_y": tf.train.Feature(float_list=tf.train.FloatList(value=[float(label)]))
            }
            feature.update({
                "features": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(t) for t in tmpline[2:]]))
            })
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            serialized = example.SerializeToString()
            tfrecord_out.write(serialized)

            line = fin.readline()

        tfrecord_out.close()


def main(_):
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)
    file_list = glob.glob(os.path.join(FLAGS.input_dir, "*.txt"))
    print("total files: %d" % len(file_list))

    pool = ThreadPool(FLAGS.threads)  # Sets the pool size
    pool.map(gen_tfrecords, file_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
