import glob
import os
from multiprocessing import Pool as ThreadPool

import numpy as np
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
LOG = tf.logging
mode = 'train'
tf.flags.DEFINE_string("input_dir", "./../alicpp/%s/remap_sample" % mode, "input dir")
tf.flags.DEFINE_string("output_dir", "./../alicpp/%s" % mode, "output dir")
tf.flags.DEFINE_integer("threads", 64, "threads num")

Common_Fileds = {'101': '638072', '121': '638073', '122': '638074', '124': '638075', '125': '638076', '126': '638077',
                 '127': '638078', '128': '638079', '129': '638080', '205': '638081', '301': '638082'}
UMH_Fileds = {'109_14': ('u_cat', '638083'), '110_14': ('u_shop', '638084'), '127_14': ('u_brand', '638085'),
              '150_14': ('u_int', '638086')}  # user multi-hot feature
Ad_Fileds = {'206': ('a_cat', '638087'), '207': ('a_shop', '638088'), '210': ('a_int', '638089'),
             '216': ('a_brand', '638090')}  # ad feature for DIN
X_Fileds = {'508': ('x_a', '638091'), '509': ('x_b', '638092'), '702': ('x_c', '638093'), '853': ('x_d', '638094')}


def gen_tfrecords(in_file):
    print(in_file)
    basename = os.path.basename(in_file) + ".tfrecord"
    out_file = os.path.join(FLAGS.output_dir, basename)
    tfrecord_out = tf.python_io.TFRecordWriter(out_file)
    num = 0
    with open(in_file) as fi:
        for line in fi:
            num += 1
            if num % 100000 == 0:
                print(num)
            fields = line.strip().split()
            y, z = fields[2], fields[3]
            y, z = [float(y)], [float(z)]
            feature = {
                "click_y": tf.train.Feature(float_list=tf.train.FloatList(value=y)),
                "conversion_y": tf.train.Feature(float_list=tf.train.FloatList(value=z))
            }
            splits = [_.split(':') for _ in fields[6:]]
            ffv = np.reshape(splits, (-1, 3))
            # 2 不需要特殊处理的特征
            feat_ids = np.array([])
            # feat_vals = np.array([])
            for f, def_id in Common_Fileds.items():
                if f in ffv[:, 0]:
                    mask = np.array(f == ffv[:, 0])
                    feat_ids = np.append(feat_ids, ffv[mask, 1])
                else:
                    feat_ids = np.append(feat_ids, def_id)
            feature.update(
                {"feat_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=feat_ids.astype(np.int)))})
            # "feat_vals": tf.train.Feature(float_list=tf.train.FloatList(value=feat_vals))})

            # 3 特殊字段单独处理
            for f, (fname, def_id) in UMH_Fileds.items():
                if f in ffv[:, 0]:
                    mask = np.array(f == ffv[:, 0])
                    feat_ids = ffv[mask, 1]
                    feat_vals = ffv[mask, 2]
                else:
                    feat_ids = np.array([def_id])
                    feat_vals = np.array([1.0])
                feature.update(
                    {fname + "ids": tf.train.Feature(int64_list=tf.train.Int64List(value=feat_ids.astype(np.int))),
                     fname + "vals": tf.train.Feature(
                         float_list=tf.train.FloatList(value=feat_vals.astype(np.float)))})

            for f, (fname, def_id) in Ad_Fileds.items():
                if f in ffv[:, 0]:
                    mask = np.array(f == ffv[:, 0])
                    feat_ids = ffv[mask, 1]
                else:
                    feat_ids = np.array([def_id])
                feature.update(
                    {fname + "ids": tf.train.Feature(int64_list=tf.train.Int64List(value=feat_ids.astype(np.int)))})
            # 4 特殊字段单独处理
            for f, (fname, def_id) in X_Fileds.items():
                if f in ffv[:, 0]:
                    mask = np.array(f == ffv[:, 0])
                    feat_ids = ffv[mask, 1]
                    feat_vals = ffv[mask, 2]
                else:
                    feat_ids = np.array([def_id])
                    feat_vals = np.array([1.0])
                feature.update(
                    {fname + "ids": tf.train.Feature(int64_list=tf.train.Int64List(value=feat_ids.astype(np.int))),
                     fname + "vals": tf.train.Feature(
                         float_list=tf.train.FloatList(value=feat_vals.astype(np.float)))})
            # serialized to Example
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            serialized = example.SerializeToString()
            tfrecord_out.write(serialized)
    tfrecord_out.close()


def main(_):
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)
    file_list = glob.glob(os.path.join(FLAGS.input_dir, "r*.txt"))
    print(file_list, )
    print("total files: %d" % (len(file_list)))

    pool = ThreadPool(FLAGS.threads)  # Sets the pool size
    pool.map(gen_tfrecords, file_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
