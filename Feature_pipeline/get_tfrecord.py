"""
tfrecord for <<Deep Interest Network for Click-Through Rate Prediction>> and <<Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate>>

by lambdaji
"""

import glob
import os
import re
from multiprocessing import Pool as ThreadPool

import numpy as np
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
LOG = tf.logging
mod = 'train'
tf.flags.DEFINE_string("input_dir", "./../alicpp/%s/remap_sample" % (mod), "input dir")
tf.flags.DEFINE_string("output_dir", "./../alicpp/%s" % (mod), "output dir")
tf.flags.DEFINE_integer("threads", 64, "threads num")

Common_Fileds = {'101': '1', '121': '2', '122': '3', '124': '4', '125': '5', '126': '6', '127': '7', '128': '8',
                 '129': '9', '205': '10', '301': '11'}
UMH_Fileds = {'109_14': ('u_cat', '12'), '110_14': ('u_shop', '13'), '127_14': ('u_brand', '14'),
              '150_14': ('u_int', '15')}  # user multi-hot feature
Ad_Fileds = {'206': ('a_cat', '16'), '207': ('a_shop', '17'), '210': ('a_int', '18'),
             '216': ('a_brand', '19')}  # ad feature for DIN
X_Fileds = {'508': ('x_a', '20'), '509': ('x_b', '21'), '702': ('x_c', '22'), '853': ('x_d', '23')}


# 40362692,0,0,216:9342395:1.0 301:9351665:1.0 205:7702673:1.0 206:8317829:1.0 207:8967741:1.0 508:9356012:2.30259 210:9059239:1.0 210:9042796:1.0 210:9076972:1.0 210:9103884:1.0 210:9063064:1.0 127_14:3529789:2.3979 127_14:3806412:2.70805
def gen_tfrecords(in_file):
    basename = os.path.basename(in_file) + ".tfrecord"
    out_file = os.path.join(FLAGS.output_dir, basename)
    tfrecord_out = tf.python_io.TFRecordWriter(out_file)
    with open(in_file) as fi:
        for line in fi:
            fields = line.strip().split(',')
            if len(fields) != 4:
                continue
            # 1 label
            y = [float(fields[2])]
            z = [float(fields[3])]
            feature = {
                "click_y": tf.train.Feature(float_list=tf.train.FloatList(value=y)),
                "conversion_y": tf.train.Feature(float_list=tf.train.FloatList(value=z))
            }

            splits = re.split('[ :]', fields[4])
            ffv = np.reshape(splits, (-1, 3))
            # common_mask = np.array([v in Common_Fileds for v in ffv[:,0]])
            # af_mask = np.array([v in Ad_Fileds for v in ffv[:,0]])
            # cf_mask = np.array([v in Context_Fileds for v in ffv[:,0]])

            # 2 不需要特殊处理的特征
            feat_ids = np.array([])
            # feat_vals = np.array([])
            for f, def_id in Common_Fileds.items():
                if f in ffv[:, 0]:
                    mask = np.array(f == ffv[:, 0])
                    feat_ids = np.append(feat_ids, ffv[mask, 1])
                    # np.append(feat_vals,ffv[mask,2].astype(np.float))
                else:
                    feat_ids = np.append(feat_ids, def_id)
                    # np.append(feat_vals,1.0)
            feature.update({"feat_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=feat_ids.astype(np.int)))})
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
                     fname + "vals": tf.train.Feature(float_list=tf.train.FloatList(value=feat_vals.astype(np.float)))})

            for f, (fname, def_id) in Ad_Fileds.items():
                if f in ffv[:, 0]:
                    mask = np.array(f == ffv[:, 0])
                    feat_ids = ffv[mask, 1]
                else:
                    feat_ids = np.array([def_id])
                feature.update(
                    {fname + "ids": tf.train.Feature(int64_list=tf.train.Int64List(value=feat_ids.astype(np.int)))})

            # serialized to Example
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            serialized = example.SerializeToString()
            tfrecord_out.write(serialized)
            # num_lines += 1
            # if num_lines % 10000 == 0:
            #    print("Process %d" % num_lines)
    tfrecord_out.close()


def main(_):
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)
    file_list = glob.glob(os.path.join(FLAGS.input_dir, "*-*"))
    print("total files: %d" % len(file_list))

    pool = ThreadPool(FLAGS.threads)  # Sets the pool size
    pool.map(gen_tfrecords, file_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
