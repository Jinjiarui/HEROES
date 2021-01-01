import tensorflow as tf
import glob
import os
from collections import defaultdict
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
tr_files = glob.glob("Criteo/t*/*.txt")
print("train_files:", tr_files)
filename_queue = tf.train.string_input_producer(tr_files,
                                                num_epochs=1)
seq_len = [2, 4]
seq_max_len = 6
a = np.random.randn(2, 6, 4)
mask = tf.sequence_mask(seq_len, seq_max_len)
b_tensor = tf.zeros(shape=(1, 2, 3))
c_tensor = tf.constant([0, 1, 2, 3, 4], dtype=tf.float32)
b = tf.boolean_mask(a, mask)
c_softmax = tf.nn.softmax(b[0][0:1])
print(b[0][0])
h = 0
ear = tf.map_fn(lambda x: tf.nn.softmax(b[x][0:2]), tf.range(0, tf.shape(b_tensor)[2]), dtype=b.dtype)
sss = tf.stack([[[1], [2]], [[2], [3]]], axis=0)
ss_reshape = tf.reshape(sss, shape=(-1,))
with tf.Session() as sess:
    print(a)
    h2 = sess.run(b)
    print(h2)
    print(sess.run(mask))
    print(sess.run(ear))
    print(sess.run(c_softmax))
    print(sess.run(tf.reduce_sum(ear, axis=0)))
    print(sess.run(sss))
    print(sess.run(ss_reshape))
