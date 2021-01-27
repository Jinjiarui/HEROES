import glob
import os
import pickle

import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = ''
tr_files = glob.glob("Criteo/t*/*.txt")
print("train_files:", tr_files)
filename_queue = tf.train.string_input_producer(tr_files,
                                                num_epochs=1)
seq_len = [2, 4]
seq_max_len = 6
a = np.random.randn(2, 6, 4)
mask = tf.sequence_mask(seq_len, seq_max_len)
b_tensor = tf.zeros(shape=(3, 2, 3))
b_test = tf.ones(shape=(1, 2, 1))
c_gate = tf.ones(shape=(3, 2, 128))
c_tensor = tf.constant([[0.3, 0.4, 0.3],
                        [0.66, 0.14, 0.2]], dtype=tf.float32)
H_c_p = tf.concat([b_tensor, tf.tile(b_test, [3, 1, 1])], axis=-1)
c_copy = tf.tile(c_tensor, [3, 1])
with tf.Session() as sess:
    print(a)
    print(sess.run(H_c_p))
    print(sess.run(c_copy))

x = np.array([0, 0, 0, 0, 0, 0, 1])
z = np.argsort(-x)
print(z)
print(x[z])
