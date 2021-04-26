import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

K.set_learning_phase(1)
is_training = K.learning_phase()
print(is_training)
p = tf.convert_to_tensor(np.random.normal(size=100))
print(p)
m_a = tf.map_fn(lambda i: tf.reduce_mean(tf.map_fn(lambda j: j + 1, tf.range(i))), tf.range(10))
inputs = tf.range(10, dtype=tf.float32)
input_sum = tf.cumsum(inputs) / tf.reshape(tf.range(4, dtype=tf.float32) + 1, (-1, 1, 1))
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(inputs))
    print(sess.run(m_a))
