import tensorflow as tf
from tensorflow.keras import backend as K

a = tf.ones((1, 13))
l = tf.keras.layers.Dense(units=10)
p = l(a)
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(p))
    print(sess.run(l.weights[0]).shape)
