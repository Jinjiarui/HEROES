import tensorflow as tf

a = tf.range(8)
l = tf.split(a, [4, 2, 2])
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.argsort(a)))
    print(sess.run(tf.gather_nd(a, tf.where(a > 4))))
    print(sess.run(tf.gather(a, [0, 1])))
    print(sess.run(tf.where((a > 1) & (a > 4), a, a - 11)))
