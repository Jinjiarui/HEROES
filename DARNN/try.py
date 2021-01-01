from load_criteo import loaddualattention
import tensorflow as tf

l = open('./../Criteo/test/test_usr.yzx.txt')
total_data, total_click_label, total_label, total_seqlen = loaddualattention(
    30, 20, 12, l)
sess = tf.Session()
a = tf.convert_to_tensor([[[0.6, 0.4], [0.3, 0.7], [0.6, 0.4]]])
b = tf.convert_to_tensor([[[1, 0], [0, 1], [1, 0]]])
h = tf.metrics.auc(b, a)
sess.run(tf.local_variables_initializer())
print(sess.run(h)[0])
