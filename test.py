import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

K.set_learning_phase(1)
is_training = K.learning_phase()
print(is_training)
input = tf.convert_to_tensor(np.random.normal(size=(3, 3)), dtype=tf.float32)
output = tf.keras.layers.BatchNormalization()
dropout = tf.keras.layers.Dropout(rate=0.5)
a1 = output(input)
b1 = dropout(a1)
K.set_learning_phase(0)
a2 = output(input)
b2 = dropout(a2)
ops = tf.get_default_graph().get_operations()
bn_update_ops = [x for x in ops if ("AssignMovingAvg" in x.name and x.type == "AssignSubVariableOp")]
tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, bn_update_ops)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
print(update_ops)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(input))
    print(sess.run([a1, b1]))
    print(sess.run([a2, b2]))
