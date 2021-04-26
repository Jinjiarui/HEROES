import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

l = tf.convert_to_tensor(np.random.normal(size=(100, 10, 10)),dtype=tf.float32)
gru = tf.keras.layers.GRU(units=4,time_major=True,return_sequences=True)

print(l)
print(gru(l))
