import tensorflow as tf


def read_from_tfrecord(filenames):
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    tfrecord_features = tf.parse_single_example(tfrecord_serialized, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'shape': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
    }, name='features')

    image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
    shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)

    image = tf.reshape(image, shape)
    label = tfrecord_features['label']
    return label, shape, image

