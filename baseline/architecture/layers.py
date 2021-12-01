import tensorflow as tf


def drop_path(x, drop_path_rate, training):
    if drop_path_rate == 0.0 or not training:
        return x
    keep_prob = 1 - drop_path_rate
    batch_size = tf.shape(x)[0]
    random_tensor = keep_prob + tf.random.uniform([batch_size, 1, 1], dtype=x.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(x, keep_prob) * binary_tensor
    return output


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_path_rate=0.0):
        super(DropPath, self).__init__()

        self.drop_path_rate = drop_path_rate

    def call(self, x, training):
        return drop_path(x, self.drop_path_rate, training)


class Identity(tf.keras.layers.Layer):
    def __init__(self, name):
        super(Identity, self).__init__(name=name)

    def call(self, x):
        return x
