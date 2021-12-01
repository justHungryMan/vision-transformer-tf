import tensorflow as tf
from tensorflow.keras.losses import Loss


class BinaryCrossentropy(Loss):
    def __init__(
        self, reduction=tf.keras.losses.Reduction.AUTO, name="sigmoid_cross_entropy"
    ):
        super().__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, logits):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=y_true
        )

        loss_sum = tf.math.reduce_sum(cross_entropy, axis=-1)

        return tf.math.reduce_mean(loss_sum)
