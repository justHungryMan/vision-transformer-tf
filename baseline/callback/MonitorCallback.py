import tensorflow as tf


class MonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, batch, logs=None):
        if callable(self.model.optimizer.learning_rate):
            lr = self.model.optimizer.learning_rate(self.model.optimizer.iterations)
        else:
            lr = self.model.optimizer.learning_rate
        logs.update({"lr": lr})
        logs.update({"iterations": self.model.optimizer.iterations})

    def on_train_batch_end(self, batch, logs=None):
        if callable(self.model.optimizer.learning_rate):
            lr = self.model.optimizer.learning_rate(self.model.optimizer.iterations)
        else:
            lr = self.model.optimizer.learning_rate
        logs.update({"lr": lr})
        logs.update({"iterations": self.model.optimizer.iterations})
