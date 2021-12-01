import tensorflow as tf

from baseline.utils import get_logger

log = get_logger(__name__)


class WarmUpExtension:
    def __init__(self, warmup_steps, init_lr=0.0, *args, **kwargs):
        warmup_steps = kwargs.pop("warmup_steps", warmup_steps)
        init_lr = kwargs.pop("init_lr", init_lr)
        super().__init__(*args, **kwargs)
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr

    @tf.function
    def __call__(self, step):
        lr = super().__call__(step)
        step = tf.cast(step, tf.float32)
        if step < self.warmup_steps:
            lr = ((lr - self.init_lr) * step / self.warmup_steps) + self.init_lr

        return lr

    def get_config(self):
        config = super().get_config()
        config.update({"warmup_steps": self.warmup_steps, "init_lr": self.init_lr})
        return config
