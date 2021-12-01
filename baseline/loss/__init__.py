import tensorflow as tf

from baseline.utils import get_logger
from .bce import BinaryCrossentropy

log = get_logger(__name__)


def create(config):
    if config["type"].lower() == "ce":
        log.info(f"[loss] create CategoricalCrossEntropy")

        return tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    elif config["type"].lower() == "bce":
        log.info(f"[loss] create BinaryCrossEntropy")

        return BinaryCrossentropy()
    else:
        raise AttributeError(f"not support loss config: {config}")
