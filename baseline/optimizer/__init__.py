import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_addons.utils import types
from typeguard import typechecked

from baseline.utils import get_logger
from . import adamw_huggingface

log = get_logger(__name__)


def create(config, model=None):
    opt_type = config["type"].lower()

    if opt_type == "sgd":
        optimizer = tf.keras.optimizers.SGD(**config["params"])
    elif opt_type == "sgdw":
        optimizer = tfa.optimizers.SGDW(**config["params"])
    elif opt_type == "adam":
        optimizer = tf.keras.optimizers.Adam(**config["params"])
    elif opt_type == "adamw":
        optimizer = tfa.optimizers.AdamW(**config["params"])
    elif opt_type == "adamw_huggingface":
        optimizer = adamw_huggingface.AdamWeightDecay(
            **config["params"],
        )
    elif opt_type == "lamb":
        optimizer = tfa.optimizers.LAMB(**config["params"])
    else:
        raise AttributeError(f"not support optimizer config: {config}")

    log.info(f"[optimizer] create {opt_type}")

    if config["ema_decay"] is not False:
        optimizer = tfa.optimizers.MovingAverage(
            optimizer, average_decay=config["ema_decay"]
        )
        log.info(f"[optimizer] use average_decay: {config['ema_decay']}")

    return optimizer
