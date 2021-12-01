import logging

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50

from baseline.utils import get_logger
from . import vit

log = get_logger(__name__)


def create(conf, num_classes=1000):
    base, architecture_name = [l.lower() for l in conf["type"].split("/")]

    if base == "resnet":
        architecture = Sequential()
        architecture.add(ResNet50(**conf["params"]))
        architecture.add(Dense(num_classes, activation="softmax"))
    elif base == "vit":
        architecture = vit.create_name(
            architecture_name, num_classes=num_classes, **conf["params"]
        )
    else:
        raise AttributeError(f"not support architecture config: {conf}")

    log.info(f"[Model] create {architecture_name}")
    return architecture
