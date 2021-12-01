import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import os
import json
import logging


from .preprocess import preprocess
from .postprocess import postprocess

LOGGER = logging.getLogger(__name__)

# Adjust depending on the available RAM.
MAX_IN_MEMORY = 200_000


def create(config, data_dir=None, seed=None, num_devices=1):
    logging.info(f'Data Directory: {os.path.join(data_dir, config["name"])}')
    builder = (
        tfds.builder(config["name"])
        if data_dir is None
        else tfds.builder(config["name"], data_dir=data_dir, try_gcs=False)
    )
    builder.download_and_prepare(
        # download_config=tfds.download.DownloadConfig(manula_dir='~/tensorflow_datasets/')
    )
    info = {
        "num_examples": builder.info.splits[config["split"]].num_examples,
        "num_shards": len(builder.info.splits[config["split"]].file_instructions),
        "num_classes": builder.info.features["label"].num_classes,
    }

    dataset = builder.as_dataset(
        split=config["split"],
        shuffle_files=config.get("shuffle", False),
        decoders={"image": tfds.decode.SkipDecoding()},
    )
    decoder = builder.info.features["image"].decode_example

    # dataset = dataset.cache()

    if config.get("repeat", False):
        dataset = dataset.repeat()

    if config.get("shuffle", False):
        dataset = dataset.shuffle(min(info["num_examples"], MAX_IN_MEMORY))

    # Preprocessing
    dataset = dataset.map(
        preprocess.create(config["preprocess"], info, decoder),
        tf.data.experimental.AUTOTUNE,
    )

    dataset = dataset.batch(
        config["batch_size"], drop_remainder=config["drop_remainder"]
    )

    # mixing
    if config.get("postprocess", None) is not None:
        dataset = dataset.map(
            postprocess.create(config["postprocess"]), tf.data.experimental.AUTOTUNE
        )

    # shape
    dataset = dataset.map(
        lambda v: (v["image"], v["label"]), tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # from deepmind's code : https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L91
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_threading.private_threadpool_size = 48
    options.experimental_threading.max_intra_op_parallelism = 1
    policy = (
        tf.data.experimental.AutoShardPolicy.FILE
        if info["num_shards"] > num_devices
        else tf.data.experimental.AutoShardPolicy.DATA
    )
    options.experimental_distribute.auto_shard_policy = policy
    dataset = dataset.with_options(options)

    return {"dataset": dataset, "info": info}
