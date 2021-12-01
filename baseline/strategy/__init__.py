import tensorflow as tf

from baseline.utils import get_logger

log = get_logger(__name__)


def create(conf):
    # https://github.com/google/automl/blob/2dbfb7984c20c24d856b96b54498799ed1b270cb/efficientdet/keras/eval.py#L56

    if conf.mode == "tpu":
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            conf.tpu_name
        )
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
        ds_strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
        if conf.mixed_precision == True:
            policy = tf.keras.mixed_precision.experimental.Policy("mixed_bfloat16")
            tf.keras.mixed_precision.experimental.set_policy(policy)

        # log.info("All devices: %s", tf.config.list_logical_devices("TPU"))
    elif conf.mode == "gpus":
        ds_strategy = tf.distribute.MirroredStrategy()

        if conf.mixed_precision == True:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)

        # log.info("All devices: %s", tf.config.list_physical_devices("GPU"))
    else:
        if tf.config.list_physical_devices("GPU"):
            ds_strategy = tf.distribute.OneDeviceStrategy("device:GPU:0")

            if conf.mixed_precision == True:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
        else:
            ds_strategy = tf.distribute.OneDeviceStrategy("device:CPU:0")

    return ds_strategy
