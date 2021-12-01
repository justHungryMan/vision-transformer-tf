import tensorflow as tf
from omegaconf import DictConfig, ListConfig

from baseline.utils import get_logger
from .MonitorCallback import MonitorCallback

log = get_logger(__name__)


def create(conf, conf_all):
    def create_callback(conf_callback):
        if conf_callback["type"] == "MonitorCallback":
            return MonitorCallback()
        elif conf_callback["type"] == "TerminateOnNaN":
            return tf.keras.callbacks.TerminateOnNaN()
        elif conf_callback["type"] == "ProgbarLogger":
            return tf.keras.callbacks.ProgbarLogger(**conf_callback["params"])
        elif conf_callback["type"] == "ModelCheckpoint":
            return tf.keras.callbacks.ModelCheckpoint(**conf_callback["params"])
        elif conf_callback["type"] == "TensorBoard":
            return tf.keras.callbacks.TensorBoard(**conf_callback["params"])
        elif conf_callback["type"] == "Wandb":
            import wandb
            from wandb.keras import WandbCallback

            if conf_callback["nested_dict"] is True:
                configs_all = replace_nested_dict(source=conf_all)
            else:
                configs_all = conf_all
            wandb.init(
                project=conf_callback["project"],
                config=None if conf_callback["hide_config"] else configs_all,
            )
            wandb.run.name = conf_all.base.project_name
            return WandbCallback(**conf_callback["params"])
        else:
            raise AttributeError(f"not support callback config: {conf_callback}")

    callbacks = []

    for single_conf in conf:
        callbacks.append(create_callback(single_conf))
    return callbacks


def replace_nested_dict(source, unnested_dict={}, unnested_keys=""):
    for idx, key in enumerate(source):
        if not isinstance(key, str):
            key = idx
        if isinstance(source[key], DictConfig):
            unnested_dict = replace_nested_dict(
                source[key],
                unnested_dict=unnested_dict,
                unnested_keys=f"{unnested_keys}_{key}",
            )
        elif isinstance(source[key], ListConfig):
            unnested_dict = replace_nested_dict(
                source[key],
                unnested_dict=unnested_dict,
                unnested_keys=f"{unnested_keys}_{key}",
            )
        else:
            unnested_dict[f"{unnested_keys}_{key}"] = source[key]
    return unnested_dict
