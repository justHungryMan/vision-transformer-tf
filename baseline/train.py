import copy
import datetime
import time
import os
import math
import scipy
import numpy as np
import tensorflow as tf

import baseline
from baseline.utils import get_logger

log = get_logger(__name__)


class Trainer:
    def __init__(self, conf):
        self.conf = copy.deepcopy(conf)
        self.debug = self.conf.base.debug
        self.strategy = baseline.strategy.create(self.conf["base"]["env"])

        self.initial_epoch = 0
        self.wandb_finish = None

    def build_dataset(self):
        dataset = baseline.dataset.create(
            conf_dataset=self.conf.dataset,
            conf_processing=self.conf.dataProcess,
            seed=self.conf.base.seed,
            num_devices=self.strategy.num_replicas_in_sync,
        )
        train_dataset = self.strategy.experimental_distribute_dataset(
            dataset["train"]["dataset"]
        )
        test_dataset = self.strategy.experimental_distribute_dataset(
            dataset["test"]["dataset"]
        )

        log.info(
            f'[train_dataset] #classes:{dataset["train"]["info"]["num_classes"]} #examples:{dataset["train"]["info"]["num_examples"]} #shards:{dataset["train"]["info"]["num_shards"]}'
        )

        return {
            "train": train_dataset,
            "test": test_dataset,
            "train_info": dataset["train"]["info"],
            "test_info": dataset["test"]["info"],
        }

    def build_model(self, steps_per_epoch, num_classes=1000):
        with self.strategy.scope():
            model = baseline.architecture.create(
                self.conf.architecture, num_classes=num_classes
            )
            model.build((None, None, None, 3))

            if self.conf.base.get("pretrained", False) is not False:
                self.load_pretrained(
                    model, self.conf.base.pretrained, self.conf.architecture
                )
            if self.conf.base.get("resume", False) is True:
                latest = tf.train.latest_checkpoint(self.conf.base.save_dir)
                assert (
                    latest is not None
                ), f"{self.conf.base.save_dir} has no checkpoint"
                model.load_weights(latest).expect_partial()
                self.initial_epoch = int(os.path.basename(latest).split("_")[-1])
                log.info(f"Training resumed from {self.initial_epoch} epochs")

            optimizer = self.build_optimizer(model=model)
            loss_fn = baseline.loss.create(self.conf["loss"])

            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=["accuracy"],
                steps_per_execution=1 if self.debug else steps_per_epoch,
            )
            if self.conf.base.get("resume", False) is False:
                tf.keras.backend.set_value(model.optimizer.iterations, 0)
            log.info(f"Model Completed. Iteration starts: {model.optimizer.iterations}")
        if self.debug:
            model.summary(print_fn=log.info)
        return model

    # https://github.com/google-research/vision_transformer/blob/39c905d2caf96a4306c9d78f05df36ddb3eb8ecb/vit_jax/checkpoint.py#L119
    def load_pretrained(self, model, pretrained_path, config):
        loaded_model = tf.keras.models.load_model(pretrained_path)
        for layer, loaded_layer in zip(model.layers, loaded_model.layers):
            layer_name = layer.get_config()["name"]
            if layer_name not in ["pre_logits", "head"]:
                layer.set_weights(loaded_layer.get_weights())

        posemb = loaded_model.pos_emb
        posemb_new = model.pos_emb
        if posemb_new.shape[1] != posemb.shape[1]:
            log.info(
                f"load_pretrained: resized variant: {posemb.shape} to {posemb_new.shape}"
            )
            ntok_new = posemb_new.shape[1]

            posemb_tok, posemb_grid = posemb[:, :1].numpy(), posemb[0, 1:].numpy()
            ntok_new -= 1

            gs_old = int(np.sqrt(len(posemb_grid)))
            gs_new = int(np.sqrt(ntok_new))
            log.info(f"load_pretrained: grid-size from {gs_old} to {gs_new}")
            posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

            zoom = (gs_new / gs_old, gs_new / gs_old, 1)
            posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
            posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
            embedding_weights = tf.convert_to_tensor(
                np.concatenate([posemb_tok, posemb_grid], axis=1)
            )
            model.pos_emb.assign(embedding_weights)
        else:
            embedding_weights = posemb
            model.pos_emb.assign(embedding_weights)

    def build_optimizer(self, model):
        lr_scheduler = baseline.scheduler.create(self.conf["scheduler"])
        params = {k: v for k, v in self.conf["optimizer"]["params"].items()}
        params["learning_rate"] = lr_scheduler
        opti_conf = {
            "type": self.conf["optimizer"]["type"],
            "params": params,
            "ema_decay": self.conf["optimizer"].get("ema_decay", False),
        }
        optimizer = baseline.optimizer.create(opti_conf, model=model)

        return optimizer

    def build_callback(self):
        for single_conf in self.conf.callbacks.modules:
            if single_conf["type"] == "ModelCheckpoint":
                single_conf["params"]["filepath"] = os.path.join(
                    self.conf.base.save_dir, "chpt_{epoch}"
                )
            elif single_conf["type"] == "TensorBoard":
                single_conf["params"][
                    "log_dir"
                ] = f"{os.path.join(self.conf.base.save_dir, 'logs/fit/')}" + datetime.datetime.now().strftime(
                    "%Y%m%d-%H%M%S"
                )
            elif single_conf["type"] == "Wandb":
                self.wandb_finish = True

        callbacks = baseline.callback.create(
            conf=self.conf.callbacks.modules, conf_all=self.conf
        )
        return callbacks

    def train_eval(self, train_dataset, model, callbacks, kwargs, val_kwargs={}):
        log.info(f"Model Fitting...")
        tic = time.time()
        history = model.fit(
            train_dataset,
            initial_epoch=self.initial_epoch,
            callbacks=callbacks,
            verbose=1,
            **kwargs,
            **val_kwargs,
        )
        toc = time.time()

        tf.keras.models.save_model(
            model,
            filepath=os.path.join(self.conf.base.save_dir, "last_checkpoint"),
            overwrite=True,
            include_optimizer=False,
            save_format="tf",
        )

        return {
            "dataset": self.conf.dataset.train.name,
            "history": {key: value for key, value in history.history.items()},
            "duration": toc - tic,
        }

    def run(self):
        tf.keras.backend.clear_session()

        if "seed" in self.conf.base:
            baseline.utils.set_seed(self.conf.base.seed)
            log.info(f"[Seed]: {self.conf.base.seed}")

        datasets = self.build_dataset()
        steps_per_epoch = max(
            1,
            datasets["train_info"]["num_examples"]
            // self.conf.dataset.train.batch_size,
        )
        epochs = self.conf.scheduler.steps // steps_per_epoch
        kwargs = {"steps_per_epoch": steps_per_epoch, "epochs": epochs}
        log.info(
            f"[train] #devices:{self.strategy.num_replicas_in_sync} batch_size:{self.conf.dataset.train.batch_size} steps_per_epoch:{steps_per_epoch} steps:{self.conf.scheduler.steps} epochs:{epochs}"
        )

        model = self.build_model(
            steps_per_epoch=steps_per_epoch,
            num_classes=datasets["train_info"]["num_classes"],
        )
        callbacks = self.build_callback()

        mode = self.conf.base.mode

        if mode == "train":
            rv = self.train_eval(
                train_dataset=datasets["train"],
                model=model,
                callbacks=callbacks,
                kwargs=kwargs,
            )
        elif mode == "train_eval":
            val_kwargs = {
                "validation_data": datasets["test"],
                "validation_steps": math.ceil(
                    datasets["test_info"]["num_examples"]
                    / self.conf.dataset.test.batch_size
                ),
            }
            rv = self.train_eval(
                train_dataset=datasets["train"],
                model=model,
                callbacks=callbacks,
                kwargs=kwargs,
                val_kwargs=val_kwargs,
            )
        elif mode == "eval":
            pass
        else:
            raise AttributeError(f"not support mode config: {mode}")

        log.info(f"{self.conf.base.project_name}: {self.conf.base.save_dir}")
        log.info(
            f'[train] loss:{rv["history"]["loss"][-1]:.4f} accuracy:{rv["history"]["accuracy"][-1]*100:.2f} duration:{rv["duration"] / 60:.2f}min'
        )
        log.info(
            f'[valid] loss:{rv["history"]["val_loss"][-1]:.4f} accuracy:{rv["history"]["val_accuracy"][-1]*100:.2f}'
        )
        log.info(
            f'[valid] best accuracy:{max(rv["history"]["val_accuracy"]) * 100:.2f}'
        )

        if self.wandb_finish:
            import wandb

            wandb.finish()

        # Return metric score for hyperparameter optimization
        optimized_metric = self.conf.get("optimized_metric")
        if optimized_metric:
            return rv["history"][optimized_metric][-1]
