# Vision Transformer(ViT) in Tensorflow2

Tensorflow2 implementation of the Vision Transformer(ViT).

This repository is for `An image is worth 16x16 words: Transformers for image recognition at scale`  and  `How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`.

## Limitations.
- Due to memory limitations, only the ti/16, s/16, and b/16 models were tested. 
- Due to memory limitations, batch_size 2048 in s16 and 1024 in b/16 (in paper, 4096).
- Due to computational resource limitations, only reproduce using imagenet1k.

All experimental results and graphs are opend in Wandb.
- https://wandb.ai/justhungryman/vit

# Model weights

This is personal project. Since it is difficult to open the model weights of all experiments, I only open the weights with the best performance for each model. But you can check the all results of experiment in `WIP`


# Install dependencies
```
pip install -r requirements
```

All experiments were done on tpu_v3-8 with the support of TRC. But you can experiment on GPU. Check `conf/config.yaml` and `conf/downstream.yaml` 

```
  # TPU options
  env:
    mode: tpu
    gcp_project: {your_project}
    tpu_name: node-1
    tpu_zone: europe-west4-a
    mixed_precision: True
  # GPU options
  # env:
  #   mode: gpu
  #   mixed_precision: True
```

# Train from scratch
```
[WIP]
```

# Downstream
```
[WIP]
```

# Board

To track metics, you can use wandb or tensorboard (default: wandb).
You can change in `conf/callbacks/{filename.yaml}`.
```
modules:
  - type: MonitorCallback
  - type: TerminateOnNaN
  - type: ProgbarLogger
    params:
      count_mode: steps
  - type: ModelCheckpoint
    params:
      filepath: ???
      save_weights_only: True
  - type: Wandb
    project: vit
    nested_dict: False
    hide_config: True
    params: 
      monitor: val_loss
      save_model: False
  # - type: TensorBoard
  #   params:
  #     log_dir: ???
  #     histogram_freq: 1
```


# TFC

This open source was assisted by TPU Research Cloud ([TRC](https://sites.research.google/trc/about/)) program  

Thank you for providing the TPU.

# Citations
```
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

```
@article{steiner2021train,
  title={How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers},
  author={Steiner, Andreas and Kolesnikov, Alexander and Zhai, Xiaohua and Wightman, Ross and Uszkoreit, Jakob and Beyer, Lucas},
  journal={arXiv preprint arXiv:2106.10270},
  year={2021}
}
```