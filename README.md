# Vision Transformer(ViT) in Tensorflow2

Tensorflow2 implementation of the Vision Transformer(ViT).

This repository is for `An image is worth 16x16 words: Transformers for image recognition at scale`  and  `How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`.

## Limitations.
- Due to memory limitations, only the ti/16, s/16, and b/16 models were tested. 
- Due to memory limitations, batch_size 2048 in s16 and 1024 in b/16 (in paper, 4096).
- Due to computational resource limitations, only reproduce using imagenet1k.

All experimental results and graphs are opend in Wandb.
- https://docs.google.com/spreadsheets/d/1j0lFlaMuqccFiHj3eQVpZYIbSoXY6Pz6oEW76x7g25M/edit?usp=sharing
- upstream: https://wandb.ai/justhungryman/vit
- downstream: https://wandb.ai/justhungryman/vit-downstream/
- In case of an experiment in which the tpu is stopped, it is resumed (duplicated experiment name but different start epoch).

# Model weights

Since this is personal project, it is hard to train with large datasets like imagenet21k. For a pretrain model with good performance, see the [official repo](https://github.com/google-research/vision_transformer). But if you really need it, contact to me.


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
# example
python run.py experiment=vit-s16-aug_light1-bs_2048-wd_0.1-do_0.1-dp_0.1-lr_1e-3 \
base.project_name=vit-s16-aug_light1-bs_2048-wd_0.1-do_0.1-dp_0.1-lr_1e-3 \
base.save_dir={your_save_dir} base.env.gcp_project={your_gcp_project} \
base.env.tpu_name={your_tpu_name} base.debug=False
```

# Downstream
```
# example
python run.py --config-name=downstream experiment=downstream-imagenet-ti16_384 \
base.pretrained={your_checkpoint} base.project_name={your_project_name} \
base.save_dir={your_save_dir} base.env.gcp_project={your_gcp_project} \
base.env.tpu_name={your_tpu_name} base.debug=False
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

This open source was supported by TPU Research Cloud ([TRC](https://sites.research.google/trc/about/)) program  

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
