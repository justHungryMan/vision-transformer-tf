import tensorflow as tf
import copy
from .mixing import mixing, cutmix_mask


def create(postprocess_list):
    def postprocess_image(conf, image, label):
        if conf["type"] == "mixing":
            features = {
                "image": image,
                "cutmix_mask": cutmix_mask(conf["params"]["cutmix_alpha"], 224, 224)
                if conf["params"]["cutmix_alpha"] > 0.0
                else None,
            }
            labels = {"label": label}
            features, labels = mixing(
                batch_size=image.shape[0],
                mixup_alpha=conf["params"]["mixup_alpha"],
                cutmix_alpha=conf["params"]["cutmix_alpha"],
                features=features,
                labels=labels,
            )
            return features["image"], labels["label"]

        else:
            raise AttributeError(f"not support dataset/preprocess config: {conf}")

    def _pp(data):
        image = data["image"]
        label = data["label"]

        for postprocess_conf in postprocess_list:
            image, label = postprocess_image(postprocess_conf, image, label)

        return {"image": image, "label": label}

    return _pp
