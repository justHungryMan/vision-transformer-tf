import tensorflow as tf
import copy
from .inception import center_crop, distort_crop, distort_color, vit_inception_crop
from .autoaugment import distort_image_with_autoaugment, distort_image_with_randaugment


def create(preprocess_list, info, decoder):
    def preprocess_image(conf, image, label):
        if conf["type"] == "resize":
            config = {"method": conf["params"]["method"]}
            config["size"] = (conf["params"]["size"][0], conf["params"]["size"][1])
            return tf.image.resize(image, **config), label
        elif conf["type"] == "random_crop":
            config = {
                "size": (
                    conf["params"]["size"][0],
                    conf["params"]["size"][1],
                    conf["params"]["size"][2],
                )
            }
            return tf.image.random_crop(image, **config), label
        elif conf["type"] == "resize_smaller_aspect_ratio":
            smaller_size = conf["params"]["size"]

            shape = tf.shape(image)
            height, width = shape[-3], shape[-2]
            if height > width:
                ratio = height / width
                resize_shape = (int(ratio * smaller_size), smaller_size)
            else:
                ratio = width / height
                resize_shape = (smaller_size, int(ratio * smaller_size))
            config = {"method": conf["params"]["method"]}
            config["size"] = resize_shape

            return tf.image.resize(image, **config), label
        elif conf["type"] == "random_flip":
            return tf.image.random_flip_left_right(image), label
        elif conf["type"] == "normalize":
            print("norm", (image - conf["params"]["mean"]) / conf["params"]["std"])
            return (image - conf["params"]["mean"]) / conf["params"]["std"], label
        elif conf["type"] == "imagenet_normalize":
            MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
            STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]
            return (image - MEAN_RGB) / STDDEV_RGB, label
        elif conf["type"] in {"labelsmooth", "label_smooth", "label_smoothing"}:
            return image, label * (1.0 - conf["params"]["epsilon"]) + (
                conf["params"]["epsilon"] / info["num_classes"]
            )
        elif conf["type"] in "sigmoid_label_smoothing":
            return image, label * (1.0 - conf["params"]["epsilon"]) + (
                conf["params"]["epsilon"]
            )
        elif conf["type"] in {"inception_random_crop"}:
            return distort_crop(image, **conf["params"]), label
        elif conf["type"] in {"inception_center_crop"}:
            return center_crop(image, **conf["params"]), label
        elif conf["type"] in {"inception_distort_color"}:
            return distort_color(image, **conf.get("params", {})), label
        elif conf["type"] in {"vit_inception_crop"}:
            return vit_inception_crop(image, **conf.get("params", {})), label
        elif conf["type"] in {"cast"}:
            dtype = tf.bfloat16 if conf["params"]["type"] == "bfloat16" else None
            dtype = tf.float32 if conf["params"]["type"] == "float32" else dtype
            dtype = tf.float16 if conf["params"]["type"] == "float16" else dtype
            dtype = tf.uint8 if conf["params"]["type"] == "uint8" else dtype
            dtype = tf.int32 if conf["params"]["type"] == "int32" else dtype
            if dtype is None:
                raise AttributeError(f"not support cast type: {conf}")
            return tf.image.convert_image_dtype(image, dtype=dtype), label
        elif conf["type"] in {"rand_augment"}:
            input_image_type = image.dtype
            image = tf.clip_by_value(image, 0.0, 255.0)
            image = tf.cast(image, dtype=tf.uint8)
            image = distort_image_with_randaugment(image, **conf.get("params", {}))
            image = tf.cast(image, dtype=input_image_type)
            return image, label

        else:
            raise AttributeError(f"not support dataset/preprocess config: {conf}")

    def _pp(data):
        data["image"] = decoder(data["image"])
        image = data["image"]

        label = data["label"]

        label = tf.one_hot(tf.reshape(label, [-1]), info["num_classes"])
        label = tf.reduce_sum(label, axis=0)

        label = tf.clip_by_value(label, 0, 1)

        for preprocess_conf in preprocess_list:
            image, label = preprocess_image(preprocess_conf, image, label)

        return {"image": image, "label": label}

    return _pp
