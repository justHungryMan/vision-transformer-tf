import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config.yaml")
def main(conf: DictConfig) -> None:

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from baseline import utils
    from baseline import Trainer

    conf = utils.set_environment(conf)

    if conf.base.get("print_config", False):
        utils.print_config(conf, resolve=True)

    trainer = Trainer(conf)
    return trainer.run()


if __name__ == "__main__":
    main()
