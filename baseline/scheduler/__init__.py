import tensorflow as tf

from baseline.utils import get_logger
from . import warmup_scheduler

log = get_logger(__name__)


def create(config):
    sched_type = config["type"].lower()

    if sched_type == "warmup_piecewise":
        scheduler = warmup_scheduler.WarmupPiecewiseConstantDecay(**config["params"])
    elif sched_type == "warmup_cosine":
        scheduler = warmup_scheduler.WarmupCosineDecay(**config["params"])
    else:
        raise AttributeError(f"not support scheduler config: {config}")
    return scheduler
