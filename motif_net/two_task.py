from network import MotifNetwork
from train import train
import task_init

model_kwargs = {"seed": 123}
model = MotifNetwork(**model_kwargs)

default_params = {
    "lr": 1e-3,
    "l2_terms": (1e-6, 1e-6),
    "model": model,
    "batch_size": 64,
    "n_iter": 500000,
    "clip_grad": True,
    "max_norm": 10,
    "task_list": [task_init.DelayedAnti, task_init.DelayedPro],
    "task_kwargs": {"gamma": 0.2},
}

train(**default_params)
