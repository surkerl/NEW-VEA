import os
import random
from functools import partial
from typing import Callable

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, use_deterministic_algorithms: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

    if use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)


def build_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_worker_init_fn(seed: int) -> Callable[[int], None]:
    return partial(seed_worker, base_seed=seed)
