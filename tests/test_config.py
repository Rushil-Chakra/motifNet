from omegaconf import OmegaConf

from hydra import initialize, compose
import pytest


def test_config_loads():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="default", overrides=["experiments=ReactAnti"])


def test_config_local():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="default", overrides=["experiments=ReactAnti", "hydra/launcher=local"]
        )
