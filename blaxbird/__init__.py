"""blaxbird: A high-level API for building and training Flax NNX models."""

__version__ = "0.0.1"

from blaxbird._src.checkpointer import get_default_checkpointer
from blaxbird._src.trainer import train_fn

__all__ = ["get_default_checkpointer", "train_fn"]
