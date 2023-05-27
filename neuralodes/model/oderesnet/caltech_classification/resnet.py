import equinox as eqx
import jax.random as jrandom
from jaxtyping import Array, Float

from .utils.modules import DownsamplingBlock, FCBlock, ResBlock


class ResBlocks(eqx.Module):
    layers: list
    def __init__(self, key, width = 64, blocks=6):
        feature_keys = jrandom.split(key, blocks)
        self.layers = [ResBlock(feature_keys[i], width) for i in range(blocks)]

    def __call__(self, x:Float[Array, "1 H W"]):
        for layer in self.layers:
            x = layer(x)
        return x


class ResNet(eqx.Module):
    layers: list
    def __init__(self, key, width=64, blocks=6):
        key0, key1, key2 = jrandom.split(key, 3)
        self.layers = [
            DownsamplingBlock(key0, width),
            ResBlocks(key1, width, blocks),
            FCBlock(key2, width)
            ]

    def __call__(self, x:Float[Array, "1 H W"]):
        for layer in self.layers:
            x = layer(x)
        return x
