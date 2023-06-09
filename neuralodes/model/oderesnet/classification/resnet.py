import equinox as eqx
import jax.random as jrandom
from jaxtyping import Array, Float

from .utils.modules import DownsamplingBlock, FCBlock, ResBlock


class ResBlocks(eqx.Module):
    layers: list
    def __init__(self, key, width = 64):
        feature_keys = jrandom.split(key, 6)
        self.layers = [ResBlock(width, width, feature_keys[i]) for i in range(6)]

    def __call__(self, x:Float[Array, "1 28 28"]):
        for layer in self.layers:
            x = layer(x)
        return x


class ResNet(eqx.Module):
    layers: list
    def __init__(self, key, width=64):
        key0, key1, key2 = jrandom.split(key, 3)

        self.layers = [DownsamplingBlock(key0, width)]
        self.layers.extend([ResBlocks(key1)])
        self.layers.extend([FCBlock(key2, width)])        

    def __call__(self, x:Float[Array, "1 28 28"]):
        for layer in self.layers:
            x = layer(x)
        return x
