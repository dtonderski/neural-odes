import equinox as eqx
import jax
import jax.random as jrandom
from jaxtyping import Array, Float

from .utils.modules import ResBlock, ClassificationFirstBlock, ClassificationFinalBlock

class ResBlocks(eqx.Module):
    layers: list
    def __init__(self, key, width = 64):
        feature_keys = jrandom.split(key, 6)
        self.layers = [ResBlock(feature_keys[i], width) for i in range(6)]

    def __call__(self, x:Float[Array, "1 H W"]):
        for layer in self.layers:
            x = layer(x)
        return x


class ResNet(eqx.Module):
    layers: list
    def __init__(self, key, width=64):
        key0, key1, key2 = jrandom.split(key, 3)
        self.layers = [
            ClassificationFirstBlock(key0, width),
            ResBlocks(key1, width),
            ClassificationFinalBlock(key2, width)
            ]

    def __call__(self, x:Float[Array, "1 H W"]):
        for layer in self.layers:
            x = layer(x)
        return x
