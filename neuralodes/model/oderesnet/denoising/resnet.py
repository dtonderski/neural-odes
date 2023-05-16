import equinox as eqx
import jax
import jax.random as jrandom
from jaxtyping import Array, Float

from ..utils.modules import ResBlock


class ResBlocks(eqx.Module):
    layers: list
    def __init__(self, key, width = 64):
        feature_keys = jrandom.split(key, 6)
        self.layers = [ResBlock(width, width, feature_keys[i]) for i in range(6)]

    def __call__(self, x:Float[Array, "1 H W"]):
        for layer in self.layers:
            x = layer(x)
        return x


class ResNet(eqx.Module):
    layers: list
    def __init__(self, key, width=64):
        key0, key1 = jrandom.split(key, 2)
        self.layers = [
            eqx.nn.Conv2d(1, width, kernel_size=3, stride=1, padding=1, key = key0),
            ResBlocks(key1, width),
            jax.nn.sigmoid
            ]

    def __call__(self, x:Float[Array, "1 H W"]):
        for layer in self.layers:
            x = layer(x)
        return x
