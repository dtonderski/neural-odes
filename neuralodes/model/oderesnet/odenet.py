import equinox as eqx
import jax.random as jrandom
from jaxtyping import Array, Float

from .utils.modules import DownsamplingBlock, FCBlock
from .utils.ode_modules import ODEBlock


class ODENet(eqx.Module):
    layers: list
    def __init__(self, key):
        key0, key1, key2 = jrandom.split(key, 3)

        self.layers = [DownsamplingBlock(key0),
                       ODEBlock(key1),
                       FCBlock(key2)]
        
    def __call__(self, x:Float[Array, "1 28 28"]):
        for layer in self.layers:
            x = layer(x)
        return x
