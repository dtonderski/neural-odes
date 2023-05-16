import equinox as eqx
import jax.random as jrandom
from jaxtyping import Array, Float
import jax

from ..utils.modules import DownsamplingBlock, FCBlock
from ..utils.ode_modules import ODEBlock, ODEBlockEulerWrapper


class ODENet(eqx.Module):
    layers: list
    def __init__(self, key, solver_name: str, width: int = 64):
        key0, key1 = jrandom.split(key, 2)
        self.layers = [
            eqx.nn.Conv2d(1, width, kernel_size=3, stride=1, padding=1, key = key0),
            ODEBlock(key1, solver_name, width),
            eqx.nn.Conv2d(width, 1, kernel_size=3, stride=1, padding=1, key = key0),
            jax.nn.sigmoid
            ]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ODENetEulerWrapper(eqx.Module):
    layers: list
    
    def __init__(self, ode_net: ODENet, steps: int = 6):
        self.layers = [
            ode_net.layers[0],
            ODEBlockEulerWrapper(ode_net.layers[1], steps),
            jax.nn.sigmoid
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x