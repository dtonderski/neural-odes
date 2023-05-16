import equinox as eqx
import jax.random as jrandom
from jaxtyping import Array, Float

from ..utils.modules import DownsamplingBlock, FCBlock
from ..utils.ode_modules import ODEBlock, ODEBlockEulerWrapper


class ODENet(eqx.Module):
    layers: list
    def __init__(self, key, solver_name: str, width: int = 64):
        key0, key1, key2 = jrandom.split(key, 3)

        self.layers = [DownsamplingBlock(key0, width),
                       ODEBlock(key1, solver_name, width),
                       FCBlock(key2, width)]
        
    def __call__(self, x:Float[Array, "1 28 28"]):
        for layer in self.layers:
            x = layer(x)
        return x

class ODENetEulerWrapper(eqx.Module):
    layers: list
    
    def __init__(self, ode_net: ODENet, steps: int = 6):
        self.layers = [
            ode_net.layers[0],
            ODEBlockEulerWrapper(ode_net.layers[1], steps),
            ode_net.layers[2]
        ]

    def __call__(self, x:Float[Array, "1 28 28"]):
        for layer in self.layers:
            x = layer(x)
        return x