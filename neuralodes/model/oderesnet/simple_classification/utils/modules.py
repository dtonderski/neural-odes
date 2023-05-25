import equinox as eqx
import jax
import jax.random as jrandom
import jax.numpy as jnp

def group_norm(dim):
    return eqx.nn.GroupNorm(groups=min(32, dim), channels=dim)

class ResBlock(eqx.Module):
    """ Here, we start with a ReLU, as the odefunc cannot do it at the end."""
    layers: list

    def __init__(self, key, width: int = 64):
        key0, key1 = jrandom.split(key, 2)
        self.layers = [
            jax.nn.relu,
            eqx.nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, key = key0),
            group_norm(width),
            jax.nn.relu,
            eqx.nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, key = key1),
            group_norm(width),
        ]

    def __call__(self, x):
        shortcut = x
        for layer in self.layers:
            x = layer(x)
        return x + shortcut


class ClassificationFinalBlock(eqx.Module):
    layers: list
    
    def __init__(self, key, width: int = 64):
        self.layers = [
            jax.nn.relu,
            eqx.nn.AdaptiveAvgPool2d((1,1)),
            jnp.ravel,
            eqx.nn.Linear(width,10, key=key),
            jax.nn.log_softmax
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ClassificationFirstBlock(eqx.Module):
    layers: list

    def __init__(self, key, width: int = 64):
        self.layers = [
            eqx.nn.Conv2d(1, width, kernel_size=3, stride=1, padding=1, key = key),
            group_norm(width),
        ]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x