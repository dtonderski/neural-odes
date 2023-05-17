import equinox as eqx
import jax
import jax.random as jrandom

class ResBlock(eqx.Module):
    layers: list

    def __init__(self, key, width: int = 64):
        key0, key1 = jrandom.split(key, 2)
        self.layers = [
            eqx.nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, key = key0),
            jax.nn.relu,
            eqx.nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, key = key1),
        ]

    def __call__(self, x):
        shortcut = x
        for layer in self.layers:
            x = layer(x)
        return x + shortcut


class DenoisingFinalBlock(eqx.Module):
    layers: list
    
    def __init__(self, key, width: int = 64):
        self.layers = [
            eqx.nn.Conv2d(width, 1, kernel_size=3, stride=1, padding=1, key = key),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class DenoisingFirstBlock(eqx.Module):
    layers: list

    def __init__(self, key, width: int = 64):
        self.layers = [
            eqx.nn.Conv2d(1, width, kernel_size=3, stride=1, padding=1, key = key),
            jax.nn.relu
        ]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x