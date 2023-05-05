from typing import Callable

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float


class ConcatConv2D(eqx.Module):
    layer: eqx.Module

    def __init__(self, dim_in, dim_out, key, ksize=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, transpose=False):
        module = eqx.nn.ConvTranspose2d if transpose else eqx.nn.Conv2d
        self.layer = module(dim_in+1, dim_out, ksize, stride, padding, 
                            dilation, groups, bias, key=key)

    def __call__(self, t, x):
        tt = jnp.ones_like(x[:1,:,:]) * t
        ttx = jnp.concatenate([tt, x], axis=0)
        return self.layer(ttx)
        
def norm(dim):
    return eqx.nn.GroupNorm(groups=min(32, dim), channels=dim)

def conv3x3(in_planes, out_planes, key, stride=1):
    return eqx.nn.Conv2d(in_planes, out_planes, kernel_size = 3, padding = 1, stride=stride, key = key)

def conv1x1(in_planes, out_planes, key, stride=1):
    return eqx.nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride=stride, key = key)

class DownsamplingBlock(eqx.Module):
    layers: list
    def __init__(self, key):
        key0, key1, key2, key3, key4 = jrandom.split(key, 5)

        self.layers = [
            eqx.nn.Conv2d(1, 64, 3, 1, key = key0),
            ResBlock(64, 64, stride=2, downsample = conv1x1(64,64, key1, stride = 2), key=key2),
            ResBlock(64, 64, stride=2, downsample = conv1x1(64,64, key3, stride = 2), key=key4),
        ]

    def __call__(self, x:Float[Array, "1 28 28"]):
        for layer in self.layers:
            x = layer(x)
        return x


class FCBlock(eqx.Module):
    layers: list
    def __init__(self, key):
        self.layers = [
            norm(64),
            jax.nn.relu,
            eqx.nn.AdaptiveAvgPool2d((1,1)),
            jnp.ravel,
            eqx.nn.Linear(64,10, key=key),
            jax.nn.log_softmax
        ]
    def __call__(self, x:Float[Array, "1 28 28"]):
        for layer in self.layers:
            x = layer(x)
        return x

class ResBlock(eqx.Module):
    layers: list
    downsample: eqx.Module

    def __init__(self, inplanes, planes, key, stride=1, downsample=None):
        key1, key2 = jrandom.split(key, 2)
        self.downsample = downsample
        self.layers = [
            norm(inplanes),
            jax.nn.relu,
            conv3x3(inplanes, planes, key1, stride),
            norm(inplanes),
            jax.nn.relu,
            conv3x3(inplanes, planes, key2)
        ]

    def __call__(self, x:Float[Array, "1 28 28"]):
        shortcut = x
        shortcut = self.downsample(shortcut) if self.downsample is not None else shortcut
        for layer in self.layers:
            x = layer(x) if layer is not None else x
        return x + shortcut
