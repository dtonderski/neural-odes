from typing import Callable

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from .modules import ConcatConv2D, norm


class ODEFunc(eqx.Module):
    norm1: eqx.Module
    relu: Callable
    conv1: ConcatConv2D
    norm2: eqx.Module
    conv2: ConcatConv2D
    norm3: eqx.Module

    def __init__(self, dim, key):
        key1, key2 = jrandom.split(key, 2)
        self.norm1 = norm(dim)
        self.relu = jax.nn.relu
        self.conv1 = ConcatConv2D(dim, dim, key1, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2D(dim, dim, key2, 3, 1, 1)
        self.norm3 = norm(dim)

    def __call__(self, t, x, args):
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv1(t, x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv2(t, x)
        x = self.norm3(x)
        return x

class ODEBlock(eqx.Module):
    odefunc: ODEFunc
    integration_time: jnp.ndarray

    def __init__(self, key):
        self.odefunc = ODEFunc(64, key)
        self.integration_time = jnp.array([0, 1])

    def __call__(self, x):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.odefunc),
            diffrax.Tsit5(),
            t0 = self.integration_time[0],
            t1 = self.integration_time[1],
            dt0 = None,
            y0 = x,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=self.integration_time),
        )
        return solution.ys[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value