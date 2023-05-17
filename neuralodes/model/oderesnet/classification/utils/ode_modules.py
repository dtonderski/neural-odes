from typing import Callable

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from diffrax.solver.base import AbstractSolver

from .modules import norm

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
    solver: AbstractSolver

    def __init__(self, key, solver_name: str = 'Tsit5', width = 64):
        self.odefunc = ODEFunc(width, key)
        self.solver = get_solver(solver_name)
        self.integration_time = jnp.array([0, 1])

    def __call__(self, x):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.odefunc),
            self.solver,
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

def get_solver(solver_name: str = 'Tsit5') -> AbstractSolver:
    match solver_name.lower():
        case 'euler':
            return diffrax.Euler()
        case 'tsit5':
            return diffrax.Tsit5()
        case 'heun':
            return diffrax.Heun()
        case 'midpoint':
            return diffrax.Midpoint()
        case 'ralston':
            return diffrax.Ralston()
        case 'bosh3':
            return diffrax.Bosh3()
        case 'dopri5':
            return diffrax.Dopri5()
        case 'dopri8':
            return diffrax.Dopri8()
        # case 'impliciteuler':
        #     return diffrax.ImplicitEuler()
        # case 'kvaerno3':
        #     return diffrax.Kvaerno3()
        # case 'kvaerno4':
        #     return diffrax.Kvaerno4()
        # case 'kvaerno5':
        #     return diffrax.Kvaerno4()

class ODEBlockEulerWrapper(eqx.Module):
    ode_block: ODEBlock
    dt: float
    """ Used so that we can use trained ODEBlocks with Euler solver.
    """
    def __init__(self, ode_block: ODEBlock, steps: int):
        self.ode_block = ode_block
        self.dt = (ode_block.integration_time[1] - ode_block.integration_time[0]) / steps
    
    def __call__(self, x):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.ode_block.odefunc),
            diffrax.Euler(),
            t0 = self.ode_block.integration_time[0],
            t1 = self.ode_block.integration_time[1],
            dt0 = self.dt,
            y0 = x,
            saveat=diffrax.SaveAt(ts=self.ode_block.integration_time),
        )
        return solution.ys[1]