from typing import Callable, List

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from diffrax.solver.base import AbstractSolver

from .modules import group_norm


class ConcatConv2D(eqx.Module):
    layer: eqx.Module

    def __init__(self, key, dim_in, dim_out, ksize=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, transpose=False):
        module = eqx.nn.ConvTranspose2d if transpose else eqx.nn.Conv2d
        self.layer = module(dim_in+1, dim_out, ksize, stride, padding, 
                            dilation, groups, bias, key=key)

    def __call__(self, t, x):
        tt = jnp.ones_like(x[:1,:,:]) * t
        ttx = jnp.concatenate([tt, x], axis=0)
        return self.layer(ttx)


class ODEFunc(eqx.Module):
    conv1: ConcatConv2D
    conv2: ConcatConv2D
    norm1: Callable
    norm2: Callable

    def __init__(self, key, width):
        key0, key1 = jrandom.split(key, 2)
        self.conv1 = ConcatConv2D(key0, width, width, 3, 1, 1)
        self.norm1 = group_norm(width)
        self.conv2 = ConcatConv2D(key1, width, width, 3, 1, 1)
        self.norm2 = group_norm(width)

    def __call__(self, t, x, args):
        x = self.conv1(t, x)
        x = self.norm1(x)
        x = jax.nn.relu(x)
        x = self.conv2(t, x)
        x = self.norm2(x)

        return x

class ODEBlock(eqx.Module):
    odefunc: ODEFunc
    integration_time: jnp.ndarray
    solver: AbstractSolver
    rtol: float
    atol: float

    def __init__(self, key, solver_name: str = 'Tsit5', width = 64, rtol=1e-1, atol=1e-2):
        self.odefunc = ODEFunc(key, width)
        self.solver = get_solver(solver_name)
        self.integration_time = jnp.array([0, 1])
        self.rtol = rtol
        self.atol = atol

    def __call__(self, x):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.odefunc),
            self.solver,
            t0 = self.integration_time[0],
            t1 = self.integration_time[1],
            dt0 = None,
            y0 = x,
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            saveat=diffrax.SaveAt(ts=self.integration_time),
            adjoint=diffrax.BacksolveAdjoint()
        )
        return solution.ys[1]
    
class ODEBlockWrapper(eqx.Module):
    """ Wraps an ODEBlock and allows to set a custom integration time."""
    ode_block: ODEBlock
    integration_time: jnp.ndarray
    def __init__(self, ode_block: ODEBlock, integration_time_end: float):
        self.ode_block = ode_block
        self.integration_time = jnp.array([0, integration_time_end])

    def __call__(self, x):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.ode_block.odefunc),
            self.ode_block.solver,
            t0 = self.integration_time[0],
            t1 = self.integration_time[1],
            dt0 = None,
            y0 = x,
            stepsize_controller=diffrax.PIDController(rtol=self.ode_block.rtol, atol=self.ode_block.atol),
            saveat=diffrax.SaveAt(ts=self.integration_time),
        )
        return solution.ys[1]

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