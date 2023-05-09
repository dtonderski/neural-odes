from typing import Callable

import diffrax
from diffrax.solver.base import AbstractSolver
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
    solver: AbstractSolver
    solver_kwargs: dict # contains dt0 if not Euler or stepsize_controller if Euler

    def __init__(self, key, solver_name: str = 'Tsit5', width = 64, steps_if_euler = 6):
        self.odefunc = ODEFunc(width, key)
        self.integration_time = jnp.array([0, 1])
        self.solver, self.solver_kwargs = get_solver_params(solver_name, steps_if_euler, self.integration_time)

    def __call__(self, x):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.odefunc),
            self.solver,
            t0 = self.integration_time[0],
            t1 = self.integration_time[1],
            y0 = x,
            saveat=diffrax.SaveAt(ts=self.integration_time),
            **self.solver_kwargs
        )
        return solution.ys[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

def get_solver_params(solver_name: str = 'Tsit5', steps_if_euler: int = 6, integration_time: jnp.ndarray = jnp.array([0, 1])) -> dict:
    match solver_name.lower():
        case 'euler':
            return diffrax.Euler(), {'dt0': (integration_time[1] - integration_time[0])/steps_if_euler} 
        case 'tsit5':
            return diffrax.Tsit5(), {'stepsize_controller': diffrax.PIDController(rtol=1e-3, atol=1e-6)}
        case 'heun':
            return diffrax.Heun(), {'stepsize_controller': diffrax.PIDController(rtol=1e-3, atol=1e-6)}
        case 'midpoint':
            return diffrax.Midpoint(), {'stepsize_controller': diffrax.PIDController(rtol=1e-3, atol=1e-6)}
        case 'ralston':
            return diffrax.Ralston(), {'stepsize_controller': diffrax.PIDController(rtol=1e-3, atol=1e-6)}
        case 'bosh3':
            return diffrax.Bosh3(), {'stepsize_controller': diffrax.PIDController(rtol=1e-3, atol=1e-6)}
        case 'dopri5':
            return diffrax.Dopri5(), {'stepsize_controller': diffrax.PIDController(rtol=1e-3, atol=1e-6)}
        case 'dopri8':
            return diffrax.Dopri8(), {'stepsize_controller': diffrax.PIDController(rtol=1e-3, atol=1e-6)}
        # case 'impliciteuler':
        #     return diffrax.ImplicitEuler()
        # case 'kvaerno3':
        #     return diffrax.Kvaerno3()
        # case 'kvaerno4':
        #     return diffrax.Kvaerno4()
        # case 'kvaerno5':
        #     return diffrax.Kvaerno4()
