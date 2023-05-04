import os
import pickle
import time

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from jaxtyping import (Array, Float,  # https://github.com/google/jaxtyping
                       Int, PyTree)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".85"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
STEPS = 300
PRINT_EVERY = 30
SEED = 5678
MODEL_NAME = "odenet"

key = jax.random.PRNGKey(SEED)

normalise_data = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_dataset = torchvision.datasets.MNIST(
    "data/MNIST",
    train=True,
    download=True,
    transform=normalise_data,
)
test_dataset = torchvision.datasets.MNIST(
    "data/MNIST",
    train=False,
    download=True,
    transform=normalise_data,
)
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)

def norm(dim):
    return eqx.nn.GroupNorm(groups=min(32, dim), channels=dim)

def conv3x3(in_planes, out_planes, key, stride=1):
    return eqx.nn.Conv2d(in_planes, out_planes, kernel_size = 3, padding = 1, stride=stride, key = key)

def conv1x1(in_planes, out_planes, key, stride=1):
    return eqx.nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride=stride, key = key)


class ResBlock(eqx.Module):
    layers: list
    downsample: eqx.Module
    
    def __init__(self, inplanes, planes, key, stride=1, downsample=None):
        key1, key2, key3 = jrandom.split(key, 3)
        self.downsample = downsample
        self.layers = [
            norm(inplanes),
            jax.nn.relu,
            conv3x3(inplanes, planes, key2, stride),
            norm(inplanes),
            jax.nn.relu,
            conv3x3(inplanes, planes, key3)            
        ]
    
    def __call__(self, x:Float[Array, "1 28 28"]):
        shortcut = x
        shortcut = self.downsample(shortcut) if self.downsample is not None else shortcut
        for layer in self.layers:
            x = layer(x) if layer is not None else x
        return x + shortcut
    
############ Downsampling Block ############
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

############ ResNet ############
class ResNet(eqx.Module):
    layers: list
    def __init__(self, key):
        feature_keys = jrandom.split(key, 6)
        self.layers = [ResBlock(64, 64, feature_keys[i]) for i in range(6)]

    def __call__(self, x:Float[Array, "1 28 28"]):
        for layer in self.layers:
            x = layer(x)
        return x

############ ODE Solver ############
class ConcatConv2D(eqx.Module):
    layer: eqx.Module
    
    def __init__(self, dim_in, dim_out, key, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        module = eqx.nn.ConvTranspose2d if transpose else eqx.nn.Conv2d
        self.layer = module(dim_in+1, dim_out, ksize, stride, padding, dilation, groups, bias, key=key)
        
    def __call__(self, t, x):
        tt = jnp.ones_like(x[:1,:,:]) * t
        ttx = jnp.concatenate([tt, x], axis=0)
        return self.layer(ttx)
        
class ODEFunc(eqx.Module):
    norm1: eqx.Module
    relu: jax.nn.relu
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
    


############ Downsampling Block ############
class Model(eqx.Module):
    layers: list
    def __init__(self, key, type:str):
        key0, key1, key2 = jrandom.split(key, 3)

        self.layers = [DownsamplingBlock(key0)]
        if type.lower() == "resnet":
            self.layers.extend([ResNet(key1)])
        elif type.lower() == "odenet":
            self.layers.extend([ODEBlock(key1)])

        self.layers.extend([FCBlock(key2)])        
    
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
        
def loss(model, x:Float[Array, "1 28 28"], y:Int[Array, " batch"]):
    # Our input has the shape (BATCH_SIZE, 1, 28, 28), but our model operations on
    # a single input input image of shape (1, 28, 28).
    #
    # Therefore, we have to use jax.vmap, which in this case maps our model over the
    # leading (batch) axis.
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)


def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    # y are the true targets, and should be integers 0-9.
    # pred_y are the log-softmax'd predictions.
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)

dummy_x, dummy_y = next(iter(trainloader))
dummy_x = dummy_x.numpy()
dummy_y = dummy_y.numpy()

model = Model(key, MODEL_NAME)

# Example loss
loss_value = loss(model, dummy_x, dummy_y)
print(loss_value.shape)  # scalar loss
# Example inference
output = jax.vmap(model)(dummy_x)
print(output.shape)  # batch of predictions

value, grads = eqx.filter_value_and_grad(loss)(model, dummy_x, dummy_y)
print(value)

loss = eqx.filter_jit(loss)  # JIT our loss function from earlier!


@eqx.filter_jit
def compute_accuracy(model, x, y):
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    pred_y = jax.vmap(model)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)

def evaluate(model, testloader: torch.utils.data.DataLoader):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    avg_loss = 0
    avg_acc = 0
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()
        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        avg_loss += loss(model, x, y)
        avg_acc += compute_accuracy(model, x, y)
    return avg_loss / len(testloader), avg_acc / len(testloader)

evaluate(model, testloader)

optim = optax.adamw(LEARNING_RATE)


def train(
    model,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
):
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model,
        opt_state,
        x,
        y,
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    train_loss_dict = {}
    test_loss_dict = {}
    test_accuracy_dict = {}
    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy()
        y = y.numpy()
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        train_loss_dict.update({step:train_loss})
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model, testloader)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )
            test_loss_dict.update({step:test_loss})
            test_accuracy_dict.update({step:test_accuracy})

    return model, train_loss_dict, test_loss_dict, test_accuracy_dict

model, train_loss_dict, test_loss_dict, test_accuracy_dict = train(model, trainloader, testloader, optim, STEPS, PRINT_EVERY)

with open(f'{MODEL_NAME}.pkl', 'wb') as f:
    pickle.dump([train_loss_dict, test_loss_dict, test_accuracy_dict], f)