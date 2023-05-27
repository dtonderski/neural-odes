import os
import pathlib

import equinox as eqx
import fire
import jax.random as jrandom
import numpy as np
import optax
import tqdm
import wandb
from neuralodes.data.caltech_classification import dataloader
from neuralodes.model.oderesnet.caltech_classification.evaluation import evaluate
from neuralodes.model.oderesnet.caltech_classification.loss import loss
from neuralodes.model.oderesnet.caltech_classification.odenet import ODENet
from neuralodes.model.oderesnet.caltech_classification.resnet import ResNet
from torch.utils.data import DataLoader

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".85"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

def train(
    model,
    trainloader: DataLoader,
    testloader: DataLoader,
    optim: optax.GradientTransformation,
    num_epochs: int,
    name: str
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

    model_dir = pathlib.Path("models", "oderesnet", "caltech", name)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        batch_losses = []

        min_test_loss = np.inf

        for i, (x,y) in enumerate(tqdm.tqdm(trainloader)):
            x = x.numpy()
            y = y.numpy()
            model, opt_state, batch_train_loss = make_step(model, opt_state, x, y)
            batch_train_loss = batch_train_loss.item()
            batch_losses.append(batch_train_loss)

            wandb.log({"batch_train_loss": batch_train_loss, "batch": i+len(trainloader)*epoch})                

        epoch_train_loss = np.mean(np.array(batch_losses))
        print(f"{epoch=}, train_loss={epoch_train_loss}")
        wandb.log({"train_loss": epoch_train_loss, "epoch": epoch})

        
        epoch_test_loss, epoch_test_accuracy = evaluate(model, testloader)
        
        eqx.tree_serialise_leaves(model_dir / f"epoch{epoch}.eqx", model)
        if epoch_test_loss < min_test_loss:
            min_test_loss = epoch_test_loss
            
            eqx.tree_serialise_leaves(model_dir / f"best.eqx", model)

        
        print(
            f"{epoch=}, epoch_test_loss={epoch_test_loss}, epoch_test_accuracy={epoch_test_accuracy}"
        )
        wandb.log({"epoch_test_loss": epoch_test_loss, "epoch_test_accuracy": epoch_test_accuracy, "epoch": epoch})

    return model

def get_name(model_type, width, solver, rtol, atol, blocks):
    match model_type:
        case "odenet":
            return f"caltech_{model_type}_{width}_{solver}_{rtol}_{atol}"
        case "resnet":
            return f"caltech_{model_type}_{width}_{blocks}"
        case _:
            raise ValueError(f"Unknown model type {model_type}")
    
def initialize_wandb(name:str):
    wandb.init(
        entity='davton',
        # set the wandb project where this run will be logged
        project="neural-odes",
        name=name,
        # track hyperparameters and run metadata
        config=locals()
    )
    wandb.define_metric("batch")
    wandb.define_metric("epoch")
    
    wandb.define_metric("batch_train_loss", step_metric="batch")
    wandb.define_metric("epoch_train_loss", step_metric="epoch")
    
    wandb.define_metric("batch_test_loss", step_metric="batch")
    wandb.define_metric("epoch_test_loss", step_metric="epoch")

def get_model(model_type: str, key, width, solver, rtol, atol, blocks):
    match model_type:
        case "odenet":
            return ODENet(key, solver, width, rtol, atol)
        case "resnet":
            return ResNet(key, width, blocks)
        case _:
            raise ValueError(f"Unknown model type {model_type}")

def get_lr_scheduler(learning_rate):
    def lr_scheduler(epoch):
        if epoch < 15:
            return learning_rate
        if epoch < 30:
            return learning_rate/10
        return learning_rate/100
    return lr_scheduler

def main(model_type: str = "odenet", width = 64, solver: str = 'Tsit5', rtol: float = 1e-1, atol: float = 1e-2, blocks: int = 6,
         learning_rate: float = 1e-1, batch_size = 64, seed: int = 5678, num_epochs = 50):
    train_dataloader, test_dataloader = dataloader.get_dataloaders(batch_size)
    key = jrandom.PRNGKey(seed)
    
    name = get_name(model_type, width, solver, rtol, atol, blocks)
    
    initialize_wandb(name)

    model = get_model(model_type, key, width, solver, rtol, atol, blocks)

    optim = optax.sgd(learning_rate = get_lr_scheduler(learning_rate), momentum=0.9)

    model = train(model, train_dataloader, test_dataloader, optim, num_epochs, name)

if __name__ == "__main__":
    fire.Fire(main)
