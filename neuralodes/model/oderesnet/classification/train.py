import os
import pathlib

import equinox as eqx
import fire
import jax.random as jrandom
import numpy as np
import optax
import tqdm
import wandb
from neuralodes.data.classification import dataloader
from neuralodes.model.oderesnet.classification.evaluation import evaluate
from neuralodes.model.oderesnet.classification.loss import loss
from neuralodes.model.oderesnet.classification.odenet import ODENet
from neuralodes.model.oderesnet.classification.resnet import ResNet
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
    evaluate_every: int,
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

    model_path = pathlib.Path("models", "oderesnet", f"{name}.eqx")
    
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
            
            if (i % evaluate_every) == 0 or (i == len(trainloader) - 1):
                batch_test_loss, batch_test_accuracy = evaluate(model, testloader)
                wandb.log({"batch_test_loss": batch_test_loss, "batch_test_accuracy": batch_test_accuracy, "batch": i+len(trainloader)*epoch})
                
                if batch_test_loss < min_test_loss:
                    min_test_loss = batch_test_loss
                    
                    eqx.tree_serialise_leaves(model_path, model)

        epoch_train_loss = np.mean(np.array(batch_losses))
        print(f"{epoch=}, train_loss={epoch_train_loss}")
        wandb.log({"train_loss": epoch_train_loss, "epoch": epoch})

        
        epoch_test_loss, epoch_test_accuracy = evaluate(model, testloader)
        print(
            f"{epoch=}, epoch_test_loss={epoch_test_loss}, epoch_test_accuracy={epoch_test_accuracy}"
        )
        wandb.log({"epoch_test_loss": epoch_test_loss, "epoch_test_accuracy": epoch_test_accuracy, "epoch": epoch})

    return model

def get_name(model_type, dataset, solver, width):
    return f"{model_type}_{dataset}_{solver}_{width}"    

def main(model_type: str = "odenet", dataset: str = 'fashionmnist', solver: str = 'Tsit5', width = 64,
         learning_rate: float = 3e-4, batch_size = 256, seed: int = 5678, num_epochs = 20, evaluate_every = 100):
    train_dataloader, test_dataloader = dataloader.get_dataloaders(dataset, batch_size)
    key = jrandom.PRNGKey(seed)
    
    name = get_name(model_type, dataset, solver, width)
    
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
    wandb.define_metric("batch_test_accuracy", step_metric="batch")
    
    wandb.define_metric("epoch_test_loss", step_metric="epoch")
    wandb.define_metric("epoch_test_accuracy", step_metric="epoch")



    match model_type:
        case "odenet":
            model = ODENet(key, solver, width)
        case "resnet":
            model = ResNet(key, width)
        case _:
            raise ValueError(f"Unknown model type {model_type}")

    optim = optax.adamw(learning_rate)

    model = train(model, train_dataloader, test_dataloader, optim, num_epochs, evaluate_every, name)

if __name__ == "__main__":
    fire.Fire(main)
