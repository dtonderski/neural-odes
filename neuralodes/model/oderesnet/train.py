import os

import equinox as eqx
import fire
import jax.random as jrandom
import optax
from torch.utils.data import DataLoader
import wandb
import tqdm

from neuralodes.data import mnist
from neuralodes.model.oderesnet.evaluation import evaluate
from neuralodes.model.oderesnet.loss import loss
from neuralodes.model.oderesnet.odenet import ODENet
from neuralodes.model.oderesnet.resnet import ResNet
import numpy as np
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

    for epoch in range(num_epochs):
        batch_losses = []

        for i, (x,y) in enumerate(tqdm.tqdm(trainloader)):
            x = x.numpy()
            y = y.numpy()
            model, opt_state, batch_train_loss = make_step(model, opt_state, x, y)
            batch_losses.append(batch_train_loss)

            wandb.log({"batch_train_loss": batch_train_loss, "batch": i+len(trainloader)*epoch})
            
            if (i % evaluate_every) == 0 or (i == len(trainloader) - 1):
                batch_test_loss, batch_test_accuracy = evaluate(model, testloader)
                wandb.log({"batch_test_loss": batch_test_loss, "batch_test_accuracy": batch_test_accuracy, "batch": i+len(trainloader)*epoch})

        epoch_train_loss = np.mean(np.array(batch_losses))
        print(f"{epoch=}, train_loss={epoch_train_loss}")
        wandb.log({"train_loss": epoch_train_loss, "epoch": epoch})

        
        epoch_test_loss, epoch_test_accuracy = evaluate(model, testloader)
        print(
            f"{epoch=}, epoch_test_loss={epoch_test_loss.item()}, epoch_test_accuracy={epoch_test_accuracy.item()}"
        )
        wandb.log({"epoch_test_loss": epoch_test_loss, "epoch_test_accuracy": epoch_test_accuracy, "epoch": epoch})

    return model


def main(model_type: str = "odenet", learning_rate: float = 3e-4, batch_size = 256, seed: int = 5678, num_epochs = 300, evaluate_every = 10):
    train_dataloader = mnist.get_dataloader_split(True, batch_size)
    test_dataloader = mnist.get_dataloader_split(False, batch_size)
    key = jrandom.PRNGKey(seed)
    
    wandb.init(
        entity='davton',
        # set the wandb project where this run will be logged
        project="neural-odes",
        name=f"{model_type}_{learning_rate}_{batch_size}",

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
            model = ODENet(key)
        case "resnet":
            model = ResNet(key)
        case _:
            raise ValueError(f"Unknown model type {model_type}")

    optim = optax.adamw(learning_rate)

    model = train(model, train_dataloader, test_dataloader, optim, num_epochs, evaluate_every)

if __name__ == "__main__":
    fire.Fire(main)
