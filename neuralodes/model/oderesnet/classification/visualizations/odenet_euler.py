from pathlib import Path

import equinox as eqx
from fire import Fire
import jax.numpy as jnp
import jax.random as jrandom
from matplotlib import pyplot as plt


from neuralodes.data.dataloader import get_dataloaders
from neuralodes.model.oderesnet.classification.evaluation import evaluate
from neuralodes.model.oderesnet.classification.odenet import ODENet, ODENetEulerWrapper
from neuralodes.model.oderesnet.classification.resnet import ResNet

def main(loaded_solver_name = 'Tsit5', max_steps = 20, save_path = None):
    save_path = Path(save_path) if save_path is not None else Path("visualizations", "odenet_euler", loaded_solver_name)
    key = jrandom.PRNGKey(0)
    model_solver_name = "tsit5"

    print("Loading models...")
    odenet = eqx.tree_deserialise_leaves(Path("models", "oderesnet", f"odenet_fashionmnist_{loaded_solver_name}_64.eqx"), ODENet(key, model_solver_name))
    resnet = eqx.tree_deserialise_leaves(Path("models", "oderesnet", f"resnet_fashionmnist_Tsit5_64.eqx"), ResNet(key))

    _, test_dataloader = get_dataloaders("fashionmnist", 256)
    
    losses_euler = []
    accs_euler = []
    from tqdm import tqdm
    steps_arr = list(range(1, max_steps))
    
    print("Evaluating Euler...")
    for steps in tqdm(range(1,max_steps)):
        odenet_euler = ODENetEulerWrapper(odenet, steps)
        loss, acc = evaluate(odenet_euler, test_dataloader)
        losses_euler.append(loss)
        accs_euler.append(acc)
        
    # Initialize subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    print("Calculating baselines...")
    # Compute baseline loss and accuracy
    loss_baseline_ode, acc_baseline_ode = evaluate(odenet, test_dataloader)
    loss_baseline_res, acc_baseline_res = evaluate(resnet, test_dataloader)

    print("Plotting...")
    # Plot accuracy on the left subplot (ax1)
    ax1.plot(steps_arr, accs_euler, '.-')
    ax1.hlines(acc_baseline_ode, 1, max_steps, 'red')
    ax1.hlines(acc_baseline_res, 1, max_steps, 'green')
    ax2.legend([f"Euler - {loaded_solver_name}", f"ODENet - {loaded_solver_name}", "ResNet"])
    ax1.set_ylim([0.8, 0.95])
    ax1.grid()


    # Plot loss on the right subplot (ax2)
    ax2.plot(steps_arr, losses_euler, '.-')
    ax2.hlines(loss_baseline_ode, 1, max_steps, 'red')
    ax2.hlines(loss_baseline_res, 1, max_steps, 'green')
    ax2.legend([f"Euler - {loaded_solver_name}", f"ODENet - {loaded_solver_name}", "ResNet"])
    ax2.set_ylim([0.2, 0.6])
    ax2.grid()

    # Display the plots
    fig.savefig(save_path, dpi=300)
    plt.show()
    
if __name__ == '__main__':
    Fire(main)