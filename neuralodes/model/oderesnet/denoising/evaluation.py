from torch.utils.data import DataLoader
from .loss import loss

def evaluate(model, testloader: DataLoader):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    avg_loss = 0
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()
        # Note that all the JAX operations happen inside `loss`,
        # and both have JIT wrappers, so this is fast.
        avg_loss += loss(model, x, y).item()
    return avg_loss / len(testloader)