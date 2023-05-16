import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


@eqx.filter_jit
def loss(model, x:Float[Array, "batch 1 H W"], y:Int[Array, "batch 1 H W"]):
    # Our input has the shape (BATCH_SIZE, 1, 28, 28), but our model operations on
    # a single input input image of shape (1, 28, 28).
    #
    # Therefore, we have to use jax.vmap, which in this case maps our model over the
    # leading (batch) axis.
    pred_y = jax.vmap(model)(x)
    return mse_loss(y, pred_y)

def mse_loss(
    y: Float[Array, "batch 1 H W"], pred_y: Float[Array, "batch 1 H W"]
) -> Float[Array, ""]:
    return jnp.mean(jnp.square(y - pred_y))