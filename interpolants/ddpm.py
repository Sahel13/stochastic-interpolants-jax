import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax import Array, random

# Create the Swill roll data set
num_data_points = 1024
rng = np.random.default_rng()

angles = rng.uniform(size=num_data_points) * 4 * np.pi
noises = rng.standard_normal((2, num_data_points), dtype=np.float32) * 0.1

xs = angles * np.cos(angles) + noises[0]
ys = angles * np.sin(angles) + noises[1]

dataset = np.stack([xs, ys], axis=0).T
index_sampler = grain.samplers.IndexSampler(
    num_records=len(dataset), shuffle=True, num_epochs=1, seed=42
)


class MLP(nnx.Module):
    def __init__(self, dim: int, time_dim: int = 256, *, rngs: nnx.Rngs):
        self.dim = dim
        self.time_dim = time_dim

        # Time embedding MLP
        self.time_mlp = nnx.Sequential(
            nnx.Linear(time_dim, time_dim * 4, rngs=rngs),
            nnx.silu,
            nnx.Linear(time_dim * 4, time_dim, rngs=rngs),
        )

        # Main network layers
        self.layer_1 = nnx.Linear(dim, 256, rngs=rngs)
        self.layer_2 = nnx.Linear(256, 256, rngs=rngs)
        self.layer_3 = nnx.Linear(256, dim, rngs=rngs)

        # Time conditioning layers (project time embedding to layer dimensions)
        self.time_proj_1 = nnx.Linear(time_dim, 256, rngs=rngs)
        self.time_proj_2 = nnx.Linear(time_dim, 256, rngs=rngs)

    def get_sinusoidal_embedding(self, timesteps: Array) -> Array:
        half_dim = self.time_dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb

    def __call__(self, x: Array, t: Array) -> Array:
        # Get sinusoidal time embeddings and process through ALP
        t_emb = self.get_sinusoidal_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # First layer with time conditioning
        x = nnx.relu(self.layer_1(x) + self.time_proj_1(t_emb))

        # Second layer with time conditioning
        x = nnx.relu(self.layer_2(x) + self.time_proj_2(t_emb))

        # Output layer
        x = self.layer_3(x)
        return x


grid_size = 1000
betas = jnp.linspace(1e-4, 2e-2, grid_size)
alphas = 1 - betas
alpha_bars = jnp.exp(jnp.cumsum(jnp.log(alphas)))


@nnx.jit
def train_step(
    model: MLP,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch: Array,
    key: Array,
):
    time_key, noise_key = random.split(key)
    batch_size = batch.shape[0]
    ts = random.choice(time_key, grid_size, (batch_size,))
    x_shape = batch[0].shape
    epsilons = random.normal(noise_key, (batch_size,) + x_shape)
    batch_alpha_bars = alpha_bars[ts][:, None]
    model_input = (
        jnp.sqrt(batch_alpha_bars) * batch + jnp.sqrt(1 - batch_alpha_bars) * epsilons
    )

    def loss_fn(model):
        predictions = model(model_input, ts)
        return optax.l2_loss(predictions, epsilons).mean()

    loss, grads = jax.value_and_grad(loss_fn)(model)
    metrics.update(loss=loss)
    optimizer.update(model, grads)


key = random.key(0)
num_epochs = 100
batch_size = 128

data_loader = grain.DataLoader(
    data_source=dataset,  # type: ignore
    sampler=index_sampler,
    operations=[grain.transforms.Batch(batch_size=batch_size)],
    worker_count=0,
)


model = MLP(2, rngs=nnx.Rngs(0))
learning_rate = 0.005
momentum = 0.9
optimizer = nnx.Optimizer(model, optax.adam(learning_rate, momentum), wrt=nnx.Param)
metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

for epoch in range(num_epochs):
    for batch in data_loader:
        key, sub_key = random.split(key)
        train_step(model, optimizer, metrics, batch, sub_key)

    for metric, value in metrics.compute().items():
        print(f"{metric}: {value:0.4f}")
    metrics.reset()
