import jax
import flax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from typing import Dict
import jax_dataloader as jdl
import numpy as np
import matplotlib.pyplot as plt 
from flax.serialization import to_bytes
import optax
import jax.nn.initializers as init  
import os 
import json
import csv 

@jax.jit
def count_unique(x):
    x = jnp.sort(x)
    return 1 + (x[1:] != x[:-1]).sum()


class AtomEmbedding(nnx.Module):
    def __init__(self, num_atom_types, embedding_dim, n_batch, rngs: nnx.Rngs = nnx.Rngs(0)):
        super(AtomEmbedding, self).__init__()

        self.embedding = nnx.Embed(
            num_embeddings=num_atom_types,
            features=embedding_dim,
            rngs=rngs
        )
        self.n_batch = n_batch

    def __call__(self, atom_types):
        ord_atom_types = jnp.unique(atom_types)
        mask = jnp.array([jnp.where(ord_atom_types==elem) for elem in atom_types.flatten()]).flatten()
        return jnp.array(jnp.split(self.embedding(mask), self.n_batch))


class R_distances(nnx.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, R):
        num_atoms = len(R[0])
        Rij = jnp.array([[[r[i] - r[j] for j in range(num_atoms)] for i in range(num_atoms)] for r in R])
        d_ij = jnp.linalg.norm(Rij, axis=-1)
        return d_ij


class RadialBasisFunctions(nnx.Module):
    def __init__(self, rbf_min, rbf_max, n_rbf, gamma=10):
        super().__init__()
        self.rbf_min = rbf_min
        self.rbf_max = rbf_max
        self.n_rbf = n_rbf
        self.gamma = gamma
        self.centers = jnp.linspace(rbf_min, rbf_max, n_rbf).reshape(1, -1)

    def __call__(self, d_ij):
        diff = d_ij[..., None] - self.centers
        return jnp.exp(-self.gamma * jnp.pow(diff, 2))


class relu_layer(nnx.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return nnx.relu(x)


class LeakyReLU(nnx.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def __call__(self, x):
        return jnp.where(x > 0, x, self.alpha * x)


class ssp_layer(nnx.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return jnp.log(0.5 * jnp.exp(x) + 0.5)


class filter_generator(nnx.Module):
    def __init__(
            self, 
            atom_embeddings_dim, 
            rbf_min, 
            rbf_max, 
            n_rbf, 
            rngs: nnx.Rngs = nnx.Rngs(0), 
            activation=ssp_layer,
            kernel_init=init.uniform(),
            bias_init=init.zeros
            ):
        super().__init__()
        self.rbf = RadialBasisFunctions(rbf_min, rbf_max, n_rbf)
        self.w_layers = nnx.Sequential(
            nnx.Linear(n_rbf, atom_embeddings_dim, rngs=rngs, kernel_init=kernel_init, bias_init=bias_init),
            activation(),
            nnx.Linear(atom_embeddings_dim, atom_embeddings_dim, rngs=rngs, kernel_init=kernel_init, bias_init=bias_init),
            activation()
        )

    def __call__(self, d_ij):
        rbfs = self.rbf(d_ij)
        Wij = self.w_layers(rbfs)
        return Wij


class CfConv(nnx.Module):
    def __init__(
            self, 
            atom_embeddings_dim, 
            rbf_min, 
            rbf_max, 
            n_rbf, 
            rngs: nnx.Rngs = nnx.Rngs(0), 
            activation=ssp_layer,
            kernel_init=init.uniform(),
            bias_init=init.zeros):
        super().__init__()

        self.rbf = RadialBasisFunctions(rbf_min, rbf_max, n_rbf)
        self.filters = filter_generator(
            atom_embeddings_dim, 
            rbf_min,
            rbf_max, 
            n_rbf, 
            rngs,
            activation,
            kernel_init=kernel_init,
            bias_init=bias_init
        )

    def __call__(self, X, d_ij):
        fij = self.filters(d_ij)
        X_j = X[:, None, :, :]
        X_ij = fij*X_j
        X_update = jnp.sum(X_ij, axis=2)
        return X_update


class InteractionBlock(nnx.Module):
    def __init__(
            self,
            atom_embeddings_dim,
            rbf_min,
            rbf_max,
            n_rbf,
            rngs: nnx.Rngs,
            activation=ssp_layer,
            kernel_init=init.uniform(),
            bias_init=init.zeros
    ):
        super().__init__()
        self.in_atom_wise = nnx.Linear(
            atom_embeddings_dim,
            atom_embeddings_dim,
            rngs=rngs,
            kernel_init=kernel_init,
            bias_init=bias_init
        )

        self.cf_conv = CfConv(
            atom_embeddings_dim,
            rbf_min,
            rbf_max,
            n_rbf,
            rngs=rngs,
            activation=activation,
            kernel_init=kernel_init,
            bias_init=bias_init
        )

        self.out_atom_wise = nnx.Sequential(
            nnx.Linear(atom_embeddings_dim, atom_embeddings_dim, rngs=rngs, kernel_init=kernel_init, bias_init=bias_init),
            activation(),
            nnx.Linear(atom_embeddings_dim, atom_embeddings_dim, rngs=rngs, kernel_init=kernel_init, bias_init=bias_init)
        )

    def __call__(self, X, R_distances):
        X_in = self.in_atom_wise(X)
        X_conv = self.cf_conv(X_in, R_distances)
        V = self.out_atom_wise(X_conv)
        return X + V


class SchNet(nnx.Module):
    def __init__(
            self,
            n_batch,
            atom_embedding_dim=16,
            n_interactions=3,
            n_atom_types=3,
            rbf_min=0.,
            rbf_max=30.,
            n_rbf=300,
            rngs: nnx.Rngs = nnx.Rngs(0),
            activation: nnx.Module = ssp_layer,
            kernel_init=init.xavier_normal(),
            bias_init=init.zeros
    ):
        super().__init__()
        self.n_atom_types = n_atom_types
        self.embedding = AtomEmbedding(n_atom_types, atom_embedding_dim, rngs=rngs, n_batch=n_batch)

        self.interactions = [
            InteractionBlock(
                atom_embedding_dim, rbf_min, rbf_max, n_rbf, rngs, activation,
                kernel_init=kernel_init, bias_init=bias_init
            )
            for _ in range(n_interactions)
        ]

        self.output_layers = nnx.Sequential(
            nnx.Linear(atom_embedding_dim, 32, rngs=rngs, kernel_init=kernel_init, bias_init=bias_init),
            activation(),
            nnx.Linear(32, 1, rngs=rngs, kernel_init=kernel_init, bias_init=bias_init)
        )

        self.distances = R_distances()

    def __call__(self, batch):
        Z = batch[0]
        R = batch[1]
        R_distances = self.distances(R)
        X = self.embedding(Z)
        X_interacted = X
        for _, interaction in enumerate(self.interactions):
            X_interacted = interaction(X_interacted, R_distances)
        
        atom_outputs = self.output_layers(X_interacted)
        predicted_energies = jnp.sum(atom_outputs, axis=1)
        return predicted_energies

    

N_frames=99000

def create_water_dataset_from_npz(data_name: str):
    water_data = np.load(data_name)
    energy = water_data["E"][:N_frames]
    n_data = len(energy)
    atom_types = jnp.array([water_data["z"][:N_frames] for _ in range(n_data)])
    positions = water_data["R"][:N_frames]
    return jdl.ArrayDataset(atom_types, positions, energy)

n_batch = 9000

dataset = create_water_dataset_from_npz("./data_water.npz")
train_size = 90000
train_data, val_data = dataset[:train_size], dataset[train_size:]

mean = jnp.mean(train_data[2])
std = jnp.std(train_data[2])
std_train_energy = (train_data[2] - mean) / std
std_val_energy= (val_data[2] - mean) / std

train_dataset = jdl.ArrayDataset(train_data[0], train_data[1], std_train_energy)
val_dataset = jdl.ArrayDataset(val_data[0], val_data[1], val_data[2])

# Crear dataloaders
train_dataloader = jdl.DataLoader(train_dataset, 'jax', batch_size=n_batch, shuffle=True)
val_dataloader = jdl.DataLoader(val_dataset, 'jax', batch_size=n_batch, shuffle=False)



learning_rate = 0.001

schedule = optax.exponential_decay(
    init_value=learning_rate,
    transition_steps=30,
    decay_rate=0.9,
    staircase=True
)

def get_current_lr(step):
    return schedule(step).item() 

model = SchNet(n_batch=n_batch, n_atom_types=3, rbf_min=0., rbf_max=30.,n_rbf=300, activation=ssp_layer)

'''optimizer = nnx.Optimizer(model, optax.chain(
    optax.clip_by_global_norm(0.5), 
    optax.adamw(learning_rate=schedule)
))'''
optimizer=nnx.Optimizer(model,optax.adamw(learning_rate=learning_rate))


def train_step(optimizer,batch):
    def loss_fn(model):
        return ((model(batch) - batch[2]) ** 2).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
    
    #grads = nnx.grad(loss_fn)(optimizer.model)
    optimizer.update(grads)
    #print('Optimizer after update:', optimizer) 
    #loss=loss_fn(model)
    
    return optimizer,loss

def validate_step(optimizer,batch):
    def loss_fn(model):
        energy=model(batch)*std+mean
        return ((energy - batch[2]) ** 2).mean()
    loss = loss_fn(optimizer.model)
    return loss

num_epochs = 200
patience = 30  
best_val_loss = float('inf')
epochs_without_improvement = 0
filters=[]
train_losses, val_losses = [], []


for epoch in range(num_epochs):
    current_lr = get_current_lr(epoch)
        #print(f"Epoch {epoch + 1}, Learning Rate: {current_lr}")
    epoch_train_loss = 0
    epoch_val_loss=0
    epoch_filters=[]
    for batch in train_dataloader:
        optimizer, train_loss = train_step(optimizer, batch)
        epoch_train_loss += train_loss
        d_ij = optimizer.model.distances(batch[1])  
        batch_filters = optimizer.model.interactions[0].cf_conv.filters.rbf(d_ij)  
        epoch_filters.append(np.array(batch_filters)) 
            
    avg_train_loss = epoch_train_loss / len(train_dataloader)
    filters.append(epoch_filters)
        
    for batch in val_dataloader:
        val_loss = validate_step(optimizer, batch)
        epoch_val_loss+=val_loss
    avg_val_loss = epoch_val_loss / len(val_dataloader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

        # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
            # Guardar el mejor modelo
        best_model = model
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stop.")
            break
        
    print(f"Epoch {epoch}: Average Training Loss = {avg_train_loss}, Validation Loss = {avg_val_loss}")



output_dir = "model_outputs"
os.makedirs(output_dir, exist_ok=True)

loss_curve_path = os.path.join(output_dir, "loss_curve.png")
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Training and Validation Loss Curves")
plt.savefig(loss_curve_path)
plt.close()


filters_csv_path = os.path.join(output_dir, "filters.csv")

# Aplanar los filtros y guardarlos como CSV
with open(filters_csv_path, "w", newline="") as archivo_csv:
    writer = csv.writer(archivo_csv)
    writer.writerow(["Epoch", "Batch", "Filter_Index", "Filter_Values"])  # Cabecera

    for epoch_idx, epoch_filters in enumerate(filters):
        for batch_idx, batch_filters in enumerate(epoch_filters):
            for filter_idx, filter_values in enumerate(batch_filters):
                writer.writerow([
                    epoch_idx + 1, 
                    batch_idx + 1, 
                    filter_idx, 
                    list(filter_values)  # Convertir los valores a lista para serializar
                ])
#filters_path = os.path.join(output_dir, "filters.json")
#with open(filters_path, "w") as archivo:
#    json.dump(list(filters), archivo)
train_loss_csv_path = os.path.join(output_dir, "train_losses.csv")
with open(train_loss_csv_path, "w", newline="") as archivo_csv:
    writer = csv.writer(archivo_csv)
    writer.writerow(["Epoch", "Training Loss"])  # Cabecera
    for epoch, loss in enumerate(train_losses, start=1):
        writer.writerow([epoch, loss])

val_loss_csv_path = os.path.join(output_dir, "val_losses.csv")
with open(val_loss_csv_path, "w", newline="") as archivo_csv:
    writer = csv.writer(archivo_csv)
    writer.writerow(["Epoch", "Validation Loss"])  # Cabecera
    for epoch, loss in enumerate(val_losses, start=1):
        writer.writerow([epoch, loss])

params = nnx.state(model, nnx.Param)
model_path = os.path.join(output_dir, "best_model_params.npy")
np.save(model_path, params)

print(f"Parámetros del modelo guardados en: {model_path}")