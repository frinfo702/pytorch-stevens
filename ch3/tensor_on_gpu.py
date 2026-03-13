# %%
import h5py
import torch

# %%
# save data on GPU's RAM
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device="cuda")

points = 2 * points  # on CPU
points_gpu = 2 * points.to(device="cuda")  # on GPU
points_gpu += 4  # on GPU

# To bring tensor back to CPU, "cpu" argument have to be given .to
points_cpu = points_gpu.to(device="cpu")

# %% interaction with NumPy
points = torch.ones(3, 4)
points_np = points.numpy()
print(points_np.dtype)
points = torch.from_numpy(points_np)
print(points.dtype)
# %% saving and loading (Serializing)
torch.save(points, "../data/p1ch3/ourpoints.t")
points = torch.load("../data/p1ch3/ourpoints.t")

# %% 相互運用性の高い形式でテンソルを保存
f = h5py.File("../data/p1ch3/ourpoints.hdf5", "w")
dataset = f.create_dataset("coords", data=points.numpy())
f.close()


