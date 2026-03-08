# %%
import torch

# %%
# save data on GPU's RAM
points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device="cuda")
