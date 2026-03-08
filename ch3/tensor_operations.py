# %%
import torch

# %%
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
print(a.shape, a_t.shape)

# %%
