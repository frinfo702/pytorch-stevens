# %%
import torch

# %%
double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)

print(double_points.dtype)
print(short_points.dtype)

# %% type casting
double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()

print(double_points.dtype)
print(short_points.dtype)

double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.zeros(10, 2).to(torch.short)

print(double_points.dtype)
print(short_points.dtype)
