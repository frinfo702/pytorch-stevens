# %%
import torch

# %%
a = torch.ones(3)
print(a)
print(a[1])
print(float(a[1]))
print(a[1].item())
# %%
points = torch.zeros(6)
points[0] = 4.0
points[1] = 1.0
points[2] = 5.0
points[3] = 3.0
points[4] = 2.0
points[5] = 1.0

# %%
points2 = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
print(points2)
print(points2.shape)
# %%
points3 = torch.zeros(3, 2)
print(points3[0][0])
print(points3[0])
# %%
points4 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(points4[1:])
print(points4[1:, :])
print(points4[1:, 0])
print(points4[None])
