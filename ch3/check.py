# %% 1.
import torch

a = torch.tensor(list(range(9)))
print(a.size())
print(a.storage_offset())
print(a.stride())

# %% a.
b = a.view(3, 3)
print(id(b.storage()) is id(a.storage()))

# %% b.
c = b[1:, 1:]
print(b.size())
print(b.storage_offset())
print(b.stride())

# %% 2.a.
a_cosine = torch.cos(a)
a_sqrt = torch.sqrt(a)
