# %%
import torch

# %%
img_t = torch.randn(3, 5, 5)  # shape [channels, rows, columns]
weights = torch.tensor(
    [0.2126, 0.7152, 0.0722]
)  # a 1D tensor of 3 channel weights like [R, G, B]
batch_t = torch.randn(2, 3, 5, 5)  # [batch, channels, rows, columns]

# %% get channels in regular
img_gray_naive = img_t.mean(-3)
batch_gray_naive = batch_t.mean(-3)
print(img_gray_naive, batch_gray_naive)
# %%
# changes shape from [3] to [3, 1, 1]
# That makes it broadcastable over image height and width
unsqeezed_weights = weights.unsqueeze(-1).unsqueeze_(-1)

# [3, H, W] * [3, 1, 1] = [3, H, W]
img_weights = img_t * unsqeezed_weights

# [N, 3, H, W] * [3, 1, 1] = [1, 3, 1, 1]
batch_weights = batch_t * unsqeezed_weights

# sum(-3) sums across the channel dimension
# For [3, H, W], -3 is the first dimension.
# So this collapsses RGB into one grayscale image; [H, W]
img_gray_weighted = img_weights.sum(-3)
batch_gray_weighted = batch_weights.sum(-3)
print(batch_weights.shape, batch_t.shape, unsqeezed_weights.shape)
# %%
x = torch.ones(32, 128)
scale = torch.ones(128) * 2
y = x * scale  # [32, 128] * [1, 128] = [32, 128] in the tensor rule
print(x.shape)
print(y)

# %%
x = torch.randn(64, 256)
bias = torch.randn(256)
y = x + bias  # [64, 256] + [1, 256] = [64, 256] in the same rule as mul
print(y.shape)
# %%
img = torch.randn(16, 3, 224, 224)  # [num, channel, H, W]
mean = img.mean((0, 2, 3), keepdim=True)
std = img.std((0, 2, 3), keepdim=True)
print(mean.shape)  # [1, 3, 1, 1]
print(std.shape)  # [1, 3, 1, 1]
normalized = (img - mean) / std
print(normalized.shape)  # [16, 3, 224, 224]
print(normalized[2, 4])

# %% name tensor
weights_named = torch.tensor([0.2126, 0.7152, 0.0722]).refine_names("channels")
print(weights_named)
img_named = img_t.refine_names("channels", "rows", "columns")
batch_named = batch_t.refine_names(..., "channels", "rows", "columns")
print("img named: ", img_named.shape, img_named.names)
print("batch named: ", batch_named.shape, batch_named.names)

# %% align_asで名前と次元を合わせることができる
weights_aligned = weights_named.align_as(img_named)
print(weights_aligned.shape, weights_aligned.names)

# %%
gray_named = (img_named * weights_aligned).sum("channels")  # sqeeze in "channel"
print(gray_named.shape, gray_named.names)

# %% Runtime error with different dimension names
gray_named = (img_named[..., :3] * weights_named).sum("channels")
# %% delete names with None
gray_plain = gray_named.rename(None)
print(gray_plain.shape, gray_plain.names)
