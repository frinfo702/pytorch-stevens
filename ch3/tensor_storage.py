# %%
import torch

# %%
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storage()
# tensorの行数列数に関係なくメモリ上では連続したサイズ6の配列になっている(常に1次元)
# tensorはインデックス先のポインタを把握しているだけ

# %%
points_storage = points.storage()
print(points_storage[0])
print(points.storage()[1])

# %%
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
print(points)
points_storage = points.storage()
points_storage[0] = 2.0
print(points)

# %% offset, size and stride
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]

print(
    second_point.storage_offset()
)  # 2つの項目を持つ最初の点をスキップしないといけないため
print(second_point.size())
print(second_point.shape)
print(
    second_point.stride()
)  # stride: 各次元でインデックスが1増加した時にスキップしなければならないストレージ内の要素の数を示すタプル
# 2次元テンソルの要素i, jにアクセスするとき
# ストレージ内のstorage_offset + stride[0]*i + stride[1]*j の要素にアクセスする

# %% How size and stride changes when indexing
second_point = points[1]
print("original tensor: \n", points)
print("subtensor: ", second_point)
print("size: ", second_point.size())
print("offset: ", second_point.storage_offset())
print("stride: ", second_point.stride())
# %% To avoid side effect, new tensor can be cloned
original_points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points = original_points
second_point = points[1].clone()
second_point[0] += 10

assert original_points is points  # true

# %%
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
print(points)

points_t = points.t()
print(points_t)
# check whether these two tensors share same storage
print(points.untyped_storage().data_ptr() == points_t.untyped_storage().data_ptr())

# %%
# strideの挙動で転置を定義すれば3次元以上の転置も定義できる
some_t = torch.ones(3, 4, 5)
transpose_t = some_t.transpose(0, 2)
print(some_t.shape)
# %% contiguous
print(points.is_contiguous())
print(points_t.is_contiguous())

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_t = points.t()
print(points_t)
print(points_t.storage())
print(points_t.stride())

points_t_cont = points_t.contiguous()
print(points_t_cont)
print(points_t_cont.storage())
print(points_t_cont.stride())
