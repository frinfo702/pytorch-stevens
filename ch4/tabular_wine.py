import csv

import numpy as np
import torch

wine_path = "data/p1ch4/tabular-wine/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
wineq = torch.from_numpy(wineq_numpy)
col_list = next(csv.reader(open(wine_path), delimiter=";"))

data = wineq[:, :-1]

target = wineq[:, -1].long()

target_onehot = torch.zeros(target.shape[0], 10)  # torch.Size([4898, 10])

# _ で終わるメソッドは破壊的
target_onehot.scatter_(
    1, target.unsqueeze(-1), 1.0
)  # indexの次元数は一致しないといけない
print(target_onehot)
# tensor([[0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         ...,
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 1., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.]])

data_mean = torch.mean(data, dim=0)
data_var = torch.var(data, dim=0)
print(data.shape, data_mean.shape, data_var.shape)
data_normalized = (data - data_mean) / torch.sqrt(data_var)

# quality <= 3を抽出
bad_indexes = target <= 3
mid_indexes = (target > 3) & (target < 7)
good_indexes = target >= 7

bad_data = data[bad_indexes]
mid_data = data[mid_indexes]
good_data = data[good_indexes]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print(i, *args)

total_sulfur_threshold = 141.83
total_sulfur_data = data[:, 6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
print(predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum())

actual_indexes = target > 5

n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
print(n_matches, n_matches / n_predicted, n_matches / n_actual)
