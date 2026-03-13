import numpy as np
import torch

bikes_numpy = np.loadtxt(
    "data/p1ch4/bike-sharing-dataset/hour-fixed.csv",
    dtype=np.float32,
    delimiter=",",
    skiprows=1,
    converters={1: lambda x: float(x[8:10])},
)

bikes = torch.from_numpy(bikes_numpy)  # [17520, 17])

# [day, hr, eatch index]の3軸に変形
daily_bikes = bikes.view(-1, 24, bikes.shape[1])  # [C, L, N]
# -1はプレースホルダーで、他の要素から自動的に計算される
print(daily_bikes.shape, daily_bikes.stride())

# [N, C, L]に変えたい
daily_bikes = daily_bikes.transpose(1, 2)
print(daily_bikes.shape, daily_bikes.stride())

# ひとまず初日のみを確認
first_day = bikes[:24].long()
weather_onehot = torch.zeros(first_day.shape[0], 4)
first_day[:, 9]

weather_onehot.scatter_(dim=1, index=first_kday[:, 9].unsqueeze(1).long() - 1, value=1.0)

torch.cat((bikes[:24], weather_onehot), 1)[:1]

daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, daily_bikes.shape[2])
# [730, 4, 24]

# one-hot encoding をC次元のテンソルに実施
daily_weather_onehot.scatter_(
    1, daily_bikes[:, 9, :].long().unsqueeze(1) - 1, 1.0
)  # [730, 4, 24]
# C次元に沿って連結
daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), dim=1)
daily_bikes[:, 9, :] = (daily_bikes[:, 9, :] - 1.0) / 3.0

temp = daily_bikes[:, 10, :]
temp_min = torch.min(temp)
temp_max = torch.max(temp)
daily_bikes[:, 10, :] = (daily_bikes[:, 10, :] - temp_min) / (temp_max - temp_min)
