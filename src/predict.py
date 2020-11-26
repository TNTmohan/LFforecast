from torch import load, Tensor
import pandas as pd
import numpy as np
from chinese_calendar import is_holiday
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter('ignore')

print('Loading data...')
df_load = pd.read_excel('../data/STLF_DATA_IN_1.xls', sheet_name=0, header=None)
df_weather = pd.read_excel('../data/STLF_DATA_IN_1.xls', sheet_name=1, header=None)


predict_date = int(input('请输入待预测日（示例：20080605）：'))
date = pd.to_datetime(predict_date, format='%Y%m%d')
load_index = df_load.loc[df_load[0] == predict_date].index.item()
load_data = df_load.iloc[load_index-7:load_index, 1:].values.reshape((1, -1)) / 7000


max_tempe = df_weather.loc[(df_weather[0] == predict_date) & (df_weather[1] == '最高温度'), [2]].values / 20
min_tempe = df_weather.loc[(df_weather[0] == predict_date) & (df_weather[1] == '最低温度'), [2]].values / 20
avg_tempe = df_weather.loc[(df_weather[0] == predict_date) & (df_weather[1] == '平均温度'), [2]].values / 20
humidity = df_weather.loc[(df_weather[0] == predict_date) & (df_weather[1] == '湿度'), [2]].values / 100
weather_data = np.concatenate([max_tempe, min_tempe, avg_tempe, humidity]).reshape((1, -1))


type_of_day = np.eye(7)[date.dayofweek]
holiday = np.eye(2)[int(is_holiday(date))]
time_data = np.concatenate([type_of_day, holiday]).reshape((1, -1))


features = np.concatenate([load_data, weather_data, time_data], axis=1).reshape(1, 685)
features = Tensor(features)

print('Loading model...')
net = load('./model.pt', map_location='cpu')

print('Start predicting...')
net.eval()
labels = net(features).detach().numpy() * 7000

print('Start ploting...')
plt.figure(figsize=(10, 8))
plt.title(predict_date)
plt.xlabel('Time')
plt.ylabel('Load/MW')
plt.plot(labels.reshape(96))
plt.grid()

print('Done!')

