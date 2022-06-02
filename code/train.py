from turtle import shape
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import glob
def createXY(dataset):
	dataX, dataY = [], []
	for i in range(168,len(dataset)-25,24):
		dataX.append(dataset[i-168:i, 0]) 
		dataY.append(dataset[i+1:i+25, 0])
	return np.array(dataX), np.array(dataY)
g_train=np.empty((1,1),dtype=float)
c_train=np.empty((1,1),dtype=float)
# 載入訓練資料集
for f in glob.glob("*.csv"):
	a = read_csv(f, usecols=[1])
	a = a.values
	a = a.astype('float32')
	g_train=np.concatenate([a,g_train])
	b = read_csv(f, usecols=[2])
	b = b.values
	b = b.astype('float32')
	c_train=np.concatenate([b,c_train])
# Normalize 資料
scaler = MinMaxScaler(feature_range=(0, 1))
g_train = scaler.fit_transform(g_train)

# Normalize 資料
scaler = MinMaxScaler(feature_range=(0, 1))
trainX, trainY = createXY(g_train)
trainX = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1 ))

# 建立及訓練 LSTM 模型
model = Sequential()
model.add(LSTM(50, input_shape=(trainX.shape[1],1)))
model.add(Dense(24))
model.compile(loss='mean_squared_error', optimizer='adam')
history=model.fit(trainX, trainY, epochs=20, verbose=2)
model.save("generation_model.h5")

scaler = MinMaxScaler(feature_range=(0, 1))
c_train = scaler.fit_transform(c_train)
# Normalize 資料
scaler = MinMaxScaler(feature_range=(0, 1))
trainX, trainY = createXY(c_train)
trainX = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1 ))
history=model.fit(trainX, trainY, epochs=20, verbose=2)
model.save("consumption_model.h5")