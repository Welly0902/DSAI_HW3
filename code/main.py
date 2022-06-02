# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()
def output(path, data):
    

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return
if __name__ == '__main__':
	args = config()
	import tensorflow as tf
	from turtle import shape
	import numpy as np
	from pandas import read_csv
	from sklearn.preprocessing import MinMaxScaler
	from datetime import datetime, timedelta


	from numpy.random import seed
	from numpy.random import randint
	import pandas as pd
	# 載入測試資料集
	g_test = read_csv(args.generation, usecols=[1])
	g_test = g_test.values
	g_test = g_test.astype('float32')
	c_test = read_csv(args.consumption, usecols=[1])
	c_test = c_test.values
	c_test = c_test.astype('float32')
	# Normalize 資料
	scaler = MinMaxScaler(feature_range=(0, 1))
	g_test = scaler.fit_transform(g_test)
	#載入generation模型
	g_model = tf.keras.models.load_model('generation_model.h5')
	# Normalize 資料
	scaler = MinMaxScaler(feature_range=(0, 1))
	c_test = scaler.fit_transform(c_test)
	#載入consumption模型
	c_model = tf.keras.models.load_model('consumption_model.h5')

	g_test=np.reshape(g_test, (1, g_test.shape[0],1))
	g_testPredict = g_model.predict(g_test)
	g_testPredict=scaler.inverse_transform(g_testPredict)
	c_test=np.reshape(c_test, (1, c_test.shape[0],1))
	c_testPredict = c_model.predict(c_test)
	c_testPredict=scaler.inverse_transform(c_testPredict)

	# print(g_testPredict)
	# print(c_testPredict)

	df = pd.DataFrame()
	df['g']=pd.DataFrame(g_testPredict[0])
	df['c']=pd.DataFrame(c_testPredict[0])
	
	gn = df['g'].sum()
	# print(gn)
	cn = df['c'].sum()
	# print(cn)
	tn = round(cn-gn,2)	
	tn2= round((tn*1.2)/24,2)
	# print(tn)

	getday=read_csv(args.consumption, usecols=[0])
	# getday['time'] = getday['time'].replace("/", "-", regex=True)
	day=getday['time'][getday.shape[0]-24]
	# print(day)
	# day = datetime.strptime(day, '%Y/%m/%d %H:%M')
	day = datetime.strptime(day, '%Y-%m-%d %H:%M:%S')
	delta = timedelta(days=1)
	day2 = day + delta
	# print(day2)
	seed(5)
	r = randint(-10, 10, 20)

	data=[]
	if tn>=0:
		for i in range(g_testPredict.shape[1]):
			insert=(day2,"buy",2.51,tn2+(randint(-10, 10)/100))
			data.append(list(insert))
			delta = timedelta(hours=1)
			day2 = day2 + delta
	elif tn<0:
		for i in range(g_testPredict.shape[1]):
			insert=(day2,"sell",0.01,tn2+(randint(-10, 10)/100))
			data.append(list(insert))
			delta = timedelta(hours=1)
			day2 = day2 + delta
	output(args.output, data)
