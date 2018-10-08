import matplotlib.pyplot as plt
import csv
import numpy as np

from sklearn.preprocessing import MinMaxScaler

def create_dataset(signal_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal_data) - look_back):
        dataX.append(signal_data[i:(i + look_back), 0])
        dataY.append(signal_data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

x=[]
signal_data=[]
c=[]
f= open("test.csv", 'r')
row = csv.reader(f)
for line in row:
    data = [float(line[0]), float(line[1])]
    c.append(data)

print(c)
look_back = 20
scaler = MinMaxScaler(feature_range=(0, 1))
signal_data = scaler.fit_transform(c)
test = signal_data[0:100]
x_test, y_test = create_dataset(test, look_back)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

print(x_test)
