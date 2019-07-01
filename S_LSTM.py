import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, regularizers
from keras.optimizers import Adam
import matplotlib.pyplot as plt

d=4096
d1=1024
d2=40
b_size=16

x_train=np.load("figure_skating/train_converted_data.npy")
train=pd.read_csv("figure_skating/training data.csv")
y_train=train['PCS']
y_train=np.array(y_train).reshape((-1,1))

x_test=np.load("figure_skating/test_converted_data.npy")
test=pd.read_csv("figure_skating/testing data.csv")
y_test=test['PCS']
y_test=np.array(y_test).reshape((-1,1))


# 1.define variables/ create placeholders
x_batch=tf.placeholder(dtype=tf.float32,shape=(None,d2,d),name='x_batch')
y_batch=tf.placeholder(dtype=tf.float32,shape=(None,1),name='y_batch')

# 2. weights

# 3. Build the model
model=Sequential()
model.add(LSTM(units=256))
model.add(Dense(units=64,kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(rate=0.5))
model.add(Dense(units=1))

model.compile(optimizer=Adam(lr=2.0e-4),loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=b_size,epochs=16)
score=model.evaluate(x_test,y_test,batch_size=b_size)
print(score)
