import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

x_train=np.load("figure_skating/S-LSTM/S_LSTM.npy")
train=pd.read_csv("training data.csv")
y_train=train['TES']

# 1.define variables/ create placeholders
d=4096
d2=40
x_batch=tf.placeholder(dtype=tf.float32,shape=(None,d2,d),name='x_batch')
y_batch=tf.placeholder(dtype=tf.float32,shape=(None,1),name='y_batch')

# 2. weights


# 3. Build the model
def build_model():
    model=Sequential()
    model.add(LSTM(units=256))
    model.add(Dense(units=64))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1))
    return model

# 4. loss function
def sample_data_batch(x_train,y_train,b_size):
    idxs = np.random.choice(len(x_train), size=b_size,replace=False)
    train_sample = []
    label_sample = []
    for idx in idxs:
        train_sample.append(x_train[idx])
        label_sample.append(y_train[idx])
    train_sample=np.array(train_sample).reshape((-1,d2,d))
    label_sample=np.array(label_sample).reshape((-1,1))
    return train_sample, label_sample

model=build_model()
y_pred=model(x_batch)
loss=tf.losses.mean_squared_error(y_batch,y_pred)

# 5. optimizer
optimizer=tf.train.AdamOptimizer().minimize(loss)


epoch_num=250
loss_cache=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch_num):
        x_train_batch,y_train_batch=sample_data_batch(x_train,y_train,b_size=32)
        feed_dict={x_batch:x_train_batch,y_batch:y_train_batch}
        _,Loss=sess.run([optimizer,loss],feed_dict)

        if i%10==0:
            print("Epoch {}: mse={}".format(i,Loss))
            loss_cache.append(Loss)

    model.save("S-LSTM model.h5")

plt.plot(range(0,250,10),loss_cache)
plt.show()