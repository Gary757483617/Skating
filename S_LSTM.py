import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, regularizers
import matplotlib.pyplot as plt

d=4096
d1=1024
d2=40

x_train=np.load("figure_skating/train_converted_data.npy")
train=pd.read_csv("figure_skating/training data.csv")
y_train=train['TES']

x_test=np.load("figure_skating/test_converted_data.npy")
test=pd.read_csv("figure_skating/testing data.csv")
y_test=test['TES']
y_test=np.array(y_test).reshape((-1,1))


# 1.define variables/ create placeholders
x_batch=tf.placeholder(dtype=tf.float32,shape=(None,d2,d),name='x_batch')
y_batch=tf.placeholder(dtype=tf.float32,shape=(None,1),name='y_batch')

# 2. weights


# 3. Build the model
def build_model():
    model=Sequential()
    model.add(LSTM(units=256))
    model.add(Dense(units=64,kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1))

    return model

model=build_model()
y_pred=model(x_batch)

# 4.Loss function
loss=tf.losses.mean_squared_error(y_batch,y_pred)
# 5. optimizer
opt=tf.train.AdamOptimizer(learning_rate=2.0e-4).minimize(loss)


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


epoch_num=200
train_loss_cache=[]
test_loss_cache=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch_num):
        x_train_batch,y_train_batch=sample_data_batch(x_train,y_train,b_size=16)
        feed_dict={x_batch:x_train_batch,y_batch:y_train_batch}
        _,Loss=sess.run([opt,loss],feed_dict)

        if i%5==0:
            print("Epoch {}: mse={}".format(i,Loss))
            train_loss_cache.append(Loss)

            feed_dict = {x_batch: x_test, y_batch: y_test}
            Loss=sess.run(loss,feed_dict)
            test_loss_cache.append(Loss)
            print("test loss:",Loss)

    model.save("S-LSTM model.h5")

plt.plot(range(0,200,5),train_loss_cache,'r-')
plt.plot(range(0,200,5),test_loss_cache,'g-')
plt.legend(['train_mse','test_mse'])
plt.title("S-LSTM: TES")
plt.show()
