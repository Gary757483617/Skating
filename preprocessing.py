import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


test=pd.read_csv("figure_skating/training data.csv")
y_test=test['TES'][360:]
y_test=np.reshape(y_test,(-1,1))
ids=test['id']
x_test=[]
for _id in ids[360:]:
    test_case=np.load("figure_skating/c3d_feat/"+str(_id)+".npy")
    frames=test_case.shape[0]
    test_case=test_case[(frames>>1)-248:(frames>>1)+248,:]
    x_test.append(test_case)


# 1.define variables/ create placeholders
d=4096
d1=1024
d2=40
batch_size=8

x_batch=tf.placeholder(dtype=tf.float32,shape=(None,496,d),name='x_batch')
y_batch=tf.placeholder(dtype=tf.float32,shape=(None,1),name='y_batch')

# 2. weights
w_s1=tf.Variable(initial_value=np.random.normal(size=(batch_size,d1,d)),dtype=tf.float32,trainable=True)
w_s2=tf.Variable(initial_value=np.random.normal(size=(batch_size,d2,d1)),dtype=tf.float32,trainable=True)

tmp=tf.sigmoid(tf.matmul(w_s1,x_batch,transpose_b=True))
tmp2=tf.sigmoid(tf.matmul(w_s2,tmp))
out=tf.matmul(tmp2,x_batch)

# 3. Build the model
def build_model():
    model=Sequential()
    model.add(LSTM(units=256))
    model.add(Dense(units=64))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1))
    return model

model=build_model()
y_pred=model(out)

# 4. loss function
loss=tf.losses.mean_squared_error(y_batch,y_pred)

# 5. optimizer
optimizer=tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

def sample_data_batch(x_train,y_train,b_size):
    idxs = np.random.choice(len(x_train), size=b_size,replace=False)
    train_sample = []
    label_sample = []
    for idx in idxs:
        train_sample.append(x_train[idx])
        label_sample.append(y_train[idx])
    train_sample=np.array(train_sample).reshape((-1,496,d))
    label_sample=np.array(label_sample).reshape((-1,1))
    # train_sample=tf.convert_to_tensor(train_sample)
    return train_sample, label_sample

epoch_num=64
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch_num):
        x_train_batch, y_train_batch = sample_data_batch(x_test, y_test, b_size=batch_size)
        feed_dict={x_batch:x_train_batch,y_batch:y_train_batch}
        _,Loss,w2=sess.run([optimizer,loss,w_s2],feed_dict)

        if i%2==0:
            print("Epoch {}: mse={}".format(i,Loss))
        if i%16==0:
            print("w2:",w2)

    feed_dict={x_batch:x_test[:8],y_batch:y_test[:8]}
    x_test_final=sess.run(out,feed_dict)
    np.save("test_data 46.npy",x_test_final)
    feed_dict = {x_batch: x_test[8:16], y_batch: y_test[8:16]}
    x_test_final = sess.run(out, feed_dict)
    np.save("test_data 47.npy", x_test_final)
    feed_dict = {x_batch: x_test[16:24], y_batch: y_test[16:24]}
    x_test_final = sess.run(out, feed_dict)
    np.save("test_data 48.npy", x_test_final)
    feed_dict = {x_batch: x_test[24:32], y_batch: y_test[24:32]}
    x_test_final = sess.run(out, feed_dict)
    np.save("test_data 49.npy", x_test_final)
    feed_dict = {x_batch: x_test[32:], y_batch: y_test[32:]}
    x_test_final = sess.run(out, feed_dict)
    np.save("test_data 50.npy", x_test_final)


# train_converted_array=np.load("test_data 1.npy")
# for i in range(2,51):
#     tmp=np.load("test_data "+str(i)+".npy")
#     train_converted_array=np.vstack([train_converted_array,tmp])
# np.save("test_converted_data.npy",train_converted_array)
