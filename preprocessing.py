import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

test=pd.read_csv("figure_skating/testing data.csv")
y_test=test['TES'][80:]
ids=test['id']

# 1.define variables/ create placeholders
d=4096
d1=1024
d2=40
x_batch=tf.placeholder(dtype=tf.float32,shape=(None,d2,d),name='x_batch')
y_batch=tf.placeholder(dtype=tf.float32,shape=(None,1),name='y_batch')

# 2. weights
x_test=[]
w_s1=tf.Variable(initial_value=tf.ones(shape=(d1,d)),dtype=tf.float32)
w_s2=tf.Variable(initial_value=tf.ones(shape=(d2,d1)),dtype=tf.float32)
for _id in ids[80:]:
    test_case=np.load("figure_skating/c3d_feat/"+str(_id)+".npy")
    tmp=tf.sigmoid(tf.matmul(w_s1,test_case,transpose_b=True))
    tmp2=tf.sigmoid(tf.matmul(w_s2,tmp))
    out=tf.matmul(tmp2,test_case)
    x_test.append(out)
x_test=tf.convert_to_tensor(x_test,dtype=tf.float32)
y_test=np.reshape(y_test,(-1,1))
print(x_test.shape,y_test.shape)

# 3. Build the model
def build_model():
    model=Sequential()
    model.add(LSTM(units=256))
    model.add(Dense(units=64))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1))
    return model

model=build_model()
y_pred=model(x_batch)

# 4. loss function
loss=tf.losses.mean_squared_error(y_batch,y_pred)

# 5. optimizer
optimizer=tf.train.AdamOptimizer().minimize(loss)

epoch_num=50
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch_num):
        feed_dict={x_batch:x_test.eval(),y_batch:y_test}
        _,Loss=sess.run([optimizer,loss],feed_dict)

        if i%2==0:
            print("Epoch {}: mse={}".format(i,Loss))

    x_test_final=x_test.eval()
    np.save("test_data 5.npy",x_test_final)