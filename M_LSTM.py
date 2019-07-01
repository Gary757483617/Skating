import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.layers import conv1d, dropout, fully_connected, l2_regularizer
from skip_rnn_cells import SkipLSTMCell
import matplotlib.pyplot as plt


train=pd.read_csv("figure_skating/training data.csv")
training_imgs=train['id']
y_train=train['PCS']
x_train=[]
for i in training_imgs:
    c = np.load("figure_skating/c3d_feat/"+str(i)+".npy")
    frames=c.shape[0]
    c=c[(frames>>1)-248:(frames>>1)+248,:]   # 最短的视频为496 frames, 调整为所有视频(496,4096)
    x_train.append(c)
print("******Finished step 1******")

test=pd.read_csv("figure_skating/testing data.csv")
testing_imgs=test['id']
y_test=test['PCS']
x_test=[]
for i in testing_imgs:
    c = np.load("figure_skating/c3d_feat/"+str(i)+".npy")
    frames=c.shape[0]
    c=c[(frames>>1)-248:(frames>>1)+248,:]
    x_test.append(c)
y_test=np.array(y_test).reshape((-1,1))
print("******Finished step 2******")

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

# 1. define placeholders
d=4096
d2=496
batch_size=20

x=tf.placeholder(dtype=tf.float32,shape=(None,d2,d),name='x_train')
y=tf.placeholder(dtype=tf.float32,shape=(None,1),name='y_true')

# 2. weights

# 3. build the model
# M-LSTM (1)

with tf.variable_scope("M-LSTM-1",reuse=tf.AUTO_REUSE):
    cell_1=SkipLSTMCell(num_units=64)
    initial_state_1 = cell_1.trainable_initial_state(batch_size=batch_size)

    hidden_1=conv1d(x,num_outputs=1,kernel_size=1,padding='VALID',stride=1)
    rnn_outputs_1,_= tf.nn.dynamic_rnn(cell_1, hidden_1, dtype=tf.float32, initial_state=initial_state_1)
    rnn_outputs_1= rnn_outputs_1.h[:,-1,:]
    hidden_2=dropout(inputs=rnn_outputs_1,keep_prob=0.7)
    output_1=fully_connected(hidden_2,num_outputs=32,weights_regularizer=l2_regularizer(0.01))

# M-LSTM (2)
with tf.variable_scope("M-LSTM-2",reuse=tf.AUTO_REUSE):
    cell_2=SkipLSTMCell(num_units=64)
    initial_state_2 = cell_2.trainable_initial_state(batch_size=batch_size)

    hidden_3=conv1d(x,num_outputs=1,kernel_size=4,padding='VALID',stride=2)
    rnn_outputs_2,_= tf.nn.dynamic_rnn(cell_2, hidden_3, dtype=tf.float32,initial_state=initial_state_2)
    rnn_outputs_2= rnn_outputs_2.h[:,-1,:]
    hidden_4=dropout(inputs=rnn_outputs_2,keep_prob=0.7)
    output_2=fully_connected(hidden_4,num_outputs=32,weights_regularizer=l2_regularizer(0.01))


# M-LSTM (3)
with tf.variable_scope("M-LSTM-3",reuse=tf.AUTO_REUSE):
    cell_3=SkipLSTMCell(num_units=64)
    initial_state_3 = cell_3.trainable_initial_state(batch_size=batch_size)

    hidden_5=conv1d(x,num_outputs=1,kernel_size=8,padding='VALID',stride=2)
    rnn_outputs_3,_= tf.nn.dynamic_rnn(cell_3, hidden_5, dtype=tf.float32,initial_state=initial_state_3)
    rnn_outputs_3= rnn_outputs_3.h[:,-1,:]
    hidden_6=dropout(inputs=rnn_outputs_3,keep_prob=0.7)
    output_3=fully_connected(hidden_6,num_outputs=32,weights_regularizer=l2_regularizer(0.01))

# concat network
output=tf.concat([output_1,output_2,output_3],axis=1)
hidden_7=dropout(output,keep_prob=0.7)
hidden_8=fully_connected(hidden_7,num_outputs=32,weights_regularizer=l2_regularizer(0.01))
y_pred=fully_connected(hidden_8,num_outputs=1,activation_fn=tf.nn.relu)


# 4. loss function
loss=tf.losses.mean_squared_error(y,y_pred)
# 5. optimizer
optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

epoch_num=150
train_loss_cache=[]
test_loss_cache=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoch_num):
        x_train_batch,y_train_batch=sample_data_batch(x_train,y_train,batch_size)
        feed_dict = {x: x_train_batch, y: y_train_batch}
        _, Loss= sess.run([optimizer, loss], feed_dict)

        if epoch%5==0:
            print("Epoch {}: mse={}".format(epoch,Loss))
            train_loss_cache.append(Loss)

        if epoch%10==0:
            avg_loss=0
            feed_dict={x:x_test[:20],y:y_test[:20]}
            Loss=sess.run(loss,feed_dict)
            avg_loss+=Loss
            feed_dict = {x: x_test[20:40], y: y_test[20:40]}
            Loss = sess.run(loss, feed_dict)
            avg_loss += Loss
            feed_dict = {x: x_test[40:60], y: y_test[40:60]}
            Loss = sess.run(loss, feed_dict)
            avg_loss += Loss
            feed_dict = {x: x_test[60:80], y: y_test[60:80]}
            Loss = sess.run(loss, feed_dict)
            avg_loss += Loss
            feed_dict = {x: x_test[80:], y: y_test[80:]}
            Loss = sess.run(loss, feed_dict)
            avg_loss += Loss
            avg_loss/=5
            test_loss_cache.append(avg_loss)
            print("test loss=",avg_loss)

plt.plot(range(0,150,5),train_loss_cache,'r-')
plt.plot(range(0,150,10),test_loss_cache,'g-')
plt.legend(['train_mse','test_mse'])
plt.title("M-LSTM: PCS")
plt.show()
