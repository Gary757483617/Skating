import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.layers import conv1d, dropout, fully_connected, \
    l1_regularizer, l2_regularizer, batch_norm
import matplotlib.pyplot as plt
from random import shuffle
from skip_rnn_cells import SkipLSTMCell
from keras.layers import RNN

d = 4096
d1 = 50
d2 = 300
batch_size = 100

train = pd.read_csv("figure_skating/training data.csv")
y_train = train['PCS']
x_train = []
for i in train['id']:
    c = np.load("figure_skating/c3d_feat/" + str(i) + ".npy")
    frames = c.shape[0]
    c = c[(frames >> 1) - 150:(frames >> 1) + 150, :]
    x_train.append(c)
y_train = np.array(y_train).reshape((-1, 1))
print("******Finished step 1******")

test = pd.read_csv("figure_skating/testing data.csv")
y_test = test['PCS']
x_test = []
for i in test['id']:
    c = np.load("figure_skating/c3d_feat/" + str(i) + ".npy")
    frames = c.shape[0]
    c = c[(frames >> 1) - 150:(frames >> 1) + 150, :]
    x_test.append(c)
x_test = np.array(x_test).reshape((-1, d2, d))
y_test = np.array(y_test).reshape((-1, 1))
print("******Finished step 2******")

# 1. define placeholders
x_batch = tf.placeholder(dtype=tf.float32, shape=(None, d2, d), name='x_batch')
y_batch = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y_batch')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')
global_step = tf.placeholder(dtype=tf.int32, name='global_steps')

# 2. weights
w_s1_0 = tf.get_variable(name='w_s1_0', shape=(1, 128, 64), dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(), trainable=True)
w_s2_0 = tf.get_variable(name='w_s2_0', shape=(1, 64, d1), dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(), trainable=True)
w_s1 = tf.tile(w_s1_0, multiples=[batch_size, 1, 1], name='w_s1')
w_s2 = tf.tile(w_s2_0, multiples=[batch_size, 1, 1], name='w_s2')

# 3. build the model
conv_output = conv1d(inputs=x_batch, num_outputs=128, kernel_size=1, stride=1)  # (batch_size,300,128)
conv_output= batch_norm(conv_output,is_training=is_training)

# Attention model
tmp = tf.tanh(tf.matmul(conv_output, w_s1))
A = tf.sigmoid(tf.matmul(tmp,w_s2))
M = tf.matmul(A, conv_output,transpose_a=True)
p = tf.norm(tf.matmul(A, A, transpose_b=True) - tf.eye(num_rows=d2), ord=2)
penalty = p * p / batch_size

# S-LSTM
with tf.variable_scope("S-LSTM", reuse=tf.AUTO_REUSE):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=256, reuse=tf.AUTO_REUSE)
    initial_state = cell.zero_state(batch_size, dtype=tf.float32)

    # rnn_output, _ = tf.nn.dynamic_rnn(cell, M, dtype=tf.float32, initial_state=initial_state)
    layer=RNN(cell,stateful=True)
    rnn_output=layer(M,initial_state=initial_state)
    # rnn_output = rnn_output[:, -1, :]
    output_s = fully_connected(rnn_output, num_outputs=64)


# M-LSTM (1)
with tf.variable_scope("M-LSTM-1", reuse=tf.AUTO_REUSE):
    cell_1 = SkipLSTMCell(num_units=256)
    # initial_state_1 = cell_1.trainable_initial_state(batch_size=batch_size)

    hidden_1 = conv1d(conv_output, num_outputs=1, kernel_size=2, padding='VALID', stride=1)
    hidden_1_norm = batch_norm(hidden_1,is_training=is_training)
    # rnn_outputs_1, _ = tf.nn.dynamic_rnn(cell_1, hidden_1_norm, dtype=tf.float32, initial_state=initial_state_1)
    layer_1 = RNN(cell_1, stateful=True)
    rnn_outputs_1 = layer_1(M)
    # rnn_outputs_1 = rnn_outputs_1.h[:, -1, :]
    output_1 = fully_connected(rnn_outputs_1, num_outputs=64)

# M-LSTM (2)
with tf.variable_scope("M-LSTM-2", reuse=tf.AUTO_REUSE):
    cell_2 = SkipLSTMCell(num_units=256)
    initial_state_2 = cell_2.trainable_initial_state(batch_size=batch_size)

    hidden_2 = conv1d(conv_output, num_outputs=1, kernel_size=4, padding='VALID', stride=2)
    hidden_2_norm = batch_norm(hidden_2, is_training=is_training)
    # rnn_outputs_2, _ = tf.nn.dynamic_rnn(cell_2, hidden_2_norm, dtype=tf.float32, initial_state=initial_state_2)
    layer_2 = RNN(cell_2, stateful=True)
    rnn_outputs_2 = layer_2(M)
    # rnn_outputs_2 = rnn_outputs_2.h[:, -1, :]
    output_2 = fully_connected(rnn_outputs_2, num_outputs=64)

# M-LSTM (3)
with tf.variable_scope("M-LSTM-3", reuse=tf.AUTO_REUSE):
    cell_3 = SkipLSTMCell(num_units=256)
    initial_state_3= cell_3.trainable_initial_state(batch_size=batch_size)

    hidden_3 = conv1d(conv_output, num_outputs=1, kernel_size=8, padding='VALID', stride=4)
    hidden_3_norm = batch_norm(hidden_3, is_training=is_training)
    # rnn_outputs_3, _ = tf.nn.dynamic_rnn(cell_3, hidden_3_norm, dtype=tf.float32, initial_state=initial_state_3)
    layer_3 = RNN(cell_3, stateful=True)
    rnn_outputs_3 = layer_3(M)
    # rnn_outputs_3 = rnn_outputs_3.h[:, -1, :]
    output_3 = fully_connected(rnn_outputs_3, num_outputs=64)

# concat network
output = tf.concat([output_1, output_2, output_3], axis=1)
hidden_7 = dropout(output, keep_prob=0.3, is_training=is_training)
hidden_8 = fully_connected(hidden_7, num_outputs=64)
y_pred = fully_connected(hidden_8, num_outputs=1, activation_fn=tf.nn.relu)

# 4. loss function
mse = tf.losses.mean_squared_error(y_pred, y_batch)
loss = mse + penalty

# 5. optimizer
# lr=tf.train.exponential_decay(learning_rate=5.0e-4,global_step=global_step,decay_steps=25,
#                               decay_rate=0.8,staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=5.0e-4).minimize(loss)


# get batch
def sample_data_batch(x_train, y_train, b_size):
    idxs = range(400)
    shuffle(idxs)
    train_batches = []
    label_batches = []
    for i in range(4):
        tr_batch = []
        lab_batch = []
        for idx in range(i * b_size, i * b_size + 100):
            tr_batch.append(x_train[idxs[idx]])
            lab_batch.append(y_train[idxs[idx]])
        tr_batch = np.array(tr_batch).reshape((-1, 300, d))
        lab_batch = np.array(lab_batch).reshape((-1, 1))
        train_batches.append(tr_batch)
        label_batches.append(lab_batch)

    return train_batches, label_batches


# training process
train_loss_cache = []
test_loss_cache = []
min_mse = 200
epoch_num = 150
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoch_num):
        x_train_batches, y_train_batches = sample_data_batch(x_train, y_train, batch_size)
        feed_dict = {x_batch: x_train_batches[0], y_batch: y_train_batches[0],
                     is_training: True, global_step: epoch}
        _ = sess.run(optimizer, feed_dict)
        feed_dict = {x_batch: x_train_batches[1], y_batch: y_train_batches[1],
                     is_training: True, global_step: epoch}
        _ = sess.run(optimizer, feed_dict)
        feed_dict = {x_batch: x_train_batches[2], y_batch: y_train_batches[2],
                     is_training: True, global_step: epoch}
        _ = sess.run(optimizer, feed_dict)
        feed_dict = {x_batch: x_train_batches[3], y_batch: y_train_batches[3],
                     is_training: True, global_step: epoch}
        _, MSE = sess.run([optimizer, mse], feed_dict)

        if epoch % 2 == 0:
            print("Epoch {}: mse={}".format(epoch, MSE))
            train_loss_cache.append(MSE)

        if epoch % 4 == 0:
            feed_dict = {x_batch: x_test[:batch_size], y_batch: y_test[:batch_size], is_training: False}
            MSE, prediction = sess.run([mse, y_pred], feed_dict)

            test_loss_cache.append(MSE)
            print("test loss=", MSE)
            min_mse = min(min_mse, MSE)
            # print (prediction)

            # np.save("figure_skating/attention_matrix_"+str(epoch)+".npy", atten_matrix)

print (min_mse)
plt.plot(range(0, epoch_num, 2), train_loss_cache, 'r-')
plt.plot(range(0, epoch_num, 4), test_loss_cache, 'g-')
plt.legend(['train_mse', 'test_mse'])
plt.title("S_LSTM+M_LSTM: PCS")
plt.show()
