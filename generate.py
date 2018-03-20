import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

'''
Tuan Anh's Tensorflow book
use: generate a time sequence
'''

#create sequence data
X_train = np.sin(np.arange(1000)/10)


n_steps = 50
n_inputs = 1
n_neurons = 100
lr = 1e-3
n_epochs = 1000


X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
Y = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
states, fin_state = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
outputs = tf.layers.dense(states,1)
loss = tf.reduce_mean(tf.square(outputs-Y))
optimizer = tf.train.AdamOptimizer(lr)
training_op = optimizer.minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        #create batch
        start_i = random.randint(0,49)
        X_batch = np.reshape(X_train[start_i:start_i+900],[-1,n_steps,n_inputs])
        Y_batch = np.reshape(X_train[start_i+1:start_i+900+1],[-1,n_steps, n_inputs])

        _train, _loss = sess.run([training_op, loss],feed_dict={X:X_batch, Y:Y_batch})

        # writer = tf.summary.FileWriter("./logs/test", sess.graph)
        # sess.run(writer, feed_dict={X: X_batch,seq_length:seq_length_batch})
        if(epoch%100==0):
            print("epoch: %d   MSE: %f" %(epoch, _loss))

    #generate sequence
    seq = [0.]*n_steps
    for i in range(500):
        X_batch = np.array(seq[-n_steps:]).reshape(1,n_steps,n_inputs)
        X_gen = sess.run(outputs, feed_dict={X:X_batch})
        seq.append(X_gen[0,-1,0])

    plt.plot(seq)
    plt.plot(np.arange(550)+50, np.sin(np.arange(550) / 10))
    plt.show()

