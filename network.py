import tensorflow as tf
import numpy as np

n_inputs = 2
n_steps = 3
n_neurons = 4

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
X_seqs = tf.unstack(tf.transpose(X, perm=[1,0,2]))
basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.nn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
outputs = tf.transpose(tf.stack(output_seqs),perm=[1,0,2])
init = tf.global_variables_initializer()

X_batch = np.random.randn(20,n_steps,n_inputs)
with tf.Session() as sess:
    init.run()
    writer = tf.summary.FileWriter("./logs/test", sess.graph)
    sess.run(writer, feed_dict={X:X_batch})
    _outputs = sess.run(outputs, feed_dict={X:X_batch})
    _states = sess.run(states, feed_dict={X:X_batch})
    print(_outputs)
    print()
    print(_states)
