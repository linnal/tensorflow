import random
import numpy as np
import tensorflow as tf
from itertools import islice
import time

batch_size = 100
num_classes = 2
timestep = 7
vocabolary = 20
emb_dim = 4
hidden_size = 200
training_samples_size = 1000


def generate_data(dimension, vocabolary_size):
  x_data = []
  y_data = []

  for _ in range(dimension):
    temp_x_data = random.sample(range(0, vocabolary_size), timestep)
    temp_y_data = [ 1 if x > p else 0 for (x, p) in zip(temp_x_data, [0]+temp_x_data) ]

    x_data.append(temp_x_data)
    y_data.append(temp_y_data)
  return np.array(x_data), np.array(y_data)


x = tf.placeholder(tf.int32, [batch_size, timestep])
y = tf.placeholder(tf.int32, [batch_size, timestep])

embeddings = tf.get_variable("embedding", [vocabolary, emb_dim])
embedding_layer = tf.nn.embedding_lookup(embeddings, x)


cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
outputs = []
state = cell.zero_state(batch_size, tf.float32)
with tf.variable_scope("RNN"):
  for time_step in range(timestep): # 5 steps
    # print(f'timestap {time_step}')
    # print(embedding_layer[:, time_step, :])
    (cell_output, state) = cell(embedding_layer[:, time_step, :], state)
    outputs.append(cell_output)
output = tf.stack(axis=1, values=outputs)
# print(f' ==> {output}')
output = tf.reshape(output, [-1, hidden_size])
softmax_w = tf.get_variable("softmax_w", [hidden_size, 2], dtype=tf.float32)
softmax_b = tf.get_variable("softmax_b", [2], dtype=tf.float32)
logits = tf.matmul(output, softmax_w) + softmax_b
# print(f' logits {logits}')
logits = tf.reshape(logits, [batch_size, timestep,2])

loss = tf.contrib.seq2seq.sequence_loss(
    logits,
    y,
    tf.ones([batch_size, timestep], dtype=tf.float32),
    average_across_timesteps=True,
    average_across_batch=False
)
# print(loss)
cost = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer().minimize(cost)

tvars = tf.trainable_variables()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for epoch in range(training_samples_size):
    epoch_loss = 0
    x_data, y_data = generate_data(batch_size, vocabolary)

    res = sess.run([embedding_layer, optimizer, loss, cost, tvars], feed_dict={x: x_data, y: y_data})

    epoch_loss += res[3]
    print(f'{time.time()} epoch: {epoch} completed out of {training_samples_size} with loss {epoch_loss}')

    for var in tvars:
      print(var.name)

  # print(a.shape)
  # print(embeddings)
  # print(outputs)
  # print(loss.shape)
  # print(f'loss = {a[1]}')
  # print(f'cost = {a[2]}')

  # tvars_vals = a[3]
