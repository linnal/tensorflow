import random
import numpy as np
import tensorflow as tf
from itertools import islice

num_steps = 5
batch_size = 10
num_classes = 2
state_size = 4
learning_rate = 0.1
dimension = 5000
timestep = 7
vocabolary = 20
emb_dim = 4
num_layers = 2
hidden_size = 200

def generate_random_data(dimension, max_number):
  data = []
  y_data = []

  for _ in range(dimension):
    temp_x_data = random.sample(range(0, max_number), max_number)
    temp_y_data = [ 1 if x > p else 0 for (x, p) in zip(temp_x_data, [0]+temp_x_data) ]

    data.append((np.array(temp_x_data), np.array(temp_y_data)))
  return data

# itera i 500 in batch di 200 ciclicamnte e fai shuffle
def generate_batch(data, batch_size):
  i = iter(data)
  piece = list(islice(i, batch_size))
  while piece:
    yield piece
    piece = list(islice(i, batch_size))

    if not piece:
      random.shuffle(data)
      i = iter(data)
      piece = list(islice(i, batch_size))


def generate_data(dimension = 5000, vocabolary_size = 20):
  x_data = []
  y_data = []

  for _ in range(dimension):
    temp_x_data = random.sample(range(0, vocabolary_size), timestep)
    temp_y_data = [ 1 if x > p else 0 for (x, p) in zip(temp_x_data, [0]+temp_x_data) ]

    x_data.append(temp_x_data)
    y_data.append(temp_y_data)

  return np.array(x_data), np.array(y_data)

def lstm_cell():
  return tf.contrib.rnn.BasicLSTMCell(
      hidden_size, forget_bias=0.0, state_is_tuple=True)

def data_type():
  return  tf.float32

x = tf.placeholder(tf.int32, [batch_size, timestep])
y = tf.placeholder(tf.int32, [batch_size, timestep])

embeddings = tf.get_variable("embedding", [vocabolary, 4])
embedding_layer = tf.nn.embedding_lookup(embeddings, x)

x_data, y_data = generate_data(batch_size, vocabolary)

cell = lstm_cell()
outputs = []
state = cell.zero_state(batch_size, data_type())
with tf.variable_scope("RNN"):
  for time_step in range(timestep): # 5 steps
    if time_step > 0:
      tf.get_variable_scope().reuse_variables()
    print(embedding_layer[:, time_step, :])
    (cell_output, state) = cell(embedding_layer[:, time_step, :], state)
    outputs.append(cell_output)

output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, hidden_size])
softmax_w = tf.get_variable(
    "softmax_w", [hidden_size, vocabolary], dtype=data_type())
softmax_b = tf.get_variable("softmax_b", [vocabolary], dtype=data_type())
logits = tf.matmul(output, softmax_w) + softmax_b
print(f' logits {logits}')
logits = tf.reshape(logits, [batch_size, timestep, vocabolary])

loss = tf.contrib.seq2seq.sequence_loss(
    logits,
    y,
    tf.ones([batch_size, timestep], dtype=data_type()),
    average_across_timesteps=False,
    average_across_batch=True
)
cost = tf.reduce_sum(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  a = sess.run([embedding_layer, loss, cost], feed_dict={x: x_data, y: y_data})
  # print(a.shape)
  print(embeddings)
  print(outputs)
  print(loss.shape)
  print(f'loss = {a[1]}')
  print(f'cost = {a[2]}')
