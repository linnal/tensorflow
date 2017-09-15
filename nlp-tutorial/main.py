from datasetProcessor import DatasetProcessor
import numpy as np
import tensorflow as tf


dp = DatasetProcessor()
x_train, y_train, x_test, y_test = dp.perpareDataset()
print(len(x_train), len(x_train[0]), len(x_test), len(x_test[0]))


batch_size = 100
timestep = 265
emb_dim = 4
hidden_size = 200
vocab_size = len(dp.vocab.keys())
training_samples_size = len(x_train) // batch_size

counter=0
def getBatchData(counter, batch_size):
  start = counter*batch_size
  end = start + batch_size
  x_data = x_train[start:end]
  y_data = y_train[start:end]
  return np.array(x_data), np.array(y_data)

x = tf.placeholder(tf.int32, [batch_size, timestep])
y = tf.placeholder(tf.int32, [batch_size, timestep])

embeddings = tf.get_variable("embedding", [vocab_size, emb_dim])
embedding_layer = tf.nn.embedding_lookup(embeddings, x)

cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
outputs = []
state = cell.zero_state(batch_size, tf.float32)

with tf.variable_scope("RNN"):
  for i in range(timestep):
    (cell_output, state) = cell(embedding_layer[:, i, :], state)
    outputs.append(cell_output)
output = tf.stack(axis=1, values=outputs)

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

cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer().minimize(cost)
tvars = tf.trainable_variables()
saver = tf.train.Saver()




#train
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for training_step in range(training_samples_size):
    training_step_loss = 0
    x_data, y_data = getBatchData(counter, batch_size)
    counter += 1

    res = sess.run([embedding_layer, optimizer, loss, cost, tvars], feed_dict={x: x_data, y: y_data})

    if training_step % 100 == 0:
      training_step_loss += res[3]
      checkpoint = saver.save(sess, './training_model', global_step=training_step)
      print(f'training_step: {training_step} completed out of {training_samples_size} with loss {training_step_loss}')

