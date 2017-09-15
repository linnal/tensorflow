from datasetProcessor import DatasetProcessor
import numpy as np
import tensorflow as tf


dp = DatasetProcessor()
x_train, y_train, x_test, y_test = dp.perpareDataset()
print(f'xtrain {len(x_train)}, {len(x_train[0])}, ytrain {len(x_test)}, {len(x_test[0])}')
print(f'vocab_word={len(dp.vocab_word.keys())}, vocab_tag={len(dp.vocab_tag.keys())}')
# print([len(x) for x in xtrain])
print("="*30)

batch_size = 100
timestep = 265
emb_dim = 128
hidden_size = 200
vocab_input_size = len(dp.vocab_word.keys())
vocab_ouput = len(dp.vocab_tag.keys())
training_samples_size = len(x_train) // batch_size
test_samples_size = len(x_test) // batch_size
TRAINING_STEPS = 6

counterTrain=0
def getBatchData(counterTrain, batch_size):
  start = counterTrain*batch_size
  end = start + batch_size
  x_data = x_train[start:end]
  y_data = y_train[start:end]

  return np.array(x_data), np.array(y_data) #100x256

counterTest=0
def getTestBatchData(counterTrain, batch_size):
  start = counterTrain*batch_size
  end = start + batch_size
  x_test_data = x_train[start:end]
  y_test_data = y_train[start:end]

  return np.array(x_test_data), np.array(y_test_data) #100x256





x = tf.placeholder(tf.int32, [batch_size, timestep]) #100x256
y = tf.placeholder(tf.int32, [batch_size, timestep]) #100x256

embeddings = tf.get_variable("embedding", [vocab_input_size, emb_dim]) #6428x128
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
softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_ouput], dtype=tf.float32)
softmax_b = tf.get_variable("softmax_b", [vocab_ouput], dtype=tf.float32)
logits = tf.matmul(output, softmax_w) + softmax_b
# print(f' logits {logits}')
logits = tf.reshape(logits, [batch_size, timestep, vocab_ouput])

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

  for training_step in range(TRAINING_STEPS):
    while counterTrain < training_samples_size:
      training_step_loss = 0
      x_data, y_data  = getBatchData(counterTrain, batch_size)
      counterTrain += 1

      res = sess.run([embedding_layer, optimizer, loss, cost, tvars], feed_dict={x: x_data, y: y_data})

    training_step_loss += res[3]
    print(f'training_step: {training_step}*10 completed out of {TRAINING_STEPS}*{training_samples_size} with loss {training_step_loss}')

    #test
    while counterTest < test_samples_size:
      x_test_data, y_test_data = getTestBatchData(counterTest, batch_size)
      test_loss = sess.run([loss, cost], feed_dict={x: x_test_data, y: y_test_data})
      print(f'testing_step {counterTest} out of {test_samples_size} with loss cost: {test_loss[1]}')
      counterTest += 1

    counterTrain = 0
    counterTest = 0
    x_train, y_train, x_test, y_test = dp.perpareDataset()


