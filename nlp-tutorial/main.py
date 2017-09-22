from datasetProcessor import DatasetProcessor
import numpy as np
import tensorflow as tf
import sys

dp = DatasetProcessor()
x_train, y_train, x_test, y_test = dp.perpareDataset()
# print([len(x) for x in xtrain])

batch_size = 100
emb_dim = 51
hidden_size = 200
vocab_input_size = len(dp.vocab_word.keys())
vocab_ouput = len(dp.vocab_tag.keys())
TRAINING_STEPS = 100


def fillMissingData():
  global x_train, y_train, x_test, y_test

  train_extra = len(x_train)%batch_size
  if train_extra > 0:
    extra_xtrain, extra_ytrain = dp.getExtraTrainData(batch_size - train_extra)
  test_extra = len(x_test)%batch_size
  if test_extra > 0:
    extra_xtest, extra_ytest = dp.getExtraTestData(batch_size - test_extra)

  x_train += extra_xtrain
  y_train += extra_ytrain
  x_test += extra_xtest
  y_test += extra_ytest


fillMissingData()

timestep = len(x_train[0])

training_samples_size = len(x_train) // batch_size
test_samples_size = len(x_test) // batch_size

print(f'xtrain {len(x_train)}, {len(x_train[0])}, ytrain {len(x_test)}, {len(x_test[0])}')
print(f'vocab_word={len(dp.vocab_word.keys())}, vocab_tag={len(dp.vocab_tag.keys())}')
print("="*30)

counterTrain=0
def getBatchData(counterTrain, batch_size):
  start = counterTrain*batch_size
  end = start + batch_size
  x_data = x_train[start:end]
  y_data = y_train[start:end]

  return x_data, y_data #100x256

counterTest=0
def getTestBatchData(counterTrain, batch_size):
  start = counterTrain*batch_size
  end = start + batch_size
  x_test_data = x_test[start:end]
  y_test_data = y_test[start:end]

  return x_test_data, y_test_data #100x256

def getIndexOfFirstEOS(ls):
  translatedLs = dp.translate(ls)
  for i, tag in enumerate(translatedLs):
    if tag == 'EOS':
      break
  return i

def truncateEOS(ls):
  i= getIndexOfFirstEOS(ls)
  return ls[:i+1]

def createWeightMask(data):
  mask = []
  for ls in data:
    i= getIndexOfFirstEOS(ls)
    res = [1.0]*i + [0.0]*(len(ls)-i)
    mask.append(res)
  return mask

def calculateAccuracy(prediction, gold):
  accNum = 0
  accDenom = 1
  for i, ls in enumerate(gold):
    truncatedLs = truncateEOS(ls)
    countCorrect = 0
    for j, nr in enumerate(truncatedLs):
      if prediction[i][j] == truncatedLs[j]:
        countCorrect += 1
    accNum += countCorrect
    accDenom += len(truncatedLs)
  return accNum/accDenom

def _build_inner_cell():
  return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)

x = tf.placeholder(tf.int64, [batch_size, timestep]) #100x256
y = tf.placeholder(tf.int64, [batch_size, timestep]) #100x256
weight = tf.placeholder(tf.float32, [batch_size, timestep]) #100x256

embeddings = tf.get_variable("embedding", [vocab_input_size, emb_dim]) #6428x128
embedding_layer = tf.nn.embedding_lookup(embeddings, x)

NUM_LAYERS = int(sys.argv[1])
cell = tf.contrib.rnn.MultiRNNCell(cells=[_build_inner_cell() for _ in range(0, NUM_LAYERS)], state_is_tuple=True)

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
    weight,
    average_across_timesteps=True,
    average_across_batch=True
)

optimizer = tf.train.AdamOptimizer().minimize(loss)
tvars = tf.trainable_variables()

predictions = tf.argmax(logits, -1)

first_sentence_pred = tf.gather(predictions, 0)
first_sentence_tag = tf.gather(y, 0)
first_sentence_word = tf.gather(x, 0)


#train
with tf.Session() as sess:
  print("global_variables_initializer")
  sess.run(tf.global_variables_initializer())
  # print(logits.shape)

  for training_step in range(TRAINING_STEPS):

    while counterTrain < training_samples_size:
      training_step_loss = 0
      x_data, y_data  = getBatchData(counterTrain, batch_size)
      counterTrain += 1

      res = sess.run([embedding_layer, #0
                      optimizer, #1
                      loss, #2
                      tvars, #3
                      first_sentence_word, #4
                      first_sentence_tag, #5
                      first_sentence_pred, #6
                      predictions], #7
                      feed_dict={x: np.array(x_data), y: np.array(y_data), weight: createWeightMask(y_data)})

    training_step_loss += res[2]
    accuracy =  calculateAccuracy(res[7], y_data)

    print('='*30 + 'TRAINING' + '='*30)
    print(f'training_step: {training_step}*10 completed out of {TRAINING_STEPS}*{training_samples_size} with loss {training_step_loss} with accuracy={accuracy}')
    print( [x for x in list(zip(dp.translate(res[4], 'word'), dp.translate(res[5]), dp.translate(res[6])))] )

    #test
    while counterTest < test_samples_size:
      x_test_data, y_test_data = getTestBatchData( counterTest,  batch_size)
      res = sess.run([loss, #0
                      predictions, #1
                      first_sentence_word, #2
                      first_sentence_tag, #3
                      first_sentence_pred], #4
                      feed_dict={x: np.array(x_test_data), y: np.array(y_test_data), weight: createWeightMask(y_data)})

      accuracy =  calculateAccuracy(res[1], y_test_data)
      print('='*30 + 'TEST' + '='*30)
      print(f'testing_step {counterTest} out of {test_samples_size} with loss cost: {res[0]} with accuracy={accuracy}')
      print( [x for x in list(zip(dp.translate(res[2], 'word'), dp.translate(res[3]), dp.translate(res[4])))] )

      counterTest += 1

      # stamp one of the array with its tag version and its predicted version
      # x/y/logits


    counterTrain = 0
    counterTest = 0
    x_train, y_train, x_test, y_test = dp.perpareDataset()
    fillMissingData()
