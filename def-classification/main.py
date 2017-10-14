import numpy as np
import tensorflow as tf

import sys
sys.path.append('src')
from dataset_handler import DatasetHandler
from utils import *


batchSize = 50
embDim = 51
hiddenSize = 200
TRAINING_STEPS = 60
NUM_LAYERS = int(sys.argv[1])

datasetHandler = DatasetHandler(batchSize, "out")

vocab_input_size = datasetHandler.vocabSize()
vocab_ouput_size = 1


print(f'Size: vocab input = {vocab_input_size}, vocab output={vocab_ouput_size}')
print('='*40)


x = tf.placeholder(tf.int64, [batchSize, None])
y = tf.placeholder(tf.float32, [batchSize])
lengths = tf.placeholder(tf.int32, [batchSize])
embeddings = tf.get_variable("embedding", [vocab_input_size, embDim])
embedding_layer = tf.nn.embedding_lookup(embeddings, x)

cell = tf.contrib.rnn.MultiRNNCell(cells=[build_inner_cell(hiddenSize) for _ in range(0, NUM_LAYERS)], state_is_tuple=True)

# dynamic rnn
output, state = tf.nn.dynamic_rnn(cell, embedding_layer, sequence_length=lengths, initial_state=cell.zero_state(batchSize, tf.float32))

# apply slice to ouput
output = extract_axis(output, lengths - 1)

softmax_w = tf.get_variable("softmax_w", [hiddenSize, vocab_ouput_size], dtype=tf.float32)
softmax_b = tf.get_variable("softmax_b", [vocab_ouput_size], dtype=tf.float32)
logits = tf.matmul(output, softmax_w) + softmax_b

predictions = tf.sigmoid(logits)
predictions = tf.reshape(predictions, [-1])
loss = tf.losses.mean_squared_error(
    y,
    predictions,
    weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
)
tf.summary.scalar("loss", loss)

correct_prediction = tf.equal(y, tf.round(predictions))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)

optimizer = tf.train.AdamOptimizer().minimize(loss)
tvars = tf.trainable_variables()



def train():
  x_train, y_train = datasetHandler.nextTrainBatch()
  train_lengths =[ getRealLength(ls, datasetHandler.eos()) for ls in x_train]
  res = sess.run([optimizer, tvars, loss, accuracy, merged],
    feed_dict={ x: np.array(x_train),
                y: np.array(y_train),
                lengths: train_lengths
                } )
  # print(f'train: loss={res[2]}, accuracy={res[3]}')

  return res[4]

def test():
  x_test, y_test = datasetHandler.nextTestBatch()
  test_lengths =[ getRealLength(ls, datasetHandler.eos()) for ls in x_test]
  res = sess.run([optimizer, tvars, loss, accuracy, merged],
    feed_dict={ x: np.array(x_test),
                y: np.array(y_test),
                lengths: test_lengths
                } )
  # print(f'test: loss={res[2]}, accuracy={res[3]}')

  return res[4]


_start = 0
with tf.Session() as sess:
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter( 'summaries/train', sess.graph)
  test_writer = tf.summary.FileWriter('summaries/test')

  sess.run(tf.global_variables_initializer())

  while _start < TRAINING_STEPS:
    _start += 1
    print(f'\n{_start}/{TRAINING_STEPS}')
    summary = train()
    train_writer.add_summary(summary, _start)

    if _start%10 == 0:
      while datasetHandler.hasNextTestBatch():
        summary = test()
        test_writer.add_summary(summary, _start)
