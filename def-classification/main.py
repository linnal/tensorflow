from dataset_gen import DatasetGen
from dataset_proc import DatasetProc
import numpy as np
import tensorflow as tf
from utils import *
import sys

batchSize = 50
embDim = 51
hiddenSize = 200
TRAINING_STEPS = 50

datasetProc = DatasetProc()
ls = datasetProc.getData()

vocab_input_size = datasetProc.vocabSize()
vocab_ouput_size = 1
timestep = datasetProc.maxSentenceLen

print(f'len of dataset={len(ls)}')
print(f'Size: vocab input = {vocab_input_size}, vocab output={vocab_ouput_size}')
print(f'timestp= {timestep}')
print('='*40)


x = tf.placeholder(tf.int64, [batchSize, timestep])
y = tf.placeholder(tf.float32, [batchSize])
lengths = tf.placeholder(tf.int32, [batchSize])
embeddings = tf.get_variable("embedding", [vocab_input_size, embDim])
embedding_layer = tf.nn.embedding_lookup(embeddings, x)

NUM_LAYERS = int(sys.argv[1])
cell = tf.contrib.rnn.MultiRNNCell(cells=[build_inner_cell(hiddenSize) for _ in range(0, NUM_LAYERS)], state_is_tuple=True)


outputs = []
state = cell.zero_state(batchSize, tf.float32)
with tf.variable_scope("RNN"):
  for i in range(timestep):
    (cell_output, state) = cell(embedding_layer[:, i, :], state)
    outputs.append(cell_output)
output = tf.stack(axis=1, values=outputs)

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

correct_prediction = tf.equal(y, predictions)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer().minimize(loss)
tvars = tf.trainable_variables()



datasetGen = DatasetGen(ls, batchSize)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for i in range(0,TRAINING_STEPS):
    x_train, y_train = datasetGen.nextTrainBatch()
    train_lengths =[ getRealLength(ls, datasetProc.eos()) for ls in x_train]
    print(f'xtrain=({len(x_train)},{len(x_train[0])}), ytrain=({len(y_train)})')
    res = sess.run([optimizer, tvars, loss, accuracy],
      feed_dict={ x: np.array(x_train),
                  y: np.array(y_train),
                  lengths: train_lengths
                  } )
    print(f'loss={res[2]}, accuracy={res[3]}')

