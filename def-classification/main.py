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
vocab_ouput_size = 2
timestep = datasetProc.maxSentenceLen

print(f'len of dataset={len(ls)}')
print(f'Size: vocab input = {vocab_input_size}, vocab output={vocab_ouput_size}')
print(f'timestp= {timestep}')
print('='*40)


x = tf.placeholder(tf.int64, [batchSize, timestep])
y = tf.placeholder(tf.int64, [batchSize])
weight = tf.placeholder(tf.float32, [batchSize, timestep])
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

output = tf.reshape(output, [-1, hiddenSize])
softmax_w = tf.get_variable("softmax_w", [hiddenSize, vocab_ouput_size], dtype=tf.float32)
softmax_b = tf.get_variable("softmax_b", [vocab_ouput_size], dtype=tf.float32)
logits = tf.matmul(output, softmax_w) + softmax_b

predictions = tf.sigmoid(logits)
loss = tf.metrics.mean_squared_error(
    y,
    predictions,
    weights=weight,
    metrics_collections=None,
    updates_collections=None,
    name=None
)

datasetGen = DatasetGen(ls, batchSize)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for i in range(0,TRAINING_STEPS):
    x_train, y_train = datasetGen.nextTrainBatch()
    print(f'xtrain=({len(x_train)},{len(x_train[0])}), ytrain=({len(y_train)})')
    res = sess.run([embedding_layer, predictions, loss],
      feed_dict={ x: np.array(x_train), 
      			y: np.array(y_train),
      			weight:  createWeightMask(x_train, datasetProc.eos())} )
    print(res[2])

