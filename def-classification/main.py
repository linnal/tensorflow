from dataset_gen import DatasetGen
from dataset_proc import DatasetProc
import numpy as np
import tensorflow as tf

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



datasetGen = DatasetGen(ls, batchSize)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for i in range(0,TRAINING_STEPS):
    x_train, y_train = datasetGen.nextTrainBatch()
    print(f'xtrain=({len(x_train)},{len(x_train[0])}), ytrain=({len(y_train)})')
    res = sess.run([embedding_layer],
      feed_dict={ x: np.array(x_train), y: np.array(y_train) } )

