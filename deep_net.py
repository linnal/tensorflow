import tensorflow as tf
from tensorflow.examples.tutorias.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10
batch_size = 100 #batches of 100 of features

x = tf.placeholder('float', [None, 784]) # 28*28 width*height of the input image
x = tf.placeholder('float')

def createHiddenLayer(previous, current):
  return {'weights': tf.Variable(tf.random_normal([previous, current])),
          'biases': tf.Variable(tf.random_normal(current))}

def setRelu(previous, layer):
  l1 = tf.add(tf.matmul(previous, layer['weights']), layer['biases'])
  return tf.nn.relu(l1)

def neural_network_model(data):
  hidden_1_layer = createHiddenLayer(784, n_nodes_hl1)
  hidden_2_layer = createHiddenLayer(n_nodes_hl1, n_nodes_hl2)
  hidden_3_layer = createHiddenLayer(n_nodes_hl2, n_nodes_hl3)
  output_layer = createHiddenLayer(n_nodes_hl3, n_classes)

  l1 = setRelu(data, hidden_1_layer)
  l2 = setRelu(l1, hidden_2_layer)
  l3 = setRelu(l2, hidden_3_layer)

  output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

  return output

def train_neural_network(x):
  prediction = neural_network_model(x)
  cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y))
  # minimize the cost
  optimizer = tf.train.AdamOptimizer().minimize(cost)
  epochs = 10 # cycles feed forward + backprop

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(epochs):
      epoch_loss = 0
      for _ in range(mnist.train.num_examples // batch_size):
        x, y = mnist.train.next_batch(batch_size)
        _, c = sess.run([optimizer, cost], feed_dict={x: x, y: y})
        epoch_loss += c

      print(f'epoch: {epoch} completed out of {epochs} with loss {epoch_loss}')

    correct = tf.eqaul(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    accuracy_eval = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    print(f'Accuracy is {accuracy_eval}')

train_neural_network(x)
