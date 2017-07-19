import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10
batch_size = 100 #batches of 100 of features

x = tf.placeholder('float', [None, 784]) # 28*28 width*height of the input image
y = tf.placeholder('float')

def create_hidden_layer(previous, current):
  return {'weights': tf.Variable(tf.random_normal([previous, current])),
          'biases': tf.Variable(tf.random_normal([current]))}

def set_relu(previous, layer):
  l1 = tf.add(tf.matmul(previous, layer['weights']), layer['biases'])
  return tf.nn.relu(l1)

def neural_network_model(data):
  hidden_1_layer = create_hidden_layer(784, n_nodes_hl1)
  hidden_2_layer = create_hidden_layer(n_nodes_hl1, n_nodes_hl2)
  hidden_3_layer = create_hidden_layer(n_nodes_hl2, n_nodes_hl3)
  output_layer = create_hidden_layer(n_nodes_hl3, n_classes)

  l1 = set_relu(data, hidden_1_layer)
  l2 = set_relu(l1, hidden_2_layer)
  l3 = set_relu(l2, hidden_3_layer)

  output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

  return output

def train_neural_network(x):
  prediction = neural_network_model(x)
  cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
  # minimize the cost
  optimizer = tf.train.AdamOptimizer().minimize(cost)
  epochs = 10 # cycles feed forward + backprop

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(epochs):
      epoch_loss = 0
      for _ in range(mnist.train.num_examples // batch_size):
        e_x, e_y = mnist.train.next_batch(batch_size)
        _, c = sess.run([optimizer, cost], feed_dict={x: e_x, y: e_y})
        epoch_loss += c

      print(f'epoch: {epoch} completed out of {epochs} with loss {epoch_loss}')

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    accuracy_eval = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    print(f'Accuracy is {accuracy_eval}')

train_neural_network(x)
