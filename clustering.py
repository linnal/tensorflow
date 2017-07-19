import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main():
  k = 3
  conjunto_puntos = generateRandomData()
  vectors = tf.constant(conjunto_puntos)  # N,2
  centroides = getKrandomCentroids(k, vectors)  # 4,2
  # print(vectors.get_shape())

  #tf.expand_dims inserts a dimension into a tensor in the one given in the argument
  #The aim is to extend both tensors from 2 dimensions to 3 dimensions to make the sizes match in order to perform a subtraction later
  expanded_vectors = tf.expand_dims(vectors, 0)  # 1,N,2
  expanded_centroides = tf.expand_dims(centroides, 1)  # 4,1,2

  #print(tf.subtract(expanded_vectors, expanded_centroides))  # 4,N,2
  #Squared Euclidean Distance
  distance = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroides)), 2)  # 4,N
  assignments = tf.argmin(distance, 0)  # N

  means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)),[1,-1])), reduction_indices=[1]) for c in range(k)], 0) # 4 2

  update_centroides = tf.assign(centroides, means) # 4 2; update centroids
  init_op = tf.initialize_all_variables()

  sess = tf.Session()
  sess.run(init_op)

  for step in range(200):
    _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])

  data = {"x": [], "y": [], "cluster": []}

  for i in range(len(assignment_values)):
    data["x"].append(conjunto_puntos[i][0])
    data["y"].append(conjunto_puntos[i][1])
    data["cluster"].append(assignment_values[i])

  plotData(data)



def getKrandomCentroids(k, vectors):
  return tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))

def plotData(data):
  df = pd.DataFrame(data)
  sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster",legend=False)
  plt.show()

def generateRandomData():
  num_puntos = 2000
  conjunto_puntos = []
  for i in range(num_puntos):
    if np.random.random() > 0.5:
      conjunto_puntos.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
      conjunto_puntos.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])
  return conjunto_puntos

if __name__ == '__main__':
  main()
