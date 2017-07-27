import random
import numpy as np

def generate_random_data(dimension = 5000, max_number = 20):
  x_data = []
  y_data = []

  for _ in range(dimension):
    temp_x_data = random.sample(range(0, max_number), max_number)
    temp_y_data = [ 1 if x > p else 0 for (x, p) in zip(temp_x_data, [0]+temp_x_data) ]

    x_data.append(temp_x_data)
    y_data.append(temp_y_data)

  return np.array(x_data), np.array(y_data)

def generate_batch(raw_data, batch_size, num_steps):
  raw_x, raw_y = raw_data
  data_length = len(raw_x)

  # partition raw data into batches and stack them vertically in a data matrix
  batch_partition_length = data_length // batch_size  # 25
  data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32) # 200x25
  data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
  for i in range(batch_size):
    data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
    data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]

  epoch_size = batch_partition_length // num_steps
  for i in range(epoch_size):
    x = data_x[:, i * num_steps:(i + 1) * num_steps]
    y = data_y[:, i * num_steps:(i + 1) * num_steps]
    yield (x, y)

def generate_epochs(n, batch_size, num_steps):
  for i in range(n):
    yield generate_batch(generate_random_data(), batch_size, num_steps)

def main():
  num_steps = 5
  batch_size = 200
  num_classes = 2
  state_size = 4
  learning_rate = 0.1

if __name__ == '__main__':
  main()
