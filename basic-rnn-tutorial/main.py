import random

def generate_random_data(dimension, max_number):
  x_data = []
  y_data = []

  for _ in range(dimension):
    temp_x_data = random.sample(range(0, max_number), max_number)
    temp_y_data = [ 1 if x > p else 0 for (x, p) in zip(temp_x_data, [0]+temp_x_data) ]

    x_data.append(temp_x_data)
    y_data.append(temp_y_data)

  return x_data, y_data

def main():
  x_data, y_data = generate_random_data(10, 20)

if __name__ == '__main__':
  main()
