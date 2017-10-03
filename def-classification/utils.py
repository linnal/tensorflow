import tensorflow as tf


def build_inner_cell(hiddenSize):
  return tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=0.0, state_is_tuple=True)


 
def createWeightMask(data, eosIndex):
  mask = []
  for ls in data:
    for i, nr in enumerate(ls):
    	if nr == eosIndex:
    		break

    real = [1]*(i+1)
    fake = [0]*(len(ls)-(i+1))
    mask.append(real + fake)

  return mask