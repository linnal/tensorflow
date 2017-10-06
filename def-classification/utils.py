import tensorflow as tf


def build_inner_cell(hiddenSize):
  return tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=0.0, state_is_tuple=True)

def getRealLength(ls, eosIndex):
  for i, nr in enumerate(ls):
    if nr == eosIndex:
      break
  return i+1


def extract_axis(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res


