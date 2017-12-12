import tensorflow as tf


def build_inner_cell(hidden_size):
    return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)


def multi_rnn_cell(hidden_size, num_layers):
    cells = [build_inner_cell(hidden_size) for _ in range(0, num_layers)]
    return tf.contrib.rnn.MultiRNNCell(cells=cells, state_is_tuple=True)


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
