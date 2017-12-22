import tensorflow as tf


class Logits():
    def __init__(self, config):
        self.vocab_input_size = config.vocab_input_size
        self.emb_dim = config.emb_dim
        self.hidden_size = config.hidden_size
        self.vocab_ouput_size = config.vocab_ouput_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.cell_type = config.cell_type

    def build(self, tensor_input, tensor_length):
        embeddings = self._get_embeddings(
            self.vocab_input_size, self.emb_dim)
        softmax_w = self._get_softmax_w(
            self.hidden_size, self.vocab_ouput_size)
        softmax_b = self._get_softmax_b(self.vocab_ouput_size)

        embedding_layer = tf.nn.embedding_lookup(embeddings, tensor_input)
        output = self._get_output(embedding_layer, tensor_length)

        logits = tf.matmul(output, softmax_w) + softmax_b
        return logits

    def _get_embeddings(self, input_size, emb_dim):
        return tf.get_variable('embedding', [input_size, emb_dim])

    def _get_softmax_w(self, hidden_size, output_size):
        return tf.get_variable(
            'softmax_w', [hidden_size, output_size], dtype=tf.float32)

    def _get_softmax_b(self, output_size):
        return tf.get_variable('softmax_b', [output_size], dtype=tf.float32)

    def _get_output(self, embedding_layer, tensor_length):
        cell = self._multi_rnn_cell(
            self.hidden_size, self.num_layers, self.cell_type)

        output, state = tf.nn.dynamic_rnn(
            cell, embedding_layer, sequence_length=tensor_length,
            initial_state=cell.zero_state(tf.shape(embedding_layer)[0], tf.float32))
        return self._extract_axis(output, tensor_length - 1)

    def _build_inner_cell(self, hidden_size, celltype):
        if celltype == "BasicLSTMCell":
            return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
        raise Exception(celltype, "not implemented.")

    def _multi_rnn_cell(self, hidden_size, num_layers, celltype):
        cells = [self._build_inner_cell(hidden_size, celltype) for _ in range(0, num_layers)]
        return tf.contrib.rnn.MultiRNNCell(cells=cells, state_is_tuple=True)

    def _extract_axis(self, data, ind):
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
