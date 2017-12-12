import rnn.util as util
import tensorflow as tf
import rnn.data as data
import os


class Main(object):
    """docstring for Rnn"""
    def __init__(self, config):
        self.sess = tf.Session()
        self.config = config

        rio_file = os.path.join(config.dataset_path, 'validation.tsv.rio')
        self.tensor = data.inputs([rio_file], config.batch_size)

        y = self.tensor['definition']
        logits = self._get_logits(self.tensor['words'], self.tensor['sentence_length'])

        predictions = tf.sigmoid(logits)
        predictions = tf.reshape(predictions, [-1])
        real_ouput = tf.cast(tf.reshape(y, [-1]), tf.float32)

        self.loss = tf.losses.mean_squared_error(
            real_ouput,
            predictions,
            weights=1.0,
            scope=None,
            loss_collection=tf.GraphKeys.LOSSES,
            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        )
        tf.summary.scalar('loss', self.loss)

        correct_prediction = tf.equal(real_ouput, tf.round(predictions))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        self.tvars = tf.trainable_variables()
        return

    def run(self, force):
        counter = 0
        with self.sess as sess:
            merged = tf.summary.merge_all()
            train_file_path = os.path.join(self.config.summaries, 'train')
            train_writer = tf.summary.FileWriter(train_file_path, sess.graph)
            init = tf.global_variables_initializer()
            sess.run(init)
            tf.train.start_queue_runners(sess=sess)

            while counter < self.config.training_steps:
                counter += 1
                res = sess.run([self.loss,
                                self.accuracy,
                                self.tvars,
                                self.optimizer,
                                merged])

                print('loss', res[0])
                print('accuracy', res[1])

                train_writer.add_summary(res[-1], counter)
        return



    def _get_logits(self, x, lengths):
        embeddings = tf.get_variable('embedding',
            [self.config.vocab_input_size, self.config.emb_dim])
        embedding_layer = tf.nn.embedding_lookup(embeddings, x)

        cell = util.multi_rnn_cell(self.config.hidden_size, self.config.num_layers)

        output, state = tf.nn.dynamic_rnn(cell, embedding_layer,
                                          sequence_length=lengths,
                                          initial_state=cell.zero_state(self.config.batch_size, tf.float32))
        output = util.extract_axis(output, lengths - 1)

        softmax_w = tf.get_variable('softmax_w',
            [self.config.hidden_size, self.config.vocab_ouput_size], dtype=tf.float32)

        softmax_b = tf.get_variable('softmax_b',
            [self.config.vocab_ouput_size], dtype=tf.float32)

        logits = tf.matmul(output, softmax_w) + softmax_b
        return logits
