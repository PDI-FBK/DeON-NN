import rnn.data as data
import tensorflow as tf
from enum import Enum
import rnn.util as util
import os
import time


class ModelType(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALIDATION = 'eval'


class Model(object):

    def __init__(self, model_type, config, graph):
        checkpoint = tf.train.latest_checkpoint(config.checkpoint_dir)
        self.sess = tf.Session()
        self.config = config
        self.graph = graph

        tensor = self._get_tensor(model_type)

        y = tensor['definition']
        logits = self._get_logits(tensor['words'], tensor['sentence_length'], model_type.value)

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

        self.global_step = self.get_or_create_global_step(graph=self.graph)

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)
        self.tvars = tf.trainable_variables()

        self.summary_path = os.path.join(self.config.summaries, model_type.value)

        with self.graph.as_default():
            saver = tf.train.Saver()
            if checkpoint is not None:
                saver.restore(self.sess, checkpoint)

        return

    def next(self):
        with self.sess as sess:
            merged = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.summary_path, sess.graph)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            flag = True

            try:
                while flag:
                    res = sess.run([self.loss,
                                    self.accuracy,
                                    self.tvars,
                                    self.optimizer,
                                    merged,
                                    self.global_step])
                    summary_writer.add_summary(res[4], res[5])
                    yield res[-1]
            except tf.errors.OutOfRangeError as oore:
                coord.request_stop(ex=oore)
            finally:
                flag = False
                coord.request_stop()
                coord.join(threads)

    def save_checkpoint(self, checkpoint_path, step):
        saver = tf.train.Saver()
        return saver.save(self.sess, checkpoint_path, global_step=step)


    def _get_tensor(self, model_type):
        if model_type == ModelType.TRAIN:
            return data.inputs([self.config.train_file], self.config.batch_size, True, 1, self.config.seed)
        if model_type == ModelType.TEST:
            return data.inputs([self.config.test_file], self.config.batch_size, True, 1, self.config.seed)
        return data.inputs([self.config.eval_file], self.config.batch_size, True, 1, self.config.seed)

    def _get_logits(self, x, lengths, scope_name):
        embeddings = tf.get_variable('embedding',
            [self.config.vocab_input_size, self.config.emb_dim])
        embedding_layer = tf.nn.embedding_lookup(embeddings, x)

        cell = util.multi_rnn_cell(self.config.hidden_size, self.config.num_layers)

        output, state = tf.nn.dynamic_rnn(cell, embedding_layer,
                                          sequence_length=lengths,
                                          initial_state=cell.zero_state(tf.shape(embedding_layer)[0], tf.float32))
        output = util.extract_axis(output, lengths - 1)

        softmax_w = tf.get_variable('softmax_w',
            [self.config.hidden_size, self.config.vocab_ouput_size], dtype=tf.float32)

        softmax_b = tf.get_variable('softmax_b',
            [self.config.vocab_ouput_size], dtype=tf.float32)

        logits = tf.matmul(output, softmax_w) + softmax_b
        return logits


    def _save_ckpt(self, step):
        ckpt = self._saver.save(self.sess, self._ckpt_name, step)
        self._ckpt_ts = time.time()
        return ckpt

    def get_or_create_global_step(self, graph=None):
        graph = graph or tf.get_default_graph()
        global_step = self.get_global_step(graph=graph)
        if global_step is None:
            with graph.as_default():
                global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
                graph.add_to_collection(tf.GraphKeys.GLOBAL_STEP, global_step)
        return global_step

    def get_global_step(self, graph=None):
        graph = graph or tf.get_default_graph()
        collection = graph.get_collection(tf.GraphKeys.GLOBAL_STEP)
        for global_step in collection:
            return global_step
        return None
