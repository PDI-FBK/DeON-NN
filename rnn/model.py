import tensorflow as tf
import logging
from rnn.logits import Logits
import rnn.data as data


class Model(object):
    """this model construct the graph of the rnn"""
    def __init__(self, checkpoint_dir):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        self.graph = tf.get_default_graph()
        self.loss = None
        self.accuracy = None
        self.global_step = None
        self.tvars = None
        self.summary_op = None
        self.summary_path = None
        self.epochs = 1
        self.logger = logging
        self._checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        self.MODE = 'TRAIN'
        self.saver = tf.train.Saver()

    def next(self):
        with self.sess as sess:
            self.summary_writer = tf.summary.FileWriter(
                self.summary_path, sess.graph)
            self.summary_writer.flush()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self._restore_from_checkpoint()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            flag = True

            try:
                while flag:
                    yield self.step()
            except tf.errors.OutOfRangeError as oore:
                self.logger.info('Model:next:except {}'.format(oore))
                coord.request_stop(ex=oore)
            finally:
                self.summary_writer.flush()
                flag = False
                coord.request_stop()
                coord.join(threads)
                if self.MODE is not 'TRAIN':
                    self.summarize()

    def step(self):
        raise NotImplementedError

    def summarize(self):
        raise NotImplementedError

    def build(self, config, file, batch_size):
        tensor = self._get_tensor(file, batch_size, config.seed)
        x = tensor['words']
        y = tensor['definition']
        real_ouput = tf.cast(tf.reshape(y, [-1]), tf.float32)
        logits = Logits(config).build(x, tensor['sentence_length'])
        predictions = self._init_prediction(logits)
        self._initialize(real_ouput, predictions)
        self._add_summary_scalar()
        self.summary_op = self._build_summary()
        pass

    def save_checkpoint(self, checkpoint_path, step):
        with tf.device('CPU:0'):
            return self.saver.save(self.sess, checkpoint_path, global_step=step)

    def _initialize(self, real_ouput, predictions):
        self.accuracy = self._init_accuracy(real_ouput, predictions)
        self.global_step = self._get_or_create_global_step(graph=self.graph)
        self.logger.info('model:_initialize \t global_step={}'.format(self.global_step))
        if self.MODE == 'TRAIN':
            self.loss = self._init_loss(real_ouput, predictions)
            self.optimizer = self._init_optimizer(self.loss, self.global_step)
            self.tvars = tf.trainable_variables()
        pass

    def _add_summary_scalar(self):
        with tf.device('CPU:0'):
            tf.summary.scalar('accuracy', self.accuracy)
            if self.MODE == 'TRAIN':
                tf.summary.scalar('loss', self.loss)

    def _restore_from_checkpoint(self):
        with self.graph.as_default(), tf.device('CPU:0'):
            saver = tf.train.Saver()
            if self._checkpoint is not None:
                saver.restore(self.sess, self._checkpoint)

    def _init_loss(self, real_ouput, predictions):
        return tf.losses.mean_squared_error(
            real_ouput,
            predictions,
            weights=1.0,
            scope=None,
            loss_collection=tf.GraphKeys.LOSSES,
            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        )

    def _init_accuracy(self, real_ouput, predictions):
        correct_prediction = tf.equal(real_ouput, tf.round(predictions))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _init_optimizer(self, loss, global_step):
        return tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

    def _init_prediction(self, logits):
        predictions = tf.sigmoid(logits)
        return tf.reshape(predictions, [-1])

    def _build_summary(self):
        with self.graph.as_default(), tf.device('CPU:0'):
            return tf.summary.merge_all()

    def _get_or_create_global_step(self, graph=None):
        device = 'CPU:0'

        graph = graph or tf.get_default_graph()
        global_step = self._get_global_step(graph=graph)
        if global_step is None:
            with graph.as_default(), tf.device(device):
                global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
                graph.add_to_collection(tf.GraphKeys.GLOBAL_STEP, global_step)
        return global_step

    def _get_global_step(self, graph=None):
        graph = graph or tf.get_default_graph()
        collection = graph.get_collection(tf.GraphKeys.GLOBAL_STEP)
        for global_step in collection:
            self.logger.info('Found global_step={}'.format(global_step))
            return global_step
        return None

    def _get_tensor(self, file, batch_size, seed):
        with tf.device('CPU:0'):
            return data.inputs(
                [file],
                batch_size,
                True,
                self.epochs,
                seed)

    def as_summary(self, kvp):
        """Turns a dictionary into a `TensorFlow` summary."""
        return tf.summary.Summary(
            value=[
                tf.summary.Summary.Value(tag=key, simple_value=value)
                for key, value in kvp.items()
            ])
