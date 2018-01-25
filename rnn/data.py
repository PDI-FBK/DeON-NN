import itertools
import tensorflow as tf


SENTENCE_LENGTH_KEY = 'sentence_length'
DEF_LENGTH_KEY = 'definition_length'
WORDS_KEY = 'words'
DEF_KEY = 'definition'



def parse(serialized):
    features = {
        SENTENCE_LENGTH_KEY: tf.FixedLenFeature([], tf.int64),
        DEF_LENGTH_KEY: tf.FixedLenFeature([], tf.int64),
        WORDS_KEY: tf.VarLenFeature(tf.int64),
        DEF_KEY: tf.VarLenFeature(tf.int64),
    }
    parsed = tf.parse_single_example(
        serialized=serialized,
        features=features)
    sentence_length = parsed[SENTENCE_LENGTH_KEY]
    formula_length = parsed[DEF_LENGTH_KEY]
    words = tf.sparse_tensor_to_dense(parsed[WORDS_KEY])
    formula = tf.sparse_tensor_to_dense(parsed[DEF_KEY])
    return words, sentence_length, formula, formula_length


def read_from_files(file_patterns, shuffle=True, num_epochs=None, seed=None):
    files = list(itertools.chain(*[tf.gfile.Glob(p) for p in file_patterns]))
    fqueue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=shuffle, name='FilenameQueue', seed=seed)
    reader = tf.TFRecordReader(name='TFRecordReader')
    _, value = reader.read(fqueue, name='Read')
    tensors = parse(value)
    return tensors


def inputs(file_patterns, batch_size, shuffle=True, num_epochs=None, seed=None):
    """Build the input pipeline."""
    min_after_dequeue = 100
    num_threads = 4
    safety_margin = 2
    capacity = (min_after_dequeue + (num_threads + safety_margin) * batch_size)

    tensors = read_from_files(file_patterns, shuffle, num_epochs, seed)
    tensors = shuffle_batch(
        tensors,
        batch_size,
        seed=seed,
        allow_smaller_final_batch=True,
        num_threads=num_threads,
        min_after_dequeue=min_after_dequeue,
        capacity=capacity
    )
    tensors = {
        WORDS_KEY: tf.cast(tensors[0], tf.int32, name=WORDS_KEY),
        SENTENCE_LENGTH_KEY: tf.cast(tensors[1], tf.int32, name=SENTENCE_LENGTH_KEY),
        DEF_KEY: tf.cast(tensors[2], tf.int64, name=DEF_KEY),
        DEF_LENGTH_KEY: tf.cast(tensors[3], tf.int32, name=DEF_LENGTH_KEY)
    }
    return tensors


def shuffle_batch(tensors,
                  batch_size,
                  capacity=32,
                  num_threads=1,
                  min_after_dequeue=16,
                  dtypes=None,
                  shapes=None,
                  seed=None,
                  enqueue_many=False,
                  dynamic_pad=True,
                  allow_smaller_final_batch=False,
                  shared_name=None,
                  name='shuffle_batch'):
    tensors = list(tensors)
    with tf.name_scope(name, tensors):
        dtypes = dtypes or list([t.dtype for t in tensors])
        shapes = shapes or list([t.get_shape() for t in tensors])
        inputs = shuffle(tensors,
                         seed=seed,
                         dtypes=dtypes,
                         capacity=capacity,
                         num_threads=num_threads,
                         min_after_dequeue=min_after_dequeue,
                         shared_name=shared_name,
                         name='shuffle')

        # fix the shapes
        for tensor, shape in zip(inputs, shapes):
            tensor.set_shape(shape)

        minibatch = tf.train.batch(
            tensors=inputs,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
            dynamic_pad=dynamic_pad,
            allow_smaller_final_batch=allow_smaller_final_batch,
            shared_name=shared_name,
            enqueue_many=enqueue_many,
            name='batch')
        return minibatch


def shuffle(tensors,
            capacity=32,
            min_after_dequeue=16,
            num_threads=1,
            dtypes=None,
            shapes=None,
            seed=None,
            shared_name=None,
            name='shuffle'):
    tensors = list(tensors)
    with tf.name_scope(name, tensors):
        dtypes = dtypes or list([t.dtype for t in tensors])
        queue = tf.RandomShuffleQueue(
            seed=seed,
            shared_name=shared_name,
            name='random_shuffle_queue',
            dtypes=dtypes,
            shapes=shapes,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        enqueue = queue.enqueue(tensors)
        runner = tf.train.QueueRunner(queue, [enqueue] * num_threads)
        tf.train.add_queue_runner(runner)
        dequeue = queue.dequeue()
        return dequeue
