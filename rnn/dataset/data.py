import tensorflow as tf
import os
from pathlib import Path
import rnn.dataset.util as util

SENTENCE_LENGTH_KEY = 'sentence_length'
FORMULA_LENGTH_KEY = 'formula_length'
WORDS_KEY = 'words'
DEF_KEY = 'definition'


def encode(words_idxs, definition_idxs):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                SENTENCE_LENGTH_KEY: tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[len(words_idxs)])),
                FORMULA_LENGTH_KEY: tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[len(definition_idxs)])),
                WORDS_KEY: tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=words_idxs)),
                DEF_KEY: tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=definition_idxs)),
            }
        )
    )
    return example


def generate_rio_dataset(dataset_path, vocabolary):
    tsv_files = util._get_tsv_files(dataset_path)
    for filename in tsv_files:
        tsv_path = os.path.join(dataset_path, filename)
        rio_path = os.path.join(dataset_path, filename + ".rio")
        if Path(rio_path).exists():
            return
        with tf.python_io.TFRecordWriter(rio_path) as writer:
            with open(tsv_path, 'r') as f:
                for line in f:
                    _, sentence, _, _, _def, _ = line.split('\t')
                    words = sentence.split()
                    w_idx = [vocabolary.index(word) for word in words]
                    writer.write(encode(w_idx, [int(_def)])
                        .SerializeToString())
    return
