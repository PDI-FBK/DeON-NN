from rnn.dataset.dictionary import Dictionary
import mock


def mock_read_file(t):
    print('args', t)
    return ["1\tAnnuity is an account where funds are withdrawn periodically .\tannuity\t0\t1\tdifference-between-annuity-and-vs-sinking-fund"]


def test_should_return_dictionary_translation_from_words_to_numbers():
    with mock.patch.object(Dictionary, 'read_file', new=mock_read_file):
        dictionary = Dictionary('some_path_file')

        result = dictionary.get_data()

        assert [([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1)] == result


def test_should_return_vocabolary_size():
    with mock.patch.object(Dictionary, 'read_file', new=mock_read_file):
        dictionary = Dictionary('some_path_file')
        dictionary.get_data()

        size = dictionary.vocab_size()

        assert 10 == size


def test_should_return_number_of_eos():
    with mock.patch.object(Dictionary, 'read_file', new=mock_read_file):
        dictionary = Dictionary('some_path_file')
        dictionary.get_data()

        eos = dictionary.eos()

        assert 9 == eos
