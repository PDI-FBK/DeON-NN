import os


def _get_tsv_files(dataset_path):
    ls = []
    for file in os.listdir(dataset_path):
        if file.endswith('.tsv'):
            ls.append(file)
    return ls
