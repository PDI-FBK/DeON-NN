#!/usr/bin/env python

"""Create a neural net for DeON."""

import click
import json
from rnn.main import Main
from rnn.config import Config
import os


@click.command()
@click.option('--config')
@click.option('--train/--no-train', default=True)
@click.option('--test/--no-test', default=True)
@click.option('--validation/--no-validation', default=True)
def run(config, train, test, validation):
    basedir, _ = tuple(os.path.split(config))
    config = Config(json.load(open(config)), basedir)

    Main(config).run(train, test, validation)
    pass

if __name__ == '__main__':
    run()
