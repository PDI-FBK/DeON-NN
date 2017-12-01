#!/usr/bin/env python

"""Create a neural net for DeON."""

import click
import json
from rnn.main import Main
import os


@click.command()
@click.option('--config')
def run(config):
    basedir, _ = tuple(os.path.split(config))
    config = json.load(open(config))
    Main(config).run(basedir)
    pass

if __name__ == '__main__':
    run()
