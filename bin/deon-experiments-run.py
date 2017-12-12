#!/usr/bin/env python

"""Create a neural net for DeON."""

import click
import json
from rnn.main import Main
from rnn.config import Config
import os


@click.command()
@click.option('--config')
@click.option('--force', is_flag=True)
def run(config, force):
    basedir, _ = tuple(os.path.split(config))
    config = Config(json.load(open(config)), basedir)

    Main(config).run(force)
    pass

if __name__ == '__main__':
    run()
