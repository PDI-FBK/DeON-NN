#!/usr/bin/env python

"""Create a neural net for DeON."""

import click
import json
from rnn.rnn import Rnn


@click.command()
@click.argument('config', type=click.File('r'))
def run(config):
    config = json.loads(config.read())
    Rnn(config).run()
    pass

if __name__ == '__main__':
    run()
