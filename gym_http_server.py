#!/usr/bin/env python3
from functools import wraps
import uuid
import gym
from gym import wrappers
import numpy as np
import six
import argparse
import sys

import logging
logger = logging.getLogger('robrocker7')
logger.setLevel(logging.ERROR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a Gym HTTP API server')
    parser.add_argument('-l', '--listen', help='interface to listen to', default='0.0.0.0')
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to bind to')

    args = parser.parse_args()
    print('Server starting at: ' + 'http://{}:{}'.format(args.listen, args.port))
    
    from server.flask_app import app
    app.run()