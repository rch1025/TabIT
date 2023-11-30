# -*- coding: utf-8 -*-

"""Top-level package for model."""

__author__ = 'DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__version__ = '0.7.2.dev0'

from model.demo import load_demo
from model.synthesizers.model import model

__all__ = (
    'model',
    'load_demo'
)
