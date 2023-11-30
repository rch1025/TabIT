"""Synthesizers module."""

from model.synthesizers.model import model

__all__ = (
    'model'
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }
