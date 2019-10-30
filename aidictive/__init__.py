"""AIdictive.

Addictive AI library built on top of others amazing libraries.
"""

from ._version import __version__
from . import models
from . import modules
from . import trainer
from . import data
from . import plot
from . import eda
from . import pandas_accessors


__author__ = "guiferviz"


def greet():
    """Print a silly sentence on the screen. """
    print("An algorithm implemented is worth two in pseudocode.")

