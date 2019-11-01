"""AIdictive.

Addictive AI library built on top of others amazing libraries.
"""

from aidictive._version import __version__
from aidictive import data
from aidictive import eda
from aidictive import models
from aidictive import modules
from aidictive import pandas_accessors
from aidictive import plot
from aidictive import trainer
from aidictive import utils
from aidictive.utils import seed
from aidictive.utils import save


__author__ = "guiferviz"


def greet():
    """Print a silly sentence on the screen. """
    print("An algorithm implemented is worth two in pseudocode.")

