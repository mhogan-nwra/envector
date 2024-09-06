from __future__ import absolute_import

from ._info import __doc__  # @UnresolvedImport
from .util import *
from .rotation import *
from .core import *
from .objects import *


__version__ = "0.2.0"


def test(*options):
    """
    Run tests for module using pytest.

    Parameters
    ----------
    *options : optional
        options to pass to pytest. The most important ones include:
        '-v', '--verbose':
            increase verbosity.
        '-q', '--quiet':
            decrease verbosity.
        '--doctest-modules':
            run doctests in all .py modules
        '--cov':
            measure coverage for .py modules
        '-h', '--help':
            show full help message and display all possible options to use.

    Returns
    -------
    exit_code: scalar
        Exit code is 0 if all tests passed without failure.

    Example
    -------
    import envector as nv
    env.test('-q', '--doctest-modules', '--cov', '--disable-warnings')
    """

    import pytest
    return pytest.main(['--pyargs', 'envector'] + list(options))
