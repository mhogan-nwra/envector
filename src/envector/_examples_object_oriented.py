from __future__ import absolute_import
from envector._examples import GETTING_STARTED
__doc__ = GETTING_STARTED  # @ReservedAssignment


if __name__ == '__main__':
    from envector._common import write_readme, test_docstrings
    test_docstrings(__file__)
    write_readme(__doc__)
