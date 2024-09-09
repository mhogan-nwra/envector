Welcome to the envector's documentation!
========================================
.. only:: html

    |pkg_img| |docs_img| |versions_img| |downloads_img|


This is the documentation of **nvector** version |release| for Python3 only released on |today|.

Bleeding edge available at: https://github.com/mhogan-nwra/envector.

Official releases are available at: http://pypi.python.org/pypi/envector.

Official homepage are available at: http://www.navlab.net/nvector/


Questions and Answers
---------------------

If you are coming from the nvector_ package, these Q-and-A can quickly explain this package.

1. *What is the difference between this package and nvector*?

    * Virtually none! The envector_ package is a fork of nvector_ with mostly aesthetic changes.
    * No methods and functions have been removed, but documented deprecated methods in nvector_ will be removed.
    * If your Python software worked with nvector_, then there is a good chance envector_ will also work. The caveats
      are that this package abandons Python2 and extends to Python3.9+ support.

2. *Why did you fork nvector*?

    * Primarily because the nvector_ Python package is broken with NumPy version 2.
    * There is no indication that the situation will change.

3. *What changes are there with from nvector*?

    * Any place there is a `import nvector` or `from nvector` statement, replace it with `import envector` or
      `from envector`, respectively.
    * The envector_ package is a Python3-only package as it embraces type-hints in most cases.
    * Documentation is expanded in some cases.
    * The docstrings have been refactored to utilize the Napoleon docstring style.

4. *When is the appropriate to switch to envector*?

    * If your Python software must support NumPy version 2,
    * If your Python software also stops supporting Python versions after its end-of-life cycle.



.. toctree::
    :maxdepth: 1
    :numbered:
    :includehidden:
    :caption: Contents:


    intro/index.rst
    tutorials/index.rst
    how-to/index.rst
    topics/index.rst
    reference/index.rst



.. only:: html

    .. toctree::
        :maxdepth: 3

        appendix/changelog.rst
        appendix/authors.rst
        appendix/license.rst
        appendix/acknowledgement.rst
        appendix/index.rst
        appendix/zreferences.rst


    .. automodule:: envector._images
       :members: __doc__


.. _envector: https://github.com/mhogan-nwra/envector
.. _nvector: https://github.com/pbrod/nvector
.. _toolbox: http://www.navlab.net/nvector/#download>