from .__init__ import __version__

__doc__ = """
Install envector
================

If you have pip installed and are online, then simply type:

    $ pip install envector

to get the lastest stable version. Using pip also has the advantage that all
requirements are automatically installed.

You can download envector and all dependencies to a folder "pkg", by the following:

   $ pip install --download=pkg envector

To install the downloaded envector, just type:

   $ pip install --no-index --find-links=pkg envector


Verifying installation
======================
To verify that envector can be seen by Python, type ``python`` from your shell.
Then at the Python prompt, try to import envector:

.. parsed-literal::

    >>> import envector as nv
    >>> print(nv.__version__)
    {release}


To test if the toolbox is working correctly paste the following in an interactive
python session::

   import envector as nv
   nv.test('--doctest-modules')

or

   $ pdm run pytest

at the command prompt.

""".format(release=__version__)