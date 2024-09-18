Welcome to the envector's documentation!
========================================
.. only:: html

    |pkg_img| |docs_img| |versions_img| |test_img| |downloads_img|


This is the documentation of the **envector** Python package version |release| for Python3-only released on |today|.

Bleeding edge available at: https://github.com/mhogan-nwra/envector.

Official releases are available at: http://pypi.python.org/pypi/envector.

Official homepage are available at: http://www.navlab.net/nvector/


Explanation of Intent
---------------------

This is a Python package that will assist you in calculating the distance between two points anywhere on, above, or
below the Earth's surface. This can also be between two, three, or many points. This is quite useful in many areas from
logistics, tracking, navigation, data analytics, data science, and research. The calculations are simple and
non-singular. Full accuracy is achieved for any global position (and for any distance).

Use Cases
---------

A use case in logistics is when you need to recommend to your customers the closest facilities or store
locations given an address or GPS coordinates. Your customers usually provide an address and you can convert that to GPS
coordinates, or geographic location in latitude and longitude. Given this information, you need to recommend the closest
locations and approximate distance in kilometers (km) or miles. You can implement the Haversine formula, which is a
reasonable estimate if you are concerned with relatively short distances less than 100 km. However, you might meed to be
aware of the difference in altitudes between two locations. On top of all that, the Haversine formula is not accurate
for longer distances as the Earth is not exactly a sphere. You can properly account for all these issues using our
envector_ package.

Another use case is for navigation and tracking. Imagine that you
have a vehicle like a ship, airplane, or off-road vehicle on a fixed course. The vehicle has an unreliable and
inaccurate GPS unit, and it is your job to ensure that the vehicle stays on course. This makes your job much harder if
you want to minimize the trip duration and vehicle fuel to maximize the number of trips possible for the day.
Fortunately, the envector_ package can help you 1) aggregate measurements to estimate the mean position, 2) interpolate
the next expected position in a fixed time interval, and 3) determine if the vehicle is veering off-course by measuring
the cross-track distance from the intended path.

These use cases and more are well supported by the envector_ package. We encourage you to check out the
:doc:`/tutorials/index` to help you maximize the utility of envector_.


Questions and Answers
---------------------

If you are coming from the nvector_ package, these Q-and-A can quickly explain this package.

1. *What is the difference between this package and nvector*?

    * Virtually none! The envector_ package is a fork of nvector_ with mostly aesthetic changes.
    * No methods and functions have been removed, but documented deprecated methods in nvector_ will be removed.
    * If your Python software worked with nvector_, then there is a good chance envector_ will also work. The caveats
      are that this package abandons Python2 and extends to Python3.9+ support.

2. *Why did you call this package "envector"*?

    * The name honors the original nvector_ Python package and the progenitor `MATLAB n-vector toolbox`_. There are
      adaptations in other languages as noted in the `nvector downloads`_ page.
    * The names nvector_ and envector_ are homophones (pronounced the same), so the name invokes the original n-vector
      origin.

3. *Why did you fork nvector*?

    * Primarily because the nvector_ Python package is broken with NumPy version 2.
    * There is no indication that the situation will change.

4. *What changes are there with from nvector*?

    * The major difference is the namespace. Any place where your project have statements like:

       .. code-block:: python
            :caption: Importing the nvector package

            import nvector as nv
            from nvector import GeoPoint

      You would replace them:

        .. code-block:: python
            :caption: Importing the envector package instead

            import envector as nv
            from envector import GeoPoint

    * The envector_ package is a Python3-only package as it embraces static typing in most cases.
    * Documentation is expanded in some cases.
    * The docstrings have been refactored to utilize the Napoleon docstring style.

5. *When is the appropriate to switch to envector*?

    * If your Python software must support NumPy version 2,
    * If your Python software also stops supporting Python versions after its end-of-life cycle.

6. *How do I make the switch to envector*?

    * If your project utilizes CPython3.9 through 3.12, inclusive, then you can simply change your `requirements.txt`,
      `environment.yml`, `pyproject.toml`, or `setup.py` file to specify envector_.

        .. code-block:: text
            :caption: `requirements.txt` format

            envector>=0

        .. code-block:: yaml
            :caption: Anaconda `environment.yml` format

            - pip:
              - envector>=0

        .. code-block:: toml
            :caption: `pyproject.toml` format

            # PEP 508 compliant
            [project]
            dependencies = [
                "envector>=0"
            ]

            # Poetry (not PEP 508 compliant)
            [tool.poetry.dependencies]
            envector = ">=0"

        .. code-block:: python
            :caption: `setup.py` format

            install_requires=['envector>=0',
                              ...
                              ]

        * Your Python code will now need to import envector_

            .. code-block:: python
                :caption: Importing envector into your module

                    import envector as nv

    * If your project uses anything less than CPython3.9, then it depends on how your project is specified. If you are
      using `pyproject.toml` or `setup.py`, then the changes are relatively simple as shown below. The other common
      Anaconda `environment.yml` and `requirements.txt` formats require you to pick one depending on the Python
      version. For Python2 to 3.8, you cannot use envector_.

        .. code-block:: toml
            :caption: `pyproject.toml` format to specify both nvector and envector

            # PEP 508 compliant
            [project]
            dependencies = [
                "envector>=0; python_version >= '3.9'",
                "nvector>=0; python_version < '3.9'",
            ]

            # Poetry (not PEP 508 compliant)
            [tool.poetry.dependencies]
            envector = { version = ">=0", python = ">=3.9" }
            nvector = { version = ">=0", python = "<3.9" }

        .. code-block:: python
            :caption: `setup.py` format to specify both nvector and envector

            install_requires=['envector>=0; python_version >= "3.9"',
                              'nvector>=0; python_version < "3.9"',
                              ...
                              ]

        * Your Python code will now need to import both envector_ and nvector_

            .. code-block:: python
                :caption: Code block to import either nvector or envector

                    try:
                        import nvector as nv
                    except (ImportError,):
                        import envector as nv


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
.. _nvector downloads: https://www.ffi.no/en/research/n-vector/n-vector-downloads
.. _MATLAB n-vector toolbox: https://github.com/FFI-no/n-vector