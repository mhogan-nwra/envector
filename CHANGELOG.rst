=========
Changelog
=========

Version 0.1.0, August 29, 2024
================================
Matt Hogan (1):
    * Initial commit to GitHub as fork of nvector


Version 0.2.0, September 06, 2024
=================================
Matt Hogan (3)
    * Update docstrings to adopt the Napoleon docstring standard
    * Add type-hints to most functions and methods
    * Update the documentation to show more than a summary.


Version 0.3.0, September 09, 2024
=================================
Matt Hogan (4)
    * Reduce matrix of supported Python versions as to reduce burden of testing
    * Add testing using GitHub Action. Windows, MacOS, and Linux (Ubuntu) is tested using 3.9 - 3.12, inclusive.
    * Added local testing support using nox.
    * Remove numpydoc from the documentation requirements as it is not used.

Version 0.3.1, September 11, 2024
=================================
Matt Hogan (6)
    * Embraced 3.9+ support with more descriptive static typing.
    * Expanded or added docstrings.
    * Fixed some pytest failures associated with matplotlib opening multiple figures simultaneously.
    * Added missing figures to the documentation.
    * Updated the noxfile to support multiple virtual environment backends.
    * Expand documentation on the reasons for the existence of envector.
