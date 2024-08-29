from envector import (_intro,  # pylint: disable=no-name-in-module
                     _documentation,
                     _installation,
                     _examples_object_oriented,
                     _acknowledgements,
                     _images)

__doc__ = (  # @ReservedAssignment
    """Introduction to envector
=======================
""" + _intro.__doc__  # @UndefinedVariable @ReservedAssignment
    + _documentation.__doc__  # @UndefinedVariable
    + _installation.__doc__  # @UndefinedVariable
    + _examples_object_oriented.__doc__
    + """Acknowledgements
================
""" + _acknowledgements.__doc__  # @UndefinedVariable
    + _images.__doc__)  # @UndefinedVariable


if __name__ == '__main__':
    from envector._common import test_docstrings
    test_docstrings(__file__)
