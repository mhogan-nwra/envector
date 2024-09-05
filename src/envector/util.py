"""
Utility functions
=================

"""
from __future__ import division, print_function

import functools
import warnings
from typing import NamedTuple, Tuple, Union, Any

import numpy as np
from numpy import rad2deg, deg2rad, ndarray, float64
from numpy.linalg import norm

from envector import license as _license
from envector._common import test_docstrings, _make_summary

__all__ = ['deg', 'rad', 'mdot', 'nthroot', 'get_ellipsoid', 'select_ellipsoid', 'unit',
           'allclose', 'eccentricity2', 'polar_radius', 'third_flattening', 'deprecate']

FINFO = np.finfo(float)
_tiny_name = 'tiny' if np.__version__ < '1.22' else 'smallest_normal'
_TINY = getattr(FINFO, _tiny_name)
_EPS = FINFO.eps  # machine precision (machine epsilon)


class Ellipsoid(NamedTuple):
    """An ellipsoid with semi-major radius, flattening, and name"""

    a: float
    """Semi-major axis in meters"""
    f: float
    """Flattening value"""
    name: str
    """A name associated with the ellipsoid"""


ELLIPSOID = {
    1: Ellipsoid(a=6377563.3960, f=1.0 / 299.3249646, name='Airy 1858'),
    2: Ellipsoid(a=6377340.189, f=1.0 / 299.3249646, name='Airy Modified'),
    3: Ellipsoid(a=6378160.0, f=1.0 / 298.25, name='Australian National'),
    4: Ellipsoid(a=6377397.155, f=1.0 / 299.1528128, name='Bessel 1841'),
    5: Ellipsoid(a=6378249.145, f=1.0 / 293.465, name='Clarke 1880'),
    6: Ellipsoid(a=6377276.345, f=1.0 / 300.8017, name='Everest 1830'),
    7: Ellipsoid(a=6377304.063, f=1.0 / 300.8017, name='Everest Modified'),
    8: Ellipsoid(a=6378166.0, f=1.0 / 298.3, name='Fisher 1960'),
    9: Ellipsoid(a=6378150.0, f=1.0 / 298.3, name='Fisher 1968'),
    10: Ellipsoid(a=6378270.0, f=1.0 / 297, name='Hough 1956'),
    11: Ellipsoid(a=6378388.0, f=1.0 / 297,
                  name='Hayford/International ellipsoid 1924/European Datum 1950/ED50'),
    12: Ellipsoid(a=6378245.0, f=1.0 / 298.3, name='Krassovsky 1938'),
    13: Ellipsoid(a=6378145.0, f=1.0 / 298.25, name='NWL-9D / WGS 66'),
    14: Ellipsoid(a=6378160.0, f=1.0 / 298.25, name='South American 1969 / SAD69'),
    15: Ellipsoid(a=6378136.0, f=1.0 / 298.257, name='Soviet Geod. System 1985'),
    16: Ellipsoid(a=6378135.0, f=1.0 / 298.26, name='WGS 72'),
    17: Ellipsoid(a=6378206.4, f=1.0 / 294.9786982138, name='Clarke 1866 / NAD27'),
    18: Ellipsoid(a=6378137.0, f=1.0 / 298.257223563, name='GRS80 / WGS84 / NAD83'),
    19: Ellipsoid(a=6378137.0, f=298.257222101, name='ETRS89 / EUREF89'),
    20: Ellipsoid(a=6377492.0176, f=1/299.15281285, name='NGO1948')
}
"""Dictionary enumeration of supported ellipsoids. Synonyms are partitioned by '/' characters"""
ELLIPSOID_IX = {'airy1858': 1,
                'airymodified': 2,
                'australiannational': 3,
                'bessel': 4,
                'bessel1841': 4,
                'clarke1880': 5,
                'everest1830': 6,
                'everestmodified': 7,
                'fisher1960': 8,
                'fisher1968': 9,
                'hough1956': 10,
                'hough': 10,
                'hayford': 11,
                'international': 11,
                'internationalellipsoid1924': 11,
                'europeandatum1950': 11,
                'ed50': 11,
                'krassovsky': 12,
                'krassovsky1938': 12,
                'nwl-9d': 13,
                'wgs66': 13,
                'southamerican1969': 14,
                'sad69': 14,
                'sovietgeod.system1985': 15,
                'wgs72': 16,
                'clarke1866': 17,
                'nad27': 17,
                'grs80': 18,
                'wgs84': 18,
                'nad83': 18,
                'euref89': 19,
                'etrs89': 19,
                'ngo1948': 20
                }
"""Inverse mapping between a name string and ellipsoid ID"""


def deprecate(*args, **kwargs):
    """Replacement for :py:func:`numpy.deprecate` if missing"""
    try:
        np_deprecate = getattr(np, "deprecate")
        warnings.warn(
            "Usage of the numpy.deprecate function is removed in NumPy version 2. The envector.util.deprecate"
            "  function will stop calling numpy.deprecate once NumPy version 1 is no longer supported."
        )
        return np_deprecate(*args, **kwargs)
    except AttributeError:
        pass

    func = args[0]
    args = args[1:]
    old_name = kwargs.pop('old_name', None)
    new_name = kwargs.pop('new_name', None)
    message = kwargs.pop('message', None)

    if old_name is None:
        old_name = func.__name__
    if new_name is None:
        depdoc = "`%s` is deprecated!" % old_name
    else:
        depdoc = "`%s` is deprecated, use `%s` instead!" % \
                 (old_name, new_name)
    if isinstance(message, str):
        depdoc += "\n" + message

    @functools.wraps(func)
    def newfunc(*args, **kwds):
        warnings.warn(depdoc, DeprecationWarning, stacklevel=2)
        return func(*args, **kwds)

    newfunc.__name__ = old_name
    doc = func.__doc__
    if doc is None:
        doc = depdoc
    else:
        doc = '\n\n'.join([depdoc, doc])
    newfunc.__doc__ = doc

    return newfunc


def eccentricity2(f: Union[float, ndarray]) -> Tuple[Union[float, ndarray], Union[float, ndarray]]:
    """Returns the first and second eccentricity squared given the flattening, f.

    Parameters
    ----------
    f : float | ndarray
        Flattening parameter

    Returns
    -------
    tuple[float | ndarray, float | ndarray]
        Tuple of the first and second eccentricities, respectively.

    Notes
    -----
    The (first) eccentricity squared is defined as e2 = f*(2-f).
    The second eccentricity squared is defined as e2m = e2 / (1 - e2).
    """
    e2 = (2.0 - f) * f  # = 1-b**2/a**
    e2m = e2 / (1.0 - e2)
    return e2, e2m


def polar_radius(
    a: Union[float, ndarray],
    f: Union[float, ndarray]
) -> Union[float, ndarray]:
    """Returns the polar radius b given the equatorial radius a and flattening f of the ellipsoid.

    Parameters
    ----------
    a : float | ndarray
        Equatorial radius
    f : float | ndarray
        Flattening parameter

    Returns
    -------
    float | ndarray
        Polar radius :math:`b`

    Notes
    -----
    The semi minor half axis (polar radius) is defined as :math:`b = (1 - f)a`
    where :math:`a` is the semi major half axis (equatorial radius) and :math:`f` is the flattening
    of the ellipsoid.

    """
    b = a * (1.0 - f)
    return b


def third_flattening(f: Union[float, ndarray]) -> Union[float, ndarray]:
    """Returns the third flattening, n, given the flattening, f.

    Parameters
    ----------
    f : float | ndarray
        Flattening parameter

    Returns
    -------
    float | ndarray
        Polar radius :math:`n`

    Notes
    -----
    The third flattening is defined as :math:`n = f / (2 - f)`.
    """

    return f / (2.0 - f)


def array_to_list_dict(data: Union[Any, ndarray, list, tuple, dict]) -> Union[Any, dict, list]:
    """Convert dict arrays to dict of lists.

    If the input is not iterable, then the data is returned untouched.

    Parameters
    ----------
    data : Any | ndarray | list | tuple | dict
        A collection of data

    Returns
    -------
    dict | list

    Examples
    --------
    >>> data1 = dict(a=np.zeros((3,)), b=(1,2,3), c=[], d=1, e='test',
    ...              f=np.nan, g=[1], h=[np.nan], i=None)
    >>> e1 = array_to_list_dict(data1)
    >>> e1 == {'a': [0.0, 0.0, 0.0],  'b': [1, 2, 3], 'c': [],'d': 1,
    ...        'e': 'test', 'f': np.nan, 'g': [1], 'h': [np.nan], 'i': None}
    True
    >>> data2 = [1, 2., None]
    >>> e2 = array_to_list_dict(data2)
    >>> e2 == [1, 2., None]
    True
    >>> e3 = array_to_list_dict(np.array([-3., -2., -1.]))
    >>> e3 == [-3., -2., -1.]
    True
    >>> e4 = array_to_list_dict(True)
    >>> e4 is True
    True
    """
    if isinstance(data, dict):
        for key in data:
            data[key] = array_to_list_dict(data[key])
    elif isinstance(data, (list, tuple)):
        data = [array_to_list_dict(item) for item in data]
    else:
        try:
            data = data.tolist()
        except AttributeError:
            pass
    return data


def isclose(
    a: Union[int, float, list, tuple, ndarray],
    b: Union[int, float, list, tuple, ndarray],
    rtol: float=1e-9,
    atol: float=0.0,
    equal_nan: bool=False
) -> ndarray:
    """
    Returns True where the two arrays `a` and `b` are element-wise equal within a tolerance.

    Parameters
    ----------
    a : int | float | list | tuple | ndarray
        First input array or number to compare.
    b : int | float | list | tuple | ndarray
        Second input array or number to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    ndarray
        Returns a boolean array of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.

    See Also
    --------
    allclose

    Notes
    -----
    For finite values, isclose uses the following equation to test whether
    two floating point values are equivalent:

     absolute(`a` - `b`) <= maximimum(`atol`, `rtol` * maximum(absolute(`a`), absolute(`b`)))

    Like the built-in `math.isclose`, the above equation is symmetric
    in `a` and `b`. Furthermore, `atol` should be carefully selected for
    the use case at hand. A zero value for `atol` will result in `False`
    if either `a` or `b` is zero.

    For NumPy version 2, the representation of its boolean types are `np.True_` and `np.False_` when returned
    whereas `True` and `False` when they are members of arrays, respectively.

    Examples
    --------
    >>> bool(np.all(isclose([1e10,1e-7], [1.00001e10,1e-8]) == [False, False]))
    True
    >>> bool(np.all(isclose([1e10,1e-8], [1.00001e10,1e-9]) == [False, False]))
    True
    >>> bool(np.all(isclose([1e10,1e-8], [1.0001e10,1e-9]) == [False, False]))
    True
    >>> bool(np.all(isclose([1.0, np.nan], [1.0, np.nan]) == [True, False]))
    True
    >>> bool(np.all(isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True) == [True, True]))
    True
    >>> bool(np.all(isclose([1e-8, 1e-7], [0.0, 0.0]) == [False, False]))
    True
    >>> bool(np.all(isclose([1e-100, 1e-7], [0.0, 0.0], atol=0.0) == [False, False]))
    True
    >>> bool(np.all(isclose([1e-10, 1e-10], [1e-20, 0.0]) == [False, False]))
    True
    >>> bool(np.all(isclose([1e-10, 1e-10], [1e-20, 0.999999e-10], atol=0.0) == [False, False]))
    True
    """
    a, b = np.broadcast_arrays(a, b)

    mask = np.isfinite(a) & np.isfinite(b)

    out = np.full(b.shape, False)
    abs_tol = np.maximum(atol, rtol*np.maximum(np.abs(a[mask]), np.abs(b[mask])))
    out[mask] = np.isclose(a[mask], b[mask], rtol=0, atol=abs_tol, equal_nan=equal_nan)
    mask = ~mask
    out[mask] = np.isclose(a[mask], b[mask], equal_nan=equal_nan)
    return out


def allclose(
    a: Union[int, float, list, tuple, ndarray],
    b: Union[int, float, list, tuple, ndarray],
    rtol: float=1.e-7,
    atol: float=1.e-14,
    equal_nan: bool=False
) -> np.bool_:
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    Parameters
    ----------
    a : int | float | list | tuple | ndarray
        First input array or number to compare.
    b : int | float | list | tuple | ndarray
        Second input array or number to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    bool
        Returns `np.True_` if the two arrays are equal within the given
        tolerance; `np.False_` otherwise.

    See Also
    --------
    isclose, all, any, equal

    Notes
    -----
    For finite values, allclose uses the following equation to test whether
    two floating point values are equivalent:

     absolute(`a` - `b`) <= maximimum(`atol`, `rtol` * maximum(absolute(`a`), absolute(`b`)))

    NaNs are treated as equal if they are in the same place and if
    ``equal_nan=True``.  Infs are treated as equal if they are in the same
    place and of the same sign in both arrays.

    The comparison of `a` and `b` uses standard broadcasting, which
    means that `a` and `b` need not have the same shape in order for
    ``allclose(a, b)`` to evaluate to True.

    For NumPy version 2, the representation of its boolean types are `np.True_` and `np.False_` when returned
    whereas `True` and `False` when they are members of arrays, respectively.

    Examples
    --------
    >>> bool(allclose([1e10, 1e-7], [1.00001e10, 1e-8]))
    False
    >>> bool(allclose([1e10, 1e-8], [1.00001e10, 1e-9]))
    False
    >>> bool(allclose([1e10, 1e-8], [1.0001e10, 1e-9]))
    False
    >>> bool(allclose([1.0, np.nan], [1.0, np.nan]))
    False
    >>> bool(allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True))
    True

    """
    return np.all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


def _nvector_check_length(
    n_E: ndarray,
    atol: float=0.1
) -> None:
    """
    Emits a warning if nvector deviates significantly from unit length.

    Parameters
    ----------
    n_E: ndarray
        3 x n array nvector
    atol: float
          The absolute tolerance parameter (see Notes), default 0.1.

    Returns
    -------
    None
        Sends a warning to the standout output if input n-vector does not have unit length.

    Notes
    -----
    All n-vector should have unit length,  i.e. norm(n_E)=1.

    A significant deviation from that value gives a warning, i.e. when
    abs(norm(n_E)-1) > atol.
    This function only depends of the direction of n-vector, thus the warning
    is included only to give a notice in cases where a wrong input is given
    unintentionally (i.e. the input is not even approximately a unit vector).

    If a matrix of n-vectors is input, only first is controlled to save time
    (assuming advanced users input correct n-vectors)
    """
    length_deviation = abs(norm(n_E[:, 0]) - 1)
    if length_deviation > atol:
        warnings.warn('n-vector should have unit length: '
                      'norm(n_E)~=1 ! Error is: {}'.format(length_deviation))


def deg(
    *rad_angles: Union[int, float, list, tuple, ndarray]
) -> Union[float64, ndarray, Tuple[ndarray, ...]]:
    """
    Converts angle in radians to degrees.

    Parameters
    ----------
    rad_angles : int | float | list | tuple | ndarray
        Angle in radians

    Returns
    -------
    deg_angles : float64 | ndarray | tuple[ndarray, ...]
        Angle in degrees

    Examples
    --------
    >>> deg_number = deg(np.pi/2)
    >>> deg_number.size == 1
    True
    >>> deg_number.item()
    90.0
    >>> degs_array = deg(np.linspace(0, np.pi, 3))
    >>> isinstance(degs_array, ndarray)
    True
    >>> bool(allclose(degs_array, [0., 90., 180.]))
    True
    >>> degs_tuple = deg(np.pi/2, [0, np.pi])
    >>> isinstance(degs_tuple, tuple)
    True
    >>> [deg.item() if (isinstance(deg, float64) or deg.size == 1) else deg for deg in degs_tuple]
    [90.0, array([  0., 180.])]

    See also
    --------
    rad

    Note
    ----
    For NumPy version 2, the representation of its numeric types has changed. The representation of float64 for `90` is
    `np.float64(90.0)`
    """
    if len(rad_angles) == 1:
        return rad2deg(rad_angles[0])
    return tuple(rad2deg(angle) for angle in rad_angles)


def rad(
    *deg_angles: Union[int, float, list, tuple, ndarray]
) -> Union[float64, ndarray, Tuple[ndarray, ...]]:
    """
    Converts angle in degrees to radians.

    Parameters
    ----------
    deg_angles : int | float | list | tuple | ndarray
        Angle in degrees

    Returns
    -------
    rad_angles : float64 | ndarray | tuple[ndarray, ...]
        Angle in radians

    Examples
    --------
    >>> bool(np.isclose(deg(rad(90)), 90))
    True
    >>> rads = deg(*rad(90, [0, 180]))
    >>> isinstance(rads, tuple)
    True
    >>> [rad.item() if (isinstance(rad, float64) or rad.size == 1) else rad for rad in rads]
    [90.0, array([  0., 180.])]

    See also
    --------
    deg

    Notes
    -----
    For NumPy version 2, the representation of its numeric types has changed. The representation of float64 for `90` is
    `np.float64(90.0)`
    """
    if len(deg_angles) == 1:
        return deg2rad(deg_angles[0])
    return tuple(deg2rad(angle) for angle in deg_angles)


def mdot(
    a: ndarray,
    b: ndarray
) -> ndarray:
    """
    Returns multiple matrix multiplications of two arrays.

    i.e. dot(a, b)[i,j,k] = sum(a[i,:,k] * b[:,j,k])

    Parameters
    ----------
    a : ndarray
        First argument.
    b : ndarray
        Second argument.

    Returns
    -------
    ndarray
        Matrix multiplication of `a` and `b`

    Notes
    -----
    if a and b have the same shape this is the same as

    np.concatenate([np.dot(a[...,i], b[...,i])[:, :, None] for i in range(n)], axis=2)

    Examples
    --------
    3 x 3 x 2 times 3 x 3 x 2 array -> 3 x 3 x 2 array
        >>> a = 1.0 * np.arange(18).reshape(3,3,2)
        >>> b = - a
        >>> t = np.concatenate(
        ...     [np.dot( a[..., i], b[:, :, i] )[:, :, None] for i in range(2)],
        ...     axis=2
        ... )
        >>> tm = mdot(a, b)
        >>> tm.shape
        (3, 3, 2)
        >>> bool(allclose(t, tm))
        True

    3 x 3 x 2 times 3 x 1 array -> 3 x 1 x 2 array
        >>> t1 = np.concatenate([np.dot(a[:, :, i], b[:,0,0][:,None])[:,:,None]
        ...                    for i in range(2)], axis=2)

        >>> tm1 = mdot(a, b[:,0,0].reshape(-1,1))
        >>> tm1.shape
        (3, 1, 2)
        >>> bool(allclose(t1, tm1))
        True

    3 x 3  times 3 x 3 array -> 3 x 3 array
        >>> tt0 = mdot(a[:, :, 0], b[...,0])
        >>> tt0.shape
        (3, 3)
        >>> bool(allclose(t[...,0], tt0))
        True

    3 x 3  times 3 x 1 array -> 3 x 1 array
        >>> tt0 = mdot(a[:, :, 0], b[: , :1, 0])
        >>> tt0.shape
        (3, 1)
        >>> bool(allclose(t[:, :1, 0], tt0))
        True

    3 x 3  times 3 x 1 x 2 array -> 3 x 1 x 2 array
        >>> tt0 = mdot(a[:, :, 0], b[:, :2, 0][:, None])
        >>> tt0.shape
        (3, 1, 2)
        >>> bool(allclose(t[:, :2, 0][:,None], tt0))
        True

    See also
    --------
    numpy.einsum
    """
    return np.einsum('ij...,jk...->ik...', a, b)


def nthroot(
    x: Union[int, float, list, tuple, ndarray],
    n: int
) -> Union[float64, ndarray]:
    """
    Returns the n'th root of x to machine precision

    Parameters
    ----------
    x : int | float | list | tuple | ndarray
        Value
    n : int
        Integral root

    Returns
    -------
    float64 | ndarray

    Examples
    --------
    >>> bool(allclose(nthroot(27.0, 3), 3.0))
    True
    >>> bool(allclose(nthroot([27.0, 64.0, 125.0], 3), [3, 4, 5]))
    True
    """
    shape = np.shape(x)
    x = np.atleast_1d(x)
    y = x**(1. / n)
    mask = (x != 0) & (_EPS * np.abs(x) < 1)
    ym = y[mask]
    y[mask] -= (ym**n - x[mask]) / (n * ym**(n - 1))
    if shape == ():
        return y[()]
    return y


def get_ellipsoid(name: Union[int, str]) -> Ellipsoid:
    """
    Returns semi-major axis (a), flattening (f) and name of reference ellipsoid as a named tuple.

    Parameters
    ----------
    name : int | string
        Name (case insensitive) of ellipsoid or integral enumeration. Valid options are:

            1) Airy 1858
            2) Airy Modified
            3) Australian National
            4) Bessel 1841
            5) Clarke 1880
            6) Everest 1830
            7) Everest Modified
            8) Fisher 1960
            9) Fisher 1968
            10) Hough 1956
            11) Hayford / International ellipsoid 1924 / European Datum 1950 / ED50
            12) Krassovsky 1938
            13) NWL-9D / WGS 66
            14) South American 1969
            15) Soviet Geod. System 1985
            16) WGS 72
            17) Clarke 1866 / NAD27
            18) GRS80 / WGS84 / NAD83
            19) ETRS89 / EUREF89
            20) NGO1948

    Returns
    -------
    Ellipsoid
        An instance of an :py:class:`envector.util.Ellipsoid` named tuple.

    Notes
    -----
    See also:
    https://en.wikipedia.org/wiki/Geodetic_datum
    https://en.wikipedia.org/wiki/Reference_ellipsoid


    Examples
    --------
    >>> get_ellipsoid(name='wgs84')
    Ellipsoid(a=6378137.0, f=0.0033528106647474805, name='GRS80 / WGS84 / NAD83')
    >>> get_ellipsoid(name='GRS80')
    Ellipsoid(a=6378137.0, f=0.0033528106647474805, name='GRS80 / WGS84 / NAD83')
    >>> get_ellipsoid(name='NAD83')
    Ellipsoid(a=6378137.0, f=0.0033528106647474805, name='GRS80 / WGS84 / NAD83')
    >>> get_ellipsoid(name=18)
    Ellipsoid(a=6378137.0, f=0.0033528106647474805, name='GRS80 / WGS84 / NAD83')

    >>> wgs72 = get_ellipsoid(name="WGS 72")
    >>> wgs72.a == 6378135.0
    True
    >>> wgs72.f == 0.003352779454167505
    True
    >>> wgs72.name
    'WGS 72'
    >>> wgs72 == (6378135.0, 0.003352779454167505, 'WGS 72')
    True
    """
    if isinstance(name, str):
        name = name.lower().replace(' ', '').partition('/')[0]
    ellipsoid_id = ELLIPSOID_IX.get(name, name)

    return ELLIPSOID[ellipsoid_id]


select_ellipsoid = deprecate(get_ellipsoid, old_name='select_ellipsoid', new_name='get_ellipsoid')


def unit(
    vector: Union[list, tuple, ndarray],
    norm_zero_vector: Union[int, float]=1,
    norm_zero_axis: int=0
) -> ndarray:
    """
    Convert input vector to a vector of unit length.

    Parameters
    ----------
    vector : ndarray
        3 x m array (m column vectors)
    norm_zero_vector : int | float
        Defines the fill value used for zero length vectors. Either 1 or NaN
    norm_zero_axis : int
        Defines the direction that zero length vectors will point after
        the normalization is done.

    Returns
    -------
    ndarray
        3 x m array, normalized unitvector(s) along axis==0.

    Notes
    -----
    The column vector(s) that have zero length will be returned as unit vector(s)
    pointing in the x-direction, i.e, [[1], [0], [0]] if norm_zero_vector is one,
    otherwise NaN.

    Examples
    --------
    >>> bool(allclose(unit([[1, 0],[1, 0],[1, 0]]), [[ 0.57735027, 1],
    ...                                              [ 0.57735027, 0],
    ...                                              [ 0.57735027, 0]]))
    True
    """
    if not (norm_zero_vector == 1 or np.isnan(norm_zero_vector)):
        warnings.warn("The `norm_zero_vector` parameter must be either 1 or NaN")
    # Scale to avoid overflow
    unit_vector = np.atleast_1d(vector) / (np.max(np.abs(vector), axis=0, keepdims=True) + _TINY)

    current_norm = norm(unit_vector, axis=0, keepdims=True)
    unit_vector /= (current_norm + _TINY)

    idx = np.flatnonzero(current_norm == 0)
    unit_vector[:, idx] = 0 * norm_zero_vector
    unit_vector[norm_zero_axis, idx] = 1 * norm_zero_vector

    return unit_vector


_odict = globals()
__doc__ = (__doc__  # @ReservedAssignment
           + _make_summary(dict((n, _odict[n]) for n in __all__))
           + 'License\n-------\n'
           + _license.__doc__)


if __name__ == "__main__":
    test_docstrings(__file__)
