"""
Core geodesic functions
=======================
This file is part of NavLab and is available from www.navlab.net/nvector

"""
# pylint: disable=invalid-name

from __future__ import division, print_function

import warnings
from typing import Union, Tuple, Dict, Any

import numpy as np
from numpy import arctan2, sin, cos, cross, dot, sqrt, ndarray, float64
from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from envector import _examples, license as _license
from envector._common import test_docstrings, use_docstring, _make_summary
from envector.karney import (geodesic_reckon as _geodesic_reckon,
                             geodesic_distance as _geodesic_distance)
from envector.rotation import E_rotation, n_E2R_EN, n_E2lat_lon, change_axes_to_E
from envector.util import mdot, nthroot, unit, eccentricity2, polar_radius

__all__ = ['closest_point_on_great_circle',
           'cross_track_distance',
           'course_over_ground',
           'euclidean_distance',
           'geodesic_distance',
           'geodesic_reckon',
           'great_circle_distance',
           'great_circle_distance_rad',
           'great_circle_normal',
           'interp_nvectors',
           'interpolate',
           'interp_nvectors',
           'intersect',
           'mean_horizontal_position',
           'lat_lon2n_E',
           'n_E2lat_lon',
           'n_EA_E_and_n_EB_E2p_AB_E',
           'n_EA_E_and_n_EB_E2p_AB_N',
           'n_EA_E_and_p_AB_E2n_EB_E',
           'n_EA_E_and_p_AB_N2n_EB_E',
           'n_EB_E2p_EB_E',
           'p_EB_E2n_EB_E',
           'n_EA_E_distance_and_azimuth2n_EB_E',
           'n_EA_E_and_n_EB_E2azimuth',
           'on_great_circle',
           'on_great_circle_path',
           ]


def lat_lon2n_E(
    latitude: Union[int, float, list, tuple, ndarray],
    longitude: Union[int, float, list, tuple, ndarray],
    R_Ee: Union[list, tuple, ndarray, None]=None
) -> ndarray:
    """
    Converts latitude and longitude to n-vector.

    Parameters
    ----------
    latitude : int | float | list | tuple | ndarray
        Geodetic latitude scalars or vector of length n given in [rad]
    longitude : int | float | list | tuple | ndarray
        Geodetic longitude scalars or vector of length n given in [rad]
    R_Ee : list | tuple | ndarray | None
        Optional 3 x 3 rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    ndarray
        3 x n array n-vector(s) [no unit] decomposed in E.

    Examples
    --------
    >>> import envector as nv
    >>> pi = 3.141592653589793

    Scalar call
    >>> bool(nv.allclose(nv.lat_lon2n_E(0, 0), [[1.],
    ...                                         [0.],
    ...                                         [0.]]))
    True

    Vectorized call
    >>> bool(nv.allclose(nv.lat_lon2n_E([0., 0.], [0., pi/2]), [[1., 0.],
    ...                                                         [0., 1.],
    ...                                                         [0., 0.]]))
    True

    Broadcasting call
    >>> bool(nv.allclose(nv.lat_lon2n_E(0., [0, pi/2]), [[1., 0.],
    ...                                                  [0., 1.],
    ...                                                  [0., 0.]]))
    True

    See also
    --------
    n_E2lat_lon
    """
    if R_Ee is None:
        R_Ee = E_rotation()
    # Equation (3) from Gade (2010):  n-vector decomposed in E with axes='e'
    n_e = np.vstack((sin(latitude) * np.ones_like(longitude),
                     cos(latitude) * sin(longitude),
                     -cos(latitude) * cos(longitude)))
    # n_E = dot(R_Ee.T, n_e)
    n_E = np.matmul(R_Ee.T, n_e)  # n-vector decomposed in E with axes 'E'
    return n_E


@use_docstring(_examples.get_examples_no_header([4], oo_solution=False))
def n_EB_E2p_EB_E(
    n_EB_E: ndarray,
    depth: Union[int, float, ndarray]=0,
    a: float=6378137.,
    f: float=1.0 / 298.257223563,
    R_Ee: Union[list, tuple, ndarray, None]=None
) -> ndarray:
    """
    Converts n-vector to Cartesian position vector in meters.

    Parameters
    ----------
    n_EB_E : ndarray
        3 x m array n-vector(s) [no unit] of position B, decomposed in E.
    depth : int | float | ndarray
        Real scalars or 1 x n array depth(s) [m] of system B, relative to the ellipsoid (depth = -height)
    a : float
        Semi-major axis of the Earth ellipsoid given in [m], default WGS-84 ellipsoid.
    f : float.
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical Earth with radius a is used instead
        of WGS-84.
    R_Ee : list | tuple | ndarray | None
        Optional 3 x 3 rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    ndarray
        3 x max(m,n) array Cartesian position vector(s) [m] from E to B, decomposed in E.

    Notes
    -----
    The position of B (typically body) relative to E (typically Earth) is
    given into this function as n-vector, `n_EB_E`. The function converts
    to cartesian position vector ("ECEF-vector"), `p_EB_E`, in meters.
    The calculation is exact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.
    The shape of the output `p_EB_E` is the broadcasted shapes of `n_EB_E`
    and `depth`.

    Examples
    --------
    {super}

    See also
    --------
    p_EB_E2n_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E
    """
    if R_Ee is None:
        R_Ee = E_rotation()

    #     n_EB_E = np.atleast_2d(n_EB_E)
    #     _nvector_check_length(n_EB_E)
    #     n_EB_e = unit(np.matmul(R_Ee, n_EB_E))

    # Make sure to rotate the coordinates so that:
    # x -> north pole and yz-plane coincides with the equatorial
    # plane before using equation 22!
    n_EB_e = change_axes_to_E(n_EB_E, R_Ee)
    b = polar_radius(a, f)  # semi-minor axis

    # The following code implements equation (22) in Gade (2010):
    scale = np.vstack((1,
                       (1 - f),
                       (1 - f)))
    denominator = norm(n_EB_e / scale, axis=0, keepdims=True)

    # We first calculate the position at the origin of coordinate system L,
    # which has the same n-vector as B (n_EL_e = n_EB_e),
    # but lies at the surface of the Earth (z_EL = 0).

    p_EL_e = b / denominator * n_EB_e / scale**2
    # rotate back to the original coordinate system
    p_EB_E = np.matmul(R_Ee.T, p_EL_e - n_EB_e * depth)

    return p_EB_E


def _compute_k(
    a: float,
    e_2: float,
    q: ndarray,
    Ryz_2: ndarray
) -> ndarray:
    """Returns the k value in equation (23) from Gade (2010)"""
    p = Ryz_2 / a ** 2
    r = (p + q - e_2 ** 2) / 6
    s = e_2 ** 2 * p * q / (4 * r ** 3)
    t = nthroot((1 + s + sqrt(s * (2 + s))), 3)
    # t = (1 + s + sqrt(s * (2 + s)))**(1. / 3)
    u = r * (1 + t + 1. / t)
    v = sqrt(u ** 2 + e_2 ** 2 * q)
    w = e_2 * (u + v - q) / (2 * v)
    return sqrt(u + v + w ** 2) - w


def _equation23(
    a: float,
    f: float,
    p_EB_E: ndarray
) -> Tuple[ndarray, ndarray, ndarray]:
    """equation (23) from Gade (2010)"""
    Ryz_2 = p_EB_E[1, :]**2 + p_EB_E[2, :]**2
    Rx_2 = p_EB_E[0, :]**2
    e_2 = eccentricity2(f)[0]
    q = (1 - e_2) / (a ** 2) * Rx_2
    Ryz = sqrt(Ryz_2)  # Ryz = component of p_EB_E in the equatorial plane
    k = _compute_k(a, e_2, q, Ryz_2)
    d = k * Ryz / (k + e_2)
    temp0 = sqrt(d ** 2 + Rx_2)
    height = (k + e_2 - 1) / k * temp0  # Calculate height:
    x_scale = 1. / temp0
    yz_scale = x_scale * k / (k + e_2)
    return x_scale, yz_scale, -height


@use_docstring(_examples.get_examples_no_header([3], oo_solution=False))
def p_EB_E2n_EB_E(
    p_EB_E: ndarray,
    a: float=6378137.,
    f: float=1.0 / 298.257223563,
    R_Ee: Union[list, tuple, ndarray, None]=None
) -> Tuple[ndarray, ndarray]:
    """
    Converts Cartesian position vector in meters to n-vector.

    Parameters
    ----------
    p_EB_E : ndarray
        3 x n array Cartesian position vector(s) [m] from E to B, decomposed in E.
    a : float
        Semi-major axis of the Earth ellipsoid given in [m], default WGS-84 ellipsoid.
    f : float
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius `a` is used instead of WGS-84, default WGS-84 ellipsoid.
    R_Ee : list | tuple | ndarray | None
        Optional 3 x 3 rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    tuple[ndarray, ndarray]
        `(n_EB_E, depth)` where `n_EB_E` is a 3 x n array n-vector(s) [no unit] of position B, decomposed in E, and
        `depth` is a 1 x n array depth(s) [m] of system B, relative to the ellipsoid (depth = -height)

    Notes
    -----
    The position of B (typically body) relative to E (typically Earth) is
    given into this function as cartesian position vector `p_EB_E`, in meters.
    ("ECEF-vector"). The function converts to n-vector, `n_EB_E` and its `depth`.
    The calculation is exact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.

    Examples
    --------
    {super}

    See also
    --------
    n_EB_E2p_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E

    """
    if R_Ee is None:
        # R_Ee selects correct E-axes, see E_rotation for details
        R_Ee = E_rotation()

    # Make sure to rotate the coordinates so that:
    # x -> north pole and yz-plane coincides with the equatorial
    # plane before using equation 23!
    p_EB_e = np.matmul(R_Ee, p_EB_E)

    # The following code implements equation (23) from Gade (2010):
    x_scale, yz_scale, depth = _equation23(a, f, p_EB_e)

    n_EB_e_x = x_scale * p_EB_e[0, :]
    n_EB_e_y = yz_scale * p_EB_e[1, :]
    n_EB_e_z = yz_scale * p_EB_e[2, :]

    n_EB_e = np.vstack((n_EB_e_x, n_EB_e_y, n_EB_e_z))
    # Rotate back to the original coordinate system.
    n_EB_E = unit(np.matmul(R_Ee.T, n_EB_e))  # Ensure unit length

    return n_EB_E, depth


@use_docstring(_examples.get_examples_no_header([1], False))
def n_EA_E_and_n_EB_E2p_AB_E(
    n_EA_E: ndarray,
    n_EB_E: ndarray,
    z_EA: Union[int, float, ndarray]=0,
    z_EB: Union[int, float, ndarray]=0,
    a: float=6378137.,
    f: float=1.0 / 298.257223563,
    R_Ee: Union[list, tuple, ndarray, None]=None
) -> ndarray:
    """
    Returns the delta vector from position A to B decomposed in E.

    Parameters
    ----------
    n_EA_E : ndarray
        3 x j array n-vector(s) [no unit] of position A, decomposed in E.
    n_EB_E : ndarray
        3 x k array n-vector(s) [no unit] of position B, decomposed in E.
    z_EA : int | float | ndarray
        3 x m array depth(s) [m] of system A, relative to the ellipsoid.
        (z_EA = -height)
    z_EB : int | float | ndarray
        3 x n array depth(s) [m] of system A and B, relative to the ellipsoid.
        (z_EB = -height)
    a : float
        Semi-major axis of the Earth ellipsoid given in [m], default WGS-84 ellipsoid.
    f : float
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius `a` is used instead of WGS-84, default WGS-84 ellipsoid.
    R_Ee : list | tuple | ndarray | None
        Optional 3 x 3 rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    ndarray
        3 x max(j,k,m,n) array Cartesian position vector(s) [m] from A to B, decomposed in E.

    Notes
    -----
    The n-vectors for positions A (`n_EA_E`) and B (`n_EB_E`) are given. The
    output is the delta vector from A to B decompose in E (`p_AB_E`).
    The calculation is exact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.
    The shape of the output `p_AB_E` is the broadcasted shapes of `n_EA_E`, `n_EB_E`,
    `z_EA` and `z_EB`.

    Examples
    --------
    {super}


    See also
    --------
    n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_N, n_EB_E2p_EB_E,

    """

    # Function 1. in Section 5.4 in Gade (2010):
    p_EA_E = n_EB_E2p_EB_E(n_EA_E, z_EA, a, f, R_Ee)
    p_EB_E = n_EB_E2p_EB_E(n_EB_E, z_EB, a, f, R_Ee)
    p_AB_E = p_EB_E - p_EA_E
    return p_AB_E


@use_docstring(_examples.get_examples_no_header([1], False))
def n_EA_E_and_n_EB_E2p_AB_N(
    n_EA_E: ndarray,
    n_EB_E: ndarray,
    z_EA: Union[int, float, ndarray]=0,
    z_EB: Union[int, float, ndarray]=0,
    a: float=6378137.,
    f: float=1.0 / 298.257223563,
    R_Ee: Union[list, tuple, ndarray, None]=None
) -> ndarray:
    """
    Returns the delta vector from position A to B decomposed in N.

    Parameters
    ----------
    n_EA_E : ndarray
        3 x j array n-vector(s) [no unit] of position A, decomposed in E.
    n_EB_E : ndarray
        3 x k array n-vector(s) [no unit] of position B, decomposed in E.
    z_EA : int | float | ndarray
        3 x m array depth(s) [m] of system A, relative to the ellipsoid.
        (z_EA = -height)
    z_EB : int | float | ndarray
        3 x n array depth(s) [m] of system A and B, relative to the ellipsoid.
        (z_EB = -height)
    a : float
        Semi-major axis of the Earth ellipsoid given in [m], default WGS-84 ellipsoid.
    f : float
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius `a` is used instead of WGS-84, default WGS-84 ellipsoid.
    R_Ee : list | tuple | ndarray | None
        Optional 3 x 3 rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    ndarray
        3 x max(j,k,m,n) array Cartesian position vector(s) [m] from A to B, decomposed in N.

    Notes
    -----
    The n-vectors for positions A (`n_EA_E`) and B (`n_EB_E`) are given. The
    output is the delta vector from A to B decomposed in N (`p_AB_N`).
    The calculation is exact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.
    The shape of the output p_AB_N is the broadcasted shapes of `n_EA_E`, `n_EB_E`,
    `z_EA` and `z_EB`.

    Examples
    --------
    {super}


    See also
    --------
    n_EA_E_and_p_AB_E2n_EB_E, p_EB_E2n_EB_E, n_EB_E2p_EB_E, n_EA_E_and_n_EB_E2p_AB_E

    """
    p_AB_E = n_EA_E_and_n_EB_E2p_AB_E(n_EA_E, n_EB_E, z_EA, z_EB, a, f, R_Ee)

    R_EN = n_E2R_EN(n_EA_E, R_Ee=R_Ee)

    # p_AB_N = dot(R_EN.T, p_AB_E)
    p_AB_N = mdot(np.swapaxes(R_EN, 1, 0), p_AB_E[:, None, ...]).reshape(3, -1)
    # (Note the transpose of R_EN: The "closest-rule" says that when
    # decomposing, the frame in the subscript of the rotation matrix that
    # is closest to the vector, should equal the frame where the vector is
    # decomposed. Thus, the calculation np.dot(R_NE, p_AB_E) is correct,
    # since the vector is decomposed in E, and E is closest to the vector.
    # In the example we only had R_EN, and thus we must transpose it:
    # R_EN'=R_NE)
    return p_AB_N


@use_docstring(_examples.get_examples_no_header([2], oo_solution=False))
def n_EA_E_and_p_AB_E2n_EB_E(
    n_EA_E: ndarray,
    p_AB_E: ndarray,
    z_EA: Union[int, float, ndarray]=0,
    a: float=6378137.,
    f: float=1.0 / 298.257223563,
    R_Ee: Union[list, tuple, ndarray, None] = None
) -> Tuple[ndarray, ndarray]:
    """
    Returns position B from position A and delta vector decomposed in E.

    Parameters
    ----------
    n_EA_E : ndarray
        3 x k array n-vector(s) [no unit] of position A, decomposed in E.
    p_AB_E : ndarray
        3 x m array Cartesian position vector(s) [m] from A to B, decomposed in E.
    z_EA : int | float | ndarray
        1 x n array Depth(s) [m] of system A, relative to the ellipsoid. (z_EA = -height)
    a : float
        Semi-major axis of the Earth ellipsoid given in [m], default WGS-84 ellipsoid.
    f : float
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius `a` is used instead of WGS-84, default WGS-84 ellipsoid.
    R_Ee : list | tuple | ndarray | None
        Optional 3 x 3 rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    tuple[ndarray, ndarray]
        `(n_EB_E, z_EB)` where `n_EB_E` is a 3 x max(k,m,n) array n-vector(s) [no unit] of position B, decomposed in E
        and `z_EB` is a 1 x max(k,m,n) array depth(s) [m] of system B, relative to the ellipsoid. (z_EB = -height)

    Notes
    -----
    The n-vector for position A (`n_EA_E`) and the delta vector from position
    A to position B decomposed in E (`p_AB_E`) are given. The output is the
    n-vector of position B (`n_EB_E`) and depth of B (`z_EB`).
    The calculation is exact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.
    The shape of the output `n_EB_E` and `z_EB` is the broadcasted shapes of
    `n_EA_E`, `p_AB_E` and `z_EA`.

    Examples
    --------
    {super}

    See also
    --------
    n_EA_E_and_n_EB_E2p_AB_E, p_EB_E2n_EB_E, n_EB_E2p_EB_E
    """
    if R_Ee is None:
        R_Ee = E_rotation()
    n_EA_E, p_AB_E = np.atleast_2d(n_EA_E, p_AB_E)
    # Function 2. in Section 5.4 in Gade (2010):
    p_EA_E = n_EB_E2p_EB_E(n_EA_E, z_EA, a, f, R_Ee)
    p_EB_E = p_EA_E + p_AB_E
    n_EB_E, z_EB = p_EB_E2n_EB_E(p_EB_E, a, f, R_Ee)
    return n_EB_E, z_EB


@use_docstring(_examples.get_examples_no_header([2], oo_solution=False))
def n_EA_E_and_p_AB_N2n_EB_E(
    n_EA_E: ndarray,
    p_AB_N: ndarray,
    z_EA: Union[int, float, ndarray]=0,
    a: float=6378137.,
    f: float=1.0 / 298.257223563,
    R_Ee: Union[list, tuple, ndarray, None] = None
) -> Tuple[ndarray, ndarray]:
    """
    Returns position B from position A and delta vector decomposed in N.

    Parameters
    ----------
    n_EA_E : ndarray
        3 x k array n-vector(s) [no unit] of position A, decomposed in E.
    p_AB_N : ndarray
        3 x m array Cartesian position vector(s) [m] from A to B, decomposed in N.
    z_EA : int | float | ndarray
        1 x n array Depth(s) [m] of system A, relative to the ellipsoid. (z_EA = -height)
    a : float
        Semi-major axis of the Earth ellipsoid given in [m], default WGS-84 ellipsoid.
    f : float
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius `a` is used instead of WGS-84, default WGS-84 ellipsoid.
    R_Ee : list | tuple | ndarray | None
        Optional 3 x 3 rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    tuple[ndarray, ndarray]
        `(n_EB_E, z_EB)` where `n_EB_E` is a 3 x max(k,m,n) array n-vector(s) [no unit] of position B, decomposed in E
        and `z_EB` is a 1 x max(k,m,n) array depth(s) [m] of system B, relative to the ellipsoid. (z_EB = -height)

    Notes
    -----
    The n-vector for position A (n_EA_E) and the delta vector from position
    A to position B decomposed in N (p_AB_N) are given. The output is the
    n-vector of position B (`n_EB_E`) and depth of B (`z_EB`).
    The calculation is exact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.
    The shape of the output `n_EB_E` and `z_EB` is the broadcasted shapes of
    `n_EA_E`, `p_AB_N` and `z_EA`.

    Examples
    --------
    {super}

    See also
    --------
    n_EA_E_and_n_EB_E2p_AB_N,
    n_EA_E_and_p_AB_E2n_EB_E,
    n_E2R_EN
    """
    if R_Ee is None:
        R_Ee = E_rotation()
    n_EA_E, p_AB_N = np.atleast_2d(n_EA_E, p_AB_N)

    R_EN = n_E2R_EN(n_EA_E, R_Ee=R_Ee)

    # p_AB_E = dot(R_EN, p_AB_N)
    p_AB_E = mdot(R_EN, p_AB_N[:, None, ...]).reshape(3, -1)

    return n_EA_E_and_p_AB_E2n_EB_E(n_EA_E, p_AB_E, z_EA, a=a, f=f, R_Ee=R_Ee)


def _interp_vectors(
    t_i : ndarray,
    t : ndarray,
    nvectors: ndarray,
    kind : Union[int, str],
    window_length: int,
    polyorder: int,
    mode: str,
    cval: Union[int, float]
) -> ndarray:
    """
    Defines the interpolation function and return values from envector data.

    Parameters
    ----------
    t_i : ndarray
        Real vector of length m. Vector of interpolation times.
    t : ndarray
        Real vector of length n. Vector of times.
    nvectors : ndarray
        3 x n array n-vectors [no unit] decomposed in E.
    kind: str | int
        Specifies the kind of interpolation as a string
        ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
        where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
        interpolation of zeroth, first, second or third order) or as an
        integer specifying the order of the spline interpolator to use.
    window_length : int
        The length of the Savitzky-Golay filter window (i.e., the number of coefficients).
        Must be positive odd integer or zero.
    polyorder : int
        The order of the polynomial used to fit the samples.
        polyorder must be less than window_length.
    mode: str
        Accepted values are 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.
        Determines the type of extension to use for the padded signal to
        which the filter is applied.  When mode is 'constant', the padding
        value is given by cval.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree polyorder polynomial is fit to the
        last window_length values of the edges, and this polynomial is
        used to evaluate the last window_length // 2 output values.
    cval: int | float
        Value to fill past the edges of the input if mode is 'constant'.

    Returns
    -------
    ndarray
        Result is a 3 x m array of interpolated n-vector(s) [no unit] decomposed in E.
    """
    if window_length > 0:
        window_length = window_length + (window_length + 1) % 2  # make sure it is an odd integer
        options = dict(axis=1, mode=mode, cval=cval)
        normals = savgol_filter(nvectors, window_length, polyorder, **options)
    else:
        normals = nvectors

    normal_i = interp1d(t, normals, axis=1, kind=kind, bounds_error=False)(t_i)
    return normal_i.reshape(nvectors.shape[0], -1)


def interp_nvectors(
    t_i: ndarray,
    t: ndarray,
    nvectors: ndarray,
    kind: Union[int, str]='linear',
    window_length: int=0,
    polyorder: int=2,
    mode: str='interp',
    cval: Union[int, float] = 0.0
) -> ndarray:
    """
    Returns interpolated values from envector data.

    Parameters
    ----------
    t_i : ndarray
        Real vector of length m. Vector of interpolation times.
    t : ndarray
        Real vector of length n. Vector of times.
    nvectors : ndarray
        3 x n array n-vectors [no unit] decomposed in E.
    kind: str | int
        Specifies the kind of interpolation as a string
        ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
        where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
        interpolation of zeroth, first, second or third order) or as an
        integer specifying the order of the spline interpolator to use.
        Default is 'linear'.
    window_length : int
        The length of the Savitzky-Golay filter window (i.e., the number of coefficients).
        Default window_length=0, i.e. no smoothing. Must be positive odd integer or zero.
    polyorder : int
        The order of the polynomial used to fit the samples.
        polyorder must be less than window_length.
    mode: str
        Accepted values are 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.
        Determines the type of extension to use for the padded signal to
        which the filter is applied.  When mode is 'constant', the padding
        value is given by cval.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree polyorder polynomial is fit to the
        last window_length values of the edges, and this polynomial is
        used to evaluate the last window_length // 2 output values.
    cval: int | float
        Value to fill past the edges of the input if mode is 'constant'.
        Default is 0.0.

    Returns
    -------
    ndarray
        Result is a 3 x m array of interpolated n-vector(s) [no unit] decomposed in E.

    Notes
    -----
    The result for spherical Earth is returned.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import envector as nv
    >>> lat, lon = nv.rad(np.arange(0, 10)), np.sin(nv.rad(np.linspace(-90, 70, 10)))
    >>> t = np.arange(len(lat))
    >>> t_i = np.linspace(0, t[-1], 100)
    >>> nvectors = nv.lat_lon2n_E(lat, lon)
    >>> nvectors_i = nv.interp_nvectors(t_i, t, nvectors, kind='cubic')
    >>> lati, loni = nv.deg(*nv.n_E2lat_lon(nvectors_i))
    >>> h = plt.plot(nv.deg(lon), nv.deg(lat), 'o', loni, lati, '-')
    >>> plt.show()  # doctest: +SKIP
    >>> plt.close()

    Interpolate noisy data
    >>> n = 50
    >>> lat = nv.rad(np.linspace(0, 9, n))
    >>> lon = np.sin(nv.rad(np.linspace(-90, 70, n))) + 0.05* np.random.randn(n)
    >>> t = np.arange(len(lat))
    >>> t_i = np.linspace(0, t[-1], 100)
    >>> nvectors = nv.lat_lon2n_E(lat, lon)
    >>> nvectors_i = nv.interp_nvectors(t_i, t, nvectors, 'cubic', 31)
    >>> [lati, loni] = nv.n_E2lat_lon(nvectors_i)
    >>> h = plt.plot(nv.deg(lon), nv.deg(lat), 'o', nv.deg(loni), nv.deg(lati), '-')
    >>> plt.show()  # doctest: +SKIP
    >>> plt.close()


    """
    normal_i = _interp_vectors(t_i, t, nvectors, kind, window_length, polyorder, mode, cval)

    return unit(normal_i, norm_zero_vector=np.nan)


def interpolate(
    path: Tuple[ndarray, ndarray],
    ti: Union[int, float]
) -> ndarray:
    """
    Returns the interpolated point along the path

    Parameters
    ----------
    path : tuple[ndarray, ndarray]
        2-tuple of n-vectors (positionA, positionB)

    ti : int | float
        Real scalar interpolation time assuming position A and B is at t0=0 and t1=1,
        respectively.

    Returns
    -------
    ndarray
        n-vector point of interpolation along path

    Notes
    -----
    The result for spherical Earth is returned.

    """

    n_EB_E_t0, n_EB_E_t1 = path
    n_EB_E_ti = unit(n_EB_E_t0 + ti * (n_EB_E_t1 - n_EB_E_t0),
                     norm_zero_vector=np.nan)
    return n_EB_E_ti


@use_docstring(_examples.get_examples_no_header([9], oo_solution=False))
def intersect(
    path_a: Tuple[ndarray, ndarray],
    path_b: Tuple[ndarray, ndarray]
) -> ndarray:
    """
    Returns the intersection(s) between the great circles of the two paths

    Parameters
    ----------
    path_a : tuple[ndarray, ndarray]
        Defining path A with shape 2 x 3 x n
    path_b : tuple[ndarray, ndarray]
        Defining path B with shape 2 x 3 x m

    Returns
    -------
    ndarray
        Array of shape 3 x max(n, m) n-vector(s) [no unit] of position C decomposed in E.
        Point(s) of intersection between paths.

    Notes
    -----
    The result for spherical Earth is returned.
    The shape of the output `n_EC_E` is the broadcasted shapes of `path_a` and `path_b`.

    Examples
    --------
    {super}

    """
    n_EA1_E, n_EA2_E = path_a
    n_EB1_E, n_EB2_E = path_b
    # Find the intersection between the two paths, n_EC_E:
    n_EC_E_tmp = unit(cross(cross(n_EA1_E, n_EA2_E, axis=0),
                            cross(n_EB1_E, n_EB2_E, axis=0), axis=0),
                      norm_zero_vector=np.nan)

    # n_EC_E_tmp is one of two solutions, the other is -n_EC_E_tmp. Select
    # the one that is closet to n_EA1_E, by selecting sign from the dot
    # product between n_EC_E_tmp and n_EA1_E:
    n_EC_E = np.sign(dot(n_EC_E_tmp.T, n_EA1_E)) * n_EC_E_tmp
    if np.any(np.isnan(n_EC_E)):
        warnings.warn('Paths are Equal. Intersection point undefined. '
                      'NaN returned.')
    return n_EC_E


def _check_window_length(window_length: int, data: ndarray) -> int:
    """Make sure window length is odd and shorter than the length of the data"""
    n = len(data)
    window_length = window_length + (window_length + 1) % 2  # make sure it is an odd integer
    if window_length >= n:
        new_length = max(n - 1 - n % 2, 1)
        warnings.warn('Window length must be smaller than {}, but got {}!'
                      ' Truncating to {}!'.format(n, window_length, new_length))
        window_length = new_length
    return window_length


def course_over_ground(
    nvectors : ndarray,
    a: float=6378137.,
    f: float=1.0 / 298.257223563,
    R_Ee: Union[list, tuple, ndarray, None]=None,
    **options : Dict[str, Any]
) -> Union[float, ndarray]:
    """Returns course over ground in radians from nvector positions.

    An optional smoothing filter is applied if the optional keyword `window_length` value is an integer and >0.

    Parameters
    ----------
    nvectors : ndarray
        3 x n array positions of vehicle given as n-vectors [no unit] decomposed in E.
    a : float
        Semi-major axis of the Earth ellipsoid given in [m], default WGS-84 ellipsoid.
    f : float
        Flattening [no unit] of the Earth ellipsoid, default WGS-84 ellipsoid. If f==0 then spherical
        Earth with radius `a` is used instead of WGS-84.
    R_Ee : list | tuple | ndarray | None
        Optional 3 x 3 array rotation matrix defining the axes of the coordinate frame E.
    **options : dict
        Optional keyword arguments to apply a Savitzky-Golay smoothing filter window if desired.
        No smoothing is applied by default.
        Valid keyword arguments are:

            window_length : int
                positive odd integer (default = 0)
                The length of the Savitzky-Golay filter window (i.e., the number of coefficients).
                Default window_length=0, i.e. no smoothing.
            polyorder : int
                The order of the polynomial used to fit the samples (default = 2)
                polyorder must be less than window_length.
            mode: str
                Determines the type of extension to use for the padded signal to
                which the filter is applied. Accepted values are 'mirror', 'constant',
                'nearest', 'wrap', or 'interp' (default = 'nearest').
                When mode is 'constant', the padding value is given by cval.
                When the 'nearest' mode is selected (the default)
                the extension contains the nearest input value.
                When the 'interp' mode is selected, no extension
                is used.  Instead, a degree polyorder polynomial is fit to the
                last window_length values of the edges, and this polynomial is
                used to evaluate the last window_length // 2 output values.
            cval: float
                Value to fill past the edges of the input if mode is 'constant'
                (default is 0.0).

    Returns
    -------
    float | ndarray
        If n >= 2, an array of length n with angle in radians clockwise from True North to the direction towards
        which the vehicle travels. Else constant scalar NaN.

    Notes
    -----
    Please be aware that this method requires the vehicle positions to be very smooth!
    If they are not you should probably smooth it by a window_length corresponding
    to a few seconds or so.

    See https://www.navlab.net/Publications/The_Seven_Ways_to_Find_Heading.pdf
    for an overview of methods to find accurate headings.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import envector as nv
    >>> lats = nv.rad(59.381509, 59.387647)
    >>> lons = nv.rad(10.496590, 10.494713)
    >>> nvec = nv.lat_lon2n_E(lats, lons)
    >>> COG_rad = np.asarray(nv.course_over_ground(nvec))
    >>> dx, dy = np.sin(COG_rad[0]), np.cos(COG_rad[0])
    >>> COG = nv.deg(COG_rad)
    >>> p_AB_N = np.asarray(n_EA_E_and_n_EB_E2p_AB_N(nvec[:, :1], nvec[:, 1:])).ravel()
    >>> ax = plt.figure().gca()
    >>> _ = ax.plot(0, 0, 'bo', label='A')
    >>> _ = ax.arrow(0,0, dx*300, dy*300, head_width=20, label='COG')
    >>> _ = ax.plot(p_AB_N[1], p_AB_N[0], 'go', label='B')
    >>> _ = ax.set_title('COG={} degrees'.format(COG))
    >>> _ = ax.set_xlabel('East [m]')
    >>> _ = ax.set_ylabel('North [m]')
    >>> _ = ax.set_xlim(-500, 200)
    >>> _ = ax.set_aspect('equal', adjustable='box')
    >>> _ = ax.legend()
    >>> plt.show()  # doctest: +SKIP
    >>> plt.close()

    See also
    --------
    n_EA_E_and_n_EB_E2azimuth
    """
    nvectors = np.atleast_2d(nvectors)
    if nvectors.shape[1] < 2:
        return np.nan
    window_length = options.pop('window_length', 0)
    if window_length > 0:
        window_length = _check_window_length(window_length, nvectors[0])
        polyorder = options.pop('polyorder', 2)
        mode = options.pop('mode', 'nearest')
        if mode not in {'nearest', 'interp'}:
            warnings.warn('Using {} is not a recommended mode for filtering headings data!'
                          ' Use "interp" or "nearest" mode instead!'.format(mode))
        cval = options.pop('cval', 0.0)
        normal = savgol_filter(nvectors, window_length, polyorder, axis=1, mode=mode, cval=cval)
    else:
        normal = nvectors
    n_vecs = np.hstack((normal[:, :1], unit(normal[:, :-1] + normal[:, 1:]), normal[:, -1:]))

    return n_EA_E_and_n_EB_E2azimuth(n_vecs[:, :-1], n_vecs[:, 1:], a=a, f=f, R_Ee=R_Ee)


def great_circle_normal(
    n_EA_E: ndarray,
    n_EB_E: ndarray
) -> ndarray:
    """
    Returns the unit normal(s) to the great circle(s)

    Parameters
    ----------
    n_EA_E : ndarray
        3 x k array n-vector [no unit] of position A, decomposed in E.
    n_EB_E : ndarray
        3 x m array n-vector [no unit] of position B, decomposed in E.

    Returns
    -------
    ndarray
        3 x max(k, m) array unit normal(s)

    Notes
    -----
    The shape of the output `normal` is the broadcasted shapes of `n_EA_E`and `n_EB_E`.
    """
    return unit(cross(n_EA_E, n_EB_E, axis=0), norm_zero_vector=np.nan)


def _euclidean_cross_track_distance(
    sin_theta: Union[float, ndarray],
    radius: Union[int, float]=1
) -> Union[float, ndarray]:
    return sin_theta * radius


def _great_circle_cross_track_distance(
    sin_theta: Union[float, ndarray],
    radius: Union[int, float] = 1
) -> Union[float, ndarray]:
    return np.arcsin(sin_theta) * radius
    # ill conditioned for small angles:
    # return (np.arccos(-sin_theta) - np.pi / 2) * radius


@use_docstring(_examples.get_examples_no_header([10], oo_solution=False))
def cross_track_distance(
    path: Tuple[ndarray, ndarray],
    n_EB_E: ndarray,
    method: str='greatcircle',
    radius: float=6371009.0
) -> ndarray:
    """Returns cross track distance between path A and position B.

    Parameters
    ----------
    path : tuple[ndarray, ndarray]
        2-tuple of n-vectors of shape 3 x k and 3 x m, respectively, positions defining path A, decomposed in E.
    n_EB_E : ndarray
        3 x n array n-vector(s) of position B to measure the cross track distance to.
    method : str
        Defining distance calculated. Options are: 'greatcircle' or 'euclidean'
    radius : float
        Radius of sphere [m]. (default 6371009.0)

    Returns
    -------
    ndarray
        array of length max(k, m, n) cross-track distance.

    Notes
    -----
    The result for spherical Earth is returned.
    The shape of the output `distance` is the broadcasted shapes of `n_EB_E` and
    the n-vectors defining path A.

    Examples
    --------
    {super}

    See also
    --------
    great_circle_normal, closest_point_on_great_circle, on_great_circle, on_great_circle_path
    """
    c_E = great_circle_normal(path[0], path[1])
    sin_theta = -np.sum(c_E * n_EB_E, axis=0)
    if method[0].lower() == 'e':
        return _euclidean_cross_track_distance(sin_theta, radius)
    return _great_circle_cross_track_distance(sin_theta, radius)


@use_docstring(_examples.get_examples_no_header([10], oo_solution=False))
def on_great_circle(
    path: Tuple[ndarray, ndarray],
    n_EB_E: ndarray,
    radius: float=6371009.0,
    atol: float=1e-8
) -> ndarray:
    """Returns True if position B is on great circle through path A.

    Parameters
    ----------
    path : tuple[ndarray, ndarray]
        2-tuple of n-vectors of shapes 3 x k and 3 x m, respectively, of positions defining path A, decomposed in E.
    n_EB_E : ndarray
        3 x n array n-vector(s) of position B to check to.
    radius : float
        Radius of sphere. (default 6371009.0)
    atol : float
        The absolute tolerance parameter (See notes).

    Returns
    -------
    ndarray
        max(k, m, n) bool array. An element is True if position B is on great circle through path A.

    Notes
    -----
    The default value of `atol` is not zero, and is used to determine what
    small values should be considered close to zero. The default value is
    appropriate for expected values of order unity. However, `atol` should
    be carefully selected for the use case at hand. Typically, the value
    should be set to the accepted error tolerance. For GPS data the error
    ranges from 0.01 m to 15 m.
    The shape of the output `on` is the broadcasted size of `n_EB_E` and `path`.

    Examples
    --------
    {super}

    See also
    --------
    cross_track_distance
    """
    distance = np.abs(cross_track_distance(path, n_EB_E, radius=radius))
    return distance <= atol


@use_docstring(_examples.get_examples_no_header([10], oo_solution=False))
def on_great_circle_path(
    path: Tuple[ndarray, ndarray],
    n_EB_E: ndarray,
    radius: float = 6371009.0,
    atol: float = 1e-8
) -> ndarray:
    """
    Returns True if position B is on great circle and between endpoints of path A.

    Parameters
    ----------
    path : tuple[ndarray, ndarray]
        2-tuple of n-vectors of shapes 3 x k and 3 x m, respectively, of positions defining path A, decomposed in E.
    n_EB_E : ndarray
        3 x n array n-vector(s) of position B to measure the cross track distance to.
    radius : float
        Radius of sphere. (default 6371009.0)
    atol : float
        The absolute tolerance parameter (See notes).

    Returns
    -------
    ndarray
        max(k, m, n) bool array. True if position B is on great circle and between endpoints of path A.

    Notes
    -----
    The default value of `atol` is not zero, and is used to determine what
    small values should be considered close to zero. The default value is
    appropriate for expected values of order unity. However, `atol` should
    be carefully selected for the use case at hand. Typically, the value
    should be set to the accepted error tolerance. For GPS data the error
    ranges from 0.01 m to 15 m.
    The shape of the output `on` is the broadcasted shapes of `n_EB_E` and `path`.

    Examples
    --------
    {super}

    See also
    --------
    cross_track_distance, on_great_circle
    """
    n_EB_E, n_EA1_E, n_EA2_E = np.atleast_2d(n_EB_E, *path)
    scale = norm(n_EA2_E - n_EA1_E, axis=0)
    ti1 = norm(n_EB_E - n_EA1_E, axis=0) / scale
    ti2 = norm(n_EB_E - n_EA2_E, axis=0) / scale
    return (ti1 <= 1) & (ti2 <= 1) & on_great_circle(path, n_EB_E, radius, atol=atol)


@use_docstring(_examples.get_examples_no_header([10], oo_solution=False))
def closest_point_on_great_circle(
    path: Tuple[ndarray, ndarray],
    n_EB_E: ndarray
) -> ndarray:
    """
    Returns closest point C on great circle path A to position B.

    Parameters
    ----------
    path : tuple[ndarray, ndarray]
        2-tuple of n-vectors of shapes 3 x k and 3 x m, respectively, of positions defining path A, decomposed in E.
    n_EB_E : ndarray
        3 x n array n-vector(s) of position B to find the closest point to.

    Returns
    -------
    ndarray
        3 x max(k, m, n) array n-vector(s) of closest position C on great circle path A

    Notes
    -----
    The shape of the output `n_EC_E` is the broadcasted shapes of `n_EB_E` and
    the n-vectors defining path A.

    Examples
    --------
    {super}

    See also
    --------
    cross_track_distance, great_circle_normal

    """
    n_EA1_E, n_EA2_E = path
    c_E = great_circle_normal(n_EA1_E, n_EA2_E)

    c2 = cross(n_EB_E, c_E, axis=0)
    n_EC_E = unit(cross(c_E, c2, axis=0))
    return n_EC_E * np.sign(np.sum(n_EC_E * n_EB_E, axis=0, keepdims=True))


def _azimuth_sphere(
    n_EA_E: ndarray,
    n_EB_E: ndarray,
    R_Ee: Union[list, tuple, ndarray, None]=None
) -> Tuple[ndarray, ndarray]:
    """Returns azimuths from A to B and B to A, relative to North on a sphere


    See also
    https://en.wikipedia.org/wiki/Azimuth
    """
    lat1, lon1 = n_E2lat_lon(n_EA_E, R_Ee)
    lat2, lon2 = n_E2lat_lon(n_EB_E, R_Ee)

    w = lon2 - lon1
    cos_b1, sin_b1 = cos(lat1), sin(lat1)
    cos_b2, sin_b2 = cos(lat2), sin(lat2)
    cos_w, sin_w = cos(w), sin(w)

    cos_az1 = cos_b1 * sin_b2 - sin_b1 * cos_b2 * cos_w
    sin_az1 = cos_b2 * sin_w

    cos_az2 = cos_b2 * sin_b1 - sin_b2 * cos_b1 * cos_w
    sin_az2 = -cos_b1 * sin_w
    return np.arctan2(sin_az1, cos_az1), np.arctan2(sin_az2, cos_az2)


def great_circle_distance_rad(
    n_EA_E: ndarray,
    n_EB_E: ndarray,
) -> ndarray:
    """
    Returns great circle distance in radians between positions A and B on a sphere

    Parameters
    ----------
    n_EA_E : ndarray
        3 x k array n-vector(s) [no unit] of position A, decomposed in E.
    n_EB_E : ndarray
        3 x m array n-vector(s) [no unit] of position B, decomposed in E.

    Returns
    -------
    ndarray
        Array of length max(k, m), great circle distance(s) in radians

    Notes
    -----
    The result for spherical Earth is returned.
    The shape of the output `distance_rad` is the broadcasted shapes of `n_EA_E`and `n_EB_E`.
    Formula is given by equation (16) in Gade (2010) and is well
    conditioned for all angles.
    See also: https://en.wikipedia.org/wiki/Great-circle_distance.

    See also
    --------
    great_circle_distance
    """
    n_EA_E, n_EB_E = np.atleast_2d(n_EA_E, n_EB_E)

    sin_theta = norm(np.cross(n_EA_E, n_EB_E, axis=0), axis=0)
    cos_theta = np.sum(n_EA_E * n_EB_E, axis=0)

    # Alternatively:
    # sin_phi = norm(n_EA_E-n_EB_E, axis=0)/2  # phi = theta/2
    # cos_phi = norm(n_EA_E+n_EB_E, axis=0)/2
    # theta = 2 * np.arctan2(sin_phi, cos_phi)

    # ill conditioned for small angles:
    # distance_rad_version1 = arccos(dot(n_EA_E,n_EB_E))

    # ill-conditioned for angles near pi/2 (and not valid above pi/2)
    # distance_rad_version2 = arcsin(norm(cross(n_EA_E,n_EB_E)))

    return np.arctan2(sin_theta, cos_theta)


@use_docstring(_examples.get_examples_no_header([5], oo_solution=False))
def great_circle_distance(
    n_EA_E: ndarray,
    n_EB_E: ndarray,
    radius: float=6371009.0
) -> ndarray:
    """
    Returns great circle distance between positions A and B on a sphere

    Parameters
    ----------
    n_EA_E : ndarray
        3 x k array n-vector(s) [no unit] of position A, decomposed in E.
    n_EB_E : ndarray
        3 x m array n-vector(s) [no unit] of position B, decomposed in E.
    radius : float
        Radius of sphere [m]. (default 6371009.0)

    Returns
    -------
    ndarray
        Array of length max(k, m), great circle distance(s) in meters

    Notes
    -----
    The result for spherical Earth is returned.
    The shape of the output `distance` is the broadcasted shapes of `n_EA_E` and `n_EB_E`.
    Formula is given by equation (16) in Gade (2010) and is well
    conditioned for all angles.
    See also: https://en.wikipedia.org/wiki/Great-circle_distance.

    Examples
    --------
    {super}

    See also
    --------
    great_circle_distance_rad
    """
    return great_circle_distance_rad(n_EA_E, n_EB_E) * radius


def geodesic_reckon(
    n_EA_E: ndarray,
    distance: Union[int, float, list, tuple, ndarray],
    azimuth: Union[int, float, list, tuple, ndarray],
    a: float=6378137.,
    f: float=1.0 / 298.257223563,
    R_Ee: Union[list, tuple, ndarray, None]=None
) -> Tuple[ndarray, Union[int, float, ndarray]]:
    """Returns position B computed from position A, distance and azimuth.

    Parameters
    ----------
    n_EA_E : ndarray
         3 x m arrays n-vector(s) [no unit] of position A, decomposed in E.
    distance : int | float | list | tuple | ndarray
        Real scalar or vector of length n ellipsoidal distance [m] between position A and B.
    azimuth : int | float | list | tuple | ndarray
        Real scalar or vector of length n azimuth [rad or deg] of line at position A.
    a : float
        Semi-major axis of the Earth ellipsoid given in [m], default WGS-84 ellipsoid.
    f : float
        Flattening [no unit] of the Earth ellipsoid, default WGS-84 ellipsoid. If f==0 then spherical
        Earth with radius `a` is used instead of WGS-84.
    R_Ee : list | tuple | ndarray | None
        Optional 3 x 3 array rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    tuple[ndarray, int | float | ndarray]
        `(n_EB_E, alpha2)` where `n_EB_E` is a 3 x max(m,n) array n-vector(s) [no unit] of position B, decomposed in E
        and `alpha2` a real scalar or vectors of length max(m,n) of the azimuth [rad or deg] of line at position B.

    """

    lat1, lon1 = n_E2lat_lon(n_EA_E, R_Ee)
    lat2, lon2, alpha2 = _geodesic_reckon(lat1, lon1, distance, azimuth, a, f)
#     n1_e = change_axes_to_E(n_EA_E, R_Ee)
#     sin_lat1 = n1_e[0, :]
#     cos_lat1 = sqrt(n1_e[1, :]**2 + n1_e[2, :]**2)

    n_EB_E = lat_lon2n_E(lat2, lon2, R_Ee)
    return n_EB_E, alpha2


def geodesic_distance(
    n_EA_E: ndarray,
    n_EB_E: ndarray,
    a: float=6378137.,
    f: float=1.0 / 298.257223563,
    R_Ee: Union[list, tuple, ndarray, None]=None
) -> Tuple[ndarray, ndarray, ndarray]:
    """Returns surface distance between positions A and B on an ellipsoid.

    Parameters
    ----------
    n_EA_E : ndarray
        3 x m array n-vector(s) [no unit] of position A, decomposed in E.
    n_EB_E : ndarray
        3 x n array n-vector(s) [no unit] of position B, decomposed in E.
    a : float
        Semi-major axis of the Earth ellipsoid given in [m], default WGS-84 ellipsoid.
    f : float
        Flattening [no unit] of the Earth ellipsoid, default WGS-84 ellipsoid. If f==0 then spherical
        Earth with radius `a` is used instead of WGS-84.
    R_Ee : list | tuple | ndarray | None
        Optional 3 x 3 array rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        `(s12, az1, az2)` where `s12` is real scalars or vectors of length max(m,n) of the surface distance [m] from A
        to B on the ellipsoid, and `az1` and `az2` are real scalars or vectors of length max(m,n) of the direction
        [rad or deg] of line at position a and b relative to North, respectively.

    Examples
    --------
    >>> import envector as nv
    >>> n_EA_E = nv.lat_lon2n_E(0,0)
    >>> n_EB_E = nv.lat_lon2n_E(*nv.rad(0.5, 179.5))
    >>> s12, az1, az2 = nv.geodesic_distance(n_EA_E, n_EB_E)
    >>> bool(np.allclose(s12, 19936288.578965))
    True
    >>> bool(np.allclose(nv.deg(az1, az2), (25.67187286829021, 154.32708546994326)))
    True

    See also
    --------
    euclidean_distance, great_circle_distance, n_EB_E2p_EB_E

    """
    # From C.F.F. Karney (2011) "Algorithms for geodesics":
    # See also https://en.wikipedia.org/wiki/Geodesics_on_an_ellipsoid
    lat1, lon1 = n_E2lat_lon(n_EA_E, R_Ee)
    lat2, lon2 = n_E2lat_lon(n_EB_E, R_Ee)
    s12, az1, az2 = _geodesic_distance(lat1, lon1, lat2, lon2, a, f)
    return s12, az1, az2


@use_docstring(_examples.get_examples_no_header([5], oo_solution=False))
def euclidean_distance(
    n_EA_E: ndarray,
    n_EB_E: ndarray,
    radius: float=6371009.0
) -> ndarray:
    """
    Returns Euclidean distance between positions A and B on a sphere

    Parameters
    ----------
    n_EA_E : ndarray
        3 x k array n-vector(s) [no unit] of position A, decomposed in E.
    n_EB_E : ndarray
        3 x m array n-vector(s) [no unit] of position B, decomposed in E.
    radius : float
        Radius of target sphere [m]. (default 6371009.0)

    Returns
    -------
    ndarray
         Array of length max(k, m) Euclidean distance(s)

    Notes
    -----
    The shape of the output `distance` is the broadcasted shapes of `n_EB_E` and `n_EB_E`.

    Examples
    --------
    {super}
    """
    n_EB_E, n_EA_E = np.atleast_2d(n_EB_E, n_EA_E)
    d_AB = norm(n_EB_E - n_EA_E, axis=0).ravel() * radius
    return d_AB


def n_EA_E_and_n_EB_E2azimuth(
    n_EA_E: ndarray,
    n_EB_E: ndarray,
    a: float=6378137.,
    f: float=1.0 / 298.257223563,
    R_Ee: Union[list, tuple, ndarray, None]=None
) -> Union[float64, ndarray]:
    """
    Returns azimuth from A to B, relative to North:

    Parameters
    ----------
    n_EA_E : ndarray
        3 x m array n-vector(s) [no unit] of position A, decomposed in E.
    n_EB_E : ndarray
        3 x n array n-vector(s) [no unit] of position B, decomposed in E.
    a: real scalar, default WGS-84 ellipsoid.
        Semi-major axis of the Earth ellipsoid given in [m].
    f: real scalar, default WGS-84 ellipsoid.
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius `a` is used instead of WGS-84.
    R_Ee : list | tuple | ndarray | None
        Optional 3 x 3 array rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    float64 | ndarray
        Array of length max(m, n) or scalar angle [rad] the line makes with a meridian, taken clockwise from north.

    Notes
    -----
    The shape of the output `azimuth` is the broadcasted shapes of `n_EA_E` and `n_EB_E`.

    See also
    --------
    great_circle_distance_rad, n_EA_E_distance_and_azimuth2n_EB_E, course_over_ground
    """
    if R_Ee is None:
        R_Ee = E_rotation()

    #  Find p_AB_N (delta decomposed in N).
    p_AB_N = n_EA_E_and_n_EB_E2p_AB_N(n_EA_E, n_EB_E, z_EA=0, z_EB=0, a=a, f=f, R_Ee=R_Ee)

    # Find the direction (azimuth) to B, relative to north:

    return arctan2(p_AB_N[1], p_AB_N[0])


@use_docstring(_examples.get_examples_no_header([8], oo_solution=False))
def n_EA_E_distance_and_azimuth2n_EB_E(
    n_EA_E: ndarray,
    distance_rad: ndarray,
    azimuth: ndarray,
    R_Ee: Union[list, tuple, ndarray, None]=None
) -> ndarray:
    """
    Returns position B from azimuth and distance from position A

    Parameters
    ----------
    n_EA_E : ndarray
        3 x k array n-vector(s) [no unit] of position A, decomposed in E.
    distance_rad : ndarray
        m array great circle distance [rad] from position A to B
    azimuth: ndarray
        n array angle [rad] the line makes with a meridian, taken clockwise from north.
    R_Ee : list | tuple | ndarray | None
        Optional 3 x 3 array rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    ndarray
        3 x max(k,m,n) array n-vector(s) [no unit] of position B decomposed in E.

    Notes
    -----
    The result for spherical Earth is returned.
    The shape of the output `n_EB_E` is the broadcasted shapes of `n_EA_E`,
    `distance_rad` and `azimuth.

    Examples
    --------
    {super}

    See also
    --------
    n_EA_E_and_n_EB_E2azimuth, great_circle_distance_rad
    """

    if R_Ee is None:
        R_Ee = E_rotation()
    n_EA_E, distance_rad, azimuth = np.atleast_1d(n_EA_E, distance_rad, azimuth)
    # Step1: Find unit vectors for north and east:
    k_east_E = unit(cross(dot(R_Ee.T, [[1], [0], [0]]), n_EA_E, axis=0))
    k_north_E = cross(n_EA_E, k_east_E, axis=0)

    # Step2: Find the initial direction vector d_E:
    d_E = k_north_E * cos(azimuth) + k_east_E * sin(azimuth)

    # Step3: Find n_EB_E:
    n_EB_E = n_EA_E * cos(distance_rad) + d_E * sin(distance_rad)

    return n_EB_E


@use_docstring(_examples.get_examples_no_header([7], oo_solution=False))
def mean_horizontal_position(n_EB_E: ndarray) -> ndarray:
    """
    Returns the n-vector of the horizontal mean position.

    Parameters
    ----------
    n_EB_E : ndarray
        3 x n array n-vectors [no unit] of positions Bi, decomposed in E.

    Returns
    -------
    ndarray
        3 x 1 array n-vector [no unit] of the mean positions of all Bi, decomposed in E.

    Notes
    -----
    The result for spherical Earth is returned.

    Examples
    --------
    {super}

    """
    n_EM_E = unit(np.sum(n_EB_E, axis=1).reshape((3, 1)))
    return n_EM_E


_odict = globals()
__doc__ = (__doc__  # @ReservedAssignment
           + _make_summary(dict((n, _odict[n]) for n in __all__))
           + 'License\n-------\n'
           + _license.__doc__)


if __name__ == "__main__":
    test_docstrings(__file__)
