"""
Created on 13. aug. 2021

References
----------
C. F. F. Karney, "Algorithms for geodesics",
J. Geodesy 87, 43-55 (2013);
https://doi.org/10.1007/s00190-012-0578-z
Addenda: https://geographiclib.sourceforge.io/geod-addenda.html

C. F. F. Karney, "Geodesics on an ellipsoid of revolution",


@author: pab
"""

import warnings
from typing import Union, Tuple, List, Callable

import numpy as np
from numpy import arctan2, sin, cos, tan, arctan, sqrt, ndarray, float64

# from scipy.special import ellipeinc, ellipkinc  # pylint: disable=no-name-in-module
from envector.util import nthroot, eccentricity2, third_flattening, polar_radius

TINY = 1e-150
"""A tiny number"""
A1_COEFFICIENTS = (1. / 256, 1. / 64., 1. / 4., 1.)
"""A1 coefficients defined in eq. 17 in Karney:"""
C1_COEFFICIENTS = (
    (-1. / 32., 3. / 16., -1. / 2, ),  # C11
    (-9. / 2048., 1. / 32., -1. / 16.),  # C12
    (3. / 256, -1. / 48.),  # C13
    (3. / 512., -5. / 512.),  # C14
    (-7. / 1280.,),  # C15
    (-7. / 2048.,),  # C16
)
"""C1 coefficients defined in eq. 18 in Karney:"""
CM1_COEFFICIENTS = (
    (205. / 1536., -9. / 32., 1. / 2, ),  # CM11
    (1335. / 4096, -37. / 96., 5. / 16.),  # CM12
    (-75. / 128, 29. / 96.),  # CM13
    (-2391. / 2560., 539. / 1536.),  # CM14
    (3467. / 7680.,),  # CM15
    (38081. / 61440.,)  # CM16
)
"""CM1 coefficients defined in eq. 21 in Karney:"""
A2_COEFFICIENTS = (25. / 256., 9. / 64., 1./4., 1)
"""A2 coefficients defined in eq. 42 in Karney:"""
C2_COEFFICIENTS = (
    (1. / 32., 1. / 16., 1./2.),  # C21
    (35. / 2048, 1./32., 3. / 16.),  # C22
    (5. / 256, 5. / 48.),  # C23
    (7. / 512., 35. / 512.),  # C24
    (63. / 1280.,),  # C25
    (77. / 2048.,),  # C26
)
"""C2 coefficients defined in eq. 43 in Karney:"""
A3_COEFFICIENTS = (
    (-3. / 128.,),
    (-2. / 64., -3. / 64.),
    (-1. / 16., -3. / 16., -1. / 16.),
    (3. / 8., -1. / 8., -1. / 4.),
    (1. / 2., -1. / 2., ),
    (1., )
)
"""A3 coefficients defined in eq. 24 in Karney:"""
C3_COEFFICIENTS = (
    ((3, 128.), (2, 5, 128.), (-1, 3, 3, 64.), (-1, 0, 1, 8.), (-1, 1, 4.)),  # C_31
    ((5, 256.), (1, 3, 128.), (-3, -2, 3, 64.), (1, -3, 2, 32.)),  # C_32
    ((7, 512.), (-10, 9, 384.), (5, -9, 5, 192.)),  # C_33
    ((7, 512.), (-14, 7, 512.)),  # C_34
    ((21, 2560.),),  # C_35
)
"""C3 coefficients defined in eq. 25 in Karney:"""


def __astroid(
    x: ndarray,
    y: ndarray
) -> ndarray:
    """
    ASTROID  Solve the astroid equation

    K = ASTROID(X, Y) solves the quartic polynomial Eq. (55)

      K^4 + 2 * K^3 - (X^2 + Y^2 - 1) * K^2 - 2*Y^2 * K - Y^2 = 0

    for the positive root K.  X and Y are arrays of the same shape
    and the returned value K has the same shape.
    """
    x, y = np.atleast_1d(x, y)
    k = np.zeros(x.shape)
    p = x**2
    q = y**2
    r = (p + q - 1) / 6
    fl1 = ~((q == 0) & (r <= 0))
    p = p[fl1]
    q = q[fl1]
    r = r[fl1]
    pq4 = p * q / 4
    r_2 = r**2
    r_3 = r * r_2
    disc = pq4 * (pq4 + 2 * r_3)
    u = r
    fl2 = disc >= 0
    t_3 = pq4[fl2] + r_3[fl2]
    t_3 = t_3 + (1 - 2 * (t_3 < 0)) * sqrt(disc[fl2])
    t = np.sign(t_3)*nthroot(np.abs(t_3), 3)
    u[fl2] = u[fl2] + t + r_2[fl2] / np.where(t != 0, t, np.inf)
    ang = arctan2(sqrt(-disc[~fl2]), -(pq4[~fl2] + r_3[~fl2]))
    u[~fl2] = u[~fl2] + 2 * r[~fl2] * cos(ang / 3)
    v = sqrt(u**2 + q)
    u_v = u + v
    fl2 = u < 0
    u_v[fl2] = q[fl2] / (v[fl2] - u[fl2])
    w = (u_v - q) / (2 * v)
    k[fl1] = u_v / (sqrt(u_v + w**2) + w)
    return k


def _astroid(
    x: ndarray,
    y: ndarray,
    f: ndarray
) -> ndarray:
    m_u = __astroid(x, y)
    # Oblate solution
    alpha1o = np.where(y == 0,
                       arctan2(-x, sqrt(np.maximum(1 - x**2, 0))),
                       arctan2(-x * m_u / (1 + m_u), y))  # Eq. 56 and 57
    # Prolate solution
    alpha1p = np.where(y == 0,
                       arctan2(sqrt(np.maximum(1 - x**2, 0)), -x),
                       arctan2(-y, x * m_u / (1 + m_u)))
    shape = alpha1o.shape
    alpha11 = np.where((f < 0)*np.ones(shape), alpha1p, alpha1o)

    return alpha11


def _eval_cij_coefs(
    coefficients: Union[Tuple[tuple, ...], List[list]],
    epsi: ndarray,
    squared: bool=True
) -> List[ndarray]:
    epsi2 = epsi**2 if squared else epsi
    factor = 1.0
    c1x = []
    for coefs in coefficients:
        factor = factor * epsi
        c1x.append(factor*np.polyval(coefs, epsi2))
    return c1x


def _cosinesum(
    c: Union[list, tuple, ndarray],
    x: ndarray,
    sine: bool=True
):
    """
    Returns the sum of the sine or cosine series using Clenshaw algorithm.

    Parameters
    ----------
    c : list | tuple | ndarray
        sine or cosine series coefficients
    x : ndarray
        argument to the sine or cosine series.
    sine : bool
        If True the sine series sum is returned otherwise the cosine series sum.

    Returns
    -------
    ndarray
        If sine is True y = sum c[i-1] * sin( 2*i * x)
        otherwise y = sum c[i-1] * cos((2*i-1) * x) for i = 1, .... n
    """

    cosx, sinx = np.atleast_1d(cos(x), sin(x))
    n = len(c)
    factor = 2 * (cosx - sinx) * (cosx + sinx)
    y_1 = np.zeros(sinx.shape)
    is_odd = n % 2
    if is_odd:
        y_0 = c[-1]
        n = n - 1
    else:
        y_0 = y_1

    for k in range(n-1, -1, -2):
        y_1 = factor * y_0 - y_1 + c[k]
        y_0 = factor * y_1 - y_0 + c[k-1]

    if sine:
        return 2 * sinx * cosx * y_0
    return cosx * (y_0 - y_1)


def _a3_coefs(n: Union[int, float]) -> list:
    """
    Returns the A3 coefficients defined in Eq. 24 in Karney evaluated at n.

    Parameters
    ----------
    n: int | float
        third flattening of the ellipsoid

    Returns
    -------
    list
        Nested evaluated polynomial
    """
    return [np.polyval(c, n) for c in A3_COEFFICIENTS]


def _c3_coefs(n: Union[int, float]) -> List[list]:
    """
    Returns the C3 coefficients defined in Eq. 25 in Karney evaluated at n.

    Parameters
    ----------
    n: int | float
        third flattening of the ellipsoid

    Returns
    -------
    list[list]
        Doubly-nested evaluated polynomial
    """
    return [[np.polyval(c[:-1], n) / c[-1] for c in coefs]
            for coefs in C3_COEFFICIENTS]


def _get_i3_fun(
    epsi: ndarray,
    n: Union[int, float, None]=None,
    a3_coefs: Union[list, None]=None,
    c3_coefs: Union[List[list], None]=None
) -> Callable[[ndarray], ndarray]:
    """
    Returns the I3 integral function defined in equation 8 in Karney

    Parameters
    ----------
    epsi : ndarray
        normalized equatorial azimuth
    n : int | float | None
        Optional third flattening of the ellipsoid
    a3_coefs : list | None
        Optional evaluated A3 polynomial coefficients. Populated if `n` is not None
    c3_coefs : list[list] | None
        Optional evaluated C3 polynomial coefficients. Populated if `n` is not None

    Returns
    -------
    (ndarray) -> ndarray
        Callable integral function I3(sigma)

    Notes
    -----
    The I3 integral is defined as
      I3(sigma) = int (2-f)/(1+(1-f)*sqrt(1+k^2*sin(x)^2) dx from 0 to sigma

    Here
    f is the flattening
    n = f/(2-f) is the third flattening
    e = sqrt(f*(2-f)) is the eccentricity
    em = e/sqrt(1-e^2) is the second eccentricity
    alpha0 is the equatorial azimuth
    k = em*cos(alpha0)
    epsi = (sqrt(1+k^2)-1)/(sqrt(1+k^2)+1)
    sigma is the spherical arc length of the auxiliary sphere


    References
    ----------
    C. F. F. Karney, "Algorithms for geodesics",
    J. Geodesy 87, 43-55 (2013);
    https://doi.org/10.1007/s00190-012-0578-z
    Addenda: https://geographiclib.sourceforge.io/geod-addenda.html

    """
    if n is not None:
        a3_coefs = _a3_coefs(n)
        c3_coefs = _c3_coefs(n)

    a_3 = np.polyval(a3_coefs, epsi)  # Eq. 24
    c3x = _eval_cij_coefs(c3_coefs, epsi, squared=False)  # Eq 25

    def i3fun(sigma):
        return a_3*(sigma + _cosinesum(c3x, sigma, sine=True))
    return i3fun


def _get_i1_fun(
    epsi: ndarray,
    return_inverse: bool=True
) -> Union[
        Callable[[ndarray], ndarray],
        Tuple[
            Callable[[ndarray], ndarray],
            Callable[[ndarray], ndarray]
        ]
]:
    """
    Returns the I1 integral and its inverse function(s) defined in equation 7 in Karney

    Parameters
    ----------
    epsi : ndarray
        normalized equatorial azimuth
    return_inverse : bool
        If True, returns both I1 and its inverse function callables. Else just return a single callable (default = True)

    Returns
    -------
    (ndarray) -> ndarray | tuple[(ndarray) -> ndarray, (ndarray) -> ndarray]
        Either a callable integral function I1(sigma) if `return_inverse` is True else a 2-tuple of callables

    Notes
    -----
    The I1 integral is defined as
      I1(sigma) = int sqrt(1+k^2*sin(x)^2) dx from 0 to sigma

    Here
    f is the flattening

    e = sqrt(f*(2-f)) is the eccentricity
    em = e/sqrt(1-e^2) is the second eccentricity
    alpha0 is the equatorial azimuth
    k = em*cos(alpha0)
    epsi = (sqrt(1+k^2)-1)/(sqrt(1+k^2)+1) = k^2/(sqrt(1+k^2)+1)^2
    sigma is the spherical arc length of the auxiliary sphere


    References
    ----------
    C. F. F. Karney, "Algorithms for geodesics",
    J. Geodesy 87, 43-55 (2013);
    https://doi.org/10.1007/s00190-012-0578-z
    Addenda: https://geographiclib.sourceforge.io/geod-addenda.html

    """

    a_1 = np.polyval(A1_COEFFICIENTS, epsi**2) / (1.0 - epsi)  # Eq 17
    c1x = _eval_cij_coefs(C1_COEFFICIENTS, epsi, squared=True)  # Eq 18

    def i1fun(sigma: ndarray) -> ndarray:
        """The I1 function"""
        return a_1 * (sigma + _cosinesum(c1x, sigma, sine=True))

    if not return_inverse:
        return i1fun

    cm1x = _eval_cij_coefs(CM1_COEFFICIENTS, epsi, squared=True)  # Eq. 21

    def invi1fun(sdb: ndarray) -> ndarray:
        """The inverse of I1 function"""
        tau = sdb / a_1
        return tau + _cosinesum(cm1x, tau, sine=True)

#     k2 = -4 * epsi / (1-epsi)**2
#
#     def i1fun(sigma):
#         """The I1 function"""
#         return ellipeinc(sigma, k2)
#
#     def invi1fun(sdb, maxiter=20, atol=1e-12):
#         """The inverse of I1 function"""
#         sigma = sdb / a_1
#
#         for _ in range(maxiter):
#             delta = (sdb - ellipeinc(sigma, k2)) / sqrt(1 - k2 * sin(sigma)**2)
#             sigma += delta
#             if np.all(np.abs(delta) < atol):
#                 break
#         return sigma

    return i1fun, invi1fun


def _get_jfun(epsi: ndarray) -> Callable[[ndarray], ndarray]:
    """Return the callable J-function defined as J(x) = I1(x) - I2(x)

    Parameters
    ----------
    epsi : ndarray
        Normalized equatorial azimuth

    Returns
    -------
    (ndarray) -> ndarray
        J-function callable

    """
    epsi2 = epsi**2

    epsim1 = 1.0 - epsi
    a_1 = np.polyval(A1_COEFFICIENTS, epsi2) / epsim1  # Eq 17
    a_2 = np.polyval(A2_COEFFICIENTS, epsi2) * epsim1  # Eq 42
    a1m2 = a_1-a_2

    c1x = _eval_cij_coefs(C1_COEFFICIENTS, epsi, squared=True)  # Eq 18
    c2x = _eval_cij_coefs(C2_COEFFICIENTS, epsi, squared=True)  # Eq 43
    c1m2x = a_1*np.array(c1x) - a_2*np.array(c2x)

    def jfun(sigma: ndarray) -> ndarray:
        """The J function defined as I1(sigma)-I2(sigma)"""
        return a1m2 * sigma + _cosinesum(c1m2x, sigma, sine=True)

#     k2 = -4 * epsi / (1-epsi)**2
#
#     def jfun(sigma):
#         """The J function defined as I1(sigma) - I2(sigma)"""
#         return ellipeinc(sigma, k2) - ellipkinc(sigma, k2)

    return jfun


def truncate_small(x: Union[int, float, ndarray], small: float=0.06) -> ndarray:
    """Truncate tiny values to zero"""
    y = np.where(np.abs(x) < small, small - (small - x), x)
    return np.where(x == 0, 0, y)


def normalize_angle(angle: Union[int, float, ndarray]) -> ndarray:
    """Normalize angle to range (-pi, pi]"""
    mask = np.isfinite(angle)
    nangle = np.copy(angle)
    nangle[mask] = np.mod(angle[mask]+np.pi, 2*np.pi)-np.pi
    return np.where(nangle <= -np.pi, np.pi, nangle)


def _normalize_equatorial_azimuth(
    cos_alpha0: Union[int, float, ndarray],
    e2m: Union[int, float, ndarray]
) -> ndarray:
    """
    Normalize the equatorial azimuth, alpha0, given the second eccentricity squared, e2m.
    """
    k_2 = e2m * cos_alpha0 ** 2
    k_1 = sqrt(1 + k_2)
    epsi = k_2 / (k_1 + 1)**2  # Eq. 16
    return epsi


def _solve_triangle_nea_direct(
    lat1: Union[int, float, ndarray],
    alpha1: Union[int, float, ndarray],
    f: float
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Returns alpha0, sigma1, w1, cos(alpha0), sin(alpha0)"""
    blat1 = arctan((1 - f) * tan(truncate_small(lat1)))  # Eq. 6
    return _solve_triangle_nea(blat1, alpha1)


def _solve_triangle_nea(
    blat1: ndarray,
    alpha1: ndarray
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    cos_alpha1, sin_alpha1 = cos(alpha1), sin(alpha1)
    cos_blat1, sin_blat1 = cos(blat1)+TINY, sin(blat1)
    sin_alpha0 = sin_alpha1 * cos_blat1  # Eq. 5
    cos_alpha0 = np.abs(cos_alpha1 + 1j * sin_alpha1 * sin_blat1)
    # alpha0 is arctan2(sin_alpha0, cos_alpha0)  # Eq 10
    sigma1 = arctan2(sin_blat1, cos_alpha1 * cos_blat1)  # Eq 11
    w_1 = arctan2(sin_alpha0 * sin(sigma1), cos(sigma1))  # Eq 12
    return sigma1, w_1, cos_alpha0, sin_alpha0


def _solve_triangle_neb_direct(
    sigma2: ndarray,
    cos_alpha0: ndarray,
    sin_alpha0: ndarray
) -> Tuple[ndarray, ndarray, ndarray]:
    """Returns alpha2, blat2, w_2"""
    cos_sigma2, sin_sigma2 = cos(sigma2), sin(sigma2)
    sin_blat2 = cos_alpha0 * sin_sigma2
    cos_blat2 = np.abs(cos_alpha0 * cos_sigma2 + 1j * sin_alpha0)
    w_2 = arctan2(sin_alpha0 * sin_sigma2, cos_sigma2)  # Eq. 12
    blat2 = arctan2(sin_blat2, cos_blat2)  # Eq. 13
    alpha2 = arctan2(sin_alpha0, cos_alpha0 * cos_sigma2)  # Eq. 14
    return alpha2, blat2, w_2


def _solve_triangle_neb(
    cos_blat1: ndarray,
    cos_blat2: ndarray,
    sin_blat2: ndarray,
    sin_alpha0: ndarray,
    alpha1: ndarray
) -> Tuple[ndarray, ndarray, ndarray]:
    # sgn is -1 to make sure sigma2==pi for antipodal points on equator:
    sgn = np.where((sin_blat2 == 0) & (cos_blat1 == 1), -1, 1)
    cos_alpha2_cos_blat2 = sgn*np.sqrt(cos(alpha1)**2 * cos_blat1**2
                                       + (cos_blat2**2 - cos_blat1**2))

    sin_alpha2_cos_blat2 = sin(alpha1)*cos_blat1
    alpha2 = arctan2(sin_alpha2_cos_blat2, cos_alpha2_cos_blat2)  # stable at both 0 and pi/2 angle
    # alpha20 = np.arccos(cos_alpha2_cos_blat2 / cos_blat2)  # Eq 45. in Karney
    sigma2 = arctan2(sin_blat2, cos_alpha2_cos_blat2)  # Eq 11
    w_2 = np.sign(sigma2)*np.abs(arctan2(sin_alpha0 * sin(sigma2), cos(sigma2)))  # Eq 12

    return sigma2, w_2, alpha2


def sphere_distance_rad(
    lat1: Union[int, float, ndarray],
    lon1: Union[int, float, ndarray],
    lat2: Union[int, float, ndarray],
    lon2: Union[int, float, ndarray],
) -> Tuple[Union[float64, ndarray], Union[float64, ndarray], Union[float64, ndarray]]:
    """Returns surface distance between positions A and B as well as the azimuths.

    Parameters
    ----------
    lat1 : int | float | ndarray
        Real scalars or vectors of length m latitude(s) [rad] of position A.
    lon1 : int | float | ndarray
        Real scalars or vectors of length m longitude(s) [rad] of position A.
    lat2 : int | float | ndarray
        Real scalars or vectors of length n latitude(s) [rad] of position B.
    lon2 : int | float | ndarray
        Real scalars or vectors of length n longitude(s) [rad] of position B.

    Returns
    -------
    tuple[float64 | ndarray, float64 | ndarray, float64 | ndarray]
        `(distance_rad, azimuth_a, azimuth_b)`  of real scalars or vectors of length max(m,n) where `distance_rad` is
        the surface distance [rad] from A to B on the sphere and `azimuth_a` and `azimuth_b` are the directions [rad]
        of line at position A and B relative to North, respectively.

    Notes
    -----
    Solves the inverse geodesic problem of finding the length and azimuths of the
    shortest geodesic between points specified by lat1, lon1, lat2, lon2 on a sphere.

    See also
    --------
    geodesic_distance
    """
    w = lon2 - lon1
    cos_b1, sin_b1 = cos(lat1), sin(lat1)
    cos_b2, sin_b2 = cos(lat2), sin(lat2)
    cos_w, sin_w = cos(w), sin(w)

    sin_a1 = cos_b2 * sin_w
    cos_a1 = cos_b1*sin_b2-sin_b1*cos_b2*cos_w
    sin_a2 = cos_b1 * sin_w
    cos_a2 = -cos_b2 * sin_b1 + sin_b2 * cos_b1 * cos_w
    cos_distance_rad = sin_b1*sin_b2+cos_b1*cos_b2*cos_w

    sin_distance_rad = np.hypot(sin_a1, cos_a1)

    azimuth_a = arctan2(sin_a1, cos_a1)  # Eq 49
    azimuth_b = arctan2(sin_a2, cos_a2)  # Eq 50
    distance_rad = arctan2(sin_distance_rad, cos_distance_rad)  # Eq 51
    return distance_rad, azimuth_a, azimuth_b


def geodesic_reckon(
    lat_a: Union[int, float, list, tuple, ndarray],
    lon_a: Union[int, float, list, tuple, ndarray],
    distance: Union[int, float, list, tuple, ndarray],
    azimuth: Union[int, float, list, tuple, ndarray],
    a: float=6378137.,
    f: float=1.0 / 298.257223563,
    long_unroll: bool=False
) -> Tuple[Union[float, ndarray], Union[float, ndarray], Union[float, ndarray]]:
    """Returns position B computed from position A, distance and azimuth.

    Parameters
    ----------
    lat_a : int | float | list | tuple | ndarray
        Scalar or length k array vectors of latitude of position A.
    lon_a : int | float | list | tuple | ndarray
        Scalar or length k array vectors of longitude of position A.
    distance : int | float | list | tuple | ndarray
        Scalar or length m array vectors of ellipsoid distance [m] between position A and B.
    azimuth : int | float | list | tuple | ndarray
        Scalar or length n array vector azimuth [rad] of line at position A.
    a : float
        Semi-major axis of the Earth ellipsoid given in [m], default WGS-84 ellipsoid.
    f : float
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius a is used in stead of WGS-84, default WGS-84 ellipsoid.
    long_unroll : bool
        Controls the treatment of longitude. If it is False then the lon2
        is reduced to the range [-180, 180). If it is True, then (lon2 - lon1)
        determines how many times and in what sense the geodesic has encircled
        the ellipsoid.

    Returns
    -------
    tuple[float | ndarray, float | ndarray, float | ndarray]
        `(lat2, lon2, alpha2)` where  `lat2` and `lon2` are arrays of length max(k,m,n) for the latitude and longitude
        of position B and `alpha2` is a real scalar or array vectors of length max(k,m,n) of the azimuth [rad] of.
        line at position B.

    """
    lat1, lon1, distance, az1, a = np.broadcast_arrays(lat_a, lon_a, distance, azimuth, a)
    alpha1 = truncate_small(az1)
    sigma1, w_1, cos_alpha0, sin_alpha0 = _solve_triangle_nea_direct(lat1, alpha1, f)

    # Determine sigma2:
    n = third_flattening(f)
    e2m = eccentricity2(f)[1]
    epsi = _normalize_equatorial_azimuth(cos_alpha0, e2m)

    i1fun, i1inv = _get_i1_fun(epsi)
    b = polar_radius(a, f)
    s_1 = b * i1fun(sigma1)  # Eq. 7. I1(sigma1)
    s_2 = s_1 + distance
    sigma2 = i1inv(s_2/b)  # Eq. 20: Inverse of I1 where I1 is defined in Eq. 7.

    alpha2, blat2, w_2 = _solve_triangle_neb_direct(sigma2, cos_alpha0, sin_alpha0)

    # Determine lamda12
    fun_i3 = _get_i3_fun(epsi, n)
    # lamda1 is w_1 - f * sin_alpha0 * fun_i3(sigma1)  # Eq. 8
    # lamda2 is w_2 - f * sin_alpha0 * fun_i3(sigma2)  # Eq. 8

    lamda12 = w_2-w_1 + f * sin_alpha0 * (fun_i3(sigma1) - fun_i3(sigma2))

    if long_unroll:
        correction = (sigma2 - arctan2(sin(sigma2), cos(sigma2))
                      - sigma1 + arctan2(sin(sigma1), cos(sigma1)))
        sgn = np.where(sin_alpha0 >= 0, 1, -1)
        lon2 = lon1 + lamda12 + sgn * correction
    else:
        lon2 = normalize_angle(lon1 + lamda12)
    lat2 = arctan(tan(blat2)/(1-f))  # Eq. 6

    if np.ndim(lat1) == 0:
        return lat2[0], lon2[0], alpha2[0]
    return lat2, lon2, alpha2


def _solve_alpha1(
    alpha1: Union[float, ndarray],
    blat1: Union[float, ndarray],
    blat2: Union[float, ndarray],
    true_lamda12: Union[float, ndarray],
    a: Union[float, ndarray],
    f: Union[float, ndarray],
    tol: float=1e-15
) -> ndarray:
    b = polar_radius(a, f)
    eta = third_flattening(f)
    e_2, e2m = eccentricity2(f)

    sin_blat1, cos_blat1 = sin(blat1)-TINY, cos(blat1)
    sin_blat2, cos_blat2 = sin(blat2), cos(blat2)

    def _newton_step(alpha1: ndarray) -> Tuple[ndarray, ndarray]:
        """ See table 5 in Karney"""
        sigma1, w_1, cos_alpha0, sin_alpha0 = _solve_triangle_nea(blat1, alpha1)

        sigma2, w_2, alpha2 = _solve_triangle_neb(cos_blat1, cos_blat2, sin_blat2,
                                                  sin_alpha0, alpha1)

        # Determine lamda12
        epsi = _normalize_equatorial_azimuth(cos_alpha0, e2m)
        fun_i3 = _get_i3_fun(epsi, eta)
        lamda1 = w_1 - f * sin_alpha0 * fun_i3(sigma1)  # Eq. 8
        lamda2 = w_2 - f * sin_alpha0 * fun_i3(sigma2)  # Eq. 8
        lamda12 = lamda2-lamda1

        # Update alpha1
        fun_j = _get_jfun(epsi)
        k_2 = e2m * cos_alpha0 ** 2
        sin_sigma1, cos_sigma1 = sin(sigma1), cos(sigma1)
        sin_sigma2, cos_sigma2 = sin(sigma2), cos(sigma2)
        k_sin_s1 = sqrt(1+k_2*sin_sigma1**2)
        k_sin_s2 = sqrt(1+k_2*sin_sigma2**2)
        delta_j = fun_j(sigma2) - fun_j(sigma1)
        m12 = b*(k_sin_s2*cos_sigma1*sin_sigma2
                 - k_sin_s1*cos_sigma2*sin_sigma1
                 - cos_sigma1*cos_sigma2*delta_j)  # Eq 38
        # M12 is (cos_sigma1 * cos_sigma2
        #        + k_sin_s2 / k_sin_s1 * sin_sigma1 * sin_sigma2
        #        - sin_sigma1 * cos_sigma2 * delta_j / k_sin_s1)  # Eq 39
        cos_alpha2 = cos(alpha2)
        dlamda12_dalpha1 = np.where(np.abs(cos_alpha2) < tol,
                                    -sqrt(1 - e_2 * cos_blat1**2) / sin_blat1 * 2,
                                    m12 / a / (cos_alpha2 * cos_blat2))
        dlamda12 = true_lamda12-lamda12

        dalpha1 = dlamda12/dlamda12_dalpha1

        return dalpha1, dlamda12

    dalpha_old = np.zeros_like(alpha1)
    for _ in range(20):
        dalpha1, dlamda12 = _newton_step(alpha1)
        if np.any(np.isnan(dalpha1)):
            dalpha_old *= 0.5
            alpha1 -= dalpha_old
        else:
            alpha1 = (alpha1 + dalpha1).clip(0, np.pi)  # alpha1 must be in [0, pi]
            dalpha_old = dalpha1
        if np.all(np.abs(dlamda12) < 1e-12):
            break
    else:
        warnings.warn('Max iterations reached. Newton method did not converge.')
    return alpha1


def _canonical(
    blat1: ndarray,
    blat2: ndarray,
    lamda12: ndarray,
) -> Tuple[ndarray, ndarray, ndarray, Callable[[ndarray, ndarray], Tuple[ndarray, ndarray]]]:
    """Return canonical problem where blat1 <=0, blat1 <= blat2 <= -blat1 and 0<=lamda12<=pi. """
    blat1, blat2 = truncate_small(blat1), truncate_small(blat2)
    swap_bmask = np.abs(blat1) < np.abs(blat2)
    blat11, blat22 = np.where(swap_bmask, blat2, blat1), np.where(swap_bmask, blat1, blat2)
    negate_blat11 = blat11 > 0
    blat11[negate_blat11] *= -1
    blat22[negate_blat11] *= -1
    true_lamda = truncate_small(normalize_angle(lamda12))
    negate_lamda = true_lamda < 0
    true_lamda[negate_lamda] *= -1

    swap_alpha = np.logical_xor(swap_bmask, negate_blat11)

    def restore(alpha1: ndarray, alpha2: ndarray) -> Tuple[ndarray, ndarray]:
        """Restore computed values"""
        az1, az2 = np.where(swap_bmask, alpha2, alpha1), np.where(swap_bmask, alpha1, alpha2)
        az1, az2 = np.where(swap_alpha, np.pi-az1, az1), np.where(swap_alpha, np.pi-az2, az2)
        az1[negate_lamda] *= -1
        az2[negate_lamda] *= -1

        return normalize_angle(az1), normalize_angle(az2)

    return blat11, blat22, true_lamda,  restore


def geodesic_distance(
    lat_a: Union[int, float, list, tuple, ndarray],
    lon_a: Union[int, float, list, tuple, ndarray],
    lat_b: Union[int, float, list, tuple, ndarray],
    lon_b: Union[int, float, list, tuple, ndarray],
    a: float=6378137.,
    f: float=1.0 / 298.257223563
) -> Tuple[Union[float, ndarray], Union[float, ndarray], Union[float, ndarray]]:
    """Returns surface distance between positions A and B on an ellipsoid.

    Parameters
    ----------
    lat_a : int | float | list | tuple | ndarray
        Scalar or length m array vectors of latitude of position A.
    lon_a : int | float | list | tuple | ndarray
        Scalar or length m array vectors of longitude of position A.
    lat_b : int | float | list | tuple | ndarray
        Scalar or length n array vectors of latitude of position B.
    lon_b : int | float | list | tuple | ndarray
        Scalar or length n array vectors of longitude of position B.
    a : float
        Semi-major axis of the Earth ellipsoid given in [m], default WGS-84 ellipsoid.
    f : float
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius a is used in stead of WGS-84, default WGS-84 ellipsoid.

    Returns
    -------
    tuple[float | ndarray, float | ndarray, float | ndarray]
        `(s12, az1, az2)`
        distance:  real scalars or vectors of length max(m,n).
            Surface distance [m] from A to B on the ellipsoid
        azimuth_a, azimuth_b: real scalars or vectors of length max(m,n).
            direction [rad or deg] of line at position a and b relative to
            North, respectively.

    Notes
    -----
    Solves the inverse geodesic problem of finding the length and azimuths of the
    shortest geodesic between points specified by lat1, lon1, lat2, lon2 on an ellipsoid.

    See also
    --------
    sphere_distance_rad

    """
    scalar_result = np.ndim(a) == 0
    broadcast = np.broadcast_arrays
    lat1, lon1, lat2, lon2, a, f = broadcast(*np.atleast_1d(lat_a, lon_a, lat_b, lon_b, a, f))
    blat1, blat2, true_lamda12, restore = _canonical(arctan((1 - f) * tan(lat1)),  # Eq 6
                                                     arctan((1 - f) * tan(lat2)),  # Eq 6
                                                     lon2 - lon1)
    b = polar_radius(a, f)
    eta = third_flattening(f)
    e_2, e2m = eccentricity2(f)

    # assume lat1<=0 and lat1 < lat2 < -lat1 and  0 <= true_lamda12 <= pi

    cos_blat1 = cos(blat1) + TINY
    sin_blat2, cos_blat2 = sin(blat2), cos(blat2)+TINY

    def vincenty() -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """See table 3 in Karney"""
        wbar = sqrt(1 - e_2 * (0.5 * (cos_blat1 + cos_blat2))**2)  # Eq. 48
        w12 = true_lamda12 / wbar
        # Determine sigma2-sigma1 on the auxiliary sphere:
        sigma12, alpha1, alpha2 = sphere_distance_rad(blat1, 0, blat2, w12)
        s12 = a * wbar * sigma12
        return s12, alpha1, alpha2, sigma12

    def _solve_astroid() -> ndarray:
        """See table 4 in Karney"""
        delta = np.where(f == 0, 1, np.abs(f * np.pi * cos_blat1**2))

        x = (true_lamda12 - np.pi) * cos_blat1 / delta
        y = (blat1 + blat2) / delta

        alpha11 = _astroid(x, y, f)

        return alpha11

    def _solve_hybrid(alpha1: ndarray) -> Tuple[ndarray, ndarray]:
        """See table 6 in Karney"""
        sigma1, w_1, cos_alpha0, sin_alpha0 = _solve_triangle_nea(blat1, alpha1)

        sigma2, w_2, alpha2 = _solve_triangle_neb(cos_blat1, cos_blat2, sin_blat2,
                                                  sin_alpha0, alpha1)

        # Determine s12:
        epsi = _normalize_equatorial_azimuth(cos_alpha0, e2m)

        i1fun = _get_i1_fun(epsi, return_inverse=False)

        s12 = b * np.abs((i1fun(sigma2) - i1fun(sigma1)))  # Eq. 7. I1(sigma2)

        fun_i3 = _get_i3_fun(epsi, eta)
        lamda1 = w_1 - f * sin_alpha0 * fun_i3(sigma1)  # Eq. 8
        lamda2 = w_2 - f * sin_alpha0 * fun_i3(sigma2)  # Eq. 8
        lamda12 = lamda2-lamda1
        if np.any(np.abs(lamda12 - true_lamda12) > 1e-8):
            warnings.warn('Some positions did not converge using newton method!....')
        return s12, alpha2

    s12, alpha1, alpha2, sigma12 = vincenty()
    finite_mask = ~np.isnan(s12)
    if not np.any(finite_mask):
        return s12, alpha1, alpha2
    tol = 1e-12
    sin_lamda12 = sin(true_lamda12)
    sphere = (f == 0)

    meridional = (np.abs(sin_lamda12) <= tol)  # alpha1 is 0 or pi #lamda12
    delta_blat = blat2 - blat1
    equatorial = ((np.abs(delta_blat) <= tol)
                  & (np.abs(blat1) <= tol)
                  & (true_lamda12 <= (1-f)*np.pi))  # alpha1 is pi/2
    oblate = (f >= 0)
    prolate = (f < 0)
    mask = equatorial & ~(meridional & oblate)
    alpha1[mask] = np.pi/2
    alpha2[mask] = alpha1[mask]
    mask = meridional & ~(equatorial & prolate)
    alpha11 = np.sign(delta_blat)*true_lamda12
    alpha1[mask] = alpha11[mask]
    alpha2[mask] = true_lamda12[mask] - alpha1[mask]  # TODO check this
    nearly_antipodal = ~sphere & ~equatorial & (sigma12 >= np.pi*(1-3*np.abs(f)*cos_blat1**2))

    alpha1 = np.where(nearly_antipodal, _solve_astroid(), alpha1)

    short_distance = (s12 < a*1e-4)
    mask = ~(equatorial | meridional | short_distance | sphere) | nearly_antipodal
    mask *= finite_mask

    alpha1[mask] = _solve_alpha1(alpha1[mask], blat1[mask], blat2[mask], true_lamda12[mask], a, f)
    mask = ~(equatorial | short_distance | sphere) | nearly_antipodal
    mask *= finite_mask

    s12[mask], alpha2[mask] = _solve_hybrid(alpha1[mask])
    az1, az2 = restore(alpha1, alpha2)

    if scalar_result and s12.size == 1:
        return s12[0], az1[0], az2[0]
    return s12, az1, az2


if __name__ == '__main__':
    pass
