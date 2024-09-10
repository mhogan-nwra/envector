"""
Object-oriented interface to geodesic functions
===============================================

"""
from __future__ import annotations
# pylint: disable=invalid-name
import warnings
from typing import Union, Tuple, Any, Self

import numpy as np
from geographiclib.geodesic import Geodesic as _Geodesic
from numpy import ndarray, float64, bool_
from numpy.linalg import norm

from envector import _examples, license as _license
from envector._common import test_docstrings, use_docstring_from, use_docstring, _make_summary
from envector.core import (lat_lon2n_E,
                           n_E2lat_lon,
                           n_EB_E2p_EB_E,
                           p_EB_E2n_EB_E,
                           closest_point_on_great_circle,
                           course_over_ground,
                           great_circle_distance,
                           euclidean_distance,
                           cross_track_distance,
                           intersect,
                           n_EA_E_distance_and_azimuth2n_EB_E,
                           E_rotation,
                           on_great_circle_path,
                           _interp_vectors)
from envector.karney import geodesic_distance, geodesic_reckon
from envector.rotation import zyx2R, n_E_and_wa2R_EL, n_E2R_EN
from envector.util import unit, mdot, get_ellipsoid, rad, deg, isclose, allclose, array_to_list_dict, deprecate

__all__ = ['delta_E', 'delta_L', 'delta_N',
           'diff_positions',
           'FrameB', 'FrameE', 'FrameN', 'FrameL',
           'GeoPath',
           'GeoPoint',
           'ECEFvector',
           'Nvector',
           'Pvector']


@use_docstring(_examples.get_examples_no_header([1]))
def delta_E(point_a, point_b):
    """
    Returns cartesian delta vector from positions a to b decomposed in E.

    Parameters
    ----------
    point_a : Nvector | GeoPoint | ECEFvector
        Position A, decomposed in E.
    point_b : Nvector | GeoPoint | ECEFvector
        Position B, decomposed in E.

    Returns
    -------
    ECEFvector
        Cartesian position vector(s) from A to B, decomposed in E.

    Notes
    -----
    The calculation is exact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).

    Examples
    --------
    {super}

    See also
    --------
    n_EA_E_and_p_AB_E2n_EB_E, p_EB_E2n_EB_E, n_EB_E2p_EB_E
    """
    # Function 1. in Section 5.4 in Gade (2010):
    p_EA_E = point_a.to_ecef_vector()
    p_EB_E = point_b.to_ecef_vector()
    p_AB_E = p_EB_E - p_EA_E
    return p_AB_E


diff_positions = deprecate(
    delta_E,
    old_name='diff_positions',
    new_name='delta_E',
    message="Will be removed by version 1.0.0"
)


def _base_angle(angle_rad: Union[int, float, float64, ndarray]) -> Union[float64, ndarray]:
    r"""Returns angle so it is between $-\pi$ and $\pi$"""
    return np.mod(angle_rad + np.pi, 2*np.pi) - np.pi


def delta_N(point_a, point_b):
    """Returns cartesian delta vector from positions a to b decomposed in N.

    Parameters
    ----------
    point_a : Nvector | GeoPoint | ECEFvector
        Position A, decomposed in E.
    point_b : Nvector | GeoPoint | ECEFvector
        Position B, decomposed in E.

    Returns
    -------
    Pvector
        Delta vector from positions A to B, decomposed in N.

    See also
    --------
    delta_E, delta_L
    """
    # p_ab_E = delta_E(point_a, point_b)
    # p_ab_N = p_ab_E.change_frame(....)
    return delta_E(point_a, point_b).change_frame(FrameN(point_a))


def _delta(self, other):
    """Returns cartesian delta vector from positions A to B decomposed in N.

    Parameters
    ----------
    self : Nvector | GeoPoint | ECEFvector
        Position A, decomposed in N.
    other : Nvector | GeoPoint | ECEFvector
        Position B, decomposed in N.

    Returns
    -------
    Pvector
        Delta vector from positions A to B, decomposed in N.

    """
    return delta_N(self, other)


def delta_L(point_a, point_b, wander_azimuth: Union[int, float]=0):
    """Returns cartesian delta vector from positions a to b decomposed in L.

    Parameters
    ----------
    point_a : Nvector | GeoPoint | ECEFvector
        Position A, decomposed in E.
    point_b : Nvector | GeoPoint | ECEFvector
        Position B, decomposed in E.
    wander_azimuth : int | float
        Angle [rad] between the x-axis of L and the north direction.

    Returns
    -------
    Pvector
        Returns cartesian delta vector from positions a to b decomposed in L.

    See also
    --------
    delta_E, delta_N
    """
    local_frame = FrameL(point_a, wander_azimuth=wander_azimuth)
    # p_ab_E = delta_E(point_a, point_b)
    # p_ab_L = p_ab_E.change_frame(....)
    return delta_E(point_a, point_b).change_frame(local_frame)



class _Common:
    """Base class that defines the common methods for geodetic vector-like and frame-like classes"""

    _NAMES: Tuple[str, ...] = tuple()
    """Sequence of attribute names for the repr"""

    def __repr__(self) -> str:
        """Represents the class as a string

        Returns
        -------
        str
            A string like `<CLASS>(<ATTR_A>=..., <ATTR_B>=..., ...)`
        """
        cname = self.__class__.__name__
        fmt = ', '
        names = self._NAMES if self._NAMES else list(self.__dict__)
        dict_params = array_to_list_dict(self.__dict__.copy())
        if 'nvector' in dict_params:
            dict_params['point'] = dict_params['nvector']
        params = fmt.join(['{}={!r}'.format(name, dict_params[name])
                           for name in names if not name.startswith('_')])

        return '{}({})'.format(cname, params)

    def __eq__(self, other: _Common) -> bool:
        """Checks for equality by comparing identity or numerical attributes and associated frame is also itself.

        Parameters
        ----------
        other : _Common
            Any subclass of _Common.

        Returns
        -------
        bool
            True if other instance is self or numerical attributes are close and associated frame is also itself.
        """
        try:
            return self is other or self._is_equal_to(other, rtol=1e-12, atol=1e-14)
        except (AttributeError, NotImplementedError):
            return False

    def __ne__(self, other: _Common) -> bool:
        """Checks for inequality by comparing identity or numerical attributes and associated frame is also itself.

        This is the logical negation of :py:meth:`envector.objects._Common.__eq__`.

        Parameters
        ----------
        other : _Common
            Any subclass of _Common.

        Returns
        -------
        bool
            False if other instance is self or numerical attributes are close and associated frame is also itself.

        See Also
        --------
        __eq__
        """
        return not self.__eq__(other)

    def _is_equal_to(self, other: _Common, rtol: float, atol: float) -> bool:
        """Compares another object attributes of the same type"""
        raise NotImplementedError


class _FrameEBase(_Common):
    """Earth-fixed frame base-class"""

    _NAMES = ('a', 'f', 'name', 'axes')
    """Sequence of attribute names for the repr"""
    a: float
    """Semi-major axis of the Earth ellipsoid given in [m]."""
    f: float
    """Flattening [no unit] of the Earth ellipsoid."""
    name: str
    """Defining the default ellipsoid."""
    axes: str
    """Define axes orientation of E frame."""

    @property
    def R_Ee(self) -> ndarray:
        """Rotation matrix R_Ee defining the axes of the coordinate frame E"""
        raise NotImplementedError

    def inverse(self, *args, **kwargs):
        raise NotImplementedError

    def direct(self, *args, **kwargs):
        raise NotImplementedError

    def GeoPoint(self, *args, **kwds):
        """Returns a :py:class:`envector.objects.GeoPoint` instance.

        Returns
        -------
        GeoPoint
        """
        raise NotImplementedError

    def Nvector(self, *args, **kwds):
        """Returns a :py:class:`envector.objects.Nvector` instance.

        Returns
        -------
        Nvector
        """
        raise NotImplementedError

    def ECEFvector(self, *args, **kwds):
        """Returns a :py:class:`envector.objects.ECEFvector` instance.

        Returns
        -------
        ECEFvector
        """
        raise NotImplementedError


class _LocalFrameBase(_Common):

    nvector: Any
    """n-vector"""

    @property
    def R_EN(self) -> ndarray:
        raise NotImplementedError

    def Pvector(self, *args, **kwargs):
        """Returns a :py:class:`envector.objects.Pvector` instance relative to the local frame.

        Returns
        -------
        Pvector

        """
        raise NotImplementedError


class GeoPoint(_Common):
    """
    Geographical position given as latitude, longitude, depth in frame E.

    Parameters
    ----------
    latitude : int | float | tuple | list | ndarray
        Real scalars or vectors of length n, geodetic latitude in [rad or deg]
    longitude : int | float | tuple | list | ndarray
        Real scalars or vectors of length n, geodetic longitude in [rad or deg]
    z : int | float | tuple | list | ndarray
        Real scalars or vectors of length n, depth(s) [m] relative to the ellipsoid (depth = -height)
    frame : FrameE
        Reference ellipsoid. The default ellipsoid model used is WGS84, but
        other ellipsoids/spheres might be specified.
    degrees : bool
        True if input are given in degrees otherwise radians are assumed.

    Attributes
    ----------
    latitude : ndarray
        Geodetic latitudes [rad]
    longitude : ndarray
        Geodetic longitudes [rad]
    z : ndarray
        Depth(s) [m] relative to the ellipsoid (depth = -height)
    frame : FrameE
        Reference ellipsoid. The default ellipsoid model used is WGS84

    Notes
    -----
    The following arithmetic and inequality operators are defined for a :py:class:`envector.objects.GeoPoint`.

    __eq__ : bool
        Equality operator (`==`), which compares the numerical values of longitude, latitude, depth, and identical
        frames.
    __ne__ : bool
        Inequality operator (`!=`), which compares the numerical values of longitude, latitude, depth, and identical
        frames.


    Examples
    --------
    Solve geodesic problems.

    The following illustrates its use

    >>> import envector as nv
    >>> wgs84 = nv.FrameE(name='WGS84')
    >>> point_a = wgs84.GeoPoint(-41.32, 174.81, degrees=True)
    >>> point_b = wgs84.GeoPoint(40.96, -5.50, degrees=True)
    >>> print(point_a)
    GeoPoint(latitude=-0.721170046924057, longitude=3.0510100654112877, z=0, frame=FrameE(a=6378137.0, f=0.0033528106647474805, name='WGS84', axes='e'))

    The geodesic inverse problem

    >>> s12, az1, az2 = point_a.distance_and_azimuth(point_b, degrees=True)
    >>> 's12 = {:5.2f}, az1 = {:5.2f}, az2 = {:5.2f}'.format(s12, az1, az2)
    's12 = 19959679.27, az1 = 161.07, az2 = 18.83'

    The geodesic direct problem

    >>> point_a = wgs84.GeoPoint(40.6, -73.8, degrees=True)
    >>> az1, distance = 45, 10000e3
    >>> point_b, az2 = point_a.displace(distance, az1, degrees=True)
    >>> lat2, lon2 = point_b.latitude_deg, point_b.longitude_deg
    >>> msg = 'lat2 = {:5.2f}, lon2 = {:5.2f}, az2 = {:5.2f}'
    >>> msg.format(lat2, lon2, az2)
    'lat2 = 32.64, lon2 = 49.01, az2 = 140.37'

    """
    _NAMES = ('latitude', 'longitude', 'z', 'frame')
    """Sequence of attribute names for the repr"""
    latitude: ndarray
    """Geodetic latitude [rad]"""
    longitude: ndarray
    """Geodetic longitude [rad]"""
    z: ndarray
    """Depth(s) [m] relative to the ellipsoid (depth = -height)"""
    frame: FrameE
    """Frame ellipsoid"""

    def __init__(
        self,
        latitude: Union[int, float, list, tuple, ndarray],
        longitude: Union[int, float, list, tuple, ndarray],
        z: Union[int, float, list, tuple, ndarray]=0,
        frame: Union[Any, None]=None,
        degrees: bool=False
    ) -> None:
        if degrees:
            latitude, longitude = rad(latitude, longitude)
        self.latitude, self.longitude, self.z = np.broadcast_arrays(latitude, longitude, z)
        self.frame = _default_frame(frame)

    def _is_equal_to(self, other: GeoPoint, rtol: float=1e-12, atol: float=1e-14) -> bool:
        def diff(angle1, angle2):
            pi2 = 2 * np.pi
            delta = (angle1 - angle2) % pi2
            return np.where(delta > np.pi, pi2 - delta, delta)

        options = dict(rtol=rtol, atol=atol)
        delta_lat = diff(self.latitude, other.latitude)
        delta_lon = diff(self.longitude, other.longitude)
        return (allclose(delta_lat, 0, **options)
                and allclose(delta_lon, 0, **options)
                and allclose(self.z, other.z, **options)
                and self.frame == other.frame)

    @property
    def latlon_deg(self) -> Tuple[ndarray, ndarray, ndarray]:
        """Returns the latitude [deg], longitude [deg], and depth [m].

        Returns
        -------
        tuple[ndarray, ndarray, ndarray]

        Notes
        -----
        The depth is defined as depth = -height
        """
        return self.latitude_deg, self.longitude_deg, self.z

    @property
    def latlon(self) -> Tuple[ndarray, ndarray, ndarray]:
        """Returns the latitude [rad], longitude [rad], and depth [m].

        Returns
        -------
        tuple[ndarray, ndarray, ndarray]

        Notes
        -----
        The depth is defined as depth = -height

        See Also
        --------
        latlon_deg
        """
        return self.latitude, self.longitude, self.z

    @property
    def latitude_deg(self) -> ndarray:
        """Latitude in degrees.

        Returns
        -------
        ndarray
        """
        return deg(self.latitude)

    @property
    def longitude_deg(self) -> ndarray:
        """Longitude in degrees

        Returns
        -------
        ndarray
            Longitude
        """
        return deg(self.longitude)

    @property
    def scalar(self) -> bool:
        """Is a scalar point

        Returns
        -------
        bool
            True if the position is a scalar point
        """
        return (np.ndim(self.z) == 0
                and np.size(self.latitude) == 1
                and np.size(self.longitude) == 1)

    def to_ecef_vector(self) -> ECEFvector:
        """Position as ECEFvector object.

        Returns
        -------
        ECEFvector
            Returns position as ECEFvector object.

        See also
        --------
        ECEFvector
        """
        return self.to_nvector().to_ecef_vector()

    def to_geo_point(self) -> Self:
        """
        Position as GeoPoint object, in this case itself.

        Returns
        -------
        GeoPoint
            Self

        See also
        --------
        GeoPoint
        """
        return self

    def to_nvector(self) -> Nvector:
        """
        Position as Nvector object.

        Returns
        -------
        Nvector
            Returns position as Nvector object.

        See also
        --------
        Nvector
        """
        latitude, longitude = self.latitude, self.longitude
        n_vector = lat_lon2n_E(latitude, longitude, self.frame.R_Ee)
        return Nvector(n_vector, self.z, self.frame)

    delta_to = _delta

    def _displace_great_circle(
        self,
        distance: Union[int, float, list, tuple, ndarray],
        azimuth: Union[int, float, list, tuple, ndarray],
        degrees: bool
    ) -> Tuple[GeoPoint, Union[float64, ndarray]]:
        """Returns the great circle solution using the nvector method.

        Parameters
        ----------
        distance : int | float | list | tuple | ndarray
        azimuth : int | float | list | tuple | ndarray
        degrees : bool

        Returns
        -------
        tuple[GeoPoint, float64 | ndarray]

        """
        n_a = self.to_nvector()
        e_a = n_a.to_ecef_vector()
        radius = e_a.length
        distance_rad = distance / radius
        azimuth_rad = azimuth if not degrees else rad(azimuth)
        normal_b = n_EA_E_distance_and_azimuth2n_EB_E(n_a.normal, distance_rad, azimuth_rad)
        point_b = Nvector(normal_b, self.z, self.frame).to_geo_point()
        azimuth_b = _base_angle(delta_N(point_b, e_a).azimuth - np.pi)
        if degrees:
            return point_b, deg(azimuth_b)
        return point_b, azimuth_b

    def displace(
        self,
        distance: Union[int, float, list, tuple, ndarray],
        azimuth: Union[int, float, list, tuple, ndarray],
        long_unroll: bool=False,
        degrees: bool=False,
        method: str='ellipsoid'
    ) -> Tuple[GeoPoint, Union[float64, ndarray]]:
        """
        Returns position b computed from current position, distance and azimuth.

        Parameters
        ----------
        distance : int | float | list | tuple | ndarray
            Real scalars or vectors of length n ellipsoidal or great circle distance [m] between position A and B.
        azimuth : int | float | list | tuple | ndarray
            Real scalars or vectors of length n azimuth [rad or deg] of line at position A.
        long_unroll : bool
            Controls the treatment of longitude when method=='ellipsoid'.
            See distance_and_azimuth method for details.
        degrees : bool
            azimuths are given in degrees if True otherwise in radians.
        method : str
            Either 'greatcircle' or 'ellipsoid', defining the path where to find position B.

        Returns
        -------
        tuple[GeoPoint, float64 | ndarray]
            point_b:  GeoPoint object
                latitude and longitude of position B.
            azimuth_b: real scalars or vectors of length n.
                azimuth [rad or deg] of line at position B.
        """
        if method[:1] == 'e':  # exact solution
            return self._displace_ellipsoid(distance, azimuth, long_unroll, degrees)
        return self._displace_great_circle(distance, azimuth, degrees)

    def _displace_ellipsoid(
        self,
        distance: Union[int, float, list, tuple, ndarray],
        azimuth: Union[int, float, list, tuple, ndarray],
        long_unroll: bool=False,
        degrees: bool=False
    ) -> Tuple[GeoPoint, Union[float64, ndarray]]:
        """Returns the exact ellipsoidal solution using the method of Karney.

        Parameters
        ----------
        distance : int | float | list | tuple | ndarray
            Real scalars or vectors of length n ellipsoidal or great circle distance [m] between position A and B.
        azimuth : int | float | list | tuple | ndarray
            Real scalars or vectors of length n azimuth [rad or deg] of line at position A.
        long_unroll : bool
            Controls the treatment of longitude when method=='ellipsoid'.
            See distance_and_azimuth method for details.
        degrees : bool
            azimuths are given in degrees if True otherwise in radians.

        Returns
        -------
        tuple[GeoPoint, float64 | ndarray]
            point_b:  GeoPoint object
                latitude and longitude of position B.
            azimuth_b: real scalars or vectors of length n.
                azimuth [rad or deg] of line at position B.
        """
        frame = self.frame
        z = self.z
        if not degrees:
            azimuth = deg(azimuth)
        lat_a, lon_a = self.latitude_deg, self.longitude_deg
        lat_b, lon_b, azimuth_b = frame.direct(lat_a, lon_a, azimuth, distance,
                                               z=z, long_unroll=long_unroll,
                                               degrees=True)

        point_b = frame.GeoPoint(latitude=lat_b, longitude=lon_b, z=z, degrees=True)
        if not degrees:
            return point_b, rad(azimuth_b)
        return point_b, azimuth_b

    def distance_and_azimuth(
        self,
        point: GeoPoint,
        degrees: bool=False,
        method: str='ellipsoid'
    ) -> Tuple[Union[float64, ndarray], Union[float64, ndarray], Union[float64, ndarray]]:
        """
        Returns ellipsoidal distance between positions as well as the direction.

        Parameters
        ----------
        point : GeoPoint
            Latitude and longitude of position B.
        degrees : bool
            Azimuths are returned in degrees if True otherwise in radians.
        method : str
            Either 'greatcircle' or 'ellipsoid' defining the path distance.

        Returns
        -------
        tuple[float64 | ndarray, float64 | ndarray, float64 | ndarray]
            s_ab: real scalar or vector of length n.
                ellipsoidal distance [m] between position a and b at their average height.
            azimuth_a, azimuth_b: real scalars or vectors of length n.
                direction [rad or deg] of line at position a and b relative to
                North, respectively.

        Notes
        -----
        Restriction on the parameters:
        * Latitudes must lie between -90 and 90 degrees.
        * Latitudes outside this range will be set to NaNs.
        * The flattening f should be between -1/50 and 1/50 inn order to retain full accuracy.

        Examples
        --------
        >>> import envector as nv
        >>> point1 = nv.GeoPoint(0, 0)
        >>> point2 = nv.GeoPoint(0.5, 179.5, degrees=True)
        >>> s_12, az1, azi2 = point1.distance_and_azimuth(point2)
        >>> bool(nv.allclose(s_12, 19936288.579))
        True

        References
        ----------
        `C. F. F. Karney, Algorithms for geodesics, J. Geodesy 87(1), 43-55 (2013)
        <https://rdcu.be/cccgm>`_

        `geographiclib <https://pypi.python.org/pypi/geographiclib>`_
        """
        _check_frames(self, point)
        if method[0] == 'e':
            return self._distance_and_azimuth_ellipsoid(point, degrees)
        return self._distance_and_azimuth_greatcircle(point, degrees)

    def _distance_and_azimuth_greatcircle(
        self,
        point: GeoPoint,
        degrees: bool
    ) -> Tuple[Union[float64, ndarray], Union[float64, ndarray], Union[float64, ndarray]]:
        """Returns great-circle distance between positions as well as the direction.

        Parameters
        ----------
        point : GeoPoint
            Other point
        degrees : bool
            Azimuths are returned in degrees if True otherwise in radians.

        Returns
        -------
        tuple[float64 | ndarray, float64 | ndarray, float64 | ndarray]
            distance: real scalar or vector of length n.
                ellipsoidal distance [m] between position a and b at their average height.
            azimuth_a, azimuth_b: real scalars or vectors of length n.
                direction [rad or deg] of line at position a and b relative to
                North, respectively.
        """
        n_a = self.to_nvector()
        n_b = point.to_nvector()
        e_a = n_a.to_ecef_vector()
        e_b = n_b.to_ecef_vector()
        radius = 0.5 * (e_a.length + e_b.length)
        distance = great_circle_distance(n_a.normal, n_b.normal, radius)
        azimuth_a = delta_N(e_a, e_b).azimuth
        azimuth_b = _base_angle(delta_N(e_b, e_a).azimuth - np.pi)

        if degrees:
            azimuth_a, azimuth_b = deg(azimuth_a), deg(azimuth_b)

        if np.ndim(radius) == 0:
            return distance[0], azimuth_a, azimuth_b  # scalar track distance
        return distance, azimuth_a, azimuth_b

    def _distance_and_azimuth_ellipsoid(
        self,
        point: GeoPoint,
        degrees: bool
    ) -> Tuple[Union[float64, ndarray], Union[float64, ndarray], Union[float64, ndarray]]:
        """

        Parameters
        ----------
        point : GeoPoint
            Other point
        degrees : bool
            Azimuths are returned in degrees if True otherwise in radians.

        Returns
        -------
        tuple[float64 | ndarray, float64 | ndarray, float64 | ndarray]

        """
        gpoint = point.to_geo_point()
        lat_a, lon_a = self.latitude, self.longitude
        lat_b, lon_b = gpoint.latitude, gpoint.longitude
        z = 0.5 * (self.z + gpoint.z)  # Average depth

        if degrees:
            lat_a, lon_a, lat_b, lon_b = deg(lat_a, lon_a, lat_b, lon_b)

        return self.frame.inverse(lat_a, lon_a, lat_b, lon_b, z, degrees)


class Nvector(_Common):
    """
    Geographical position(s) given as n-vector(s) and depth(s) in frame E

    Parameters
    ----------
    normal : ndarray
        3 x n array n-vector(s) [no unit] decomposed in E.
    z : int | float | ndarray
        Real scalar or vector of length n depth(s) [m] relative to the ellipsoid (depth = -height)
    frame : FrameE
        Reference ellipsoid. The default ellipsoid model used is WGS84, but
        other ellipsoids/spheres might be specified.

    Attributes
    ----------
    normal : ndarray
        Normal vector(s) [no unit] decomposed in E.
    z : int | float | ndarray
        Depth(s) [m] relative to the ellipsoid (depth = -height)
    frame : FrameE
        Reference ellipsoid. The default ellipsoid model used is WGS84

    Notes
    -----
    The position of B (typically body) relative to E (typically Earth) is
    given into this function as n-vector, n_EB_E and a depth, z relative to the
    ellipsiod.

    The following arithmetic and inequality operators are defined for a :py:class:`envector.objects.Nvector`.

    __eq__ : bool
        Equality operator (`==`), which compares the numerical values of the normal vector, depth (`z`), and identical
        frames.
    __ne__ : bool
        Inequality operator (`!=`), which compares the numerical values of the normal vector, depth (`z`), and identical
        frames.
    __neg__ : Nvector
        Negation operator (`-`) that returns a new Nvector with negated normal vector and depth, see
        :py:meth:`envector.objects.Nvector.__neg__` for more details.
    __add__ : Nvector
        Additional operator (`+`) adds self with another Nvector in the same frame, see
        :py:meth:`envector.objects.Nvector.__add__` for more details.
    __sub__ : Nvector
        Subtraction operator (`-`) subtract self with another Nvector in the same frame, see
        :py:meth:`envector.objects.Nvector.__add__` for more details.
    __mul__ : Nvector
        Multiplication operator (`*`) multiplies the `normal` and `z` attributes with a scalar,
        see :py:meth:`envector.objects.Nvector.__mul__` for more details.
    __div__ : Nvector
        Division operator (`/`) divides the `normal` and `z` attributes with a scalar,
        see :py:meth:`envector.objects.Nvector.__mul__` for more details.
    __radd__ : Nvector
        Right-hand-side addition operator (`+`) is also the :py:meth:`envector.objects.Nvector.__add__` operator.
    __rmul__ : Nvector
        Right-hand-side subtraction operator (`-`) is also the :py:meth:`envector.objects.Nvector.__mul__` operator.
    __truediv__ : Nvector
        The true division operator (`//`) is also :py:meth:`envector.objects.Nvector.__div__` operator.

    Examples
    --------
    >>> import envector as nv
    >>> wgs84 = nv.FrameE(name='WGS84')
    >>> point_a = wgs84.GeoPoint(-41.32, 174.81, degrees=True)
    >>> point_b = wgs84.GeoPoint(40.96, -5.50, degrees=True)
    >>> nv_a = point_a.to_nvector()
    >>> print(nv_a)
    Nvector(normal=[[-0.7479546170813224], [0.06793758070955484], [-0.6602638683996461]], z=0, frame=FrameE(a=6378137.0, f=0.0033528106647474805, name='WGS84', axes='e'))


    See also
    --------
    GeoPoint, ECEFvector, Pvector
    """

    _NAMES = ('normal', 'z', 'frame')
    """Sequence of attribute names for the repr"""
    normal: ndarray
    """Normal vector(s)"""
    z: Union[int, float, ndarray]
    """Depth(s) [m]"""
    frame: FrameE
    """Reference ellipsoid"""

    def __init__(
        self,
        normal: ndarray,
        z: Union[int, float, ndarray]=0,
        frame: Union[_FrameEBase, None]=None
    ) -> None:
        self.normal = normal
        self.z = z
        self.frame = _default_frame(frame)

    @property
    def scalar(self) -> bool:
        """Is the position is a scalar point?

        Returns
        -------
        bool
            True if the position is a scalar point (i.e. depth `z` has no dimensions and the n-vector is shape is
            `(3, 1)`)
        """
        return np.ndim(self.z) == 0 and self.normal.shape[1] == 1

    def interpolate(
        self,
        t_i: ndarray,
        t: ndarray,
        kind: Union[int, str]='linear',
        window_length: int=0,
        polyorder: int=2,
        mode: str='interp',
        cval: Union[int, float]=0.0
    ) -> Nvector:
        """
        Returns interpolated values from nvector data.

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
            Must be positive odd integer or zero. Default window_length=0, i.e. no smoothing.
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
        Nvector
            Interpolated Nvector decomposed in E.

        Notes
        -----
        The result for spherical Earth is returned.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import envector as nv
        >>> lat = np.arange(0, 10)
        >>> lon = nv.deg(np.sin(nv.rad(np.linspace(-90, 70, 10))))
        >>> nvectors = nv.GeoPoint(lat, lon, degrees=True).to_nvector()
        >>> t = np.arange(10)
        >>> t_i = np.linspace(0, t[-1], 100)
        >>> nvectors_i = nvectors.interpolate(t_i, t, kind='cubic')
        >>> lati, loni, zi = nvectors_i.to_geo_point().latlon_deg
        >>> h = plt.plot(lon, lat, 'o', loni, lati, '-')
        >>> plt.show() # doctest: +SKIP
        >>> plt.close()
        """
        vectors = np.vstack((self.normal, self.z))
        vectors_i = _interp_vectors(t_i, t, vectors, kind, window_length, polyorder, mode, cval)
        normal = unit(vectors_i[:3], norm_zero_vector=np.nan)
        return Nvector(normal, z=vectors_i[3], frame=self.frame)

    def to_ecef_vector(self) -> ECEFvector:
        """
        Returns position as ECEFvector object.

        Returns
        -------
        ECEFvector

        See also
        --------
        ECEFvector
        """
        frame = self.frame
        a, f, R_Ee = frame.a, frame.f, frame.R_Ee
        pvector = n_EB_E2p_EB_E(self.normal, depth=self.z, a=a, f=f, R_Ee=R_Ee)
        scalar = self.scalar
        return ECEFvector(pvector, self.frame, scalar=scalar)

    def to_geo_point(self) -> GeoPoint:
        """
        Returns position as GeoPoint object.

        See also
        --------
        GeoPoint
        """
        latitude, longitude = n_E2lat_lon(self.normal, R_Ee=self.frame.R_Ee)

        if self.scalar:
            return GeoPoint(latitude[0], longitude[0], self.z, self.frame)  # Scalar geo_point
        return GeoPoint(latitude, longitude, self.z, self.frame)

    def to_nvector(self) -> Self:
        """Position as Nvector object, in this case, self.

        Returns
        -------
        Nvector
            Self

        See Also
        --------
        Nvector
        """
        return self

    delta_to = _delta

    def unit(self) -> None:
        """Normalizes self to unit vector(s)"""
        self.normal = unit(self.normal)

    def course_over_ground(self, **options) -> Union[float64, ndarray]:
        """Returns course over ground in radians from nvector positions

        Parameters
        ----------
        **options : dict
            Optional keyword arguments for the :py:function:`envector.core.course_over_ground` function.
            Supported keyword arguments areL

                window_length: positive odd integer
                    The length of the Savitzky-Golay filter window (i.e., the number of coefficients).
                    Default window_length=0, i.e. no smoothing.
                polyorder: int {2}
                    The order of the polynomial used to fit the samples.
                    polyorder must be less than window_length.
                mode: 'mirror', 'constant', {'nearest'}, 'wrap' or 'interp'.
                    Determines the type of extension to use for the padded signal to
                    which the filter is applied.  When mode is 'constant', the padding
                    value is given by cval. When the 'nearest' mode is selected (the default)
                    the extension contains the nearest input value.
                    When the 'interp' mode is selected, no extension
                    is used.  Instead, a degree polyorder polynomial is fit to the
                    last window_length values of the edges, and this polynomial is
                    used to evaluate the last window_length // 2 output values.
                cval: scalar, optional
                    Value to fill past the edges of the input if mode is 'constant'.
                    Default is 0.0.

        Returns
        -------
        float64 | ndarray
            Angle in radians clockwise from True North to the direction towards
            which the vehicle travels.

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
        >>> points = nv.GeoPoint((59.381509, 59.387647),(10.496590, 10.494713), degrees=True)
        >>> nvec = points.to_nvector()
        >>> COG_rad = nvec.course_over_ground()
        >>> dx, dy = np.sin(COG_rad[0]), np.cos(COG_rad[0])
        >>> COG = nv.deg(COG_rad)
        >>> p_AB_N = nv.n_EA_E_and_n_EB_E2p_AB_N(nvec.normal[:, :1], nvec.normal[:, 1:]).ravel()
        >>> ax = plt.figure().gca()
        >>> _ = ax.plot(0, 0, 'bo', label='A')
        >>> _ = ax.arrow(0,0, dx*300, dy*300, head_width=20)
        >>> _ = ax.plot(p_AB_N[1], p_AB_N[0], 'go', label='B')
        >>> _ = ax.set_title('COG={} degrees'.format(COG))
        >>> _ = ax.set_xlabel('East [m]')
        >>> _ = ax.set_ylabel('North [m]')
        >>> _ = ax.set_xlim(-500, 200)
        >>> _ = ax.set_aspect('equal', adjustable='box')
        >>> _ = ax.legend()
        >>> plt.show() # doctest: +SKIP
        >>> plt.close()
        """
        frame = self.frame
        return course_over_ground(self.normal, a=frame.a, f=frame.f, R_Ee=frame.R_Ee, **options)

    def mean(self) -> Nvector:
        """Mean position of the n-vectors

        Returns
        -------
        Nvector
            Mean n-vector
        """
        average_nvector = unit(np.sum(self.normal, axis=1).reshape((3, 1)))
        return self.frame.Nvector(average_nvector, z=np.mean(self.z))

    mean_horizontal_position = deprecate(mean,
                                         old_name='mean_horizontal_position',
                                         new_name='mean',
                                         message="Will be removed in version 1.0.0")

    def _is_equal_to(self, other: Nvector, rtol=1e-12, atol=1e-14):
        options = dict(rtol=rtol, atol=atol)
        return (allclose(self.normal, other.normal, **options)
                and allclose(self.z, other.z, **options)
                and self.frame == other.frame)

    def __add__(self, other: Nvector) -> Nvector:
        """Vector addition between to n-vectors

        Parameters
        ----------
        other : Nvector
            Another n-vector

        Returns
        -------
        Nvector
            A new n-vector
        """
        _check_frames(self, other)
        return self.frame.Nvector(self.normal + other.normal, self.z + other.z)

    def __sub__(self, other: Nvector) -> Nvector:
        """Vector subtraction between to n-vectors

        Parameters
        ----------
        other : Nvector
            Another n-vector

        Returns
        -------
        Nvector
            A new n-vector
        """
        _check_frames(self, other)
        return self.frame.Nvector(self.normal - other.normal, self.z - other.z)

    def __neg__(self) -> Nvector:
        """Vector negation of a n-vector

        Returns
        -------
        Nvector
            A new n-vector
        """
        return self.frame.Nvector(-self.normal, -self.z)

    def __mul__(self, scalar: Union[int, float, float64]) -> Union[Nvector, NotImplemented]:
        """Elementwise multiplication with a scalar

        Parameters
        ----------
        scalar : int | float | float64
            A scalar-like quantity that supports ndarray.__mul__

        Returns
        -------
        Nvector | NotImplemented
            A scaled n-vector if implemented. Else :py:obj:`NotImplemented`.
        """

        if not isinstance(scalar, Nvector):
            return self.frame.Nvector(self.normal * scalar, self.z * scalar)
        return NotImplemented  # 'Only scalar multiplication is implemented'

    def __div__(self, scalar: Union[int, float, float64]) -> Union[Nvector, NotImplemented]:
        """Elementwise division with a scalar

        Parameters
        ----------
        scalar : int | float | float64
            A scalar-like quantity that supports ndarray.__div__

        Returns
        -------
        Nvector | NotImplemented
            A scaled n-vector if implemented. Else :py:obj:`NotImplemented`.
        """
        if not isinstance(scalar, Nvector):
            return self.frame.Nvector(self.normal / scalar, self.z / scalar)
        return NotImplemented  # 'Only scalar division is implemented'

    __truediv__ = __div__
    __radd__ = __add__
    __rmul__ = __mul__


class Pvector(_Common):
    """Geographical position given as cartesian position vector in a frame.

    Parameters
    ----------
    pvector : list | tuple | ndarray
        3 x n array cartesian position vector(s) [m] from E to B, decomposed in E.
    frame : FrameN | FrameB | FrameL
        Local frame
    scalar : bool
        Input p-vector a scalar? If None, then determined by shape of pvector

    Attributes
    ----------
    pvector : ndarray
        3 x n array cartesian position vector(s) [m] from E to B, decomposed in E.
    frame : FrameN | FrameB | FrameL
        Reference ellipsoid
    scalar : bool
        False if input `pvector` has shape (3, >=2) or forced at init.

    Notes
    -----
    The following arithmetic and inequality operators are defined for a :py:class:`envector.objects.Pvector`.

    __eq__ : bool
        Equality operator (`==`), which compares the numerical values of the pvector, and identical
        frames.
    __ne__ : bool
        Inequality operator (`!=`), which compares the numerical values of the pvector, and identical
        frames.

    """

    _NAMES = ('pvector', 'frame', 'scalar')
    """Sequence of attribute names for the repr"""
    pvector: ndarray
    """Position array-like, must be shape (3, n, m, ...) with n>0"""
    frame: Union[_LocalFrameBase, _FrameEBase]
    """Reference ellipsoid"""
    scalar: bool
    """False if input `pvector` has shape (3, >=2)"""

    def __init__(
        self,
        pvector: Union[list, tuple, ndarray],
        frame: _LocalFrameBase,
        scalar: Union[bool, None]=None
    ) -> None:
        if scalar is None:
            scalar = np.shape(pvector)[1] == 1
        self.pvector = np.asarray(pvector)
        self.frame = frame
        self.scalar = scalar

    @property
    def length(self) -> Union[float64, ndarray]:
        """Length of the pvector.

        Returns
        -------
        float64 | ndarray
            A scalar if :py:meth:`envector.objects.Pvector.scaler` is True, else an array of lengths.
        """
        lengths = norm(self.pvector, axis=0)
        if self.scalar:
            return lengths[0]
        return lengths

    @property
    def azimuth_deg(self) -> Union[float64, ndarray]:
        """Azimuth in degrees clockwise relative to the x-axis.

        Returns
        -------
        float64 | ndarray
            A scalar if :py:meth:`envector.objects.Pvector.scaler` is True, else an array of lengths.
        """
        return deg(self.azimuth)

    @property
    def azimuth(self) -> Union[float64, ndarray]:
        """Azimuth in radians clockwise relative to the x-axis.

        Returns
        -------
        float64 | ndarray
            A scalar if :py:meth:`envector.objects.Pvector.scaler` is True, else an array of lengths.
        """
        p_AB_N = self.pvector
        if self.scalar:
            return np.arctan2(p_AB_N[1], p_AB_N[0])[0]
        return np.arctan2(p_AB_N[1], p_AB_N[0])

    @property
    def elevation_deg(self) -> Union[float64, ndarray]:
        """Elevation in degrees relative to the xy-plane. (Positive downwards in a NED frame)

        Returns
        -------
        float64 | ndarray
            A scalar if :py:meth:`envector.objects.Pvector.scaler` is True, else an array of lengths.
        """
        return deg(self.elevation)

    @property
    def elevation(self) -> Union[float64, ndarray]:
        """Elevation in radians relative to the xy-plane. (Positive downwards in a NED frame)

        Returns
        -------
        float64 | ndarray
            A scalar if :py:meth:`envector.objects.Pvector.scaler` is True, else an array of lengths.
        """
        z = self.pvector[2]
        if self.scalar:
            return np.arcsin(z / self.length)[0]
        return np.arcsin(z / self.length)

    def to_ecef_vector(self) -> ECEFvector:
        """Returns position as ECEFvector object.

        Returns
        -------
        ECEFvector
            P-vector in the ECEF-frame

        See also
        --------
        ECEFvector
        """

        n_frame = self.frame
        p_AB_N = self.pvector
        # alternatively: np.dot(n_frame.R_EN, p_AB_N)
        p_AB_E = mdot(n_frame.R_EN, p_AB_N[:, None, ...]).reshape(3, -1)
        return ECEFvector(p_AB_E, frame=n_frame.nvector.frame, scalar=self.scalar)

    def to_nvector(self) -> Nvector:
        """Returns position as Nvector object.

        See also
        --------
        Nvector
            Position as Nvector
        """

        return self.to_ecef_vector().to_nvector()

    def to_geo_point(self) -> GeoPoint:
        """
        Returns position as GeoPoint object.

        Returns
        -------
        GeoPoint
            Position as GeoPoint.

        See also
        --------
        GeoPoint
        """
        return self.to_ecef_vector().to_geo_point()

    delta_to = _delta

    def _is_equal_to(self, other: Pvector, rtol: float=1e-12, atol: float=1e-14) -> bool:
        options = dict(rtol=rtol, atol=atol)
        return (allclose(self.pvector, other.pvector, **options)
                and self.frame == other.frame)


@use_docstring(_examples.get_examples_no_header([3, 4]))
class ECEFvector(Pvector):
    """
    Geographical position given as cartesian position vector in frame E

    Parameters
    ----------
    pvector : list | tuple | ndarray
        3 x n array cartesian position vector(s) [m] from E to B, decomposed in E.
    frame: FrameE object
        reference ellipsoid. The default ellipsoid model used is WGS84, but
        other ellipsoids/spheres might be specified.

    Attributes
    ----------
    pvector : ndarray
        3 x n array cartesian position vector(s) [m] from E to B, decomposed in E.
    frame : FrameN | FrameB | FrameL
        Reference ellipsoid
    scalar : bool
        False if input `pvector` has shape (3, >=2) or forced at init.

    Notes
    -----
    The position of B (typically body) relative to E (typically Earth) is
    given into this function as p-vector, p_EB_E relative to the center of the
    frame.

    The following arithmetic and inequality operators are defined for a :py:class:`envector.objects.ECEFvector`.

    __eq__ : bool
        Equality operator (`==`), which compares the numerical values of the p-vector and identical
        frames.
    __ne__ : bool
        Inequality operator (`!=`), which compares the numerical values of the p-vector and identical
        frames.
    __neg__ : Nvector
        Negation operator (`-`) that returns a new ECEFvector with negated p-vector and frame, see
        :py:meth:`envector.objects.ECEFvector.__neg__` for more details.
    __add__ : Nvector
        Additional operator (`+`) adds self with another ECEFvector in the same frame, see
        :py:meth:`envector.objects.ECEFvector.__add__` for more details.
    __sub__ : Nvector
        Subtraction operator (`-`) subtract self with another ECEFvector in the same frame, see
        :py:meth:`envector.objects.ECEFvector.__add__` for more details.

    Examples
    --------
    {super}

    See also
    --------
    GeoPoint, ECEFvector, Pvector
    """

    def __init__(
        self,
        pvector: Union[list, tuple, ndarray],
        frame: Union[_FrameEBase, None]=None,
        scalar: Union[bool, None]=None
    ) -> None:
        super(ECEFvector, self).__init__(pvector, _default_frame(frame), scalar)

    def change_frame(self, frame: Union[FrameB, FrameN, FrameL]) -> Pvector:
        """
        Converts to Cartesian position vector in another frame

        Parameters
        ----------
        frame : FrameB | FrameN | FrameL
            local frame M used to convert p_AB_E (position vector from A to B,
            decomposed in E) to a cartesian vector p_AB_M decomposed in M.

        Returns
        -------
        Pvector
            Position vector from A to B, decomposed in frame M.

        See also
        --------
        n_EB_E2p_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E.
        """
        _check_frames(self, frame.nvector)
        p_AB_E = self.pvector
        p_AB_N = mdot(np.swapaxes(frame.R_EN, 1, 0), p_AB_E[:, None, ...])
        return Pvector(p_AB_N.reshape(3, -1), frame=frame, scalar=self.scalar)

    def to_ecef_vector(self) -> Self:
        """Returns position as ECEFvector object, in this case, itself.

        Returns
        -------
        ECEFvector
            Self

        """
        return self

    def to_geo_point(self) -> GeoPoint:
        """
        Returns position as GeoPoint object.

        Returns
        -------
        GeoPoint
            Position as GeoPoint

        See also
        --------
        GeoPoint
        """
        return self.to_nvector().to_geo_point()

    def to_nvector(self) -> Nvector:
        """Returns position as Nvector object.

        Returns
        -------
        Nvector
            Position as Nvector

        See also
        --------
        Nvector
        """
        frame = self.frame
        p_EB_E = self.pvector
        R_Ee = frame.R_Ee
        n_EB_E, depth = p_EB_E2n_EB_E(p_EB_E, a=frame.a, f=frame.f, R_Ee=R_Ee)
        if self.scalar:
            return Nvector(n_EB_E, z=depth[0], frame=frame)
        return Nvector(n_EB_E, z=depth, frame=frame)

    delta_to = _delta

    def __add__(self, other: ECEFvector) -> ECEFvector:
        """Adds two ECEFvector objects p-vectors in the same frame.

        Parameters
        ----------
        other : ECEFvector
            Another ECEFvector

        Returns
        -------
        ECEFvector
            A new ECEFvector
        """
        _check_frames(self, other)
        scalar = self.scalar and other.scalar
        return ECEFvector(self.pvector + other.pvector, self.frame, scalar)

    def __sub__(self, other: ECEFvector) -> ECEFvector:
        """Subtracts two ECEFvector objects p-vectors in the same frame.

        Parameters
        ----------
        other : ECEFvector
            Another ECEFvector

        Returns
        -------
        ECEFvector
            A new ECEFvector
        """
        _check_frames(self, other)
        scalar = self.scalar and other.scalar
        return ECEFvector(self.pvector - other.pvector, self.frame, scalar)

    def __neg__(self) -> ECEFvector:
        """Negates the ECEFvector p-vector in the same frame.

        Returns
        -------
        ECEFvector
            A new ECEFvector
        """
        return ECEFvector(-self.pvector, self.frame, self.scalar)


class _GeoPathBase:

    point_a : Union[Nvector, GeoPoint, ECEFvector]
    """Starting point of path, position A, decomposed in E."""
    point_b: Union[Nvector, GeoPoint, ECEFvector]
    """Ending point of path, position B, decomposed in E."""



@use_docstring(_examples.get_examples_no_header([5, 6, 9, 10]))
class GeoPath(_GeoPathBase):
    """
    Geographical path between two positions in Frame E

    Parameters
    ----------
     point_a : Nvector | GeoPoint | ECEFvector
        Starting point of path, position A, decomposed in E.
     point_b : Nvector | GeoPoint | ECEFvector
        Ending point of path, position B, decomposed in E.

    Attributes
    ----------
    point_a : Nvector | GeoPoint | ECEFvector
       Starting point of path, position A, decomposed in E.
    point_b : Nvector | GeoPoint | ECEFvector
       Ending point of path, position B, decomposed in E.

    Notes
    -----
    Please note that either position A or B or both might be a vector of points.
    In this case the GeoPath instance represents all the paths between the positions
    of A and the corresponding positions of B.

    The following arithmetic and inequality operators are defined for a :py:class:`envector.objects.GeoPath`.

    __eq__ : bool
        Equality operator (`==`), which compares the numerical attributes and frames of point A and point B.
    __ne__ : bool
        Inequality operator (`!=`), which compares the numerical attributes and frames of point A and point B.

    Examples
    --------
    {super}
    """

    def __init__(
        self,
        point_a: Union[Nvector, GeoPoint, ECEFvector],
        point_b: Union[Nvector, GeoPoint, ECEFvector],
    ) -> None:
        super(GeoPath, self).__init__()
        self.point_a = point_a
        self.point_b = point_b

    @property
    def positionA(self) -> Union[Nvector, GeoPoint, ECEFvector]:
        """positionA is deprecated, use point_a instead!"""  # @ReservedAssignment
        warnings.warn("positionA is deprecated, use point_a instead!",
                      category=DeprecationWarning, stacklevel=2)
        return self.point_a

    @property
    def positionB(self) -> Union[Nvector, GeoPoint, ECEFvector]:
        """positionB is deprecated, use point_b instead!"""  # @ReservedAssignment
        warnings.warn("positionB is deprecated, use point_b instead!",
                      category=DeprecationWarning, stacklevel=2)
        return self.point_b

    def nvectors(self) -> Tuple[Nvector, Nvector]:
        """Returns point_a and point_b as n-vectors

        Returns
        -------
        tuple[Nvector, Nvector]
            A 2-tuple of point A and point B, in that order, each as a Nvector.

        See Also
        --------
        Nvector
        """
        return self.point_a.to_nvector(), self.point_b.to_nvector()

    def geo_points(self) -> Tuple[GeoPoint, GeoPoint]:
        """Returns point_a and point_b as geo-points

        Returns
        -------
        tuple[GeoPoint, GeoPoint]
            A 2-tuple of point A and point B, in that order, each as a GeoPoint.

        See Also
        --------
        GeoPoint
        """
        return self.point_a.to_geo_point(), self.point_b.to_geo_point()

    def ecef_vectors(self) -> Tuple[ECEFvector, ECEFvector]:
        """Returns point_a and point_b as ECEF-vectors

        Returns
        -------
        tuple[ECEFvector, ECEFvector]
            A 2-tuple of point A and point B, in that order, each as a ECEFvector.

        See Also
        --------
        ECEFvector
        """
        return self.point_a.to_ecef_vector(), self.point_b.to_ecef_vector()

    def nvector_normals(self) -> Tuple[ndarray, ndarray]:
        """Returns n-vector normals for position a and b

        Returns
        -------
        tuple[ndarray, ndarray]
            A 2-tuple of normal vectors of point A and point B, in that order.

        See Also
        --------
        Nvector, nvectors
        """
        nvector_a, nvector_b = self.nvectors()
        return nvector_a.normal, nvector_b.normal

    def _get_average_radius(self) -> Union[float, float64, ndarray]:
        """Calculates the average radius of an imaginary sphere along the path in meters

        Returns
        -------
        float | float64 | ndarray
            A scalar or array in meters
        """
        p_E1_E, p_E2_E = self.ecef_vectors()
        radius = (p_E1_E.length + p_E2_E.length) / 2
        return radius

    def cross_track_distance(
        self,
        point: Union[Nvector, GeoPoint, ECEFvector],
        method: str='greatcircle',
        radius: Union[int, float, None]=None
    ) -> Union[float64, ndarray]:
        """
        Returns cross track distance from path to point.

        Parameters
        ----------
        point : Nvector | GeoPoint | ECEFvector
            position to measure the cross track distance to.
        method: 'greatcircle' or 'euclidean'
            defining distance calculated.
        radius : int | float | None
            Real scalar radius of a sphere in meters. If None, default is the average height of points A and B

        Returns
        -------
        float64 | ndarray
            Real scalar or vector distance in [m]

        Notes
        -----
        The result for spherical Earth is returned.
        """
        if radius is None:
            radius = self._get_average_radius()
        path = self.nvector_normals()
        n_c = point.to_nvector().normal
        distance = cross_track_distance(path, n_c, method=method, radius=radius)
        if np.ndim(radius) == 0 and distance.size == 1:
            return distance[0]  # scalar cross track distance
        return distance

    def track_distance(
        self,
        method: str='greatcircle',
        radius: Union[int, float, None]=None
    ) -> Union[float64, ndarray]:
        """
        Returns the path distance computed at the average height.

        Parameters
        ----------
        method : str
            'greatcircle', 'euclidean' or 'ellipsoidal' defining distance calculated.
        radius : int | float | None
            Real scalar radius of a sphere in meters. If None, default is the average height of points A and B

        Returns
        -------
        float64 | ndarray
            path distance computed at the average height in meters

        """
        if method[:2] in {'ex', 'el'}:  # exact or ellipsoidal
            point_a, point_b = self.geo_points()
            s_ab, _angle1, _angle2 = point_a.distance_and_azimuth(point_b)
            return s_ab
        if radius is None:
            radius = self._get_average_radius()
        normal_a, normal_b = self.nvector_normals()

        distance_fun = euclidean_distance if method[:2] == "eu" else great_circle_distance
        distance = distance_fun(normal_a, normal_b, radius)
        if np.ndim(radius) == 0:
            return distance[0]  # scalar track distance
        return distance

    def intersect(self, path: GeoPath) -> Nvector:
        """
        Returns the intersection(s) between the great circles of the two paths

        Parameters
        ----------
        path : GeoPath
            Path to intersect

        Returns
        -------
        Nvector
            Intersection(s) between the great circles of the two paths

        Notes
        -----
        The result for spherical Earth is returned at the average height.
        """
        frame = self.point_a.frame
        point_a1, point_a2 = self.nvectors()
        point_b1, point_b2 = path.nvectors()
        path_a = (point_a1.normal, point_a2.normal)  # self.nvector_normals()
        path_b = (point_b1.normal, point_b2.normal)  # path.nvector_normals()
        normal_c = intersect(path_a, path_b)  # nvector
        depth = (point_a1.z + point_a2.z + point_b1.z + point_b2.z) / 4.
        return frame.Nvector(normal_c, z=depth)

    intersection = deprecate(intersect,
                             old_name='intersection',
                             new_name='intersect',
                             message="Will remove in version 1")

    def _on_ellipsoid_path(
        self,
        point: Union[Nvector, GeoPoint, ECEFvector],
        rtol: float=1e-6,
        atol: float=1e-8
    ) -> Union[bool_, ndarray]:
        """Determines if a point is on this ellipsoidal path.

        Parameters
        ----------
        point : Nvector | GeoPoint | ECEFvector
            A point that can be converted to a GeoPoint
        rtol : float
            Relative tolerance (default = 1e-6)
        atol : float
            Absolute tolerance (default = 1e-8)

        Returns
        -------
        bool_ | ndarray
            A boolean scalar or ndarray of booleans.
        """
        point_a, point_b = self.geo_points()
        point_c = point.to_geo_point()
        z = (point_a.z + point_b.z) * 0.5
        distance_ab, azimuth_ab, _azi_ba = point_a.distance_and_azimuth(point_b)
        distance_ac, azimuth_ac, _azi_ca = point_a.distance_and_azimuth(point_c)
        return (isclose(z, point_c.z, rtol=rtol, atol=atol)
                & (isclose(distance_ac, 0, atol=atol)
                   | ((distance_ab >= distance_ac)
                      & isclose(azimuth_ac, azimuth_ab, rtol=rtol, atol=atol))))

    def on_great_circle(
        self,
        point: Union[GeoPoint, Nvector, ECEFvector],
        atol: float=1e-8
    ) -> Union[bool_, ndarray]:
        """Returns True if point is on the great circle within a tolerance.

        Parameters
        ----------
        point : Nvector | GeoPoint | ECEFvector
            Position to measure the cross track distance to.
        atol : float
            Absolute tolerance (default=1.e-8)

        Returns
        -------
        bool_ | ndarray
            A boolean scalar or ndarray of booleans.
        """
        distance = np.abs(self.cross_track_distance(point))
        result = isclose(distance, 0, atol=atol)
        if np.ndim(result) == 0:
            return result[()]
        return result

    def _on_great_circle_path(
        self,
        point : Union[Nvector, GeoPoint, ECEFvector],
        radius: Union[int, float, None]=None,
        rtol: float=1e-9,
        atol: float=1e-8
    ) -> Union[bool_, ndarray]:
        """Returns True if point is on the great circle within a relative and absolute tolerance.

        Parameters
        ----------
        point : Nvector | GeoPoint | ECEFvector
            Point that might be on the great circle
        radius : int | float | None
            Real scalar radius of a sphere in meters. If None, default is the average height of points A and B
        rtol : float
            Relative tolerance, default = 1e-9
        atol : float
            Absolute tolerance, default = 1e-8

        Returns
        -------
        bool_ | ndarray
            A boolean scalar or ndarray of booleans.
        """
        if radius is None:
            radius = self._get_average_radius()
        n_a, n_b = self.nvectors()
        path = (n_a.normal, n_b.normal)
        n_c = point.to_nvector()
        same_z = isclose(n_c.z, (n_a.z + n_b.z) * 0.5, rtol=rtol, atol=atol)
        result = on_great_circle_path(path, n_c.normal, radius, atol=atol) & same_z
        if np.ndim(radius) == 0 and result.size == 1:
            return result[0]  # scalar outout
        return result

    def on_path(
        self,
        point: Union[Nvector, GeoPoint, ECEFvector],
        method: str='greatcircle',
        rtol: float=1e-6,
        atol: float=1e-8
    ) -> Union[bool_, ndarray]:
        """
        Returns True if point is on the path between A and B witin a tolerance.

        Parameters
        ----------
        point : Nvector | GeoPoint | ECEFvector
            Point to test
        method : 'greatcircle' or 'ellipsoid'
            defining the path.
        rtol : float
            The relative tolerance parameter.
        atol : float
            The absolute tolerance parameter.

        Returns
        -------
        bool_ | ndarray
            Boolean scalar or boolean vector, true if the point is on the path at its average height.

        Notes
        -----
        The result for spherical Earth is returned for method='greatcircle'.

        Examples
        --------
        >>> import envector as nv
        >>> wgs84 = nv.FrameE(name='WGS84')
        >>> pointA = wgs84.GeoPoint(89, 0, degrees=True)
        >>> pointB = wgs84.GeoPoint(80, 0, degrees=True)
        >>> path = nv.GeoPath(pointA, pointB)
        >>> pointC = path.interpolate(0.6).to_geo_point()
        >>> bool(path.on_path(pointC))
        True
        >>> bool(path.on_path(pointC, 'ellipsoid'))
        True
        >>> pointD = path.interpolate(1.000000001).to_geo_point()
        >>> bool(path.on_path(pointD))
        False
        >>> bool(path.on_path(pointD, 'ellipsoid'))
        False
        >>> pointE = wgs84.GeoPoint(85, 0.0001, degrees=True)
        >>> bool(path.on_path(pointE))
        False
        >>> pointC = path.interpolate(-2).to_geo_point()
        >>> bool(path.on_path(pointC))
        False
        >>> bool(path.on_great_circle(pointC))
        True
        """
        if method[:2] in {'ex', 'el'}:  # exact or ellipsoid
            return self._on_ellipsoid_path(point, rtol=rtol, atol=atol)
        return self._on_great_circle_path(point, rtol=rtol, atol=atol)

    def _closest_point_on_great_circle(
        self,
        point: GeoPoint
    ) -> Nvector:
        """Returns closest point on great circle path to the point.

        Parameters
        ----------
        point : GeoPoint

        Returns
        -------
        Nvector

        """
        point_c = point.to_nvector()
        point_a, point_b = self.nvectors()
        path = (point_a.normal, point_b.normal)
        z = (point_a.z + point_b.z) * 0.5
        normal_d = closest_point_on_great_circle(path, point_c.normal)
        return point_c.frame.Nvector(normal_d, z)

    def closest_point_on_great_circle(
        self,
        point: GeoPoint
    ) -> GeoPoint:
        """
        Returns closest point on great circle path to the point.

        Parameters
        ----------
        point : GeoPoint
            Point of intersection between paths

        Returns
        -------
        GeoPoint
            Closest point on path.

        Notes
        -----
        The result for spherical Earth is returned at the average depth.

        Examples
        --------
        >>> import envector as nv
        >>> wgs84 = nv.FrameE(name='WGS84')
        >>> point_a = wgs84.GeoPoint(51., 1., degrees=True)
        >>> point_b = wgs84.GeoPoint(51., 2., degrees=True)
        >>> point_c = wgs84.GeoPoint(51., 2.9, degrees=True)
        >>> path = nv.GeoPath(point_a, point_b)
        >>> point = path.closest_point_on_great_circle(point_c)
        >>> bool(path.on_path(point))
        False
        >>> bool(nv.allclose((point.latitude_deg, point.longitude_deg),
        ...                  (50.99270338, 2.89977984)))
        True

        >>> bool(nv.allclose(GeoPath(point_c, point).track_distance(),  810.76312076))
        True

        """

        point_d = self._closest_point_on_great_circle(point)
        return point_d.to_geo_point()

    def closest_point_on_path(self, point: GeoPoint) -> GeoPoint:
        """
        Returns closest point on great circle path segment to the point.

        If the point is within the extent of the segment, the point returned is
        on the segment path otherwise, it is the closest endpoint defining the
        path segment.

        Parameters
        ----------
        point : GeoPoint
            point of intersection between paths

        Returns
        -------
        GeoPoint
            Closest point on path segment.

        Examples
        --------
        >>> import envector as nv
        >>> wgs84 = nv.FrameE(name='WGS84')
        >>> pointA = wgs84.GeoPoint(51., 1., degrees=True)
        >>> pointB = wgs84.GeoPoint(51., 2., degrees=True)
        >>> pointC = wgs84.GeoPoint(51., 1.9, degrees=True)
        >>> path = nv.GeoPath(pointA, pointB)
        >>> point = path.closest_point_on_path(pointC)
        >>> np.allclose((point.latitude_deg, point.longitude_deg),
        ...             (51.00038411380564, 1.900003311624411))
        True
        >>> np.allclose(GeoPath(pointC, point).track_distance(),  42.67368351)
        True
        >>> pointD = wgs84.GeoPoint(51.0, 2.1, degrees=True)
        >>> pointE = path.closest_point_on_path(pointD) # 51.0000, 002.0000
        >>> float(pointE.latitude_deg), float(pointE.longitude_deg)
        (51.0, 2.0)
        """
        # TODO: vectorize this
        return self._closest_point_on_path(point)

    def _closest_point_on_path(
        self,
        point: GeoPoint
    ) -> GeoPoint:
        """Returns closest point on great circle path segment to the point.

        If the point is within the extent of the segment, the point returned is
        on the segment path otherwise, it is the closest endpoint defining the
        path segment.

        Parameters
        ----------
        point : GeoPoint
            point of intersection between paths

        Returns
        -------
        GeoPoint
            Closest point on path segment.

        """
        point_c = self._closest_point_on_great_circle(point)
        if self.on_path(point_c):
            return point_c.to_geo_point()
        n0 = point.to_nvector().normal
        n1, n2 = self.nvector_normals()
        radius = self._get_average_radius()
        d1 = great_circle_distance(n1, n0, radius)
        d2 = great_circle_distance(n2, n0, radius)
        if d1 < d2:
            return self.point_a.to_geo_point()
        return self.point_b.to_geo_point()

    def interpolate(self, ti: Union[int, float, ndarray]) -> Nvector:
        """
        Returns the interpolated point along the path

        Parameters
        ----------
        ti: real scalar
            interpolation time assuming position A and B is at t0=0 and t1=1,
            respectively.

        Returns
        -------
        Nvector
            Point of interpolation along path
        """
        point_a, point_b = self.nvectors()
        point_c = point_a + (point_b - point_a) * ti
        point_c.normal = unit(point_c.normal, norm_zero_vector=np.nan)
        return point_c



class FrameE(_FrameEBase):
    """Earth-fixed frame

    Parameters
    ----------
    a : float | None
        Semi-major axis of the Earth ellipsoid given in [m]. If None, determined by `name`.
    f : float | None
        Flattening [no unit] of the Earth ellipsoid. If None, determined by `name`.
    name : str
        Defining the default ellipsoid, default is WGS-84 ellipsoid
    axes : str
        Either `'E'` or `'e'`. Define axes orientation of E frame. Default is axes='e' which means
        that the orientation of the axis is such that:
        z-axis -> North Pole, x-axis -> Latitude=Longitude=0.

    Attributes
    ----------
    a : float
        Semi-major axis of the Earth ellipsoid given in [m]. If None, determined by `name`.
    f : float
        Flattening [no unit] of the Earth ellipsoid. If None, determined by `name`.
    name : str
        Defining the default ellipsoid, default is WGS-84 ellipsoid
    axes : str
        Either `'E'` or `'e'`. Define axes orientation of E frame. Default is axes='e' which means
        that the orientation of the axis is such that:
        z-axis -> North Pole, x-axis -> Latitude=Longitude=0.

    Notes
    -----
    The frame is Earth-fixed (rotates and moves with the Earth) where the
    origin coincides with Earth's centre (geometrical centre of ellipsoid
    model).

    The following arithmetic and inequality operators are defined for a :py:class:`envector.objects.FrameE`.

    __eq__ : bool
        Equality operator (`==`), which compares the numerical values of `a`, `f`, and `R_Ee`.
    __ne__ : bool
        Inequality operator (`!=`), which compares the numerical values of `a`, `f`, and `R_Ee`.

    See also
    --------
    FrameN, FrameL, FrameB
    """

    def __init__(
        self,
        a: Union[float, None]=None,
        f: Union[float, None]=None,
        name: str='WGS84',
        axes: str='e'
    ) -> None:
        if a is None or f is None:
            a, f, _ = get_ellipsoid(name)
        self.a = a
        self.f = f
        self.name = name
        self.axes = axes

    @property
    def R_Ee(self) -> ndarray:
        """Rotation matrix R_Ee defining the axes of the coordinate frame E"""
        return E_rotation(self.axes)

    def _is_equal_to(self, other: FrameE, rtol: float=1e-12, atol: float=1e-14) -> bool:
        return (allclose(self.a, other.a, rtol=rtol, atol=atol)
                and allclose(self.f, other.f, rtol=rtol, atol=atol)
                and allclose(self.R_Ee, other.R_Ee, rtol=rtol, atol=atol))

    def inverse(
        self,
        lat_a: Union[int, float, list, tuple, ndarray],
        lon_a: Union[int, float, list, tuple, ndarray],
        lat_b: Union[int, float, list, tuple, ndarray],
        lon_b: Union[int, float, list, tuple, ndarray],
        z: Union[int, float, list, tuple, ndarray]=0,
        degrees: bool=False
    ) -> Tuple[Union[float64, ndarray], Union[float64, ndarray], Union[float64, ndarray]]:
        """
        Returns ellipsoidal distance between positions as well as the direction.

        Parameters
        ----------
        lat_a : int | float | list | tuple | ndarray
            Scalar or vectors of latitude of position A.
        lon_a : int | float | list | tuple | ndarray
            Scalar or vectors of longitude of position A.
        lat_b : int | float | list | tuple | ndarray
            Scalar or vectors of latitude of position B.
        lon_b : int | float | list | tuple | ndarray
            Scalar or vectors of longitude of position B.
        z : int | float | list | tuple | ndarray
            Scalar or vectors of depth relative to Earth ellipsoid (default = 0)
        degrees : bool
            Angles are given in degrees if True otherwise in radians.

        Returns
        -------
        tuple[float64 | ndarray, float64 | ndarray, float64 | ndarray]
            s_ab: real scalar or vector
                ellipsoidal distance [m] between position A and B.
            azimuth_a, azimuth_b:  real scalars or vectors.
                direction [rad or deg] of line at position A and B relative to
                North, respectively.

        Notes
        -----
        Restriction on the parameters:

          * Latitudes must lie between -90 and 90 degrees.
          * Latitudes outside this range will be set to NaNs.
          * The flattening f should be between -1/50 and 1/50 inn order to retain full accuracy.

        References
        ----------
        `C. F. F. Karney, Algorithms for geodesics, J. Geodesy 87(1), 43-55 (2013)
        <https://rdcu.be/cccgm>`_

        `geographiclib <https://pypi.python.org/pypi/geographiclib>`_

        """
        a1, f = self.a - z, self.f

        if degrees:
            lat_a, lon_a, lat_b, lon_b = rad(lat_a, lon_a, lat_b, lon_b)

        s_ab, azimuth_a, azimuth_b = geodesic_distance(lat_a, lon_a, lat_b, lon_b, a1, f)
        if degrees:
            azimuth_a, azimuth_b = deg(azimuth_a, azimuth_b)
        return s_ab, azimuth_a, azimuth_b
# TODO: remove this:
#         if not degrees:
#             lat_a, lon_a, lat_b, lon_b = deg(lat_a, lon_a, lat_b, lon_b)
#
#         lat_a, lon_a, lat_b, lon_b, z = np.broadcast_arrays(lat_a, lon_a, lat_b, lon_b, z)
#         fun = self._inverse
#         items = zip(*np.atleast_1d(lat_a, lon_a, lat_b, lon_b, z))
#         sab, azia, azib = np.transpose([fun(lat_ai, lon_ai, lat_bi, lon_bi, z=zi)
#                                         for lat_ai, lon_ai, lat_bi, lon_bi, zi in items])
#
#         if not degrees:
#             s_ab, azimuth_a, azimuth_b = sab.ravel(), rad(azia.ravel()), rad(azib.ravel())
#         else:
#             s_ab, azimuth_a, azimuth_b = sab.ravel(), azia.ravel(), azib.ravel()
#
#         if np.ndim(lat_a) == 0:
#             return s_ab[0], azimuth_a[0], azimuth_b[0]
#         return s_ab, azimuth_a, azimuth_b
#
#     def _inverse(self, lat_a, lon_a, lat_b, lon_b, z=0):
#         geo = _Geodesic(self.a - z, self.f)
#         result = geo.Inverse(lat_a, lon_a, lat_b, lon_b, outmask=_Geodesic.STANDARD)
#         return result['s12'], result['azi1'], result['azi2']

    @staticmethod
    def _outmask(long_unroll: bool) -> int:
        if long_unroll:
            return _Geodesic.STANDARD | _Geodesic.LONG_UNROLL
        return _Geodesic.STANDARD

    def direct(
        self,
        lat_a: Union[int, float, list, tuple, ndarray],
        lon_a: Union[int, float, list, tuple, ndarray],
        azimuth: Union[int, float, list, tuple, ndarray],
        distance: Union[int, float, list, tuple, ndarray],
        z: Union[int, float, list, tuple, ndarray]=0,
        long_unroll: bool=False,
        degrees: bool=False
    ) -> Tuple[Union[float64, ndarray], Union[float64, ndarray], Union[float64, ndarray]]:
        """
        Returns position B computed from position A, distance and azimuth.

        Parameters
        ----------
        lat_a : int | float | list | tuple | ndarray
            Real scalar or length n vector of latitude of position A.
        lon_a : int | float | list | tuple | ndarray
            Real scalar or length n vector of longitude of position A.
        azimuth : int | float | list | tuple | ndarray
            Real scalar or length n vector azimuth [rad or deg] of line at position A relative to North.
        distance : int | float | list | tuple | ndarray
            Real scalar or length n vector ellipsoidal distance [m] between position A and B.
        z : int | float | list | tuple | ndarray
            Real scalar or length n vector depth relative to Earth ellipsoid (default = 0).
        long_unroll : bool
            Controls the treatment of longitude. If it is False then the lon_a and lon_b
            are both reduced to the range [-180, 180). If it is True, then lon_a
            is as given in the function call and (lon_b - lon_a) determines how many times
            and in what sense the geodesic has encircled the ellipsoid.
        degrees : bool
            angles are given in degrees if True otherwise in radians.

        Returns
        -------
        tuple[float64 | ndarray, float64 | ndarray, float64 | ndarray]
            lat_b, lon_b:  real scalars or vectors of length n
                Latitude and longitude of position b.
            azimuth_b: real scalar or vector of length n.
                azimuth [rad or deg] of line at position B relative to North.

        Notes
        -----
        Restriction on the parameters:

          * Latitudes must lie between -90 and 90 degrees.
          * Latitudes outside this range will be set to NaNs.
          * The flattening f should be between -1/50 and 1/50 inn order to retain full accuracy.

        References
        ----------
        `C. F. F. Karney, Algorithms for geodesics, J. Geodesy 87(1), 43-55 (2013)
        <https://rdcu.be/cccgm>`_

        `geographiclib <https://pypi.python.org/pypi/geographiclib>`_
        """
        a1, f = self.a-z, self.f
        lat1, lon1, az1, distance, a1 = np.broadcast_arrays(lat_a, lon_a, azimuth, distance, a1)
        if degrees:
            lat1, lon1, az1 = rad(lat1, lon1, az1)
            lat2, lon2, az2 = deg(*geodesic_reckon(lat1, lon1, distance, az1, a1, f, long_unroll))
        else:
            lat2, lon2, az2 = geodesic_reckon(lat1, lon1, distance, az1, a1, f, long_unroll)

        return lat2, lon2, az2

# TODO: remove this:
#         if not degrees:
#             lat_a, lon_a, azimuth = deg(lat_a, lon_a, azimuth)
#
#         broadcast = np.broadcast_arrays
#         lat_a, lon_a, azimuth, distance, z = broadcast(lat_a, lon_a, azimuth, distance, z)
#         fun = partial(self._direct, outmask=self._outmask(long_unroll))
#
#         items = zip(*np.atleast_1d(lat_a, lon_a, azimuth, distance, z))
#         lab, lob, azib = np.transpose([fun(lat_ai, lon_ai, azimuthi, distancei, z=zi)
#                                        for lat_ai, lon_ai, azimuthi, distancei, zi in items])
#         if not degrees:
#             latb, lonb, azimuth_b = rad(lab.ravel(), lob.ravel(), azib.ravel())
#         else:
#             latb, lonb, azimuth_b = lab.ravel(), lob.ravel(), azib.ravel()
#         if np.ndim(lat_a) == 0:
#             return latb[0], lonb[0], azimuth_b[0]
#         return latb, lonb, azimuth_b
#
#     def _direct(self, lat_a, lon_a, azimuth, distance, z=0, outmask=None):
#         geo = _Geodesic(self.a - z, self.f)
#         result = geo.Direct(lat_a, lon_a, azimuth, distance, outmask=outmask)
#         latb, lonb, azimuth_b = result['lat2'], result['lon2'], result['azi2']
#         return latb, lonb, azimuth_b

    @use_docstring_from(GeoPoint)
    def GeoPoint(self, *args, **kwds) -> GeoPoint:
        """{super}"""
        kwds.pop('frame', None)
        return GeoPoint(*args, frame=self, **kwds)

    @use_docstring_from(Nvector)
    def Nvector(self, *args, **kwds) -> Nvector:
        """{super}"""
        kwds.pop('frame', None)
        return Nvector(*args, frame=self, **kwds)

    @use_docstring_from(ECEFvector)
    def ECEFvector(self, *args, **kwds) -> ECEFvector:
        """{super}"""
        kwds.pop('frame', None)
        return ECEFvector(*args, frame=self, **kwds)


class _LocalFrame(_LocalFrameBase):

    def Pvector(self, pvector: Union[list, tuple, ndarray]) -> Pvector:
        """Returns Pvector relative to the local frame.

        Parameters
        ----------
        pvector : list | tuple | ndarray
            3 x n array cartesian position vector(s) [m] from E to B, decomposed in E.

        Returns
        -------
        Pvector
            The pvector in the local frame
        """
        return Pvector(pvector, frame=self)


@use_docstring(_examples.get_examples_no_header([1]))
class FrameN(_LocalFrame):
    """
    North-East-Down frame

    Parameters
    ----------
    point : ECEFvector | GeoPoint | Nvector
        Position of the vehicle (B) which also defines the origin of the local
        frame N. The origin is directly beneath or above the vehicle (B), at
        Earth's surface (surface of ellipsoid model).

    Attributes
    ----------
    nvector : Nvector
        Normal vector from input `point.to_nvector()`

    Notes
    -----
    The Cartesian frame is local and oriented North-East-Down, i.e.,
    the x-axis points towards north, the y-axis points towards east (both are
    horizontal), and the z-axis is pointing down.

    When moving relative to the Earth, the frame rotates about its z-axis
    to allow the x-axis to always point towards north. When getting close
    to the poles this rotation rate will increase, being infinite at the
    poles. The poles are thus singularities and the direction of the
    x- and y-axes are not defined here. Hence, this coordinate frame is
    NOT SUITABLE for general calculations.

    The following arithmetic and inequality operators are defined for a :py:class:`envector.objects.FrameN`.

    __eq__ : bool
        Equality operator (`==`), which compares the numerical values of `R_Ee` and `nvector` associated attributes.
    __ne__ : bool
        Inequality operator (`!=`), which compares the numerical values of `R_Ee` and `nvector` associated attributes.

    Examples
    --------
    {super}

    See also
    --------
    FrameE, FrameL, FrameB
    """

    _NAMES = ('point',)
    """Sequence of attribute names for the repr"""
    nvector: Nvector
    """n-vector"""

    def __init__(self, point: Union[ECEFvector, GeoPoint, Nvector]) -> None:
        nvector = point.to_nvector()
        self.nvector = Nvector(nvector.normal, z=0, frame=nvector.frame)

    @property
    def R_EN(self) -> ndarray:
        """Rotation matrix to go between E and N frame"""
        nvector = self.nvector
        return n_E2R_EN(nvector.normal, nvector.frame.R_Ee)

    def _is_equal_to(self, other: FrameN, rtol: float=1e-12, atol: float=1e-14) -> bool:
        return (allclose(self.R_EN, other.R_EN, rtol=rtol, atol=atol)
                and self.nvector == other.nvector)


class FrameL(FrameN):
    """Local level, Wander azimuth frame

    Parameters
    ----------
    point : ECEFvector | GeoPoint | Nvector
        Position of the vehicle (B) which also defines the origin of the local
        frame L. The origin is directly beneath or above the vehicle (B), at
        Earth's surface (surface of ellipsoid model).
    wander_azimuth : int | float | list | tuple | ndarray
        Real scalar or vector angle [rad] between the x-axis of L and the north direction.

    Attributes
    ----------
    nvector : Nvector
        Normal vector from input `point.to_nvector()`
    wander_azimuth : int | float | list | tuple | ndarray
        Angle [rad] between the x-axis of L and the north direction.

    Notes
    -----
    The Cartesian frame is local and oriented Wander-azimuth-Down. This means
    that the z-axis is pointing down. Initially, the x-axis points towards
    north, and the y-axis points towards east, but as the vehicle moves they
    are not rotating about the z-axis (their angular velocity relative to the
    Earth has zero component along the z-axis).

    (Note: Any initial horizontal direction of the x- and y-axes is valid
    for L, but if the initial position is outside the poles, north and east
    are usually chosen for convenience.)

    The L-frame is equal to the N-frame except for the rotation about the
    z-axis, which is always zero for this frame (relative to E). Hence, at
    a given time, the only difference between the frames is an angle
    between the x-axis of L and the north direction; this angle is called
    the wander azimuth angle. The L-frame is well suited for general
    calculations, as it is non-singular.

    The following arithmetic and inequality operators are defined for a :py:class:`envector.objects.FrameL`.

    __eq__ : bool
        Equality operator (`==`), which compares the numerical values of `R_Ee` and `nvector` associated attributes.
    __ne__ : bool
        Inequality operator (`!=`), which compares the numerical values of `R_Ee` and `nvector` associated attributes.

    See also
    --------
    FrameE, FrameN, FrameB
    """
    _NAMES = ('point', 'wander_azimuth')
    """Sequence of attribute names for the repr"""
    wander_azimuth: Union[int, float, list, tuple, ndarray]
    """Angle [rad] between the x-axis of L and the north direction."""

    def __init__(
        self,
        point: Union[ECEFvector, GeoPoint, Nvector],
        wander_azimuth: Union[int, float, list, tuple, ndarray]=0
    ) -> None:
        super(FrameL, self).__init__(point)
        self.wander_azimuth = wander_azimuth

    @property
    def R_EN(self) -> ndarray:
        """Rotation matrix to go between E and L frame"""
        n_EA_E = self.nvector.normal
        R_Ee = self.nvector.frame.R_Ee
        return n_E_and_wa2R_EL(n_EA_E, self.wander_azimuth, R_Ee=R_Ee)


@use_docstring(_examples.get_examples_no_header([2]))
class FrameB(_LocalFrame):
    """Body frame

    Parameters
    ----------
    point : ECEFvector | GeoPoint | Nvector
        Position of the vehicle's reference point which also coincides with
        the origin of the frame B.
    yaw : int | float | float64 | ndarray
        Yaw defining the orientation of frame B in [deg] or [rad].
    pitch : int | float | float64 | ndarray
        Pitch defining the orientation of frame B in [deg] or [rad].
    roll : int | float | float64 | ndarray
        Roll defining the orientation of frame B in [deg] or [rad].
    degrees : bool
        if True yaw, pitch, roll are given in degrees otherwise in radians

    Attributes
    ----------
    nvector : Nvector
        Normal vector from input `point.to_nvector()`
    yaw : int | float | float64 | ndarray
        Yaw defining the orientation of frame B in [rad].
    pitch : int | float | float64 | ndarray
        Pitch defining the orientation of frame B in [rad].
    roll : int | float | float64 | ndarray
        Roll defining the orientation of frame B in [rad].

    Notes
    -----
    The frame is fixed to the vehicle where the x-axis points forward, the
    y-axis to the right (starboard) and the z-axis in the vehicle's down
    direction.

    The following arithmetic and inequality operators are defined for a :py:class:`envector.objects.FrameB`.

    __eq__ : bool
        Equality operator (`==`), which compares the numerical values of `yaw`, `pitch`, `roll`, `R_Ee`, and `nvector`
        associated attributes.
    __ne__ : bool
        Inequality operator (`!=`), which compares the numerical values of `yaw`, `pitch`, `roll`, `R_Ee`, and `nvector`
        associated attributes.

    Examples
    --------
    {super}

    See also
    --------
    FrameE, FrameL, FrameN
    """

    _NAMES = ('point', 'yaw', 'pitch', 'roll')
    """Names for the class repr"""
    nvector: Nvector
    """Normal vector from input `point.to_nvector()`"""
    yaw: Union[int, float, float64, ndarray]
    """Yaw defining the orientation of frame B in [rad]."""
    pitch: Union[int, float, float64, ndarray]
    """Pitch defining the orientation of frame B in [rad]."""
    roll: Union[int, float, float64, ndarray]
    """Roll defining the orientation of frame B in [rad]."""

    def __init__(
        self,
        point: Union[ECEFvector, GeoPoint, Nvector],
        yaw: Union[int, float, float64, ndarray]=0,
        pitch: Union[int, float, float64, ndarray]=0,
        roll: Union[int, float, float64, ndarray]=0,
        degrees: bool=False
    ) -> None:
        self.nvector = point.to_nvector()
        if degrees:
            yaw, pitch, roll = rad(yaw), rad(pitch), rad(roll)
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

    @property
    def R_EN(self) -> ndarray:
        """Rotation matrix to go between E and B frame"""
        R_NB = zyx2R(self.yaw, self.pitch, self.roll)
        n_EB_E = self.nvector.normal
        R_EN = n_E2R_EN(n_EB_E, self.nvector.frame.R_Ee)
        return mdot(R_EN, R_NB)  # rotation matrix

    def _is_equal_to(self, other: FrameB, rtol: float=1e-12, atol: float=1e-14) -> bool:
        return (allclose(self.yaw, other.yaw, rtol=rtol, atol=atol)
                and allclose(self.pitch, other.pitch, rtol=rtol, atol=atol)
                and allclose(self.roll, other.roll, rtol=rtol, atol=atol)
                and allclose(self.R_EN, other.R_EN, rtol=rtol, atol=atol)
                and self.nvector == other.nvector)


def _check_frames(
    self: Union[GeoPoint, Nvector, Pvector, ECEFvector],
    other: Union[GeoPoint, Nvector, Pvector, ECEFvector],
) -> None:
    """Validates that two frames are equal, else raise an error.

    Parameters
    ----------
    self : GeoPoint | Nvector | Pvector | ECEFvector
        Self point or vector
    other : GeoPoint | Nvector | Pvector | ECEFvector
        A different point or vector

    Returns
    -------
    None

    Raises
    ------
    ValueError
        When the two frames are unequal.
    """
    if not self.frame == other.frame:
        raise ValueError('Frames are unequal')


def _default_frame(
    frame: Union[FrameB, FrameE, FrameL, FrameN, None],
) -> Union[FrameB, FrameE, FrameL, FrameN]:
    """Get the default frame if None or return itself

    Parameters
    ----------
    frame : FrameB | FrameE | FrameL | FrameN | None
        A frame instance or None

    Returns
    -------
    FrameB | FrameE | FrameL | FrameN
        Instantiated frame
    """
    if frame is None:
        return FrameE()
    return frame


_ODICT = globals()
__doc__ = (__doc__  # @ReservedAssignment
           + _make_summary(dict((n, _ODICT[n]) for n in __all__))
           + 'License\n-------\n'
           + _license.__doc__)


if __name__ == "__main__":
    test_docstrings(__file__)
