# pyre-strict
import logging
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union

import numpy as np
import pyproj
import warnings

from numpy.typing import NDArray
from osgeo import osr, gdal
from opensfm import pymap

Scalars = Union[float, NDArray]

WGS84_a = 6378137.0
WGS84_b = 6356752.314245

DEFAULT_GPS_STD: NDArray = np.array([5.0, 5.0, 15.0])

logger = logging.getLogger(__name__)


def log_vertical_datum(crs: str) -> None:
    if not crs or crs.upper() in ("WGS84", "EPSG:4326"):
        logger.warning(
            f"GCPs: using implicit ellipsoid height ({crs or 'WGS84'}). Geoid effects are ignored."
        )
        return
    elif "4979" in crs:
        logger.warning(
            f"GCPs: using explicit ellipsoid height ({crs}). Geoid effects are ignored."
        )
        return

    try:
        parsed = pymap.parse_gcp_projection_string(crs)
        if not parsed:
            logger.warning(
                "GCPs: using implicit ellipsoid height (WGS84). Geoid effects are ignored.")
            return

        pyproj_crs = pyproj.CRS(parsed)
        if pyproj_crs.is_compound:
            vert_crs = pyproj_crs.sub_crs_list[1]
            logger.info(
                f"GCPs: using vertical datum '{vert_crs.name}' from compound CRS."
            )
        elif hasattr(pyproj_crs, "has_z") and pyproj_crs.has_z:
            logger.warning(
                f"GCPs: using 3D CRS '{pyproj_crs.name}'. If this represents an ellipsoid height, geoid effects are ignored."
            )
    except Exception as e:
        logger.debug(f"Could not parse GCP vertical datum for logging: {e}")


def nicify_crs(crs_string: str) -> Tuple[str, str]:
    """Produce a standardized string representation of horizontal and vertical CRS.

    Returns:
        Tuple[str, str]: A tuple specifying (Horizontal CRS, Vertical CRS).
    """
    if not crs_string or crs_string.upper() == "WGS84":
        return "WGS84", "WGS 84 Ellipsoid"

    upper_crs = crs_string.upper()
    if upper_crs == "EPSG:4326":
        return "WGS84", "WGS 84 Ellipsoid (implicit)"

    if upper_crs == "EPSG:4979":
        return "WGS84", "WGS 84 Ellipsoid (explicit)"

    if upper_crs.startswith("WGS84 UTM"):
        # e.g., "WGS84 UTM 17N"
        parts = upper_crs.split()
        return f"UTM {parts[-1]}", "WGS 84 Ellipsoid"

    if "+" in upper_crs and upper_crs.startswith("EPSG:"):
        # Compound EPSG: EPSG:4979+5773
        parts = upper_crs.split("+")
        horiz = parts[0]
        vert = f"EPSG:{parts[1]}" if not parts[1].startswith(
            "EPSG:") else parts[1]
        return horiz, vert

    try:
        parsed = pymap.parse_gcp_projection_string(crs_string)
        if not parsed:
            return crs_string, "WGS 84 Ellipsoid"

        pyproj_crs = pyproj.CRS(parsed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            crs_dict = pyproj_crs.to_dict()
            ellps = crs_dict.get("ellps", "Ellipsoid")
            vert = f"{ellps.upper()}" if ellps != "Ellipsoid" else "WGS 84 Ellipsoid"
            if "towgs84" in crs_dict and np.array(crs_dict["towgs84"]).any():
                vert += f" ({','.join(map(str, crs_dict['towgs84']))})"

        if crs_string.strip().startswith("+proj="):
            if crs_dict.get("proj") == "utm":
                zone = crs_dict.get("zone", "")
                hemisphere = "S" if "south" in crs_dict else "N"
                horiz = f"UTM {zone}{hemisphere}"
            else:
                horiz = crs_string
            return str(horiz), str(vert)

        if pyproj_crs.coordinate_operation and pyproj_crs.coordinate_operation.name:
            op_name = pyproj_crs.coordinate_operation.name
            if op_name.startswith("UTM zone "):
                horiz = "UTM " + op_name.replace("UTM zone ", "")
                return str(horiz), str(vert)

        horiz = pyproj_crs.name if pyproj_crs.name != "unknown" else crs_string

        if pyproj_crs.is_compound:
            horiz = pyproj_crs.sub_crs_list[0].name
            vert = pyproj_crs.sub_crs_list[1].name

        return str(horiz), str(vert)
    except Exception:
        return crs_string, "WGS 84 Ellipsoid"


@overload
def ecef_from_lla(lat: float, lon: float,
                  alt: float) -> Tuple[float, float, float]: ...


@overload
def ecef_from_lla(
    lat: NDArray, lon: NDArray, alt: NDArray
) -> Tuple[NDArray, NDArray, NDArray]: ...


def ecef_from_lla(
    lat: Scalars, lon: Scalars, alt: Scalars
) -> Tuple[Scalars, Scalars, Scalars]:
    """
    Compute ECEF XYZ from latitude, longitude and altitude.

    All using the WGS84 model.
    Altitude is the distance to the WGS84 ellipsoid.
    Check results here http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm

    >>> lat, lon, alt = 10, 20, 30
    >>> x, y, z = ecef_from_lla(lat, lon, alt)
    >>> np.allclose(lla_from_ecef(x,y,z), [lat, lon, alt])
    True
    """
    a2 = WGS84_a**2
    b2 = WGS84_b**2
    lat = np.radians(lat)
    lon = np.radians(lon)
    L = 1.0 / np.sqrt(a2 * np.cos(lat) ** 2 + b2 * np.sin(lat) ** 2)
    x = (a2 * L + alt) * np.cos(lat) * np.cos(lon)
    y = (a2 * L + alt) * np.cos(lat) * np.sin(lon)
    z = (b2 * L + alt) * np.sin(lat)
    return x, y, z


@overload
def lla_from_ecef(x: float, y: float,
                  z: float) -> Tuple[float, float, float]: ...


@overload
def lla_from_ecef(
    x: NDArray, y: NDArray, z: NDArray
) -> Tuple[NDArray, NDArray, NDArray]: ...


def lla_from_ecef(
    x: Scalars, y: Scalars, z: Scalars
) -> Tuple[Scalars, Scalars, Scalars]:
    """
    Compute latitude, longitude and altitude from ECEF XYZ.

    All using the WGS84 model.
    Altitude is the distance to the WGS84 ellipsoid.
    """
    a = WGS84_a
    b = WGS84_b
    ea = np.sqrt((a**2 - b**2) / a**2)
    eb = np.sqrt((a**2 - b**2) / b**2)
    # pyre-ignore[6]: pyre ignores that x,y are all either scalars or arrays
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)
    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + eb**2 * b * np.sin(theta) ** 3, p - ea**2 * a * np.cos(theta) ** 3
    )
    N = a / np.sqrt(1 - ea**2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), alt


def ecef_from_topocentric_transform(lat: float, lon: float, alt: float) -> NDArray:
    """
    Transformation from a topocentric frame at reference position to ECEF.

    The topocentric reference frame is a metric one with the origin
    at the given (lat, lon, alt) position, with the X axis heading east,
    the Y axis heading north and the Z axis vertical to the ellipsoid.
    >>> a = ecef_from_topocentric_transform(30, 20, 10)
    >>> b = ecef_from_topocentric_transform_finite_diff(30, 20, 10)
    >>> np.allclose(a, b)
    True
    """
    x, y, z = ecef_from_lla(lat, lon, alt)
    sa = np.sin(np.radians(lat))
    ca = np.cos(np.radians(lat))
    so = np.sin(np.radians(lon))
    co = np.cos(np.radians(lon))
    return np.array(
        [
            [-so, -sa * co, ca * co, x],
            [co, -sa * so, ca * so, y],
            [0, ca, sa, z],
            [0, 0, 0, 1],
        ]
    )


def ecef_from_topocentric_transform_finite_diff(
    lat: float, lon: float, alt: float
) -> NDArray:
    """
    Transformation from a topocentric frame at reference position to ECEF.

    The topocentric reference frame is a metric one with the origin
    at the given (lat, lon, alt) position, with the X axis heading east,
    the Y axis heading north and the Z axis vertical to the ellipsoid.
    """
    eps = 1e-2
    x, y, z = ecef_from_lla(lat, lon, alt)
    v1 = (
        (
            np.array(ecef_from_lla(lat, lon + eps, alt))
            - np.array(ecef_from_lla(lat, lon - eps, alt))
        )
        / 2
        / eps
    )
    v2 = (
        (
            np.array(ecef_from_lla(lat + eps, lon, alt))
            - np.array(ecef_from_lla(lat - eps, lon, alt))
        )
        / 2
        / eps
    )
    v3 = (
        (
            np.array(ecef_from_lla(lat, lon, alt + eps))
            - np.array(ecef_from_lla(lat, lon, alt - eps))
        )
        / 2
        / eps
    )
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v3 /= np.linalg.norm(v3)
    return np.array(
        [
            [v1[0], v2[0], v3[0], x],
            [v1[1], v2[1], v3[1], y],
            [v1[2], v2[2], v3[2], z],
            [0, 0, 0, 1],
        ]
    )


@overload
def topocentric_from_lla(
    lat: float,
    lon: float,
    alt: float,
    reflat: float,
    reflon: float,
    refalt: float,
) -> Tuple[float, float, float]: ...


@overload
def topocentric_from_lla(
    lat: NDArray,
    lon: NDArray,
    alt: NDArray,
    reflat: float,
    reflon: float,
    refalt: float,
) -> Tuple[NDArray, NDArray, NDArray]: ...


def topocentric_from_lla(
    lat: Scalars,
    lon: Scalars,
    alt: Scalars,
    reflat: float,
    reflon: float,
    refalt: float,
) -> Tuple[Scalars, Scalars, Scalars]:
    """
    Transform from lat, lon, alt to topocentric XYZ.

    >>> lat, lon, alt = -10, 20, 100
    >>> np.allclose(topocentric_from_lla(lat, lon, alt, lat, lon, alt),
    ...     [0,0,0])
    True
    >>> x, y, z = topocentric_from_lla(lat, lon, alt, 0, 0, 0)
    >>> np.allclose(lla_from_topocentric(x, y, z, 0, 0, 0),
    ...     [lat, lon, alt])
    True
    """
    T = np.linalg.inv(ecef_from_topocentric_transform(reflat, reflon, refalt))
    # pyre-ignore[6]: pyre gets confused with Scalar vs float vs NDarray
    x, y, z = ecef_from_lla(lat, lon, alt)
    tx = T[0, 0] * x + T[0, 1] * y + T[0, 2] * z + T[0, 3]
    ty = T[1, 0] * x + T[1, 1] * y + T[1, 2] * z + T[1, 3]
    tz = T[2, 0] * x + T[2, 1] * y + T[2, 2] * z + T[2, 3]
    return tx, ty, tz


@overload
def lla_from_topocentric(
    x: float,
    y: float,
    z: float,
    reflat: float,
    reflon: float,
    refalt: float,
) -> Tuple[float, float, float]: ...


@overload
def lla_from_topocentric(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    reflat: float,
    reflon: float,
    refalt: float,
) -> Tuple[NDArray, NDArray, NDArray]: ...


def lla_from_topocentric(
    x: Scalars,
    y: Scalars,
    z: Scalars,
    reflat: float,
    reflon: float,
    refalt: float,
) -> Tuple[Scalars, Scalars, Scalars]:
    """
    Transform from topocentric XYZ to lat, lon, alt.
    """
    T = ecef_from_topocentric_transform(reflat, reflon, refalt)
    ex = T[0, 0] * x + T[0, 1] * y + T[0, 2] * z + T[0, 3]
    ey = T[1, 0] * x + T[1, 1] * y + T[1, 2] * z + T[1, 3]
    ez = T[2, 0] * x + T[2, 1] * y + T[2, 2] * z + T[2, 3]
    return lla_from_ecef(ex, ey, ez)


@overload
def gps_distance(latlon_1: Sequence[float],
                 latlon_2: Sequence[float]) -> float: ...


@overload
def gps_distance(
    latlon_1: Sequence[NDArray], latlon_2: Sequence[NDArray]
) -> NDArray: ...
@overload
def gps_distance(latlon_1: NDArray, latlon_2: NDArray) -> NDArray: ...


class TopocentricConverter:
    """Convert to and from a topocentric reference frame."""

    def __init__(self, reflat: float, reflon: float, refalt: float) -> None:
        """Init the converter given the reference origin."""
        self.lat = reflat
        self.lon = reflon
        self.alt = refalt

    @overload
    def to_topocentric(
        self, lat: float, lon: float, alt: float
    ) -> Tuple[float, float, float]: ...

    @overload
    def to_topocentric(
        self, lat: NDArray, lon: NDArray, alt: NDArray
    ) -> Tuple[NDArray, NDArray, NDArray]: ...

    def to_topocentric(
        self, lat: Scalars, lon: Scalars, alt: Scalars
    ) -> Tuple[Scalars, Scalars, Scalars]:
        """Convert lat, lon, alt to topocentric x, y, z."""
        # pyre-ignore[6]: pyre gets confused with Scalar vs float vs NDarray
        return topocentric_from_lla(lat, lon, alt, self.lat, self.lon, self.alt)

    @overload
    def to_lla(self, x: float, y: float,
               z: float) -> Tuple[float, float, float]: ...

    @overload
    def to_lla(
        self, x: NDArray, y: NDArray, z: NDArray
    ) -> Tuple[NDArray, NDArray, NDArray]: ...

    def to_lla(
        self, x: Scalars, y: Scalars, z: Scalars
    ) -> Tuple[Scalars, Scalars, Scalars]:
        """Convert topocentric x, y, z to lat, lon, alt."""
        # pyre-ignore[6]: pyre gets confused with Scalar vs float vs NDarray
        return lla_from_topocentric(x, y, z, self.lat, self.lon, self.alt)

    def __eq__(self, o: "TopocentricConverter") -> bool:
        return np.allclose([self.lat, self.lon, self.alt], (o.lat, o.lon, o.alt))


def opk_from_ypr(
    lat: float,
    lon: float,
    alt: float,
    yaw: float,
    pitch: float,
    roll: float,
    apply_pitch_offset: bool = False,
    tc: Optional[TopocentricConverter] = None,
) -> Optional[Dict[str, float]]:
    """Convert Yaw, Pitch, Roll to Omega, Phi, Kappa."""
    if tc is None:
        tc = TopocentricConverter(lat, lon, alt)
    y, p, r = np.radians([yaw, pitch, roll])

    # YPR rotation matrix
    cnb = np.array(
        [
            [
                np.cos(y) * np.cos(p),
                np.cos(y) * np.sin(p) * np.sin(r) - np.sin(y) * np.cos(r),
                np.cos(y) * np.sin(p) * np.cos(r) + np.sin(y) * np.sin(r),
            ],
            [
                np.sin(y) * np.cos(p),
                np.sin(y) * np.sin(p) * np.sin(r) + np.cos(y) * np.cos(r),
                np.sin(y) * np.sin(p) * np.cos(r) - np.cos(y) * np.sin(r),
            ],
            [-np.sin(p), np.cos(p) * np.sin(r), np.cos(p) * np.cos(r)],
        ]
    )

    if apply_pitch_offset:
        cnb = cnb.dot(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))

    # Convert between image and body coordinates
    cbb = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    delta = 1e-10
    p1 = np.array(tc.to_topocentric(lat + delta, lon, alt))
    p2 = np.array(tc.to_topocentric(lat - delta, lon, alt))
    xnp = p1 - p2
    m = np.linalg.norm(xnp)

    if m == 0:
        return None

    # Unit vector pointing north
    xnp /= m

    znp = np.array([0, 0, -1]).T
    ynp = np.cross(znp, xnp)
    cen = np.array([xnp, ynp, znp]).T

    # OPK rotation matrix
    ceb = cen.dot(cnb).dot(cbb)

    return {
        "omega": float(np.degrees(np.arctan2(-ceb[1][2], ceb[2][2]))),
        "phi": float(np.degrees(np.arcsin(ceb[0][2]))),
        "kappa": float(np.degrees(np.arctan2(-ceb[0][1], ceb[0][0]))),
    }


def construct_proj_transformer(proj_str: str, inverse: bool = False) -> pyproj.Transformer:
    """
    Construct a pyproj Transformer object, converting between the given projection antod WGS84 (EPSG:4326).
    If inverse is True, the transformation is from WGS84 to the given projection.
    """
    crs_4326 = pyproj.CRS.from_epsg(4326)
    if inverse:
        return pyproj.Transformer.from_proj(crs_4326, pyproj.CRS(proj_str))
    else:
        return pyproj.Transformer.from_proj(pyproj.CRS(proj_str), crs_4326)


def transform_to_proj(
    point: Sequence[float], reference: TopocentricConverter, projection: pyproj.Transformer
) -> List[float]:
    """
    Transform a point defined wrt. the local topocentric frame to a projection
    defined by the given Transformer. We assume the Transformer goes from
    WGS84 to the desired projection.
    """
    assert projection.source_crs.to_epsg(
    ) == 4326, "Transformer source CRS must be WGS84 (EPSG:4326)"

    lat, lon, altitude = reference.to_lla(point[0], point[1], point[2])
    easting, northing = projection.transform(lat, lon)
    return [easting, northing, altitude]


def get_proj_transform_matrix(
    reference: TopocentricConverter, projection: pyproj.Transformer
) -> NDArray:
    """Get the linear transform from reconstruction coords to geocoords."""
    p = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    q = [transform_to_proj(point, reference, projection) for point in p]

    transformation = np.array(
        [
            [q[0][0] - q[3][0], q[1][0] - q[3][0], q[2][0] - q[3][0], q[3][0]],
            [q[0][1] - q[3][1], q[1][1] - q[3][1], q[2][1] - q[3][1], q[3][1]],
            [q[0][2] - q[3][2], q[1][2] - q[3][2], q[2][2] - q[3][2], q[3][2]],
            [0, 0, 0, 1],
        ]
    )
    return transformation


def transform_reconstruction_with_matrix(
    reconstruction: "types.Reconstruction", transformation: NDArray
) -> None:
    """Apply a rigid transformation to a reconstruction in-place."""
    A, b = transformation[:3, :3], transformation[:3, 3]
    A1 = np.linalg.inv(A)

    for rig_instance in reconstruction.rig_instances.values():
        R = rig_instance.pose.get_rotation_matrix()
        rig_instance.pose.set_rotation_matrix(np.dot(R, A1))
        rig_instance.pose.set_origin(np.dot(A, rig_instance.pose.get_origin()) + b)

    for point in reconstruction.points.values():
        point.coordinates = list(np.dot(A, point.coordinates) + b)


def transform_reconstruction_with_proj(
    reconstruction: "types.Reconstruction", transformation: pyproj.Transformer
) -> None:
    """Apply a proj Transformer to a reconstruction in-place."""
    eps = 1e-3
    for rig_instance in reconstruction.rig_instances.values():
        origin = rig_instance.pose.get_origin()

        # Jacobian for rotation update
        p0 = np.array(transform_to_proj(
            origin, reconstruction.reference, transformation))
        px = np.array(transform_to_proj(
            origin + [eps, 0, 0], reconstruction.reference, transformation))
        py = np.array(transform_to_proj(
            origin + [0, eps, 0], reconstruction.reference, transformation))
        pz = np.array(transform_to_proj(
            origin + [0, 0, eps], reconstruction.reference, transformation))
        J = np.column_stack(
            ((px - p0) / eps, (py - p0) / eps, (pz - p0) / eps))

        rig_instance.pose.set_origin(p0)
        rig_instance.pose.set_rotation_matrix(
            rig_instance.pose.get_rotation_matrix() @ np.linalg.inv(J))

    for point in reconstruction.points.values():
        point.coordinates = transform_to_proj(
            point.coordinates, reconstruction.reference, transformation
        )


def gps_distance(
    latlon_1: Union[Sequence[Scalars], NDArray],
    latlon_2: Union[Sequence[Scalars], NDArray],
) -> Scalars:
    """
    Distance between two (lat,lon) pairs.

    >>> p1 = (42.1, -11.1)
    >>> p2 = (42.2, -11.3)
    >>> 19000 < gps_distance(p1, p2) < 20000
    True
    """
    x1, y1, z1 = ecef_from_lla(latlon_1[0], latlon_1[1], 0.0)
    x2, y2, z2 = ecef_from_lla(latlon_2[0], latlon_2[1], 0.0)

    dis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    return dis


# ─────────────────────────────────────────────────────────────────────────
#  CRS helpers and GeoTIFF raster I/O (GDAL / OSR)
#
#  All DSM/ortho GeoTIFF reading and writing lives here so the dataset layer
#  only owns file *paths* and delegates the raster work.
# ─────────────────────────────────────────────────────────────────────────

#: No-data value written to (and recognised in) DSM GeoTIFF bands.
DSM_NODATA: float = -9999.0


def utm_zone_from_lonlat(lon: float, lat: float) -> Tuple[int, bool]:
    """UTM zone number and hemisphere for a position in degrees.

    Returns ``(zone, north)`` where ``north`` is True for the northern
    hemisphere.
    """
    zone = int((lon + 180.0) / 6.0) + 1
    return zone, lat >= 0.0


def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """WGS84 / UTM EPSG code for the zone containing the given lon/lat.

    Northern hemisphere → 326xx, southern → 327xx.
    """
    zone, north = utm_zone_from_lonlat(lon, lat)
    return (32600 if north else 32700) + zone


def is_projected_crs(crs_str: Optional[str]) -> bool:
    """True if ``crs_str`` is (or has a) projected horizontal CRS.

    Geographic CRS (lat/lon degrees, e.g. WGS84 / EPSG:4326 / EPSG:4979) and
    unparseable strings return False.  Compound CRS are judged by their
    horizontal component.
    """
    if not crs_str or crs_str.upper() in ("WGS84", "EPSG:4326", "EPSG:4979"):
        return False
    try:
        parsed = pymap.parse_gcp_projection_string(crs_str)
        crs = pyproj.CRS(parsed if parsed else crs_str)
        if crs.is_projected:
            return True
        if crs.is_compound and crs.sub_crs_list:
            return bool(crs.sub_crs_list[0].is_projected)
        return False
    except Exception:
        return False


def decide_output_crs(
    gcp_crs: Optional[str], reference: "TopocentricConverter"
) -> str:
    """Decide the CRS for georeferenced products (LAS/LAZ, DSM/ortho, report).

    Use the GCP coordinate system when it is *projected*; otherwise (no GCP, or
    GCPs given in a geographic CRS that can't make a metric product) fall back to
    the UTM zone of the reconstruction reference.  Returns a CRS string usable by
    both pyproj and OSR (the GCP string verbatim, or ``"EPSG:326xx"`` for UTM).
    """
    if gcp_crs is not None and is_projected_crs(gcp_crs):
        return gcp_crs
    return f"EPSG:{utm_epsg_from_lonlat(reference.lon, reference.lat)}"


def crs_to_wkt(crs_str: str) -> str:
    """WKT for a CRS given as EPSG code / proj string / WKT (via OSR)."""

    srs = osr.SpatialReference()
    if srs.SetFromUserInput(crs_str) != 0:
        parsed = pymap.parse_gcp_projection_string(crs_str)
        if not parsed or srs.SetFromUserInput(parsed) != 0:
            raise ValueError(f"Could not parse CRS: {crs_str}")
    return srs.ExportToWkt()


def _bottom_up_geotransform(
    origin_x: float, origin_y: float, gsd: float, height: int
) -> Tuple[float, float, float, float, float, float]:
    """GDAL geotransform for a north-up grid whose row 0 is the BOTTOM row.

    ``origin_x`` / ``origin_y`` are the left / bottom edges of the grid.  GeoTIFF
    rows run top→bottom, so the top-left Y is ``origin_y + height * gsd`` and the
    pixel height is negative.
    """
    top_left_y = origin_y + height * gsd
    return (origin_x, gsd, 0.0, top_left_y, 0.0, -gsd)


def save_dsm_geotiff(
    path: str,
    grid: NDArray,
    origin_x: float,
    origin_y: float,
    gsd: float,
    srs_wkt: Optional[str] = None,
    nodata: float = DSM_NODATA,
) -> None:
    """Save a DSM grid as a single-band float32 GeoTIFF.

    Args:
        path: output file path.
        grid: (H, W) float32 array, NaN = no data (row 0 = bottom).
        origin_x: X coordinate of the left edge of the grid.
        origin_y: Y coordinate of the bottom edge of the grid.
        gsd: ground sample distance in world units per pixel.
        srs_wkt: CRS as WKT. None = no CRS written.
        nodata: value written for NaN cells.
    """

    h, w = grid.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, w, h, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(_bottom_up_geotransform(origin_x, origin_y, gsd, h))
    if srs_wkt:
        ds.SetProjection(srs_wkt)

    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    out = np.where(np.isnan(grid), nodata, grid).astype(np.float32)
    # Flip vertically: GeoTIFF row 0 = top, our grid row 0 = bottom.
    band.WriteArray(np.flipud(out))
    band.FlushCache()
    ds = None  # close file


def load_dsm_geotiff(path: str) -> Tuple[NDArray, Tuple[float, ...], str]:
    """Load a DSM GeoTIFF.

    Returns:
        (grid, geotransform, crs_wkt): grid has NaN for no-data cells and is
        flipped back to our convention (row 0 = bottom).
    """

    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"DSM not found: {path}")
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr = band.ReadAsArray().astype(np.float32)
    # Flip back: GeoTIFF row 0 = top → our convention row 0 = bottom.
    arr = np.flipud(arr)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    gt = ds.GetGeoTransform()
    wkt = ds.GetProjection() or ""
    ds = None
    return arr, gt, wkt


def read_geotiff_downsampled(
    path: str, max_size: int
) -> Tuple[NDArray, Optional[float]]:
    """Read a GeoTIFF decimated so its longest side is at most ``max_size`` px.

    The raster is decimated on read (GDAL RasterIO buffer resize), so the full
    resolution grid is never materialised — handy for cheap thumbnails of large
    DSM/ortho products.  Orientation is kept as stored in the file (GeoTIFF row 0
    = top), i.e. ready for display.

    Args:
        path: GeoTIFF path.
        max_size: longest output side in pixels; values <= 0 read at full size.

    Returns:
        (array, nodata): ``array`` is (H, W) for a single-band raster or
        (H, W, bands) for a multi-band one, in the raster's native dtype.
        ``nodata`` is band 1's NoData value (None if unset).
    """
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"GeoTIFF not found: {path}")
    w, h = ds.RasterXSize, ds.RasterYSize
    longest = max(w, h)
    if max_size > 0 and longest > max_size:
        out_w = max(1, int(round(w * max_size / longest)))
        out_h = max(1, int(round(h * max_size / longest)))
    else:
        out_w, out_h = w, h
    arr = ds.ReadAsArray(buf_xsize=out_w, buf_ysize=out_h)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    ds = None  # close file
    if arr.ndim == 3:
        # GDAL returns (bands, H, W); switch to (H, W, bands) for display.
        arr = np.transpose(arr, (1, 2, 0))
    return arr, nodata


def save_ortho_geotiff(
    path: str,
    image: NDArray,
    origin_x: float,
    origin_y: float,
    gsd: float,
    srs_wkt: Optional[str] = None,
    nodata_mask: Optional[NDArray] = None,
) -> None:
    """Save an ortho image as a GeoTIFF (uint8 RGB, optional alpha).

    Args:
        image: (H, W, 3) uint8 array (row 0 = bottom).
        origin_x: X coordinate of the left edge.
        origin_y: Y coordinate of the bottom edge.
        gsd: ground sample distance.
        srs_wkt: CRS as WKT. None = no CRS written.
        nodata_mask: optional (H, W) bool, True where there is no surface.  When
            given, a 4th alpha band is written (0 = no-data / transparent,
            255 = valid) so no-data is distinguishable from a genuine black pixel
            and GIS tools can fill it.
    """

    h, w = image.shape[:2]
    n_bands = 4 if nodata_mask is not None else 3
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, w, h, n_bands, gdal.GDT_Byte)
    ds.SetGeoTransform(_bottom_up_geotransform(origin_x, origin_y, gsd, h))
    if srs_wkt:
        ds.SetProjection(srs_wkt)

    # Flip vertically and write each band.
    flipped = np.flipud(image)
    for band_idx in range(3):
        band = ds.GetRasterBand(band_idx + 1)
        band.WriteArray(flipped[:, :, band_idx])
        band.FlushCache()

    if nodata_mask is not None:
        alpha = np.where(np.flipud(nodata_mask), 0, 255).astype(np.uint8)
        aband = ds.GetRasterBand(4)
        aband.SetColorInterpretation(gdal.GCI_AlphaBand)
        aband.WriteArray(alpha)
        aband.FlushCache()
    ds = None


def save_dsm_ortho_streamed_geotiff(
    dsm_path: str,
    ortho_path: str,
    gh: int,
    gw: int,
    origin_x: float,
    origin_y: float,
    gsd: float,
    fill_band: Callable[[int, int], Tuple[NDArray, NDArray]],
    band_rows: int,
    srs_wkt: Optional[str] = None,
    nodata: float = DSM_NODATA,
    geotransform: Optional[Tuple[float, float,
                                 float, float, float, float]] = None,
    gcps: Optional[List["gdal.GCP"]] = None,
) -> int:
    """Write dsm + ortho GeoTIFFs band-by-band, never holding the full grid.

    ``fill_band(row_start, row_end)`` returns this horizontal band's
    ``(dsm_band (bh, gw) float32 with NaN = no-data, ortho_band (bh, gw, 3)
    uint8)`` in our grid convention (row 0 = bottom).  This function owns the
    GeoTIFF georeferencing, the vertical flip to GeoTIFF top-down order, the DSM
    no-data value, and the ortho alpha band.  BIGTIFF is enabled so the rasters
    can exceed 4 GB.  Returns the count of valid (non-NaN) cells.

    Georeferencing is either ``geotransform`` (north-up affine; defaults to the
    bottom-up transform from ``origin``/``gsd``) or ``gcps`` (ground control
    points mapping pixel→world, used for a non-linear topocentric→projected
    mapping that a later ``gdal.Warp`` resolves); ``gcps`` takes precedence.
    """

    opts = ["BIGTIFF=IF_SAFER"]
    drv = gdal.GetDriverByName("GTiff")
    dsm_ds = drv.Create(
        dsm_path, int(gw), int(gh), 1, gdal.GDT_Float32, options=opts
    )
    ortho_ds = drv.Create(
        ortho_path, int(gw), int(gh), 4, gdal.GDT_Byte, options=opts
    )
    if dsm_ds is None or ortho_ds is None:
        raise RuntimeError("Failed to create DSM/ortho GeoTIFF(s)")

    if gcps is not None:
        dsm_ds.SetGCPs(gcps, srs_wkt or "")
        ortho_ds.SetGCPs(gcps, srs_wkt or "")
    else:
        gt = geotransform or _bottom_up_geotransform(
            origin_x, origin_y, gsd, int(gh))
        dsm_ds.SetGeoTransform(gt)
        ortho_ds.SetGeoTransform(gt)
        if srs_wkt:
            dsm_ds.SetProjection(srs_wkt)
            ortho_ds.SetProjection(srs_wkt)

    dsm_band = dsm_ds.GetRasterBand(1)
    dsm_band.SetNoDataValue(nodata)
    ortho_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

    n_valid = 0
    for rs in range(0, int(gh), int(band_rows)):
        re_ = min(int(gh), rs + int(band_rows))
        dsm_b, ortho_b = fill_band(rs, re_)
        valid = ~np.isnan(dsm_b)
        n_valid += int(np.count_nonzero(valid))
        # GeoTIFF row 0 = top; our row 0 = bottom → flip and place this band's
        # window at yoff = gh - re_ (see save_dsm_geotiff for the full-grid case).
        yoff = int(gh) - re_
        out_dsm = np.where(valid, dsm_b, nodata).astype(np.float32)
        dsm_band.WriteArray(np.flipud(out_dsm), 0, yoff)
        of = np.flipud(ortho_b)
        for b in range(3):
            ortho_ds.GetRasterBand(b + 1).WriteArray(
                np.ascontiguousarray(of[:, :, b]), 0, yoff
            )
        alpha = np.where(np.flipud(valid), 255, 0).astype(np.uint8)
        ortho_ds.GetRasterBand(4).WriteArray(alpha, 0, yoff)

    dsm_ds.FlushCache()
    ortho_ds.FlushCache()
    dsm_ds = None
    ortho_ds = None
    return n_valid


def transform_points_to_proj(
    reference: "TopocentricConverter",
    transformer: pyproj.Transformer,
    pts: NDArray,
) -> NDArray:
    """Exact topocentric → projected transform of ``pts`` (N, 3), vectorized.

    Uses the full PROJ pipeline per point (``to_lla`` then the projection), so it
    is free of the tangent-plane / projection-distortion error that a single
    affine linearization accumulates over large extents.  ``transformer`` must go
    from WGS84 (EPSG:4326) to the target CRS (``construct_proj_transformer(...,
    inverse=True)``).  Returns ``(easting, northing, altitude)`` columns.
    """
    lat, lon, alt = reference.to_lla(pts[:, 0], pts[:, 1], pts[:, 2])
    easting, northing = transformer.transform(lat, lon)
    return np.column_stack([easting, northing, alt])


def transform_points_normals_to_proj(
    reference: "TopocentricConverter",
    transformer: pyproj.Transformer,
    pts: NDArray,
    normals: NDArray,
    eps: float = 1.0,
) -> Tuple[NDArray, NDArray]:
    """Exact transform of points and their normals to the projected CRS.

    Positions go through :func:`transform_points_to_proj`.  Normals are rotated
    by the *local* Jacobian of the transform at each point (finite differences
    with step ``eps`` metres), then renormalized — so they stay correct as the
    local frame rotates across the extent (Earth curvature + meridian
    convergence), unlike a single global rotation.  Returns ``(pts_proj (N, 3),
    normals_proj (N, 3))``.
    """
    base = transform_points_to_proj(reference, transformer, pts)
    jcols = []
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = eps
        jcols.append(
            (transform_points_to_proj(reference, transformer, pts + step) - base)
            / eps
        )
    # J[n] = [dcol0 dcol1 dcol2]; rotate each normal: n' = J @ n.
    jac = np.stack(jcols, axis=2)                       # (N, 3, 3)
    rotated = np.einsum("nij,nj->ni", jac, normals)
    norms = np.linalg.norm(rotated, axis=1, keepdims=True)
    np.divide(rotated, norms, out=rotated, where=norms > 0)
    return base, rotated


def _exact_pixel_gcps(
    reference: "TopocentricConverter",
    transformer: pyproj.Transformer,
    gh: int,
    gw: int,
    origin_x: float,
    origin_y: float,
    gsd: float,
    n: int,
) -> List["gdal.GCP"]:
    """Build an ``n×n`` grid of GCPs mapping source pixel → exact projected X/Y.

    Pixel ``(col, row)`` (top-down, as written by the streamed writer) maps to
    topocentric ``(origin_x + col*gsd, top_left_y - row*gsd)`` and then through
    the exact PROJ transform.  Horizontal position is height-independent for the
    supported CRS, so ``z = 0`` is used for the GCP coordinates.
    """
    top_left_y = origin_y + gh * gsd
    cols = np.linspace(0.0, gw, n)
    rows = np.linspace(0.0, gh, n)
    cc, rr = np.meshgrid(cols, rows)
    cc, rr = cc.ravel(), rr.ravel()
    tx = origin_x + cc * gsd
    ty = top_left_y - rr * gsd
    lat, lon, _ = reference.to_lla(tx, ty, np.zeros_like(tx))
    easting, northing = transformer.transform(lat, lon)
    return [
        gdal.GCP(float(easting[i]), float(northing[i]), 0.0,
                 float(cc[i]), float(rr[i]))
        for i in range(cc.size)
    ]


def _gcp_warp_method(
    gcps: List["gdal.GCP"],
    srs_wkt: str,
    gh: int,
    gw: int,
    reference: "TopocentricConverter",
    transformer: pyproj.Transformer,
    origin_x: float,
    origin_y: float,
    gsd: float,
    tol: float,
) -> Tuple[Dict[str, Any], float]:
    """Pick the GCP warp fit: order-3 polynomial, or TPS if its residual is high.

    Evaluates GDAL's order-3 GCP polynomial against the exact PROJ mapping on a
    dense check grid; returns the ``gdal.WarpOptions`` kwargs for the chosen
    method and the order-3 max residual (metres).  The polynomial is exact for
    realistic survey extents; TPS is the fallback for anything that isn't.
    """
    # A 1×1 MEM raster carries only the GCPs — the polynomial fit ignores raster
    # size, so this stays O(1) in memory regardless of the real raster.
    mem = gdal.GetDriverByName("MEM").Create("", 1, 1, 1, gdal.GDT_Byte)
    mem.SetGCPs(gcps, srs_wkt)
    tr = gdal.Transformer(
        mem, None, ["DST_SRS=" + srs_wkt, "METHOD=GCP_POLYNOMIAL",
                    "MAX_GCP_ORDER=3"])

    top_left_y = origin_y + gh * gsd
    g = np.linspace(0.0, 1.0, 21)
    cc, rr = np.meshgrid(g * gw, g * gh)
    cc, rr = cc.ravel(), rr.ravel()
    lat, lon, _ = reference.to_lla(
        origin_x + cc * gsd, top_left_y - rr * gsd, np.zeros_like(cc))
    ex_e, ex_n = transformer.transform(lat, lon)
    pts = np.column_stack([cc, rr, np.zeros_like(cc)]).tolist()
    fit = np.array(tr.TransformPoints(0, pts)[0])
    residual = float(np.max(
        np.hypot(fit[:, 0] - ex_e, fit[:, 1] - ex_n)))
    mem = None

    if residual <= tol:
        return {"polynomialOrder": 3}, residual
    logger.warning(
        f"GCP order-3 fit residual {residual:.3f} m > {tol} m; using TPS")
    return {"tps": True}, residual


def save_dsm_ortho_streamed_georeferenced(
    dsm_path: str,
    ortho_path: str,
    gh: int,
    gw: int,
    origin_x: float,
    origin_y: float,
    gsd: float,
    fill_band: Callable[[int, int], Tuple[NDArray, NDArray]],
    band_rows: int,
    reference: "TopocentricConverter",
    output_crs: str,
    tmp_dsm_path: str,
    tmp_ortho_path: str,
    n_gcp: int = 11,
    gcp_residual_tol: float = 0.05,
) -> int:
    """Write dsm + ortho GeoTIFFs georeferenced (north-up) in ``output_crs``.

    The grid is built in the topocentric east/north frame; the topocentric →
    projected map is non-linear (Earth curvature + projection distortion), so a
    single affine misplaces cells far from the reference (metres of error per
    ~10 km, mostly vertical).  Instead this:

    * writes a source raster whose pixels are the topocentric grid, carrying an
      ``n_gcp × n_gcp`` grid of GCPs computed with the **exact** PROJ transform
      and per-cell **exact altitudes** (``to_lla``), then
    * ``gdal.Warp`` resolves the GCPs (order-3 polynomial, or TPS if the fit
      residual exceeds ``gcp_residual_tol``) into a north-up raster.

    Returns the valid-cell count.
    """
    transformer = construct_proj_transformer(output_crs, inverse=True)
    wkt = crs_to_wkt(output_crs)
    gcps = _exact_pixel_gcps(
        reference, transformer, gh, gw, origin_x, origin_y, gsd, n_gcp)
    warp_method, _ = _gcp_warp_method(
        gcps, wkt, gh, gw, reference, transformer,
        origin_x, origin_y, gsd, gcp_residual_tol)

    cols_x = origin_x + (np.arange(gw) + 0.5) * gsd  # topo x per column centre

    def fill_band_geo(rs: int, re_: int) -> Tuple[NDArray, NDArray]:
        dsm_b, ortho_b = fill_band(rs, re_)
        bh = re_ - rs
        rows_y = origin_y + (rs + np.arange(bh) + 0.5) * gsd  # topo y per row
        valid = np.isfinite(dsm_b) & (dsm_b != DSM_NODATA)
        xx = np.broadcast_to(cols_x, (bh, gw))
        yy = np.broadcast_to(rows_y[:, None], (bh, gw))
        # Exact projected altitude per cell (z 0 where invalid is irrelevant).
        _, _, alt = reference.to_lla(xx, yy, np.where(valid, dsm_b, 0.0))
        return np.where(valid, alt, np.nan).astype(np.float32), ortho_b

    n_valid = save_dsm_ortho_streamed_geotiff(
        tmp_dsm_path, tmp_ortho_path, gh, gw, origin_x, origin_y, gsd,
        fill_band_geo, band_rows, srs_wkt=wkt, gcps=gcps,
    )

    common = dict(
        dstSRS=wkt, xRes=gsd, yRes=gsd, multithread=True,
        creationOptions=["BIGTIFF=IF_SAFER"], **warp_method,
    )
    gdal.Warp(
        dsm_path, tmp_dsm_path,
        options=gdal.WarpOptions(
            resampleAlg="bilinear", srcNodata=DSM_NODATA, dstNodata=DSM_NODATA,
            **common,
        ),
    )
    gdal.Warp(
        ortho_path, tmp_ortho_path,
        options=gdal.WarpOptions(resampleAlg="bilinear", **common),
    )

    drv = gdal.GetDriverByName("GTiff")
    drv.Delete(tmp_dsm_path)
    drv.Delete(tmp_ortho_path)
    return n_valid
