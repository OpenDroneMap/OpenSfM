import numpy as np
import pyproj
from opensfm import geo, pygeo


def test_ecef_lla_consistency() -> None:
    lla_before = [46.5274109, 6.5722075, 402.16]
    ecef = geo.ecef_from_lla(lla_before[0], lla_before[1], lla_before[2])
    lla_after = geo.lla_from_ecef(ecef[0], ecef[1], ecef[2])
    assert np.allclose(lla_after, lla_before)


def test_ecef_lla_topocentric_consistency() -> None:
    lla_ref = [46.5, 6.5, 400]
    lla_before = [46.5274109, 6.5722075, 402.16]
    enu = geo.topocentric_from_lla(
        lla_before[0], lla_before[1], lla_before[2], lla_ref[0], lla_ref[1], lla_ref[2]
    )
    lla_after = geo.lla_from_topocentric(
        enu[0], enu[1], enu[2], lla_ref[0], lla_ref[1], lla_ref[2]
    )
    assert np.allclose(lla_after, lla_before)


def test_ecef_lla_consistency_pygeo() -> None:
    lla_before = [46.5274109, 6.5722075, 402.16]
    ecef = pygeo.ecef_from_lla(lla_before[0], lla_before[1], lla_before[2])
    lla_after = pygeo.lla_from_ecef(ecef[0], ecef[1], ecef[2])
    assert np.allclose(lla_after, lla_before)


def test_ecef_lla_topocentric_consistency_pygeo() -> None:
    lla_ref = [46.5, 6.5, 400]
    lla_before = [46.5274109, 6.5722075, 402.16]
    enu = pygeo.topocentric_from_lla(
        lla_before[0], lla_before[1], lla_before[2], lla_ref[0], lla_ref[1], lla_ref[2]
    )
    lla_after = pygeo.lla_from_topocentric(
        enu[0], enu[1], enu[2], lla_ref[0], lla_ref[1], lla_ref[2]
    )
    assert np.allclose(lla_after, lla_before)

def test_eq_geo() -> None:
    assert geo.TopocentricConverter(40,30,0) == geo.TopocentricConverter(40,30,0)
    assert geo.TopocentricConverter(40,32,0) != geo.TopocentricConverter(40,30,0)


def test_transform_to_proj_matches_pyproj() -> None:
    lat, lon, alt = 41.38946, 2.18378, 12.3
    reference = geo.TopocentricConverter(lat, lon, alt)
    proj = "+proj=utm +zone=31 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    transformer = geo.construct_proj_transformer(proj, inverse=True)
    expected = pyproj.Transformer.from_proj(
        pyproj.CRS.from_epsg(4326),
        pyproj.CRS(proj),
    ).transform(lat, lon)

    easting, northing, altitude = geo.transform_to_proj([0, 0, 0], reference, transformer)

    assert np.allclose((easting, northing), expected)
    assert altitude == alt
