# pyre-strict
import os

import numpy as np
import pytest
from numpy.typing import NDArray
from osgeo import gdal

from opensfm import geo, io, pygeo, pygeometry, pymap, types


def test_ecef_lla_consistency() -> None:
    lla_before = [46.5274109, 6.5722075, 402.16]
    ecef = geo.ecef_from_lla(lla_before[0], lla_before[1], lla_before[2])
    lla_after = geo.lla_from_ecef(ecef[0], ecef[1], ecef[2])
    assert np.allclose(np.array(lla_after), lla_before)


def test_ecef_lla_topocentric_consistency() -> None:
    lla_ref = [46.5, 6.5, 400]
    lla_before = [46.5274109, 6.5722075, 402.16]
    enu = geo.topocentric_from_lla(
        lla_before[0], lla_before[1], lla_before[2], lla_ref[0], lla_ref[1], lla_ref[2]
    )
    lla_after = geo.lla_from_topocentric(
        enu[0], enu[1], enu[2], lla_ref[0], lla_ref[1], lla_ref[2]
    )
    assert np.allclose(np.array(lla_after), lla_before)


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
    assert geo.TopocentricConverter(
        40, 30, 0) == geo.TopocentricConverter(40, 30, 0)
    assert geo.TopocentricConverter(
        40, 32, 0) != geo.TopocentricConverter(40, 30, 0)


def test_parse_projection() -> None:
    proj_str = pymap.parse_gcp_projection_string("WGS84")
    assert proj_str is None

    proj_str = pymap.parse_gcp_projection_string("WGS84 UTM 31N")
    assert proj_str is not None

    proj = geo.construct_proj_transformer(proj_str)
    easting, northing = 431760, 4582313.7
    lat, lon = 41.38946, 2.18378
    plat, plon = proj.transform(easting, northing)
    assert np.allclose((lat, lon), (plat, plon))


# ── gps_distance (vectorized) ───────────────────────────────────────


def test_gps_distance_scalar() -> None:
    """Scalar GPS distance between two close points is in plausible range."""
    p1 = (42.1, -11.1)
    p2 = (42.2, -11.3)
    d = geo.gps_distance(p1, p2)
    assert 19000 < d < 20000


def test_gps_distance_same_point() -> None:
    """Distance of a point to itself is zero."""
    p = (48.8566, 2.3522)
    assert np.isclose(geo.gps_distance(p, p), 0.0)


def test_gps_distance_vectorized() -> None:
    """Vectorized GPS distance returns an array of distances."""
    lats1 = np.array([42.1, 42.0])
    lons1 = np.array([-11.1, -11.0])
    lats2 = np.array([42.2, 42.0])
    lons2 = np.array([-11.3, -11.0])
    d = geo.gps_distance(np.array([lats1, lons1]), np.array([lats2, lons2]))
    assert d.shape == (2,)
    assert d[1] < 1.0  # same point


# ── construct_proj_transformer inverse ───────────────────────────────


def test_construct_proj_transformer_inverse() -> None:
    """Inverse transformer goes from WGS84 to projection."""
    proj_str = pymap.parse_gcp_projection_string("WGS84 UTM 31N")
    assert proj_str is not None

    fwd = geo.construct_proj_transformer(proj_str)  # proj → WGS84
    inv = geo.construct_proj_transformer(
        proj_str, inverse=True)  # WGS84 → proj

    easting, northing = 431760, 4582313.7
    lat, lon = fwd.transform(easting, northing)
    e2, n2 = inv.transform(lat, lon)
    assert np.allclose((easting, northing), (e2, n2), atol=1e-2)


# ── transform_to_proj ───────────────────────────────────────────────


def test_transform_to_proj_origin() -> None:
    """Origin of topocentric frame maps to the reference's projection coords."""
    proj_str = pymap.parse_gcp_projection_string("WGS84 UTM 31N")
    assert proj_str is not None
    inv = geo.construct_proj_transformer(proj_str, inverse=True)

    ref = geo.TopocentricConverter(41.38946, 2.18378, 0)
    result = geo.transform_to_proj([0, 0, 0], ref, inv)
    # The easting/northing should be close to known values
    assert len(result) == 3
    assert abs(result[2]) < 1.0  # altitude ~ 0


# ── get_proj_transform_matrix ────────────────────────────────────────


def test_get_proj_transform_matrix_shape() -> None:
    """Transform matrix is 4x4."""
    proj_str = pymap.parse_gcp_projection_string("WGS84 UTM 31N")
    assert proj_str is not None
    inv = geo.construct_proj_transformer(proj_str, inverse=True)
    ref = geo.TopocentricConverter(41.38946, 2.18378, 0)
    mat = geo.get_proj_transform_matrix(ref, inv)
    assert mat.shape == (4, 4)
    assert np.allclose(mat[3], [0, 0, 0, 1])


# ── TopocentricConverter vectorized ─────────────────────────────────


def test_topocentric_converter_vectorized() -> None:
    """TopocentricConverter handles array inputs."""
    ref = geo.TopocentricConverter(46.5, 6.5, 400)
    lats = np.array([46.5, 46.51])
    lons = np.array([6.5, 6.51])
    alts = np.array([400.0, 410.0])
    x, y, z = ref.to_topocentric(lats, lons, alts)
    assert isinstance(x, np.ndarray)
    assert x.shape == (2,)
    # Origin maps to (0, 0, 0)
    assert np.allclose([x[0], y[0], z[0]], [0, 0, 0], atol=1e-3)
    # Round-trip
    lat2, lon2, alt2 = ref.to_lla(x, y, z)
    assert np.allclose(lat2, lats, atol=1e-8)
    assert np.allclose(lon2, lons, atol=1e-8)


# ── UTM zone / EPSG helpers ──────────────────────────────────────────


def test_utm_zone_from_lonlat() -> None:
    assert geo.utm_zone_from_lonlat(2.35, 48.85) == (31, True)    # Paris
    assert geo.utm_zone_from_lonlat(18.4, -33.9) == (34, False)   # Cape Town
    # Zone boundaries (each zone spans 6° of longitude, zone 1 starts at -180°).
    assert geo.utm_zone_from_lonlat(-180.0, 0.0)[0] == 1
    # still band [-180, -174)
    assert geo.utm_zone_from_lonlat(-174.01, 0.0)[0] == 1
    assert geo.utm_zone_from_lonlat(-173.99, 0.0)[0] == 2
    assert geo.utm_zone_from_lonlat(0.0, 0.0)[0] == 31      # band [0, 6)
    assert geo.utm_zone_from_lonlat(6.01, 0.0)[0] == 32


def test_utm_epsg_from_lonlat() -> None:
    assert geo.utm_epsg_from_lonlat(2.35, 48.85) == 32631   # north → 326xx
    assert geo.utm_epsg_from_lonlat(18.4, -33.9) == 32734   # south → 327xx


# ── CRS classification + decision ────────────────────────────────────


def test_is_projected_crs() -> None:
    assert not geo.is_projected_crs(None)
    assert not geo.is_projected_crs("")
    assert not geo.is_projected_crs("WGS84")
    assert not geo.is_projected_crs("EPSG:4326")   # geographic 2D
    assert not geo.is_projected_crs("EPSG:4979")   # geographic 3D
    assert not geo.is_projected_crs("not-a-real-crs")
    assert geo.is_projected_crs("EPSG:32631")      # UTM 31N
    assert geo.is_projected_crs(
        "+proj=utm +zone=10 +north +datum=WGS84 +units=m +no_defs"
    )
    # Compound (projected horizontal + vertical) is judged by its horizontal part.
    assert geo.is_projected_crs("EPSG:32631+5773")


def test_decide_output_crs() -> None:
    paris = geo.TopocentricConverter(48.85, 2.35, 35.0)
    # No GCP, or geographic GCP → UTM from the reference LLA.
    assert geo.decide_output_crs(None, paris) == "EPSG:32631"
    assert geo.decide_output_crs("WGS84", paris) == "EPSG:32631"
    assert geo.decide_output_crs("EPSG:4326", paris) == "EPSG:32631"
    # Projected GCP → used verbatim.
    assert geo.decide_output_crs("EPSG:2154", paris) == "EPSG:2154"
    # Southern hemisphere reference → 327xx.
    cape = geo.TopocentricConverter(-33.9, 18.4, 10.0)
    assert geo.decide_output_crs(None, cape) == "EPSG:32734"


def test_crs_to_wkt() -> None:
    wkt = geo.crs_to_wkt("EPSG:32631")
    assert "UTM zone 31N" in wkt
    # proj strings are also accepted.
    assert "PROJCS" in geo.crs_to_wkt(
        "+proj=utm +zone=31 +datum=WGS84 +units=m")
    with pytest.raises(ValueError):
        geo.crs_to_wkt("definitely-not-a-crs")


# ── DSM GeoTIFF save / load round-trip ───────────────────────────────


def test_dsm_geotiff_roundtrip(tmp_path) -> None:
    grid = np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0]], dtype=np.float32)
    path = str(tmp_path / "dsm.tif")
    geo.save_dsm_geotiff(
        path, grid, origin_x=100.0, origin_y=200.0, gsd=0.5,
        srs_wkt=geo.crs_to_wkt("EPSG:32631"),
    )
    arr, gt, wkt = geo.load_dsm_geotiff(path)
    assert np.allclose(arr, grid, equal_nan=True)  # NaN no-data preserved
    # Bottom-up convention: top-left Y = origin_y + H*gsd; pixel height negative.
    assert np.allclose(gt, (100.0, 0.5, 0.0, 200.0 + 2 * 0.5, 0.0, -0.5))
    assert "UTM zone 31N" in wkt


def test_dsm_geotiff_no_crs(tmp_path) -> None:
    path = str(tmp_path / "dsm.tif")
    geo.save_dsm_geotiff(path, np.zeros((2, 2), np.float32), 0.0, 0.0, 1.0)
    _, _, wkt = geo.load_dsm_geotiff(path)
    assert wkt == ""  # topocentric output is left untagged


def test_load_dsm_geotiff_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        geo.load_dsm_geotiff(str(tmp_path / "nope.tif"))


# ── Ortho GeoTIFF (RGB + optional alpha) ─────────────────────────────


def test_ortho_geotiff_rgb_only(tmp_path) -> None:
    img = np.zeros((2, 3, 3), np.uint8)
    img[..., 0] = 10
    img[..., 1] = 20
    path = str(tmp_path / "ortho.tif")
    geo.save_ortho_geotiff(path, img, 0.0, 0.0, 1.0)
    ds = gdal.Open(path)
    assert ds.RasterCount == 3
    # Stored flipped (raster row 0 = our top row); red band matches flipud(img).
    assert np.array_equal(ds.GetRasterBand(
        1).ReadAsArray(), np.flipud(img[..., 0]))
    ds = None


def test_ortho_geotiff_alpha(tmp_path) -> None:
    img = np.full((2, 3, 3), 50, np.uint8)
    # row 0 = bottom
    mask = np.array([[True, False, False], [False, False, False]])
    path = str(tmp_path / "ortho.tif")
    geo.save_ortho_geotiff(path, img, 0.0, 0.0, 1.0, nodata_mask=mask)
    ds = gdal.Open(path)
    assert ds.RasterCount == 4
    assert ds.GetRasterBand(4).GetColorInterpretation() == gdal.GCI_AlphaBand
    alpha = ds.GetRasterBand(4).ReadAsArray()
    assert alpha[-1, 0] == 0     # masked bottom-left → transparent
    assert alpha[0, 0] == 255    # unmasked → opaque
    ds = None


# ── Streamed DSM/ortho writer ────────────────────────────────────────


def test_streamed_geotiff_roundtrip(tmp_path) -> None:
    gh, gw, gsd = 5, 4, 2.0
    full = np.arange(gh * gw, dtype=np.float32).reshape(gh, gw)
    full[0, 0] = np.nan
    ortho = np.zeros((gh, gw, 3), np.uint8)

    def fill_band(rs: int, re_: int):
        return full[rs:re_].copy(), ortho[rs:re_].copy()

    dsm_p = str(tmp_path / "dsm.tif")
    ortho_p = str(tmp_path / "ortho.tif")
    n = geo.save_dsm_ortho_streamed_geotiff(
        dsm_p, ortho_p, gh, gw, 10.0, 20.0, gsd, fill_band, band_rows=2,
        srs_wkt=geo.crs_to_wkt("EPSG:32631"),
    )
    assert n == int(np.count_nonzero(~np.isnan(full)))
    arr, _, wkt = geo.load_dsm_geotiff(dsm_p)
    assert np.allclose(arr, full, equal_nan=True)
    assert "UTM zone 31N" in wkt
    assert gdal.Open(ortho_p).RasterCount == 4


def test_streamed_geotiff_custom_geotransform(tmp_path) -> None:
    gh, gw = 3, 3
    full = np.ones((gh, gw), np.float32)

    def fill_band(rs: int, re_: int):
        return full[rs:re_].copy(), np.zeros((re_ - rs, gw, 3), np.uint8)

    dsm_p = str(tmp_path / "d.tif")
    rotated = (500.0, 1.0, 0.5, 400.0, 0.5, -1.0)  # has rotation terms
    geo.save_dsm_ortho_streamed_geotiff(
        dsm_p, str(tmp_path / "o.tif"), gh, gw, 0.0, 0.0, 1.0, fill_band, 2,
        geotransform=rotated,
    )
    _, gt, _ = geo.load_dsm_geotiff(dsm_p)
    assert np.allclose(gt, rotated)


# ── Exact topocentric → projected transform (points + normals) ───────


def test_transform_points_to_proj_matches_pointwise() -> None:
    ref = geo.TopocentricConverter(45.0, 6.0, 300.0)
    tr = geo.construct_proj_transformer("EPSG:32632", inverse=True)
    pts = (np.random.RandomState(0).rand(50, 3) - 0.5) * 20000.0
    vec = geo.transform_points_to_proj(ref, tr, pts)
    pointwise = np.array([geo.transform_to_proj(p, ref, tr) for p in pts])
    assert np.allclose(vec, pointwise, atol=1e-6)


def test_transform_points_to_proj_is_nonlinear() -> None:
    """Over a large extent the exact transform must differ from a single affine
    by metres (the error the affine path used to introduce)."""
    ref = geo.TopocentricConverter(45.0, 6.0, 300.0)
    tr = geo.construct_proj_transformer("EPSG:32632", inverse=True)
    affine = geo.get_proj_transform_matrix(ref, tr)
    pts = np.array([[20000.0, 20000.0, 20.0], [-15000.0, 8000.0, 5.0]])
    exact = geo.transform_points_to_proj(ref, tr, pts)
    approx = pts @ affine[:3, :3].T + affine[:3, 3]
    # Mostly a vertical (tangent-plane curvature) gap of tens of metres.
    assert abs(exact[0, 2] - approx[0, 2]) > 10.0


def test_transform_points_normals_to_proj() -> None:
    ref = geo.TopocentricConverter(45.0, 6.0, 300.0)
    tr = geo.construct_proj_transformer("EPSG:32632", inverse=True)
    pts = (np.random.RandomState(1).rand(100, 3) - 0.5) * 20000.0
    normals = np.tile([0.0, 0.0, 1.0], (100, 1)).astype(np.float32)
    proj, nrm = geo.transform_points_normals_to_proj(ref, tr, pts, normals)
    # Positions match the points-only path; normals stay unit-length and ~up
    # (the local vertical is nearly preserved by the metric transform).
    assert np.allclose(proj, geo.transform_points_to_proj(ref, tr, pts))
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-5)
    assert np.all(nrm[:, 2] > 0.99)


# ── Georeferenced (GCP-warp) DSM/ortho ───────────────────────────────


def test_dsm_ortho_georeferenced_warp(tmp_path) -> None:
    ref = geo.TopocentricConverter(48.85, 2.35, 35.0)
    gh, gw, gsd = 30, 40, 0.5
    full = np.full((gh, gw), 10.0, np.float32)   # flat surface at topo z = 10
    full[0, :] = full[-1, :] = full[:, 0] = full[:, -1] = np.nan  # NaN border
    ortho = np.zeros((gh, gw, 3), np.uint8)
    ortho[..., 1] = 200

    def fill_band(rs: int, re_: int):
        return full[rs:re_].copy(), ortho[rs:re_].copy()

    dsm_p = str(tmp_path / "dsm.tif")
    ortho_p = str(tmp_path / "ortho.tif")
    n = geo.save_dsm_ortho_streamed_georeferenced(
        dsm_p, ortho_p, gh, gw, 0.0, 0.0, gsd, fill_band, band_rows=16,
        reference=ref, output_crs="EPSG:32631",
        tmp_dsm_path=dsm_p + ".topo.tif", tmp_ortho_path=ortho_p + ".topo.tif",
    )
    assert n == int(np.count_nonzero(~np.isnan(full)))
    # Temp (rotated) sources are cleaned up.
    assert not os.path.exists(dsm_p + ".topo.tif")
    assert not os.path.exists(ortho_p + ".topo.tif")

    ds = gdal.Open(dsm_p)
    gt = ds.GetGeoTransform()
    assert ds.GetSpatialRef().GetAuthorityCode(None) == "32631"
    assert abs(gt[2]) < 1e-9 and abs(gt[4]) < 1e-9          # north-up
    assert gt[0] > 440000 and gt[3] > 5.40e6               # located in UTM 31N
    arr = ds.GetRasterBand(1).ReadAsArray()
    valid = arr[arr != ds.GetRasterBand(1).GetNoDataValue()]
    # Heights become projected altitude: topo z (10) + reference altitude (35).
    assert np.allclose(valid, 45.0, atol=0.5)
    ds = None

    od = gdal.Open(ortho_p)
    assert od.RasterCount == 4
    assert od.GetRasterBand(4).GetColorInterpretation() == gdal.GCI_AlphaBand
    od = None


def test_dsm_georeferenced_heights_follow_curvature(tmp_path) -> None:
    """Over a 20 km extent a flat topocentric plane must come out as exact
    *altitudes* that rise with distance (tangent-plane curvature) — the metres of
    error the old single-affine path could not represent."""

    ref = geo.TopocentricConverter(45.0, 6.0, 300.0)
    half, gsd = 10000.0, 50.0
    gw = gh = int(2 * half / gsd)            # 400 x 400
    flat = np.zeros((gh, gw), np.float32)    # topocentric plane z = 0
    ortho = np.zeros((gh, gw, 3), np.uint8)

    def fill_band(rs: int, re_: int):
        return flat[rs:re_].copy(), ortho[rs:re_].copy()

    dsm_p = str(tmp_path / "dsm.tif")
    ortho_p = str(tmp_path / "ortho.tif")
    geo.save_dsm_ortho_streamed_georeferenced(
        dsm_p, ortho_p, gh, gw, -half, -half, gsd, fill_band, band_rows=128,
        reference=ref, output_crs=geo.decide_output_crs(None, ref),
        tmp_dsm_path=dsm_p + ".t.tif", tmp_ortho_path=ortho_p + ".t.tif",
    )
    ds = gdal.Open(dsm_p)
    arr = ds.GetRasterBand(1).ReadAsArray()
    valid = arr[arr != ds.GetRasterBand(1).GetNoDataValue()]
    ds = None

    # ≈ 300 (reference alt)
    origin_alt = ref.to_lla(0.0, 0.0, 0.0)[2]
    corner_alt = ref.to_lla(half, half, 0.0)[2]          # lifted by curvature
    assert corner_alt - origin_alt > 10.0                # curvature is real here
    # Warped heights span the exact altitude range (a single affine would be flat).
    assert abs(valid.min() - origin_alt) < 1.0
    assert abs(valid.max() - corner_alt) < 1.0


# ── transform_reconstruction_* keep cameras consistent with points ──


def _nadir_reconstruction(ref: geo.TopocentricConverter) -> types.Reconstruction:
    """Nadir cameras 1 km from the origin at 350 m, looking at ground points."""
    rec = types.Reconstruction()
    rec.reference = ref
    camera = pygeometry.Camera.create_perspective(0.8, 0.0, 0.0)
    camera.id = "cam"
    camera.width, camera.height = 4000, 3000
    rec.add_camera(camera)
    for i, (x, y, yaw) in enumerate(
        [(1000, 0, 0.0), (1000, 60, 0.3), (0, 1000, -0.2), (-700, -700, 2.0)]
    ):
        # 180 deg about x (camera looks down), then a yaw about the vertical.
        c, s = np.cos(yaw), np.sin(yaw)
        R_cw = np.diag([1.0, -1.0, -1.0]) @ np.array(
            [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
        )
        pose = pygeometry.Pose()
        pose.set_rotation_matrix(R_cw)
        pose.set_origin(np.array([x, y, 350.0]))
        rec.create_shot(str(i), camera.id, pose)
        for dx in (-80, 0, 80):
            for dy in (-60, 0, 60):
                rec.create_point(f"{i}_{dx}_{dy}", [x + dx, y + dy, 0.0])
    return rec


def _projections(rec: types.Reconstruction) -> NDArray:
    """Pixel projections of the ground points below each shot."""
    return np.array(
        [
            rec.shots[s].project(np.array(rec.points[p].coordinates)) * 4000
            for s in sorted(rec.shots)
            for p in sorted(rec.points)
            if p.startswith(s + "_")
        ]
    )


def _json_round_trip(rec: types.Reconstruction) -> types.Reconstruction:
    return io.reconstruction_from_json(io.reconstruction_to_json(rec))


def test_transform_reconstruction_with_matrix_keeps_projections() -> None:
    """Converted cameras still see the converted points where they did before."""
    proj_str = pymap.parse_gcp_projection_string("WGS84 UTM 31N")
    assert proj_str is not None
    ref = geo.TopocentricConverter(41.38946, 2.18378, 300.0)
    rec = _nadir_reconstruction(ref)
    before = _projections(rec)

    t = geo.get_proj_transform_matrix(
        ref, geo.construct_proj_transformer(proj_str, inverse=True)
    )
    t[:2, 3] -= np.round(t[:2, 3])  # offset removed, as ODM does
    geo.transform_reconstruction_with_matrix(rec, t)
    rec = _json_round_trip(rec)

    # The only change left is the grid scale (horizontal only), < 0.5 px here.
    assert np.abs(_projections(rec) - before).max() < 0.5  # pixels
    for shot in rec.shots.values():
        origin = _nadir_reconstruction(ref).shots[shot.id].pose.get_origin()
        expected = t[:3, :3] @ origin + t[:3, 3]
        assert np.linalg.norm(shot.pose.get_origin() - expected) < 1e-3  # metres


def test_transform_reconstruction_with_proj_keeps_projections() -> None:
    """Same check for the projected (non-linear) conversion."""
    proj_str = pymap.parse_gcp_projection_string("WGS84 UTM 31N")
    assert proj_str is not None
    ref = geo.TopocentricConverter(41.38946, 2.18378, 300.0)
    rec = _nadir_reconstruction(ref)
    before = _projections(rec)

    transformer = geo.construct_proj_transformer(proj_str, inverse=True)
    geo.transform_reconstruction_with_proj(rec, transformer)
    rec = _json_round_trip(rec)

    assert np.abs(_projections(rec) - before).max() < 0.5  # pixels
    for shot in rec.shots.values():
        origin = _nadir_reconstruction(ref).shots[shot.id].pose.get_origin()
        expected = geo.transform_to_proj(origin, ref, transformer)
        assert np.linalg.norm(shot.pose.get_origin() - expected) < 1e-3  # metres
