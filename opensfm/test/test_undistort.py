# pyre-strict
import itertools

import numpy as np
from opensfm import pygeometry, pymap, types, undistort
from unittest.mock import MagicMock


def test_perspective_views_of_a_panorama() -> None:
    reconstruction = types.Reconstruction()
    camera = pygeometry.Camera.create_spherical()
    camera.id = "spherical_camera"
    camera.width = 8000
    camera.height = 4000
    reconstruction.add_camera(camera)
    pose = pygeometry.Pose(np.array([1, 2, 3]), np.array([4, 5, 6]))
    spherical_shot = reconstruction.create_shot("shot1", camera.id, pose=pose)

    urec = types.Reconstruction()
    rig_instance_count = itertools.count()
    undistort.perspective_views_of_a_panorama(
        spherical_shot, 800, urec, "jpg", rig_instance_count
    )

    assert len(urec.rig_cameras) == 6
    assert len(urec.rig_instances) == 1
    assert len(urec.rig_instances["0"].shots) == 6
    front_found = False
    for shot in urec.rig_instances["0"].shots.values():
        assert np.allclose(shot.pose.get_origin(),
                           spherical_shot.pose.get_origin())
        if shot.rig_camera_id == "front":
            front_found = True
            assert np.allclose(shot.pose.rotation,
                               spherical_shot.pose.rotation)
        else:
            assert not np.allclose(
                shot.pose.rotation, spherical_shot.pose.rotation)
    assert front_found


# ── perspective_camera_from_brown ────────────────────────────────────


def test_perspective_camera_from_brown() -> None:
    """Brown camera converts to perspective with averaged focal."""
    brown = pygeometry.Camera.create_brown(
        0.8, 1.1,
        np.array([0.01, -0.02]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    brown.id = "brown_cam"
    brown.width = 1000
    brown.height = 800

    result = undistort.perspective_camera_from_brown(brown)
    assert result.projection_type == "perspective"
    assert result.id == "brown_cam"
    assert result.width == 1000
    assert result.height == 800
    expected_focal = 0.8 * (1 + 1.1) / 2.0
    assert abs(result.focal - expected_focal) < 1e-10


# ── perspective_camera_from_fisheye_opencv ───────────────────────────


def test_perspective_camera_from_fisheye_opencv() -> None:
    """Fisheye opencv converts to perspective with averaged focal."""
    fisheye_cv = pygeometry.Camera.create_fisheye_opencv(
        0.7, 1.05,
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
    )
    fisheye_cv.id = "fisheye_cv_cam"
    fisheye_cv.width = 1920
    fisheye_cv.height = 1080

    result = undistort.perspective_camera_from_fisheye_opencv(fisheye_cv)
    assert result.projection_type == "perspective"
    expected_focal = 0.7 * (1 + 1.05) / 2.0
    assert abs(result.focal - expected_focal) < 1e-10


# ── perspective_camera_from_fisheye62 ────────────────────────────────


def test_perspective_camera_from_fisheye62() -> None:
    """Fisheye62 converts to perspective with averaged focal."""
    fisheye62 = pygeometry.Camera.create_fisheye62(
        0.6, 1.2,
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    fisheye62.id = "fisheye62_cam"
    fisheye62.width = 2000
    fisheye62.height = 1500

    result = undistort.perspective_camera_from_fisheye62(fisheye62)
    assert result.projection_type == "perspective"
    expected_focal = 0.6 * (1 + 1.2) / 2.0
    assert abs(result.focal - expected_focal) < 1e-10


# ── scale_image ──────────────────────────────────────────────────────


def test_scale_image_noop() -> None:
    """Image smaller than max_size is not scaled."""
    img = np.ones((50, 80, 3), dtype=np.uint8)
    result = undistort.scale_image(img, 100)
    assert result.shape == (50, 80, 3)


def test_scale_image_downscales() -> None:
    """Image larger than max_size is scaled down."""
    img = np.ones((200, 400, 3), dtype=np.uint8)
    result = undistort.scale_image(img, 200)
    assert result.shape[1] == 200
    assert result.shape[0] == 100


def test_scale_image_grayscale() -> None:
    """Grayscale image is also handled."""
    img = np.ones((300, 600), dtype=np.uint8)
    result = undistort.scale_image(img, 300)
    assert result.shape[1] == 300
    assert result.shape[0] == 150


# ── add_image_format_extension ───────────────────────────────────────


def test_add_image_format_extension_adds() -> None:
    assert undistort.add_image_format_extension("photo", "jpg") == "photo.jpg"


def test_add_image_format_extension_no_duplicate() -> None:
    assert undistort.add_image_format_extension(
        "photo.jpg", "jpg") == "photo.jpg"


def test_add_image_format_extension_different() -> None:
    assert undistort.add_image_format_extension(
        "photo.png", "jpg") == "photo.png.jpg"


# ── _validity_from_remap ────────────────────────────────────────────


def _make_undistorted_shot(rec: types.Reconstruction, shot_id: str) -> pymap.Shot:
    """Helper to create a simple undistorted shot."""
    cam = pygeometry.Camera.create_perspective(0.5, 0.0, 0.0)
    cam.id = f"cam_{shot_id}"
    cam.width = 10
    cam.height = 10
    rec.add_camera(cam)
    return rec.create_shot(shot_id, cam.id)


def test_validity_from_remap_in_bounds() -> None:
    """Pixels mapping inside original image bounds are valid."""
    rec = types.Reconstruction()
    shot = _make_undistorted_shot(rec, "s1")

    # All coordinates are within [0, width-1) x [0, height-1)
    map1 = np.full((10, 10), 5.0, dtype=np.float32)
    map2 = np.full((10, 10), 5.0, dtype=np.float32)
    remap = {"s1": (map1, map2)}

    result = undistort._validity_from_remap(remap, 20, 20, [shot], 1000)
    assert "s1" in result
    assert np.all(result["s1"] == 255)


def test_validity_from_remap_out_of_bounds() -> None:
    """Pixels mapping outside original image bounds are invalid."""
    rec = types.Reconstruction()
    shot = _make_undistorted_shot(rec, "s1")

    map1 = np.full((10, 10), -1.0, dtype=np.float32)  # out of bounds
    map2 = np.full((10, 10), 5.0, dtype=np.float32)
    remap = {"s1": (map1, map2)}

    result = undistort._validity_from_remap(remap, 20, 20, [shot], 1000)
    assert np.all(result["s1"] == 0)


def test_validity_from_remap_no_remap() -> None:
    """Shot without remap entry gets all-valid mask."""
    rec = types.Reconstruction()
    shot = _make_undistorted_shot(rec, "s1")

    result = undistort._validity_from_remap({}, 20, 20, [shot], 1000)
    assert "s1" in result
    assert np.all(result["s1"] == 255)

def test_image_filter_is_applied() -> None:
    """Check that imageFilter is called during undistortion."""
    rec = types.Reconstruction()
    shot = _make_undistorted_shot(rec, "s1")

    data = MagicMock()
    data.config = {"undistorted_image_max_size": 1000}
    original_image = np.ones((10, 10, 3), dtype=np.uint8)
    data.load_image.return_value = original_image
    data.load_mask.return_value = None
    data.load_segmentation.return_value = None
    data.load_detection.return_value = None

    udata = MagicMock()

    filter_called = False
    def mock_filter(shot_id, img):
        nonlocal filter_called
        filter_called = True
        assert shot_id == "s1"
        return img * 2

    arguments = (shot, [shot], data, udata, mock_filter, {})
    undistort.undistort_image_and_masks(arguments)
    assert filter_called
