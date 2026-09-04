# pyre-strict
import datetime
import json
import logging
import math
import os
import random
import cv2
import statistics
from collections import defaultdict
from functools import lru_cache
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import scipy.spatial as spatial
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch

plt.set_loglevel('info')

import numpy as np
from numpy.typing import NDArray
from opensfm import feature_loader, features, geo, geometry, report, io, multiview, pygeometry, pymap, types
from opensfm.dataset import DataSet, DataSetBase

logger: logging.Logger = logging.getLogger(__name__)

RESIDUAL_PIXEL_CUTOFF = 4

# #05CB63 — Mapillary green
_CLR_ACCENT = (0.02, 0.80, 0.39)
_CLR_BAD = np.array(report.COLOR_GRADE_BAD) / \
    255.0         # #E05252 — muted red
_CLR_AVG = np.array(report.COLOR_GRADE_AVG) / \
    255.0         # #D4A843 — warm amber
_CLR_GOOD = np.array(report.COLOR_GRADE_GOOD) / \
    255.0       # #3CB371 — medium sea-green

_REPORT_SEQ_CMAP = colors.LinearSegmentedColormap.from_list(
    "opensfm_seq",
    [_CLR_BAD, _CLR_AVG, _CLR_GOOD],
)

_REPORT_SEQ_CMAP_INV = colors.LinearSegmentedColormap.from_list(
    "opensfm_quality",
    [_CLR_GOOD, _CLR_AVG, _CLR_BAD],
)


def _norm2d(point: NDArray) -> float:
    return math.sqrt(point[0] * point[0] + point[1] * point[1])


def _length_histogram(
    tracks_manager: pymap.TracksManager, points: Dict[str, pymap.Landmark]
) -> Tuple[List[str], List[int]]:
    hist = defaultdict(int)
    for point in points.values():
        obs_count = point.number_of_observations()
        if not obs_count:
            obs_count = len(tracks_manager.get_track_observations(point.id))
        hist[obs_count] += 1
    return list(hist.keys()), list(hist.values())


def _gps_errors(reconstruction: types.Reconstruction) -> List[NDArray]:
    errors = []
    for shot in reconstruction.shots.values():
        if shot.metadata.gps_position.has_value:
            bias = reconstruction.biases[shot.camera.id]
            gps = shot.metadata.gps_position.value
            unbiased_gps = bias.transform(gps)
            optical_center = shot.pose.get_origin()
            errors.append(np.array(optical_center - unbiased_gps))
    return errors


def _compute_value_at_p_level(samples: NDArray, p: float = 0.90) -> float:
    """Based on the method described in "Computation of scalar accuracy metrics LE, CE, and SE as both predictive and sample-based statistics"
    https://masseranolabs.com/IGTF2016-000255.pdf"""
    n = len(samples)
    if n < 1:
        return -1
    y = np.sort(samples)
    if n < 7:
        return float(y[-1])

    kprime = int(np.floor((n + 1) * p)) - 1  # arrays are 0-indexed

    if not ((n + 1) * p).is_integer():
        kml = kprime + 0.5
    else:
        kml = kprime

    i = int(np.floor(kml)) - 1
    j = int(np.ceil(kml)) - 1

    return 0.5 * (y[i] + y[j])


def _compute_cele_stats(errors: Optional[NDArray]) -> Dict[str, Any]:
    """Compute CE90 (horizontal) and LE90 (vertical) accuracy"""
    if errors is None or len(errors) == 0:
        return {}

    errors = np.asarray(errors)
    ce_errors = np.sqrt(errors[:, 0] ** 2 + errors[:, 1] ** 2)
    le_errors = np.abs(errors[:, 2])
    return {
        "ce90": _compute_value_at_p_level(ce_errors, 0.90),
        "le90": _compute_value_at_p_level(le_errors, 0.90),
    }


def _gps_gcp_opk_errors_stats(errors: Optional[NDArray], names: List[str]) -> Dict[str, Any]:
    if errors is None or len(errors) == 0:
        return {}

    stats = {}
    squared = np.multiply(errors, errors)
    m_squared = np.mean(squared, 0)
    mean = np.mean(errors, 0)
    std_dev = np.std(errors, 0)
    average = np.average(np.linalg.norm(errors, axis=1))

    stats["mean"] = {names[0]: mean[0], names[1]: mean[1], names[2]: mean[2]}
    stats["std"] = {names[0]: std_dev[0], names[1]                    : std_dev[1], names[2]: std_dev[2]}
    stats["error"] = {
        names[0]: math.sqrt(m_squared[0]),
        names[1]: math.sqrt(m_squared[1]),
        names[2]: math.sqrt(m_squared[2]),
    }
    stats["average_error"] = average
    return stats


def gps_errors(reconstructions: List[types.Reconstruction]) -> Dict[str, Any]:
    all_errors = []
    all_gps_std = []
    for rec in reconstructions:
        all_errors += _gps_errors(rec)
        for shot in rec.shots.values():
            if shot.metadata.gps_position.has_value:
                if shot.metadata.gps_accuracy.has_value:
                    all_gps_std.append(
                        np.array(shot.metadata.gps_accuracy.value))
                else:
                    all_gps_std.append(geo.DEFAULT_GPS_STD)
    errors = np.array(all_errors)
    stats = _gps_gcp_opk_errors_stats(errors, ["x", "y", "z"])
    stats.update(_compute_cele_stats(errors))
    if all_gps_std:
        avg_std = np.mean(all_gps_std, axis=0)
        stats["average_gps_std"] = {
            "x": float(avg_std[0]), "y": float(avg_std[1]), "z": float(avg_std[2])
        }
    return stats


def _opk_errors(reconstruction: types.Reconstruction) -> List[NDArray]:
    errors = []
    for shot in reconstruction.shots.values():
        if shot.metadata.opk_angles.has_value:
            opk_exif = np.array(shot.metadata.opk_angles.value)
            rotation_computed = shot.pose.get_rotation_matrix()

            # Extract OPK from computed rotation
            opk_computed = np.degrees(
                np.array(geometry.opk_from_rotation(rotation_computed))
            )

            # Compute difference per-angle and normalize to [-180, 180]
            opk_diff = opk_computed - opk_exif
            opk_diff = (opk_diff + 180) % 360 - 180
            errors.append(opk_diff)
    return errors


def opk_errors(reconstructions: List[types.Reconstruction]) -> Dict[str, Any]:
    all_errors = []
    for rec in reconstructions:
        all_errors += _opk_errors(rec)
    return _gps_gcp_opk_errors_stats(np.array(all_errors), ["omega", "phi", "kappa"])


def gcp_errors(
    data: DataSetBase, reconstructions: List[types.Reconstruction]
) -> Dict[str, Any]:
    reference = data.load_reference()
    gcps = data.load_ground_control_points()
    if not gcps:
        return {}

    gcp_horizontal_sd = data.config["gcp_horizontal_sd"]
    gcp_vertical_sd = data.config["gcp_vertical_sd"]

    all_errors = []
    gcp_details: List[Dict[str, Any]] = []
    gcp_stats = []
    for gcp in gcps:
        if not gcp.lla:
            continue

        result = None
        for rec in reconstructions:
            result = multiview.triangulate_gcp(
                gcp, rec.shots, data.config["gcp_reprojection_error_threshold"]
            )
            if result is not None:
                break

        gcp_enu = np.array(reference.to_topocentric(*gcp.lla_vec))

        # Determine the std dev for this point
        if gcp.std_dev is not None:
            sd = gcp.std_dev
            sigma_xyz = {"x": float(sd[0]), "y": float(
                sd[1]), "z": float(sd[2])}
        else:
            sigma_xyz = {"x": gcp_horizontal_sd,
                         "y": gcp_horizontal_sd, "z": gcp_vertical_sd}

        # Determine role string
        role_str = "gcp" if gcp.role == pymap.GroundControlPointRole.GCP else "checkpoint"

        if result is None:
            # Count total projections with a valid shot
            n_total = sum(
                1 for obs in gcp.observations
                if any(obs.shot_id in rec.shots for rec in reconstructions)
            )
            gcp_details.append({
                "id": gcp.id,
                "error": None,
                "n_inliers": 0,
                "n_total": n_total,
                "role": role_str,
                "sigma": sigma_xyz,
            })
            continue

        triangulated, inliers_mask = result
        error = triangulated - gcp_enu
        all_errors.append(error)
        gcp_details.append({
            "id": gcp.id,
            "error": {"x": float(error[0]), "y": float(error[1]), "z": float(error[2])},
            "n_inliers": sum(inliers_mask),
            "n_total": len(inliers_mask),
            "role": role_str,
            "sigma": sigma_xyz,
        })

        # Export GCP stats for ODM (ground_control_points.json)
        observations = []
        for obs in gcp.observations:
            if obs.shot_id not in rec.shots:
                continue
            shot = rec.shots[obs.shot_id]

            reprojected = shot.project(gcp_enu)
            annotated = obs.projection

            r_pixel = features.denormalized_image_coordinates(np.array([[reprojected[0], reprojected[1]]]), shot.camera.width, shot.camera.height)[0]
            r_pixel[0] /= shot.camera.width
            r_pixel[1] /= shot.camera.height

            a_pixel = features.denormalized_image_coordinates(np.array([[annotated[0], annotated[1]]]), shot.camera.width, shot.camera.height)[0]
            a_pixel[0] /= shot.camera.width
            a_pixel[1] /= shot.camera.height

            observations.append({
                'shot_id': obs.shot_id,
                'annotated': list(a_pixel),
                'reprojected': list(r_pixel)
            })

        gcp_stats.append({
            'id': gcp.id,
            'coordinates': list(gcp_enu),
            'observations': observations,
            'error': list(error)
        })

    # Separate GCP-only and CP-only errors
    gcp_only_errors = [
        e for e, d in zip(all_errors, [dd for dd in gcp_details if dd["error"] is not None])
        if d["role"] == "gcp"
    ]
    cp_only_errors = [
        e for e, d in zip(all_errors, [dd for dd in gcp_details if dd["error"] is not None])
        if d["role"] == "checkpoint"
    ]

    errors = np.array(all_errors) if all_errors else np.array([])
    stats = _gps_gcp_opk_errors_stats(errors, ["x", "y", "z"])
    stats.update(_compute_cele_stats(errors))
    stats["details"] = gcp_details

    # Add separate stats for GCP and CP
    stats["gcp_only"] = _gps_gcp_opk_errors_stats(
        np.array(gcp_only_errors) if gcp_only_errors else np.array(
            []), ["x", "y", "z"]
    )
    stats["cp_only"] = _gps_gcp_opk_errors_stats(
        np.array(cp_only_errors) if cp_only_errors else np.array(
            []), ["x", "y", "z"]
    )

    crs = data.load_gcp_coordinate_system()
    if crs:
        stats["coordinate_system"] = crs

    with data.io_handler.open_wt(os.path.join(data.data_path, "stats", "ground_control_points.json")) as f:
        f.write(json.dumps(gcp_stats, indent=4))

    return stats


def td_errors(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    reconstructions: List[types.Reconstruction],
) -> Dict[str, Any]:
    errors = []
    reproj_threshold = data.config["triangulation_threshold"]
    min_ray_angle_degrees = data.config["triangulation_min_ray_angle"]
    min_depth = data.config["triangulation_min_depth"]

    for rec in reconstructions:
        reproj_errors = rec.map.compute_reprojection_errors(
            tracks_manager, pymap.ErrorType.Pixel
        )

        # For each point (cap to 1000 samples)
        # get the first (up to) 3 cameras that
        # triangulate a point and sample around
        # the projection error radius (4 points)
        # by computing all triangulation permutations
        max_samples = 1000
        points = list(rec.points.values())

        if len(points) > max_samples:
            # Deterministic sampling
            sampler = random.Random(42)
            points = sampler.sample(points, max_samples)

        for p in points:
            track_obs = tracks_manager.get_track_observations(p.id)

            # Build the per-camera reprojection-error-box corners (up to 3 cams).
            err_perms = []
            for shot_id, obs in track_obs.items():
                if shot_id not in reproj_errors:
                    continue

                rerr = reproj_errors[shot_id][p.id]
                err_perms.append([
                    rerr * np.array([1, 1]),
                    rerr * np.array([-1, 1]),
                    rerr * np.array([1, -1]),
                    rerr * np.array([-1, -1]),
                ])
                if len(err_perms) >= 3:
                    break

            # Calculate the cartesian product (try all possibilities)
            err_products = np.array(list(product(*err_perms)))

            ray_errors = []
            for err_prod in err_products:
                origins = []
                bearings = []
                i = 0

                for shot_id, obs in track_obs.items():
                    if shot_id not in reproj_errors:
                        continue

                    shot = rec.shots[shot_id]
                    origins.append(shot.pose.get_origin())

                    reprojected_obs = obs.point + err_prod[i]
                    b = shot.camera.pixel_bearing(np.array(reprojected_obs))
                    r = shot.pose.get_rotation_matrix().T
                    bearings.append(r.dot(b))

                    i += 1

                    if i >= 3:
                        break

                if len(origins) >= 2:
                    thresholds = len(origins) * [reproj_threshold]
                    valid_triangulation, X = pygeometry.triangulate_bearings_midpoint(
                        np.array(origins),
                        np.array(bearings),
                        thresholds,
                        np.radians(min_ray_angle_degrees),
                        min_depth,
                    )
                    if valid_triangulation:
                        ray_errors.append(X - p.coordinates)

            # Take the max. This is the maximum 3D error estimate
            # for this point
            if len(ray_errors) > 0:
                errors.append(np.max(np.array(ray_errors), axis=0))

    errors = np.array(errors) if errors else np.array([])
    stats = _gps_gcp_opk_errors_stats(errors, ["x", "y", "z"])
    stats.update(_compute_cele_stats(errors))
    return stats


def _compute_errors(
    reconstructions: List[types.Reconstruction], tracks_manager: pymap.TracksManager
) -> Callable[[int, pymap.ErrorType, Optional[Tuple[str, ...]]], Dict[str, Dict[str, NDArray]]]:
    @lru_cache(10)
    def _compute_errors_cached(
        index: int, error_type: pymap.ErrorType, shot_ids: Optional[Tuple[str, ...]] = None
    ) -> Dict[str, Dict[str, NDArray]]:
        return reconstructions[index].map.compute_reprojection_errors(
            tracks_manager,
            error_type,
            list(shot_ids) if shot_ids is not None else []
        )

    return _compute_errors_cached


def _get_valid_observations(
    reconstructions: List[types.Reconstruction], tracks_manager: pymap.TracksManager
) -> Callable[[int, Optional[Tuple[str, ...]]], Dict[str, Dict[str, pymap.Observation]]]:
    @lru_cache(10)
    def _get_valid_observations_cached(
        index: int, shot_ids: Optional[Tuple[str, ...]] = None
    ) -> Dict[str, Dict[str, pymap.Observation]]:
        return reconstructions[index].map.get_valid_observations(
            tracks_manager,
            list(shot_ids) if shot_ids is not None else []
        )

    return _get_valid_observations_cached


THist = Tuple[NDArray, NDArray]


def _projection_error(
    tracks_manager: pymap.TracksManager, reconstructions: List[types.Reconstruction]
) -> Tuple[float, float, float, THist, THist, THist]:
    max_shots = 4000

    # Pre-select random shots per reconstruction
    rec_shots = []
    for rec in reconstructions:
        shot_ids = list(rec.shots.keys())
        if len(shot_ids) > max_shots:
            shot_ids = random.sample(shot_ids, max_shots)
        rec_shots.append(tuple(shot_ids))

    bins = 30

    # 1. Pixel errors
    all_errors_pixels = []
    to_skip = defaultdict(set)
    average_error_pixels = 0.0
    for i in range(len(reconstructions)):
        if not rec_shots[i]:
            continue
        errors_unnormalized = _compute_errors(reconstructions, tracks_manager)(
            i, pymap.ErrorType.Pixel, rec_shots[i]
        )
        for shot_id, shot_errors in errors_unnormalized.items():
            shot = reconstructions[i].get_shot(shot_id)
            normalizer = max(shot.camera.width, shot.camera.height)
            for lm_id, error_unnormalized in shot_errors.items():
                norm_pixels = _norm2d(error_unnormalized * normalizer)
                if norm_pixels > RESIDUAL_PIXEL_CUTOFF:
                    to_skip[shot_id].add(lm_id)
                    continue
                average_error_pixels += norm_pixels
                all_errors_pixels.append(norm_pixels)

    error_count_pixels = len(all_errors_pixels)
    avg_pixels = average_error_pixels / \
        error_count_pixels if error_count_pixels > 0 else -1.0
    hist_pixels = np.histogram(all_errors_pixels, bins) if error_count_pixels > 0 else (
        np.array([]), np.array([]))
    del all_errors_pixels

    # 2. Normalized errors
    all_errors_normalized = []
    average_error_normalized = 0.0
    for i in range(len(reconstructions)):
        if not rec_shots[i]:
            continue
        errors_normalized = _compute_errors(reconstructions, tracks_manager)(
            i, pymap.ErrorType.Normalized, rec_shots[i]
        )
        for shot_id, shot_errors in errors_normalized.items():
            for lm_id, error_normalized in shot_errors.items():
                if lm_id in to_skip[shot_id]:
                    continue
                norm_normalized = _norm2d(error_normalized)
                average_error_normalized += norm_normalized
                all_errors_normalized.append(norm_normalized)

    error_count_normalized = len(all_errors_normalized)
    avg_normalized = average_error_normalized / \
        error_count_normalized if error_count_normalized > 0 else -1.0
    hist_normalized = np.histogram(
        all_errors_normalized, bins) if error_count_normalized > 0 else (np.array([]), np.array([]))
    del all_errors_normalized

    # 3. Angular errors
    all_errors_angular = []
    average_error_angular = 0.0
    for i in range(len(reconstructions)):
        if not rec_shots[i]:
            continue
        errors_angular = _compute_errors(reconstructions, tracks_manager)(
            i, pymap.ErrorType.Angular, rec_shots[i]
        )
        for shot_id, shot_errors in errors_angular.items():
            for lm_id, error_angular in shot_errors.items():
                norm_angle = error_angular[0]
                if math.isnan(norm_angle) or lm_id in to_skip[shot_id]:
                    continue
                average_error_angular += norm_angle
                all_errors_angular.append(norm_angle)

    error_count_angular = len(all_errors_angular)
    avg_angular = average_error_angular / \
        error_count_angular if error_count_angular > 0 else -1.0
    hist_angular = np.histogram(all_errors_angular, bins) if error_count_angular > 0 else (
        np.array([]), np.array([]))
    del all_errors_angular

    if error_count_normalized == 0 and error_count_pixels == 0 and error_count_angular == 0:
        dummy = (np.array([]), np.array([]))
        return (-1.0, -1.0, -1.0, dummy, dummy, dummy)

    return (
        avg_normalized,
        avg_pixels,
        avg_angular,
        hist_normalized,
        hist_pixels,
        hist_angular,
    )


def _compute_gsd(
    reconstruction: types.Reconstruction,
    tracks_manager: pymap.TracksManager,
    num_shots: int = 100,
    num_pairs: int = 2000,
) -> float:
    """Estimate Ground Sampling Distance (GSD) for a single reconstruction.

    Samples *num_pairs* pairs of reconstructed points per shot across up to
    *num_shots* randomly chosen shots, and averages the ratio
        (3-D Euclidean distance) / (2-D pixel distance)
    over all sampled pairs.  Returns -1.0 when no valid pair is found.
    """
    all_ratios: List[float] = []

    shot_ids = list(reconstruction.shots.keys())
    if len(shot_ids) > num_shots:
        shot_ids = random.sample(shot_ids, num_shots)

    all_points = reconstruction.points
    tm_shot_ids = set(tracks_manager.get_shot_ids())

    for shot_id in shot_ids:
        if shot_id not in tm_shot_ids:
            continue

        shot = reconstruction.shots[shot_id]
        w = shot.camera.width
        h = shot.camera.height
        normalizer = max(w, h)
        center = np.array([w / 2.0, h / 2.0])

        obs_dict = tracks_manager.get_shot_observations(shot_id)

        # Keep only observations whose point is in the reconstruction.
        valid_obs: List[Tuple[str, pymap.Observation]] = [
            (pid, obs) for pid, obs in obs_dict.items() if pid in all_points
        ]
        if len(valid_obs) < 2:
            continue

        n_valid = len(valid_obs)
        n_possible = n_valid * (n_valid - 1) // 2

        if n_possible <= num_pairs:
            pairs = [(i, j) for i in range(n_valid)
                     for j in range(i + 1, n_valid)]
        else:
            pairs_set: set[Tuple[int, int]] = set()
            while len(pairs_set) < num_pairs:
                a = random.randint(0, n_valid - 1)
                b = random.randint(0, n_valid - 1)
                if a != b:
                    pairs_set.add((min(a, b), max(a, b)))
            pairs = list(pairs_set)

        for i, j in pairs:
            pid_a, obs_a = valid_obs[i]
            pid_b, obs_b = valid_obs[j]

            # 3-D Euclidean distance between the two reconstructed points.
            dist_3d = float(
                np.linalg.norm(
                    all_points[pid_a].coordinates -
                    all_points[pid_b].coordinates
                )
            )

            # 2-D pixel distance between the two observations.
            px_a = obs_a.point * normalizer + center
            px_b = obs_b.point * normalizer + center
            dist_2d = float(np.linalg.norm(px_a - px_b))

            if dist_2d > 1.0:
                all_ratios.append(dist_3d / dist_2d)

    if not all_ratios:
        return -1.0

    return float(np.mean(all_ratios))


def reconstruction_statistics(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    reconstructions: List[types.Reconstruction],
) -> Dict[str, Any]:
    stats = {}

    stats["components"] = len(reconstructions)
    gps_count = 0
    for rec in reconstructions:
        for shot in rec.shots.values():
            gps_count += shot.metadata.gps_position.has_value
    stats["has_gps"] = gps_count > 2
    stats["has_gcp"] = True if data.load_ground_control_points() else False

    stats["initial_points_count"] = tracks_manager.num_tracks()
    stats["initial_shots_count"] = len(data.images())

    stats["reconstructed_points_count"] = 0
    stats["reconstructed_shots_count"] = 0
    stats["observations_count"] = 0
    hist_agg = defaultdict(int)

    for rec in reconstructions:
        if len(rec.points) > 0:
            stats["reconstructed_points_count"] += len(rec.points)
        stats["reconstructed_shots_count"] += len(rec.shots)

        # get tracks length distrbution for current reconstruction
        hist, values = _length_histogram(tracks_manager, rec.points)

        # update aggregrated histogram
        for length, count_tracks in zip(hist, values):
            hist_agg[length] += count_tracks

    # observations total and average tracks lengths
    hist_agg = sorted(hist_agg.items(), key=lambda x: x[0])
    lengths, counts = (
        np.array([int(x[0]) for x in hist_agg]),
        np.array([x[1] for x in hist_agg]),
    )

    points_count = stats["reconstructed_points_count"]
    points_count_over_two = sum(counts[1:])
    stats["observations_count"] = int(sum(lengths * counts))
    stats["average_track_length"] = (
        (stats["observations_count"] /
         points_count) if points_count > 0 else -1
    )
    stats["average_track_length_over_two"] = (
        (int(sum(lengths[1:] * counts[1:])) / points_count_over_two)
        if points_count_over_two > 0
        else -1
    )
    stats["histogram_track_length"] = {k: v for k, v in hist_agg}

    (
        avg_normalized,
        avg_pixels,
        avg_angular,
        (hist_normalized, bins_normalized),
        (hist_pixels, bins_pixels),
        (hist_angular, bins_angular),
    ) = _projection_error(tracks_manager, reconstructions)
    stats["reprojection_error_normalized"] = avg_normalized
    stats["reprojection_error_pixels"] = avg_pixels
    stats["reprojection_error_angular"] = avg_angular
    stats["reprojection_histogram_normalized"] = (
        list(map(float, hist_normalized)),
        list(map(float, bins_normalized)),
    )
    stats["reprojection_histogram_pixels"] = (
        list(map(float, hist_pixels)),
        list(map(float, bins_pixels)),
    )
    stats["reprojection_histogram_angular"] = (
        list(map(float, hist_angular)),
        list(map(float, bins_angular)),
    )

    # Ground Sampling Distance (average across all reconstruction components).
    gsd_values = [
        _compute_gsd(rec, tracks_manager) for rec in reconstructions
    ]
    valid_gsds = [g for g in gsd_values if g > 0]
    stats["gsd"] = float(np.mean(valid_gsds)) if valid_gsds else -1.0

    return stats


def processing_statistics(
    data: DataSet, reconstructions: List[types.Reconstruction]
) -> Dict[str, Any]:
    steps = {
        "Feature Extraction": "features.json",
        "Features Matching": "matches.json",
        "Tracks Merging": "tracks.json",
        "Reconstruction": "reconstruction.json",
    }

    optional_steps = {
        "Dense Clustering": "dense_clustering.json",
        "Dense Equalize": "dense_equalize.json",
        "Dense Depthmaps": "dense_depthmaps.json",
        "Dense Fusion": "dense_fusion.json",
        "Dense Merging": "dense_merging.json",
    }

    steps_times = {}
    for step_name, report_file in steps.items():
        try:
            report_str = data.load_report(report_file)
            obj = io.json_loads(report_str)
        except (IOError, OSError):
            obj = {}
        if "wall_time" in obj:
            steps_times[step_name] = obj["wall_time"]
        elif "wall_times" in obj:
            steps_times[step_name] = sum(obj["wall_times"].values())
        else:
            steps_times[step_name] = -1

    for step_name, report_file in optional_steps.items():
        try:
            report_str = data.load_report(report_file)
            obj = io.json_loads(report_str)
        except (IOError, OSError):
            continue
        if "wall_time" in obj:
            steps_times[step_name] = obj["wall_time"]
        elif "wall_times" in obj:
            steps_times[step_name] = sum(obj["wall_times"].values())

    stats = {}
    stats["steps_times"] = steps_times
    stats["steps_times"]["Total Time"] = sum(
        filter(lambda x: x >= 0, steps_times.values())
    )

    try:
        stats["date"] = datetime.datetime.fromtimestamp(
            data.io_handler.timestamp(data._reconstruction_file(None))
        ).strftime("%d/%m/%Y at %H:%M:%S")
    except FileNotFoundError:
        stats["date"] = "unknown"

    start_ct, end_ct = start_end_capture_time(reconstructions)
    if start_ct is not None and end_ct is not None:
        stats["start_date"] = datetime.datetime.utcfromtimestamp(start_ct).strftime("%d/%m/%Y at %H:%M:%S")
        stats["end_date"] = datetime.datetime.utcfromtimestamp(end_ct).strftime("%d/%m/%Y at %H:%M:%S")
    else:
        stats["start_date"] = "unknown"
        stats["end_date"] = "unknown"

    default_max = 1e30
    min_x, min_y, max_x, max_y = default_max, default_max, 0, 0
    for rec in reconstructions:
        for shot in rec.shots.values():
            o = shot.pose.get_origin()
            min_x = min(min_x, o[0])
            min_y = min(min_y, o[1])
            max_x = max(max_x, o[0])
            max_y = max(max_y, o[1])
    stats["area"] = (max_x - min_x) * \
        (max_y - min_y) if min_x != default_max else -1
    return stats


def features_statistics(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    reconstructions: List[types.Reconstruction],
) -> Dict[str, Any]:
    stats = {}
    detected = []
    images = {s for r in reconstructions for s in r.shots}
    for im in images:
        features_data = feature_loader.instance.load_all_data(
            data, im, False, False)
        if not features_data:
            continue
        detected.append(len(features_data.points))
    if len(detected) > 0:
        stats["detected_features"] = {
            "min": min(detected),
            "max": max(detected),
            "mean": int(np.mean(detected)),
            "median": int(np.median(detected)),
        }
    else:
        stats["detected_features"] = {"min": -1,
                                      "max": -1, "mean": -1, "median": -1}

    per_shots = defaultdict(int)
    for rec in reconstructions:
        all_points_keys = set(rec.points.keys())
        for shot_id in rec.shots:
            if shot_id not in tracks_manager.get_shot_ids():
                continue
            for point_id in tracks_manager.get_shot_observations(shot_id):
                if point_id not in all_points_keys:
                    continue
                per_shots[shot_id] += 1
    per_shots = list(per_shots.values())

    stats["reconstructed_features"] = {
        "min": int(min(per_shots)) if len(per_shots) > 0 else -1,
        "max": int(max(per_shots)) if len(per_shots) > 0 else -1,
        "mean": int(np.mean(per_shots)) if len(per_shots) > 0 else -1,
        "median": int(np.median(per_shots)) if len(per_shots) > 0 else -1,
    }
    return stats


def _cameras_statistics(camera_model: pygeometry.Camera) -> Dict[str, Any]:
    camera_stats = {}
    for param_type, param_value in camera_model.get_parameters_map().items():
        camera_stats[str(param_type).split(".")[1]] = param_value
    return camera_stats


def cameras_statistics(
    data: DataSetBase, reconstructions: List[types.Reconstruction]
) -> Dict[str, Any]:
    stats = {}
    permutation = np.argsort([-len(r.shots) for r in reconstructions])
    for camera_id, camera_model in data.load_camera_models().items():
        stats[camera_id] = {
            "initial_values": _cameras_statistics(camera_model)}

    for idx in permutation:
        rec = reconstructions[idx]
        for camera in rec.cameras.values():
            if camera.id not in stats:
                continue
            if "optimized_values" in stats[camera.id]:
                continue
            stats[camera.id]["optimized_values"] = _cameras_statistics(camera)
            stats[camera.id]["bias"] = io.bias_to_json(rec.biases[camera.id])

    for camera_id in data.load_camera_models():
        if "optimized_values" not in stats[camera_id]:
            del stats[camera_id]
        else:
            # Compute relative difference (%) between initial and optimized
            initial = stats[camera_id]["initial_values"]
            optimized = stats[camera_id]["optimized_values"]
            rel_diff = {}
            for param, init_val in initial.items():
                if abs(init_val) > 1e-12:
                    rel_diff[param] = abs(
                        optimized[param] - init_val) / abs(init_val) * 100.0
                else:
                    rel_diff[param] = 0.0
            stats[camera_id]["relative_difference"] = rel_diff

    return stats


def rig_statistics(
    data: DataSetBase, reconstructions: List[types.Reconstruction]
) -> Dict[str, Any]:
    stats = {}
    permutation = np.argsort([-len(r.shots) for r in reconstructions])
    rig_cameras = data.load_rig_cameras()
    cameras = data.load_camera_models()
    for rig_camera_id, rig_camera in rig_cameras.items():
        # we skip per-camera rig camera for now
        if rig_camera_id in cameras:
            continue
        stats[rig_camera_id] = {
            "initial_values": {
                "rotation": list(rig_camera.pose.rotation),
                "translation": list(rig_camera.pose.translation),
            }
        }

    for idx in permutation:
        rec = reconstructions[idx]
        for rig_camera in rec.rig_cameras.values():
            if rig_camera.id not in stats:
                continue
            if "optimized_values" in stats[rig_camera.id]:
                continue
            stats[rig_camera.id]["optimized_values"] = {
                "rotation": list(rig_camera.pose.rotation),
                "translation": list(rig_camera.pose.translation),
            }

    for rig_camera_id in rig_cameras:
        if rig_camera_id not in stats:
            continue
        if "optimized_values" not in stats[rig_camera_id]:
            del stats[rig_camera_id]

    return stats


def compute_all_statistics(
    data: DataSet,
    tracks_manager: pymap.TracksManager,
    reconstructions: List[types.Reconstruction],
) -> Dict[str, Any]:
    stats = {}

    stats["processing_statistics"] = processing_statistics(
        data, reconstructions)
    stats["features_statistics"] = features_statistics(
        data, tracks_manager, reconstructions
    )
    stats["reconstruction_statistics"] = reconstruction_statistics(
        data, tracks_manager, reconstructions
    )
    stats["camera_errors"] = cameras_statistics(data, reconstructions)
    stats["rig_errors"] = rig_statistics(data, reconstructions)
    stats["gps_errors"] = gps_errors(reconstructions)
    stats["gcp_errors"] = gcp_errors(data, reconstructions)
    stats["3d_errors"] = td_errors(data, tracks_manager, reconstructions)
    stats["opk_errors"] = opk_errors(reconstructions)
    stats["overlap"] = overlap_statistics(reconstructions, tracks_manager)

    # CRS the georeferenced products are written in (GCP CRS if projected, else
    # UTM from the reference) — the single decision shared with LAS/LAZ + DSM/ortho.
    try:
        stats["output_coordinate_system"] = data.output_coordinate_system()
    except Exception:
        logger.debug(
            "Could not determine output coordinate system", exc_info=True)

    return stats


def _grid_buckets(camera: pygeometry.Camera) -> Tuple[int, int]:
    buckets = 40
    if camera.projection_type == "spherical":
        return 2 * buckets, buckets
    else:
        return buckets, buckets


def _heatmap_buckets(camera: pygeometry.Camera) -> Tuple[int, int]:
    buckets = 500
    if camera.projection_type == "spherical":
        return 2 * buckets, buckets
    else:
        return buckets, int(buckets / camera.width * camera.height)


def _get_gaussian_kernel(radius: int, ratio: float) -> NDArray:
    std_dev = radius / ratio
    half_kernel = list(range(1, radius + 1))
    kernel = np.array(half_kernel + [radius + 1] + list(reversed(half_kernel)))
    kernel = np.exp(np.outer(kernel.T, kernel) / (2 * std_dev * std_dev))
    return kernel / sum(kernel.flatten())


def save_matchgraph(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    reconstructions: List[types.Reconstruction],
    output_path: str,
    io_handler: io.IoFilesystemBase,
) -> None:
    all_shots = []
    all_points = []
    shot_component = {}
    for i, rec in enumerate(reconstructions):
        all_points += rec.points
        all_shots += rec.shots
        for shot in rec.shots:
            shot_component[shot] = i

    connectivity = tracks_manager.get_all_pairs_connectivity(
        all_shots, all_points)
    all_values = connectivity.values()
    lowest = np.percentile(list(all_values), 5)
    highest = np.percentile(list(all_values), 75)

    min_matches: int = 2 * data.config["resection_min_inliers"]
    edges_json: List[Dict[str, Any]] = []
    for (node1, node2), edge in connectivity.items():
        if edge < min_matches:
            continue
        comp1 = shot_component.get(node1)
        comp2 = shot_component.get(node2)
        if comp1 is None or comp2 is None or comp1 != comp2:
            continue
        edges_json.append(
            {"shot1": node1, "shot2": node2, "matches": int(edge)})

    matchgraph_data: Dict[str, Any] = {"edges": edges_json}
    with io_handler.open_wt(os.path.join(output_path, "matchgraph.json")) as fjson:
        io.json_dump(matchgraph_data, fjson)

    plt.clf()
    cmap = _REPORT_SEQ_CMAP
    for (node1, node2), edge in sorted(connectivity.items(), key=lambda x: x[1]):
        if edge < min_matches:
            continue
        comp1 = shot_component[node1]
        comp2 = shot_component[node2]
        if comp1 != comp2:
            continue
        o1 = reconstructions[comp1].shots[node1].pose.get_origin()
        o2 = reconstructions[comp2].shots[node2].pose.get_origin()
        c = max(0, min(1.0, (float(edge) - lowest) / (highest - lowest)))
        plt.plot([o1[0], o2[0]], [o1[1], o2[1]], linestyle="-", color=cmap(c))

    for i, rec in enumerate(reconstructions):
        for shot in rec.shots.values():
            o = shot.pose.get_origin()
            c = i / len(reconstructions)
            plt.plot(o[0], o[1], linestyle="", marker="o", color=cmap(c))

    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    for b in ["top", "bottom", "left", "right"]:
        ax.spines[b].set_visible(False)

    norm = colors.Normalize(vmin=lowest, vmax=highest)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(
        sm,
        orientation="horizontal",
        label="Number of matches between images",
        pad=0.0,
        ax=ax,
    )

    with io_handler.open_wb(os.path.join(output_path, "matchgraph.png")) as fwb:
        plt.savefig(
            fwb,
            dpi=300,
            bbox_inches="tight",
        )


def save_residual_histogram(
    stats: Dict[str, Any],
    output_path: str,
    io_handler: io.IoFilesystemBase,
) -> None:
    backup = dict(mpl.rcParams)
    fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=(15, 3))

    h_norm, b_norm = stats["reconstruction_statistics"][
        "reprojection_histogram_normalized"
    ]
    n, _, p_norm = axs[0].hist(b_norm[:-1], b_norm, weights=h_norm)
    n = n.astype("int")
    seq_cmap = _REPORT_SEQ_CMAP
    for i in range(len(p_norm)):
        p_norm[i].set_facecolor(seq_cmap(n[i] / max(n)))

    h_pixel, b_pixel = stats["reconstruction_statistics"][
        "reprojection_histogram_pixels"
    ]
    n, _, p_pixel = axs[1].hist(b_pixel[:-1], b_pixel, weights=h_pixel)
    n = n.astype("int")
    for i in range(len(p_pixel)):
        p_pixel[i].set_facecolor(seq_cmap(n[i] / max(n)))

    h_angular, b_angular = stats["reconstruction_statistics"][
        "reprojection_histogram_angular"
    ]
    (
        n,
        _,
        p_angular,
    ) = axs[
        2
    ].hist(b_angular[:-1], b_angular, weights=h_angular)
    n = n.astype("int")
    for i in range(len(p_angular)):
        p_angular[i].set_facecolor(seq_cmap(n[i] / max(n)))

    axs[0].set_title("Normalized Residual")
    axs[1].set_title("Pixel Residual")
    axs[2].set_title("Angular Residual")

    with io_handler.open_wb(os.path.join(output_path, "residual_histogram.png")) as fwb:
        plt.savefig(
            fwb,
            dpi=300,
            bbox_inches="tight",
        )
    mpl.rcParams = backup


def save_topview(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    reconstructions: List[types.Reconstruction],
    output_path: str,
    io_handler: io.IoFilesystemBase,
) -> None:
    # limit splatting to 100K random points for efficiency
    max_points = 100000
    points = []
    colors = []
    for rec in reconstructions:
        # pick random subset of points if there are too many
        if len(rec.points) > max_points:
            sampled_ids = random.sample(list(rec.points.keys()), max_points)
        else:
            sampled_ids = list(rec.points.keys())
        for point_id in sampled_ids:
            point = rec.points[point_id]
            track = tracks_manager.get_track_observations(point.id)
            if len(track) < 2:
                continue
            coords = point.coordinates
            points.append(coords)

            r, g, b = [], [], []
            for obs in track.values():
                r.append(obs.color[0])
                g.append(obs.color[1])
                b.append(obs.color[2])
            colors.append(
                (statistics.median(r), statistics.median(g), statistics.median(b))
            )

    all_x = []
    all_y = []
    for rec in reconstructions:
        for shot in rec.shots.values():
            o = shot.pose.get_origin()
            all_x.append(o[0])
            all_y.append(o[1])
            if not shot.metadata.gps_position.has_value:
                continue
            gps = shot.metadata.gps_position.value
            all_x.append(gps[0])
            all_y.append(gps[1])

    # compute camera's XY bounding box
    low_x, high_x = np.min(all_x), np.max(all_x)
    low_y, high_y = np.min(all_y), np.max(all_y)

    # get its size
    size_x = high_x - low_x
    size_y = high_y - low_y

    # expand bounding box by some margin
    margin = 0.05
    low_x -= size_x * margin
    high_x += size_y * margin
    low_y -= size_x * margin
    high_y += size_y * margin

    # update size
    size_x = high_x - low_x
    size_y = high_y - low_y

    im_size_x = 2000
    im_size_y = int(im_size_x * size_y / size_x)
    topview = np.zeros((im_size_y, im_size_x, 3))

    # splat points using gaussian + max-pool
    splatting = 15
    size = 2 * splatting + 1
    kernel = _get_gaussian_kernel(splatting, 2)
    kernel /= kernel[splatting, splatting]
    for point, color in zip(points, colors):
        x, y = (
            int((point[0] - low_x) / size_x * im_size_x),
            int((point[1] - low_y) / size_y * im_size_y),
        )
        if not ((0 < x < (im_size_x - 1)) and (0 < y < (im_size_y - 1))):
            continue

        k_low_x, k_low_y = -min(x - splatting, 0), -min(y - splatting, 0)
        k_high_x, k_high_y = (
            size - max(x + splatting - (im_size_x - 2), 0),
            size - max(y + splatting - (im_size_y - 2), 0),
        )
        h_low_x, h_low_y = max(x - splatting, 0), max(y - splatting, 0)
        h_high_x, h_high_y = (
            min(x + splatting + 1, im_size_x - 1),
            min(y + splatting + 1, im_size_y - 1),
        )

        for i in range(3):
            current = topview[h_low_y:h_high_y, h_low_x:h_high_x, i]
            splat = kernel[k_low_y:k_high_y, k_low_x:k_high_x]
            topview[h_low_y:h_high_y, h_low_x:h_high_x, i] = np.maximum(
                splat * (color[i] / 255.0), current
            )

    # reverse X axis of the image so that it corresponds to the common map orientation (North up, East right)
    topview = np.flip(topview, axis=0)

    plt.clf()
    plt.imshow(topview)

    # display computed camera's XY
    linewidth = 1
    markersize = 4
    for rec in reconstructions:
        sorted_shots = sorted(
            rec.shots.values(), key=lambda x: x.metadata.capture_time.value
        )
        c_camera = _CLR_ACCENT
        c_gps = _CLR_BAD
        for j, shot in enumerate(sorted_shots):
            o = shot.pose.get_origin()
            x, y = (
                int((o[0] - low_x) / size_x * im_size_x),
                int((o[1] - low_y) / size_y * im_size_y),
            )
            y = im_size_y - y  # reverse Y axis to match common map orientation
            plt.plot(
                x,
                y,
                linestyle="",
                marker="o",
                color=c_camera,
                markersize=markersize,
                linewidth=1,
            )

            # also display camera path using capture time
            if j < len(sorted_shots) - 1:
                n = sorted_shots[j + 1].pose.get_origin()
                nx, ny = (
                    int((n[0] - low_x) / size_x * im_size_x),
                    int((n[1] - low_y) / size_y * im_size_y),
                )
                ny = im_size_y - ny  # reverse Y axis to match common map orientation
                plt.plot(
                    [x, nx], [y, ny], linestyle="-", color=c_camera, linewidth=linewidth
                )

            # display GPS error
            if not shot.metadata.gps_position.has_value:
                continue
            gps = shot.metadata.gps_position.value
            gps_x, gps_y = (
                int((gps[0] - low_x) / size_x * im_size_x),
                int((gps[1] - low_y) / size_y * im_size_y),
            )
            gps_y = im_size_y - gps_y  # reverse Y axis to match common map orientation
            plt.plot(
                gps_x,
                gps_y,
                linestyle="",
                marker="v",
                color=c_gps,
                markersize=markersize,
                linewidth=1,
            )
            plt.plot(
                [x, gps_x], [y, gps_y], linestyle="-", color=c_gps, linewidth=linewidth
            )

    plt.xticks(
        [0, im_size_x / 2, im_size_x],
        [0, f"{int(size_x / 2):.0f}", f"{size_x:.0f} meters"],
        fontsize="small",
    )
    plt.yticks(
        [im_size_y, im_size_y / 2, 0],
        [f"{size_y:.0f} meters", f"{int(size_y / 2):.0f}", 0],
        fontsize="small",
    )
    plt.gca().invert_yaxis()

    with io_handler.open_wb(os.path.join(output_path, "topview.png")) as fwb:
        plt.savefig(
            fwb,
            dpi=300,
            bbox_inches="tight",
        )


def save_heatmap(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    reconstructions: List[types.Reconstruction],
    output_path: str,
    io_handler: io.IoFilesystemBase,
) -> None:
    all_projections = {}

    splatting = 15
    size = 2 * splatting + 1
    kernel = _get_gaussian_kernel(splatting, 2)

    all_cameras = {}
    for rec in reconstructions:
        for camera in rec.cameras.values():
            all_projections[camera.id] = []
            all_cameras[camera.id] = camera

    for i in range(len(reconstructions)):
        valid_observations = _get_valid_observations(
            reconstructions, tracks_manager)(i)
        for shot_id, observations in valid_observations.items():
            shot = reconstructions[i].get_shot(shot_id)
            w = shot.camera.width
            h = shot.camera.height
            center = np.array([w / 2.0, h / 2.0])
            normalizer = max(shot.camera.width, shot.camera.height)

            buckets_x, buckets_y = _heatmap_buckets(shot.camera)
            w_bucket = buckets_x / w
            h_bucket = buckets_y / h

            shots_projections = []
            for observation in observations.values():
                bucket = observation.point * normalizer + center
                x = max([0, min([int(bucket[0] * w_bucket), buckets_x - 1])])
                y = max([0, min([int(bucket[1] * h_bucket), buckets_y - 1])])
                shots_projections.append((x, y))
            all_projections[shot.camera.id] += shots_projections

    for camera_id, projections in all_projections.items():
        buckets_x, buckets_y = _heatmap_buckets(rec.cameras[camera_id])
        camera_heatmap = np.zeros((buckets_y, buckets_x))
        for x, y in projections:
            k_low_x, k_low_y = -min(x - splatting, 0), -min(y - splatting, 0)
            k_high_x, k_high_y = (
                size - max(x + splatting - (buckets_x - 2), 0),
                size - max(y + splatting - (buckets_y - 2), 0),
            )
            h_low_x, h_low_y = max(x - splatting, 0), max(y - splatting, 0)
            h_high_x, h_high_y = (
                min(x + splatting + 1, buckets_x - 1),
                min(y + splatting + 1, buckets_y - 1),
            )
            camera_heatmap[h_low_y:h_high_y, h_low_x:h_high_x] += kernel[
                k_low_y:k_high_y, k_low_x:k_high_x
            ]

        highest = np.max(camera_heatmap)
        lowest = np.min(camera_heatmap)

        plt.clf()
        plt.imshow(
            (camera_heatmap - lowest) / (highest - lowest),
            cmap=_REPORT_SEQ_CMAP,
        )

        plt.title(
            f"Detected features heatmap for camera {camera_id}",
            fontsize="x-small",
        )

        camera = all_cameras[camera_id]
        w = camera.width
        h = camera.height

        plt.xticks(
            [0, buckets_x / 2, buckets_x],
            [0, int(w / 2), w],
            fontsize="x-small",
        )
        plt.yticks(
            [buckets_y, buckets_y / 2, 0],
            [0, int(h / 2), h],
            fontsize="x-small",
        )

        with io_handler.open_wb(
            os.path.join(
                output_path, "heatmap_" +
                str(camera_id.replace("/", "_")) + ".npy"
            )
        ) as fwb:
            np.save(fwb, camera_heatmap)
    with io_handler.open_wb(
        os.path.join(
            output_path, "heatmap_" + str(camera_id.replace("/", "_")) + ".png"
        )
    ) as fwb:
        plt.savefig(
            fwb,
            dpi=300,
            bbox_inches="tight",
        )


def save_residual_grids(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    reconstructions: List[types.Reconstruction],
    output_path: str,
    io_handler: io.IoFilesystemBase,
) -> None:
    all_errors = {}
    max_shots = 2000
    scaling = 4

    for rec in reconstructions:
        for camera_id in rec.cameras:
            all_errors[camera_id] = []

    for i in range(len(reconstructions)):
        shot_ids = list(reconstructions[i].shots.keys())
        ramdom_shot_ids = tuple(random.sample(
            shot_ids, min(max_shots, len(shot_ids))))

        valid_observations = _get_valid_observations(
            reconstructions, tracks_manager)(i, ramdom_shot_ids)
        errors_scaled = _compute_errors(reconstructions, tracks_manager)(
            i, pymap.ErrorType.Normalized, ramdom_shot_ids
        )
        errors_unscaled = _compute_errors(reconstructions, tracks_manager)(
            i, pymap.ErrorType.Pixel, ramdom_shot_ids
        )

        for shot_id, shot_errors in errors_scaled.items():
            shot = reconstructions[i].get_shot(shot_id)
            w = shot.camera.width
            h = shot.camera.height
            center = np.array([w / 2.0, h / 2.0])
            normalizer = max(shot.camera.width, shot.camera.height)

            buckets_x, buckets_y = _grid_buckets(shot.camera)
            w_bucket = buckets_x / w
            h_bucket = buckets_y / h

            shots_errors = []
            for error_scaled, error_unscaled, observation in zip(
                shot_errors.values(),
                errors_unscaled[shot_id].values(),
                valid_observations[shot_id].values(),
            ):
                if _norm2d(error_unscaled * normalizer) > RESIDUAL_PIXEL_CUTOFF:
                    continue

                bucket = observation.point * normalizer + center
                x = max([0, min([int(bucket[0] * w_bucket), buckets_x - 1])])
                y = max([0, min([int(bucket[1] * h_bucket), buckets_y - 1])])

                shots_errors.append((x, y, error_scaled))
            all_errors[shot.camera.id] += shots_errors

        del errors_scaled
        del errors_unscaled

    for camera_id, errors in all_errors.items():
        if not errors:
            continue
        buckets_x, buckets_y = _grid_buckets(rec.cameras[camera_id])
        camera_array_res = np.zeros((buckets_y, buckets_x, 2))
        camera_array_count = np.full((buckets_y, buckets_x, 1), 1)
        for x, y, e in errors:
            camera_array_res[y, x] += e
            camera_array_count[y, x, 0] += 1
        camera_array_res = np.divide(camera_array_res, camera_array_count)
        camera = rec.get_camera(camera_id)
        w, h = camera.width, camera.height
        normalizer = max(w, h)

        clamp = 0.15
        res_colors = np.linalg.norm(camera_array_res[:, :, :2], axis=2)
        lowest = np.percentile(res_colors, 100 * clamp)
        highest = np.percentile(res_colors, 100 * (1 - clamp))
        np.clip(res_colors, lowest, highest, res_colors)
        res_colors /= highest - lowest

        plt.clf()
        plt.figure(figsize=(12, 10))
        Q = plt.quiver(
            camera_array_res[:, :, 0] * scaling,
            camera_array_res[:, :, 1] * scaling,
            res_colors,
            units="xy",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.1,
            cmap=_REPORT_SEQ_CMAP_INV,
        )

        scale = highest - lowest
        plt.quiverkey(
            Q,
            X=0.1,
            Y=1.04,
            U=scale * scaling,
            label=f"Residual grid scale : {scale:.2f}",
            labelpos="E",
        )
        plt.title(
            "                      ",
            fontsize="large",
        )

        norm = colors.Normalize(vmin=lowest, vmax=highest)
        cmap = _REPORT_SEQ_CMAP_INV
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        plt.colorbar(
            mappable=sm,
            orientation="horizontal",
            label="Residual Norm",
            pad=0.08,
            aspect=40,
            ax=plt.gca(),
        )

        plt.xticks(
            [0, buckets_x / 2, buckets_x], [0, int(w / 2), w], fontsize="x-small"
        )
        plt.yticks(
            [0, buckets_y / 2, buckets_y], [0, int(h / 2), h], fontsize="x-small"
        )

        with io_handler.open_wb(
            os.path.join(
                output_path, "residuals_" +
                str(camera_id.replace("/", "_")) + ".npy"
            )
        ) as fwb:
            np.save(fwb, camera_array_res)

        with io_handler.open_wb(
            os.path.join(
                output_path, "residuals_" +
                str(camera_id.replace("/", "_")) + ".png"
            )
        ) as fwb:
            plt.savefig(
                fwb,
                dpi=300,
                bbox_inches="tight",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Overlap computation
# ─────────────────────────────────────────────────────────────────────────────


def _compute_ground_plane_z(
    reconstructions: List[types.Reconstruction],
) -> float:
    """Compute median Z (altitude) of all reconstruction 3D points."""
    zs = []
    for rec in reconstructions:
        for point in rec.points.values():
            zs.append(point.coordinates[2])
    if not zs:
        return 0.0
    return float(np.median(zs))


def _compute_shot_footprint(
    shot: pymap.Shot, ground_z: float
) -> Optional[NDArray]:
    """Compute the ground footprint of a shot by ray-plane intersection.

    Returns a (4, 2) array of XY world coordinates, or None if degenerate.
    """
    if pygeometry.Camera.is_panorama(shot.camera.projection_type):
        return None

    w = shot.camera.width
    h = shot.camera.height
    if w <= 0 or h <= 0:
        return None

    # Image corners in normalized coords
    size = max(w, h)
    corners_px = np.array([
        [0.0, 0.0],
        [w - 1.0, 0.0],
        [w - 1.0, h - 1.0],
        [0.0, h - 1.0],
    ])
    corners_norm = np.empty((4, 2))
    corners_norm[:, 0] = (corners_px[:, 0] + 0.5 - w / 2.0) / size
    corners_norm[:, 1] = (corners_px[:, 1] + 0.5 - h / 2.0) / size

    # Get bearing rays in camera frame
    bearings = shot.camera.pixel_bearing_many(corners_norm)  # (4, 3)

    # Transform to world frame
    R = shot.pose.get_rotation_matrix()
    origin = shot.pose.get_origin()

    # Ray-plane intersection: find t such that (origin + t * direction).z = ground_z
    world_bearings = (R.T @ bearings.T).T  # (4, 3) in world frame

    footprint = np.empty((4, 2))
    for i in range(4):
        dz = world_bearings[i, 2]
        if abs(dz) < 1e-9:
            return None  # ray parallel to ground
        t = (ground_z - origin[2]) / dz
        if t < 0:
            return None  # camera looking away from ground
        footprint[i, 0] = origin[0] + t * world_bearings[i, 0]
        footprint[i, 1] = origin[1] + t * world_bearings[i, 1]

    return footprint


def _directional_overlap_ratio(
    fp_a: NDArray, fp_b: NDArray, direction: NDArray
) -> float:
    """Compute 1D overlap ratio by projecting footprints onto a direction vector."""
    proj_a = fp_a @ direction
    proj_b = fp_b @ direction

    min_a, max_a = proj_a.min(), proj_a.max()
    min_b, max_b = proj_b.min(), proj_b.max()

    len_a = max_a - min_a
    len_b = max_b - min_b

    if len_a < 1e-9 or len_b < 1e-9:
        return 0.0

    inter_min = max(min_a, min_b)
    inter_max = min(max_a, max_b)

    inter_len = inter_max - inter_min
    if inter_len <= 0:
        return 0.0

    return float(inter_len / min(len_a, len_b))


def _compute_front_overlap(
    reconstructions: List[types.Reconstruction],
    ground_z: float,
    max_samples: int = 100,
) -> List[float]:
    """Compute overlap ratio between time-successive shot pairs.

    Randomly samples at most *max_samples* consecutive pairs to limit
    computation time while preserving the true front overlap distribution.
    """
    overlaps = []
    for rec in reconstructions:
        shots_with_time = [
            s for s in rec.shots.values()
            if s.metadata.capture_time.has_value
        ]
        if len(shots_with_time) < 2:
            continue
        sorted_shots = sorted(
            shots_with_time, key=lambda s: s.metadata.capture_time.value
        )
        n = len(sorted_shots)

        # Build list of valid consecutive pairs (both have footprints)
        footprints: Dict[int, NDArray] = {}
        for i, s in enumerate(sorted_shots):
            fp = _compute_shot_footprint(s, ground_z)
            if fp is not None:
                footprints[i] = fp

        positions = np.array(
            [sorted_shots[i].pose.get_origin()[:2] for i in range(n)]
        )

        valid_pairs: List[Tuple[int, int]] = []
        for i in range(n - 1):
            if i in footprints and (i + 1) in footprints:
                valid_pairs.append((i, i + 1))

        # Sample randomly if too many pairs
        if len(valid_pairs) > max_samples:
            valid_pairs = random.sample(valid_pairs, max_samples)

        for i, j in valid_pairs:
            d = positions[j] - positions[i]
            norm = math.sqrt(d[0]**2 + d[1]**2)
            if norm < 1e-9:
                continue
            direction = d / norm
            overlaps.append(_directional_overlap_ratio(
                footprints[i], footprints[j], direction))
    return overlaps


def _compute_side_overlap(
    reconstructions: List[types.Reconstruction],
    ground_z: float,
    alignment_threshold: float = 0.5,
    max_samples: int = 100,
) -> List[float]:
    """Compute overlap between lateral (cross-strip) shot pairs.

    For each shot, computes the local flight direction, then finds the nearest
    neighbor (by ground footprint centroid distance) whose direction is NOT
    aligned with the flight path (|dot| < alignment_threshold). Using footprint
    centroids ensures we find shots that look at nearby ground areas, even if
    their camera XY positions are far apart.

    Samples at most *max_samples* shots to limit computation time.
    """

    overlaps = []
    for rec in reconstructions:
        shots_with_time = [
            s for s in rec.shots.values()
            if s.metadata.capture_time.has_value
        ]
        if len(shots_with_time) < 4:
            continue
        sorted_shots = sorted(
            shots_with_time, key=lambda s: s.metadata.capture_time.value
        )
        n = len(sorted_shots)

        # Precompute footprints and their centroids (representative ground points)
        footprints: Dict[int, NDArray] = {}
        centroids = np.full((n, 2), np.nan)
        for i, s in enumerate(sorted_shots):
            fp = _compute_shot_footprint(s, ground_z)
            if fp is not None:
                footprints[i] = fp
                centroids[i] = fp.mean(axis=0)

        # Filter to shots with valid footprints
        valid_indices = [i for i in range(n) if i in footprints]
        if len(valid_indices) < 2:
            continue

        # Camera positions for flight direction (use camera XY, not ground)
        positions = np.array(
            [sorted_shots[i].pose.get_origin()[:2] for i in range(n)])

        # Compute local flight direction for each shot (central difference)
        flight_dirs = np.zeros((n, 2))
        for i in range(n):
            if i == 0:
                d = positions[1] - positions[0]
            elif i == n - 1:
                d = positions[n - 1] - positions[n - 2]
            else:
                d = positions[i + 1] - positions[i - 1]
            norm = math.sqrt(d[0]**2 + d[1]**2)
            if norm > 1e-9:
                flight_dirs[i] = d / norm

        # Build kd-tree on footprint centroids (representative ground points)
        valid_centroids = centroids[valid_indices]
        tree = spatial.cKDTree(valid_centroids)

        # Select indices to evaluate (sample randomly if too many)
        if len(valid_indices) > max_samples:
            eval_indices = random.sample(valid_indices, max_samples)
        else:
            eval_indices = valid_indices

        # For each sampled shot, find nearest neighbor perpendicular to flight direction
        k = min(30, len(valid_indices))
        for idx in eval_indices:
            fd = flight_dirs[idx]
            if fd[0] == 0 and fd[1] == 0:
                continue

            # Find position of idx in valid_indices for tree query
            centroid = centroids[idx]
            distances, tree_neighbors = tree.query(centroid, k=k)
            if isinstance(tree_neighbors, int):
                tree_neighbors = [tree_neighbors]
                distances = [distances]

            for d, tree_n in zip(distances, tree_neighbors):
                if tree_n >= len(valid_indices):
                    continue
                n_idx = valid_indices[tree_n]
                if n_idx == idx:
                    continue
                if d < 1e-9:
                    continue
                # Direction from current centroid to neighbor centroid
                to_neighbor = centroids[n_idx] - centroid
                to_neighbor_norm = math.sqrt(
                    to_neighbor[0]**2 + to_neighbor[1]**2)
                if to_neighbor_norm < 1e-9:
                    continue
                to_neighbor = to_neighbor / to_neighbor_norm
                # Check alignment: low |dot| means perpendicular to flight path
                alignment = abs(fd[0] * to_neighbor[0] +
                                fd[1] * to_neighbor[1])
                if alignment < alignment_threshold:
                    ortho_dir = np.array([-fd[1], fd[0]])
                    overlap = _directional_overlap_ratio(
                        footprints[idx], footprints[n_idx], ortho_dir
                    )
                    overlaps.append(overlap)
                    break  # take nearest perpendicular neighbor only
    return overlaps


def overlap_statistics(
    reconstructions: List[types.Reconstruction],
    tracks_manager: pymap.TracksManager,
) -> Dict[str, Any]:
    """Compute front/side overlap stats."""
    ground_z = _compute_ground_plane_z(reconstructions)
    front = _compute_front_overlap(reconstructions, ground_z)
    side = _compute_side_overlap(reconstructions, ground_z)
    stats: Dict[str, Any] = {"ground_z": ground_z}
    if front:
        stats["front_overlap_mean"] = float(np.mean(front)) * 100.0
        stats["front_overlap_median"] = float(np.median(front)) * 100.0
    else:
        stats["front_overlap_mean"] = 0.0
        stats["front_overlap_median"] = 0.0
    if side:
        stats["side_overlap_mean"] = float(np.mean(side)) * 100.0
        stats["side_overlap_median"] = float(np.median(side)) * 100.0
    else:
        stats["side_overlap_mean"] = 0.0
        stats["side_overlap_median"] = 0.0
    return stats


def save_overlap_map(
    reconstructions: List[types.Reconstruction],
    output_path: str,
    io_handler: io.IoFilesystemBase,
) -> None:
    """Rasterize camera footprints and save a color-coded overlap map PNG."""

    ground_z = _compute_ground_plane_z(reconstructions)

    # Collect all valid footprints
    footprints = []
    for rec in reconstructions:
        for shot in rec.shots.values():
            fp = _compute_shot_footprint(shot, ground_z)
            if fp is not None:
                footprints.append(fp)

    if not footprints:
        return

    # Compute world extent
    all_pts = np.vstack(footprints)
    min_x, min_y = all_pts[:, 0].min(), all_pts[:, 1].min()
    max_x, max_y = all_pts[:, 0].max(), all_pts[:, 1].max()
    extent_x = max_x - min_x
    extent_y = max_y - min_y
    if extent_x < 1e-6 or extent_y < 1e-6:
        return

    # Add margin
    margin = 0.05
    min_x -= extent_x * margin
    min_y -= extent_y * margin
    max_x += extent_x * margin
    max_y += extent_y * margin
    extent_x = max_x - min_x
    extent_y = max_y - min_y

    # Grid resolution: target ~1000px on longest side
    target_px = 1000
    if extent_x > extent_y:
        nx = target_px
        ny = max(1, int(target_px * extent_y / extent_x))
    else:
        ny = target_px
        nx = max(1, int(target_px * extent_x / extent_y))

    # Build grid of sample points
    xs = np.linspace(min_x, max_x, nx)
    ys = np.linspace(min_y, max_y, ny)
    xv, yv = np.meshgrid(xs, ys)
    grid_points = np.column_stack([xv.ravel(), yv.ravel()])  # (ny*nx, 2)

    # Count how many footprints cover each cell
    counts = np.zeros(ny * nx, dtype=np.int32)
    for fp in footprints:
        path = MplPath(fp)
        inside = path.contains_points(grid_points)
        counts += inside.astype(np.int32)

    counts_2d = counts.reshape(ny, nx)

    # Color map using the unified palette
    rgba = np.ones((ny, nx, 4), dtype=np.float32)  # white = no coverage
    # 1 view: light grey (insufficient)
    rgba[counts_2d == 1] = [0.78, 0.78, 0.78, 1.0]
    # 2 views: bad (muted red)
    rgba[counts_2d == 2] = [*_CLR_BAD, 1.0]
    # 3 views: average (warm amber)
    rgba[counts_2d == 3] = [*_CLR_AVG, 1.0]
    # 4 views: good (medium sea-green)
    rgba[counts_2d == 4] = [*_CLR_GOOD, 1.0]
    # 5+ views: accent (Mapillary green)
    rgba[counts_2d >= 5] = [*_CLR_ACCENT, 1.0]

    # Create figure with legend
    fig, ax = plt.subplots(1, 1, figsize=(10, 10 * ny / nx))
    ax.imshow(rgba, origin="lower", extent=[
              min_x, max_x, min_y, max_y], aspect="equal")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("Overlap Map")

    # Legend

    legend_elements = [
        Patch(facecolor=(0.78, 0.78, 0.78), label="1 view"),
        Patch(facecolor=_CLR_BAD, label="2 views"),
        Patch(facecolor=_CLR_AVG, label="3 views"),
        Patch(facecolor=_CLR_GOOD, label="4 views"),
        Patch(facecolor=_CLR_ACCENT, label="5+ views"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    with io_handler.open_wb(os.path.join(output_path, "overlap_map.png")) as fwb:
        plt.savefig(fwb, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Hard-coded longest side, in pixels, for the DSM/ortho report thumbnails.
THUMBNAIL_MAX_SIZE = 1000


def _save_thumbnail_jpeg(
    io_handler: io.IoFilesystemBase, path: str, bgr: NDArray
) -> None:
    """Encode a BGR uint8 image as JPEG and write it through ``io_handler``."""
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError(f"Failed to JPEG-encode thumbnail: {path}")
    with io_handler.open_wb(path) as fwb:
        fwb.write(buf.tobytes())


def save_dsm_thumbnail(
    data: DataSet,
    output_path: str,
    io_handler: io.IoFilesystemBase,
    max_size: int = THUMBNAIL_MAX_SIZE,
) -> bool:
    """Render a colourised JPEG thumbnail of the dense DSM for the report.

    Reads ``undistorted/depthmaps/dsm.tif`` (decimated to ``max_size``), maps the
    elevation range through the ``terrain`` colormap and writes ``dsm_thumb.jpg``
    into ``output_path``, leaving no-data cells white.  Returns False (and writes
    nothing) when the DSM has not been produced.
    """
    dsm_path = data.undistorted_dataset().dsm_file()
    if not io_handler.isfile(dsm_path):
        return False

    grid, nodata = geo.read_geotiff_downsampled(dsm_path, max_size)
    grid = grid.astype(np.float32)
    valid = np.isfinite(grid)
    if nodata is not None:
        valid &= grid != nodata
    if not valid.any():
        return False

    lo, hi = np.percentile(grid[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((grid - lo) / (hi - lo), 0.0, 1.0)
    rgb = (mpl.colormaps["terrain"](norm)[:, :, :3] * 255.0).astype(np.uint8)
    rgb[~valid] = 255  # white for no-data cells

    _save_thumbnail_jpeg(
        io_handler,
        os.path.join(output_path, "dsm_thumb.jpg"),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    )
    return True


def save_ortho_thumbnail(
    data: DataSet,
    output_path: str,
    io_handler: io.IoFilesystemBase,
    max_size: int = THUMBNAIL_MAX_SIZE,
) -> bool:
    """Render a JPEG thumbnail of the dense orthophoto for the report.

    Reads ``undistorted/depthmaps/ortho.tif`` (decimated to ``max_size``),
    composites any alpha band over white (JPEG has no transparency) and writes
    ``ortho_thumb.jpg`` into ``output_path``.  Returns False (and writes nothing)
    when the orthophoto has not been produced.
    """
    ortho_path = data.undistorted_dataset().ortho_file()
    if not io_handler.isfile(ortho_path):
        return False

    arr, _ = geo.read_geotiff_downsampled(ortho_path, max_size)
    if arr.ndim == 2:  # grayscale fallback
        rgb = np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)
    else:
        rgb = arr[:, :, :3].astype(np.float32)
        if arr.shape[2] >= 4:  # composite RGB over white using the alpha band
            alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
            rgb = rgb * alpha + 255.0 * (1.0 - alpha)
        rgb = rgb.astype(np.uint8)

    _save_thumbnail_jpeg(
        io_handler,
        os.path.join(output_path, "ortho_thumb.jpg"),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    )
    return True


def decimate_points(
    reconstructions: List[types.Reconstruction], max_num_points: int
) -> None:
    """
    Destructively decimate the points in a reconstruction
    if they exceed max_num_points by removing points
    at random
    """
    for rec in reconstructions:
        if len(rec.points) > max_num_points:
            all_points = rec.points
            random_ids = list(all_points.keys())
            random.shuffle(random_ids)
            random_ids = set(random_ids[: len(all_points) - max_num_points])

            for point_id in random_ids:
                rec.remove_point(point_id)


def start_end_capture_time(reconstructions):
    end_ct = float('-inf')
    start_ct = float('inf')

    for reconstruction in reconstructions:
        for shot in reconstruction.shots.values():
            if shot.metadata.capture_time.has_value:
                v = shot.metadata.capture_time.value
                if v > end_ct:
                    end_ct = v
                if v < start_ct:
                    start_ct = v

    if end_ct != float('-inf') and start_ct != float('inf'):
        return (start_ct, end_ct)
    else:
        return (None, None)
