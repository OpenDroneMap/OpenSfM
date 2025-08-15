import logging
import os
from opensfm import types

import numpy as np
import pyproj
from opensfm import io
from opensfm.dataset import DataSet, UndistortedDataSet
from opensfm.geo import TopocentricConverter
from opensfm.reconstruction import bundle_shot_poses
from typing import List, Sequence

logger: logging.Logger = logging.getLogger(__name__)


def run_dataset(
    data: DataSet,
    proj: str,
    transformation: bool,
    image_positions: bool,
    reconstruction: bool,
    dense : bool,
    output: str,
    offset = (0, 0),
    mode = "affine"
) -> None:
    """Export reconstructions in geographic coordinates

    Args:
        proj: PROJ.4 projection string
        transformation : print cooordinate transformation matrix'
        image_positions : export image positions
        reconstruction : export reconstruction.json
        dense : export dense point cloud (depthmaps/merged.ply)
        output : path of the output file relative to the dataset
        offset : offset to substract from the translation (optional)
        mode : Method of georeferencing (optional)
    """

    if not (transformation or image_positions or reconstruction or dense):
        logger.info("Nothing to do. At least on of the options: ")
        logger.info(" --transformation, --image-positions, --reconstruction, --dense")

    reference = data.load_reference()

    projection = pyproj.Proj(proj)
    t = _get_transformation(reference, projection, offset)

    if transformation:
        output = output or "geocoords_transformation.txt"
        output_path = os.path.join(data.data_path, output)
        _write_transformation(t, output_path)

    if image_positions:
        reconstructions = data.load_reconstruction()
        output = output or "image_geocoords.tsv"
        output_path = os.path.join(data.data_path, output)
        _transform_image_positions(reconstructions, t, output_path)

    if reconstruction:
        reconstructions = data.load_reconstruction()
        if mode == "affine":
            for r in reconstructions:
                _transform_reconstruction(r, t)
        elif mode == "projected":
            for r in reconstructions:
                _transform_reconstruction_projected(r, t, offset[0], offset[1], reference, projection, data)
        else:
            raise Exception(f"Invalid mode: {mode}")
        
        output = output or "reconstruction.geocoords.json"
        data.save_reconstruction(reconstructions, output)

    if dense:
        output = output or "undistorted/depthmaps/merged.geocoords.ply"
        output_path = os.path.join(data.data_path, output)
        udata = data.undistorted_dataset()
        _transform_dense_point_cloud(udata, t, output_path)


def _get_transformation(reference: TopocentricConverter, projection: pyproj.Proj, offset) -> np.ndarray:
    """Get the linear transform from reconstruction coords to geocoords."""
    p = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    q = [_transform(point, reference, projection) for point in p]

    transformation = np.array(
        [
            [q[0][0] - q[3][0], q[1][0] - q[3][0], q[2][0] - q[3][0], q[3][0] - offset[0]],
            [q[0][1] - q[3][1], q[1][1] - q[3][1], q[2][1] - q[3][1], q[3][1] - offset[1]],
            [q[0][2] - q[3][2], q[1][2] - q[3][2], q[2][2] - q[3][2], q[3][2]],
            [0, 0, 0, 1],
        ]
    )
    return transformation


def _write_transformation(transformation: np.ndarray, filename: str) -> None:
    """Write the 4x4 matrix transformation to a text file."""
    with io.open_wt(filename) as fout:
        for row in transformation:
            fout.write(u" ".join(map(str, row)))
            fout.write(u"\n")


def _transform(point: Sequence, reference: TopocentricConverter, projection: pyproj.Proj) -> List[float]:
    """Transform on point from local coords to a proj4 projection."""
    lat, lon, altitude = reference.to_lla(point[0], point[1], point[2])
    easting, northing = projection(lon, lat)
    return [easting, northing, altitude]


def _transform_image_positions(
    reconstructions: List[types.Reconstruction], transformation: np.ndarray, output: str
) -> None:
    A, b = transformation[:3, :3], transformation[:3, 3]

    rows = ["Image\tX\tY\tZ"]
    for r in reconstructions:
        for shot in r.shots.values():
            o = shot.pose.get_origin()
            to = np.dot(A, o) + b
            row = [shot.id, to[0], to[1], to[2]]
            rows.append("\t".join(map(str, row)))

    text = "\n".join(rows + [""])
    with open(output, "w") as fout:
        fout.write(text)


def _transform_reconstruction(
    reconstruction: types.Reconstruction, transformation: np.ndarray
) -> None:
    """Apply a transformation to a reconstruction in-place."""
    A, b = transformation[:3, :3], transformation[:3, 3]
    A1 = np.linalg.inv(A)

    for shot in reconstruction.shots.values():
        R = shot.pose.get_rotation_matrix()
        shot.pose.set_rotation_matrix(np.dot(R, A1))
        shot.pose.set_origin(np.dot(A, shot.pose.get_origin()) + b)

    for point in reconstruction.points.values():
        point.coordinates = list(np.dot(A, point.coordinates) + b)

def _transform_points_projected(pts, offset_x, offset_y, reference, projection):
    lat, lon, alt = reference.to_lla(pts[:,0], pts[:,1], pts[:,2])
    easting, northing = projection(lon, lat)
    return easting - offset_x, northing - offset_y, alt

def _transform_reconstruction_projected(
    reconstruction: types.Reconstruction, transformation: np.ndarray, offset_x, offset_y, reference, projection, data
) -> None:
    """Apply a transformation to a reconstruction in-place by projection."""
    
    # Points
    pts =  np.array([p.coordinates for p in reconstruction.points.values()])
    easting, northing, alt = _transform_points_projected(pts, offset_x, offset_y, reference, projection)

    for i, point in enumerate(reconstruction.points.values()):
        point.coordinates = [easting[i], northing[i], alt[i]]

    # Cameras
    A, b = transformation[:3, :3], transformation[:3, 3]
    A1 = np.linalg.inv(A)

    pts = np.array([shot.pose.get_origin() for shot in reconstruction.shots.values()])
    easting, northing, alt = _transform_points_projected(pts, offset_x, offset_y, reference, projection)
    
    for i, shot in enumerate(reconstruction.shots.values()):
        R = shot.pose.get_rotation_matrix()
        shot.pose.set_rotation_matrix(np.dot(R, A1))
        shot.pose.set_origin([easting[i], northing[i], alt[i]])
    
    logger.info("Bundle shot poses")
    camera_priors = data.load_camera_models()
    rig_camera_priors = data.load_rig_cameras()
    bundle_shot_poses(reconstruction, set(reconstruction.shots.keys()), camera_priors, rig_camera_priors, data.config)
    

def _transform_dense_point_cloud(
    udata: UndistortedDataSet, transformation: np.ndarray, output_path: str
) -> None:
    """Apply a transformation to the merged point cloud."""
    A, b = transformation[:3, :3], transformation[:3, 3]
    input_path = udata.point_cloud_file()
    with io.open_rt(input_path) as fin:
        with io.open_wt(output_path) as fout:
            for i, line in enumerate(fin):
                if i < 13:
                    fout.write(line)
                else:
                    x, y, z, nx, ny, nz, red, green, blue = line.split()
                    x, y, z = np.dot(A, map(float, [x, y, z])) + b
                    nx, ny, nz = np.dot(A, map(float, [nx, ny, nz]))
                    fout.write(
                        "{} {} {} {} {} {} {} {} {}\n".format(
                            x, y, z, nx, ny, nz, red, green, blue
                        )
                    )
