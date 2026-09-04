# pyre-strict
import logging
import os
from typing import List, Sequence

import numpy as np
import pyproj
from numpy.typing import NDArray
from opensfm import io, types
from opensfm import geo
from opensfm.dataset import DataSet, UndistortedDataSet
from opensfm.reconstruction import bundle_shot_poses

logger: logging.Logger = logging.getLogger(__name__)


def run_dataset(
    data: DataSet,
    proj: str,
    transformation: bool,
    image_positions: bool,
    reconstruction: bool,
    dense: bool,
    output: str,
    offset=(0, 0),
    mode="affine"
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
        mode : Method of georeferencing (optional) [affine|projected]

    """

    if not (transformation or image_positions or reconstruction or dense):
        logger.info("Nothing to do. At least on of the options: ")
        logger.info(" --transformation, --image-positions, --reconstruction, --dense")

    reference = data.load_reference()

    projection = geo.construct_proj_transformer(proj, inverse=True)
    t = geo.get_proj_transform_matrix(reference, projection)
    t[0, 3] -= offset[0]
    t[1, 3] -= offset[1]

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
                geo.transform_reconstruction_with_matrix(r, t)
        elif mode == "projected":
            camera_priors = data.load_camera_models()
            rig_camera_priors = data.load_rig_cameras()
            for r in reconstructions:
                geo.transform_reconstruction_with_proj(r, projection)
                if offset != (0, 0):
                    for point in r.points.values():
                        point.coordinates[0] -= offset[0]
                        point.coordinates[1] -= offset[1]
                    for rig_instance in r.rig_instances.values():
                        o = rig_instance.pose.get_origin()
                        rig_instance.pose.set_origin(
                            [o[0] - offset[0], o[1] - offset[1], o[2]])
                
                logger.info("Bundle shot poses")
                bundle_shot_poses(r, set(r.shots.keys()), camera_priors, rig_camera_priors, data.config)
        else:
            raise Exception(f"Invalid mode: {mode}")
        
        output = output or "reconstruction.geocoords.json"
        data.save_reconstruction(reconstructions, output)

    if dense:
        output = output or "undistorted/depthmaps/merged.geocoords.ply"
        output_path = os.path.join(data.data_path, output)
        udata = data.undistorted_dataset()
        _transform_dense_point_cloud(udata, t, output_path)


def _write_transformation(transformation: NDArray, filename: str) -> None:
    """Write the 4x4 matrix transformation to a text file."""
    with io.open_wt(filename) as fout:
        for row in transformation:
            fout.write(" ".join(map(str, row)))
            fout.write("\n")


def _transform_image_positions(
    reconstructions: List[types.Reconstruction], transformation: NDArray, output: str
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


def _transform_dense_point_cloud(
    udata: UndistortedDataSet, transformation: NDArray, output_path: str
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
                    # pyre-fixme[6]: For 2nd argument expected `Union[Sequence[Sequen...
                    x, y, z = np.dot(A, map(float, [x, y, z])) + b
                    # pyre-fixme[6]: For 2nd argument expected `Union[Sequence[Sequen...
                    nx, ny, nz = np.dot(A, map(float, [nx, ny, nz]))
                    fout.write(
                        "{} {} {} {} {} {} {} {} {}\n".format(
                            x, y, z, nx, ny, nz, red, green, blue
                        )
                    )
