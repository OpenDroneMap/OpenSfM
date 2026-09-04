# pyre-strict
import gzip
import hashlib
import json
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from opensfm import config, features, geo, io, masking, pygeometry, pymap, rig, types, pysfm
from opensfm.dataset_base import DataSetBase
from PIL.PngImagePlugin import PngImageFile
from lz4 import frame as lz4_frame

logger: logging.Logger = logging.getLogger(__name__)


class DataSet(DataSetBase):
    """Accessors to the main input and output data.

    Data include input images, masks, and segmentation as well
    temporary data such as features and matches and the final
    reconstructions.

    All data is stored inside a single folder with a specific subfolder
    structure.

    It is possible to store data remotely or in different formats
    by subclassing this class and overloading its methods.
    """

    io_handler: io.IoFilesystemBase = io.IoFilesystemDefault()
    config: Dict[str, Any] = {}
    image_files: Dict[str, str] = {}
    mask_files: Dict[str, str] = {}
    image_list: List[str] = []

    def __init__(
        self, data_path: str, io_handler: Optional[io.IoFilesystemBase] = None
    ) -> None:
        """Init dataset associated to a folder."""
        if io_handler is not None:
            self.io_handler = io_handler
        self.data_path = data_path
        self.load_config()
        self.load_image_list()
        self.load_mask_list()

    def _config_file(self) -> str:
        return os.path.join(self.data_path, "config.yaml")

    def load_config(self) -> None:
        config_file_path = self._config_file()
        if self.io_handler.isfile(config_file_path):
            with self.io_handler.open_rt(config_file_path) as f:
                self.config = config.load_config_from_fileobject(f)
        else:
            self.config = config.default_config()

    def _image_list_file(self) -> str:
        return os.path.join(self.data_path, "image_list.txt")

    def load_image_list(self) -> None:
        """Load image list from image_list.txt or list images/ folder."""
        image_list_file = self._image_list_file()
        image_list_path = os.path.join(self.data_path, "images")

        if self.io_handler.isfile(image_list_file):
            with self.io_handler.open_rt(image_list_file) as fin:
                lines = fin.read().splitlines()
            self._set_image_list(lines)
        else:
            self._set_image_path(image_list_path)

        if self.data_path and not self.image_list:
            raise IOError("No Images found in {}".format(image_list_path))

    def images(self) -> List[str]:
        """List of file names of all images in the dataset."""
        return self.image_list

    def _image_file(self, image: str) -> str:
        """Path to the image file."""
        return self.image_files[image]

    def open_image_file(self, image: str) -> BinaryIO:
        """Open image file and return file object."""
        return self.io_handler.open_rb(self._image_file(image))

    def load_image(
        self,
        image: str,
        unchanged: bool = False,
        anydepth: bool = False,
        grayscale: bool = False,
    ) -> NDArray:
        """Load image pixels as numpy array.

        The array is 3D, indexed by y-coord, x-coord, channel.
        The channels are in RGB order.
        """
        return self.io_handler.imread(
            self._image_file(image),
            unchanged=unchanged,
            anydepth=anydepth,
            grayscale=grayscale,
        )

    def image_size(self, image: str) -> Tuple[int, int]:
        """Height and width of the image."""
        return self.io_handler.image_size(self._image_file(image))

    def load_mask_list(self) -> None:
        """Load mask list from mask_list.txt or list masks/ folder."""
        mask_list_file = os.path.join(self.data_path, "mask_list.txt")
        if self.io_handler.isfile(mask_list_file):
            with self.io_handler.open_rt(mask_list_file) as fin:
                lines = fin.read().splitlines()
            self._set_mask_list(lines)
        else:
            self._set_mask_path(os.path.join(self.data_path, "masks"))

    def load_mask(self, image: str) -> Optional[NDArray]:
        """Load image mask if it exists, otherwise return None."""
        if image in self.mask_files:
            mask_path = self.mask_files[image]
            mask = self.io_handler.imread(mask_path, grayscale=True)
            if mask is None:
                raise IOError(
                    "Unable to load mask for image {} " "from file {}".format(
                        image, mask_path
                    )
                )
        else:
            mask = None
        return mask

    def _instances_path(self) -> str:
        return os.path.join(self.data_path, "instances")

    def _instances_file(self, image: str) -> str:
        return os.path.join(self._instances_path(), image + ".png")

    def load_instances(self, image: str) -> Optional[NDArray]:
        """Load image instances file if it exists, otherwise return None."""
        instances_file = self._instances_file(image)
        if self.io_handler.isfile(instances_file):
            instances = self.io_handler.imread(instances_file, grayscale=True)
        else:
            instances = None
        return instances

    def _segmentation_path(self) -> str:
        return os.path.join(self.data_path, "segmentations")

    def _segmentation_file(self, image: str) -> str:
        return os.path.join(self._segmentation_path(), image + ".png")

    def segmentation_labels(self) -> List[Dict[str, Any]]:
        return []

    def load_segmentation(self, image: str) -> Optional[NDArray]:
        """Load image segmentation if it exists, otherwise return None."""
        segmentation_file = self._segmentation_file(image)
        if self.io_handler.isfile(segmentation_file):
            with self.io_handler.open_rb(segmentation_file) as fp:
                with PngImageFile(fp) as png_image:
                    # TODO: We do not write a header tag in the metadata. Might be good safety check.
                    data = np.array(png_image)
                    if data.ndim == 2:
                        return data
                    elif data.ndim == 3:
                        return data[:, :, 0]

                        # TODO we can optionally return also the instances and scores:
                        # instances = (
                        #     data[:, :, 1].astype(np.int16) + data[:, :, 2].astype(np.int16) * 256
                        # )
                        # scores = data[:, :, 3].astype(np.float32) / 256.0
                    else:
                        raise IndexError
        else:
            segmentation = None
        return segmentation

    def segmentation_ignore_values(self, image: str) -> List[int]:
        """List of label values to ignore.

        Pixels with these label values will be masked out and won't be
        processed when extracting and matching features.
        """
        return self.config.get("segmentation_ignore_values", [])

    def undistorted_segmentation_ignore_values(self, image: str) -> List[int]:
        """List of label values to ignore on undistorted images

        Pixels with these label values will be masked out and won't be
        processed when computing depthmaps.
        """
        return self.config.get(
            "undistorted_segmentation_ignore_values",
            self.segmentation_ignore_values(image),
        )

    def _is_image_file(self, filename: str) -> bool:
        extensions = {"jpg", "jpeg", "png", "tif", "tiff", "pgm", "pnm", "gif"}
        return filename.split(".")[-1].lower() in extensions

    def _set_image_path(self, path: str) -> None:
        """Set image path and find all images in there"""
        self.image_list = []
        self.image_files = {}
        if self.io_handler.exists(path):
            for name in self.io_handler.ls(path):
                if self._is_image_file(name):
                    self.image_list.append(name)
                    self.image_files[name] = os.path.join(path, name)

    def _set_image_list(self, image_list: List[str]) -> None:
        self.image_list = []
        self.image_files = {}
        for line in image_list:
            path = os.path.join(self.data_path, line)
            name = os.path.basename(path)
            self.image_list.append(name)
            self.image_files[name] = path

    def _set_mask_path(self, path: str) -> None:
        """Set mask path and find all masks in there"""
        self.mask_files = {}
        if self.io_handler.isdir(path):
            files = set(self.io_handler.ls(path))
            for image in self.images():
                mask = image + ".png"
                if mask in files:
                    self.mask_files[image] = os.path.join(path, mask)

    def _set_mask_list(self, mask_list_lines: List[str]) -> None:
        self.mask_files = {}
        for line in mask_list_lines:
            image, relpath = line.split(None, 1)
            path = os.path.join(self.data_path, relpath.strip())
            self.mask_files[image.strip()] = path

    def _exif_path(self) -> str:
        """Return path of extracted exif directory"""
        return os.path.join(self.data_path, "exif")

    def _exif_file(self, image: str) -> str:
        """
        Return path of exif information for given image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self._exif_path(), image + ".exif")

    def load_exif(self, image: str) -> Dict[str, Any]:
        """Load pre-extracted image exif metadata."""
        with self.io_handler.open_rt(self._exif_file(image)) as fin:
            return json.load(fin)

    def save_exif(self, image: str, data: Dict[str, Any]) -> None:
        self.io_handler.mkdir_p(self._exif_path())
        with self.io_handler.open_wt(self._exif_file(image)) as fout:
            io.json_dump(data, fout)

    def exif_exists(self, image: str) -> bool:
        return self.io_handler.isfile(self._exif_file(image))

    def feature_type(self) -> str:
        """Return the type of local features (e.g. AKAZE, SURF, SIFT)"""
        feature_name = self.config["feature_type"].lower()
        if self.config["feature_root"]:
            feature_name = "root_" + feature_name
        return feature_name

    def _feature_path(self) -> str:
        """Return path of feature descriptors and FLANN indices directory"""
        return os.path.join(self.data_path, "features")

    def _feature_file(self, image: str) -> str:
        """
        Return path of feature file for specified image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self._feature_path(), image + ".features.npz")

    def _feature_file_legacy(self, image: str) -> str:
        """
        Return path of a legacy feature file for specified image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self._feature_path(), image + ".npz")

    def _save_features(
        self, filepath: str, features_data: features.FeaturesData
    ) -> None:
        self.io_handler.mkdir_p(self._feature_path())
        with self.io_handler.open_wb(filepath) as fwb:
            features_data.save(fwb, self.config)

    def features_exist(self, image: str) -> bool:
        return self.io_handler.isfile(
            self._feature_file(image)
        ) or self.io_handler.isfile(self._feature_file_legacy(image))

    def load_features(self, image: str) -> Optional[features.FeaturesData]:
        features_filepath = (
            self._feature_file_legacy(image)
            if self.io_handler.isfile(self._feature_file_legacy(image))
            else self._feature_file(image)
        )
        with self.io_handler.open_rb(features_filepath) as f:
            return features.FeaturesData.from_file(f, self.config)

    def save_features(self, image: str, features_data: features.FeaturesData) -> None:
        self._save_features(self._feature_file(image), features_data)

    def _words_file(self, image: str) -> str:
        return os.path.join(self._feature_path(), image + ".words.npz")

    def words_exist(self, image: str) -> bool:
        return self.io_handler.isfile(self._words_file(image))

    def load_words(self, image: str) -> NDArray:
        with self.io_handler.open_rb(self._words_file(image)) as f:
            s = np.load(f)
            return s["words"].astype(np.int32)

    def save_words(self, image: str, words: NDArray) -> None:
        with self.io_handler.open_wb(self._words_file(image)) as f:
            np.savez_compressed(f, words=words.astype(np.uint16))

    def _matches_path(self) -> str:
        """Return path of matches directory"""
        return os.path.join(self.data_path, "matches")

    def _matches_file(self, image: str) -> str:
        """File for matches for an image"""
        return os.path.join(self._matches_path(), "{}_matches.pkl.gz".format(image))

    def matches_exists(self, image: str) -> bool:
        return self.io_handler.isfile(self._matches_file(image))

    def load_matches(self, image: str) -> Dict[str, NDArray]:
        # Prevent pickling of anything except what we strictly need
        # as 'pickle.load' is RCE-prone. Will raise on any class other
        # than the numpy ones we allow.
        class MatchingUnpickler(pickle.Unpickler):
            # Handle both numpy <2.0 (np.core) and numpy >=2.0 (np._core)
            _multiarray = (
                np.core.multiarray if hasattr(
                    np, "core") else np._core.multiarray
            )
            modules_map = {
                "numpy.core.multiarray._reconstruct": _multiarray,
                "numpy.core.multiarray.scalar": _multiarray,
                "numpy._core.multiarray._reconstruct": _multiarray,
                "numpy._core.multiarray.scalar": _multiarray,
                "numpy.ndarray": np,
                "numpy.dtype": np,
            }

            def find_class(self, module, name):
                classname = f"{module}.{name}"
                allowed_module = classname in self.modules_map
                if not allowed_module:
                    raise pickle.UnpicklingError(
                        "global '%s.%s' is forbidden" % (module, name)
                    )
                return getattr(self.modules_map[classname], name)

        with self.io_handler.open_rb(self._matches_file(image)) as fin:
            matches = MatchingUnpickler(
                BytesIO(gzip.decompress(fin.read()))).load()
        return matches

    def save_matches(self, image: str, matches: Dict[str, NDArray]) -> None:
        self.io_handler.mkdir_p(self._matches_path())

        with BytesIO() as buffer:
            with gzip.GzipFile(fileobj=buffer, mode="w") as fzip:
                pickle.dump(matches, fzip)
            with self.io_handler.open_wb(self._matches_file(image)) as fw:
                fw.write(buffer.getvalue())

    def find_matches(self, im1: str, im2: str) -> NDArray:
        if self.matches_exists(im1):
            im1_matches = self.load_matches(im1)
            if im2 in im1_matches:
                return im1_matches[im2]
        if self.matches_exists(im2):
            im2_matches = self.load_matches(im2)
            if im1 in im2_matches:
                if len(im2_matches[im1]):
                    return im2_matches[im1][:, [1, 0]]
        return np.array([])

    def _tracks_manager_file(self, filename: Optional[str] = None) -> str:
        """Return path of tracks file"""
        return os.path.join(self.data_path, filename or "tracks.csv")

    def load_tracks_manager(
        self, filename: Optional[str] = None
    ) -> pymap.TracksManager:
        """Return the tracks manager"""
        path = self._tracks_manager_file(filename)
        if isinstance(self.io_handler, io.IoFilesystemDefault):
            return pymap.TracksManager.instanciate_from_file(path)
        with self.io_handler.open_rt(path) as f:
            return pymap.TracksManager.instanciate_from_string(f.read())

    def tracks_exists(self, filename: Optional[str] = None) -> bool:
        return self.io_handler.isfile(self._tracks_manager_file(filename))

    def save_tracks_manager(
        self, tracks_manager: pymap.TracksManager, filename: Optional[str] = None
    ) -> None:
        path = self._tracks_manager_file(filename)
        if isinstance(self.io_handler, io.IoFilesystemDefault):
            tracks_manager.write_to_file(path)
            return
        with self.io_handler.open_wt(path) as fw:
            fw.write(tracks_manager.as_string())

    def _reconstruction_file(self, filename: Optional[str]) -> str:
        """Return path of reconstruction file"""
        return os.path.join(self.data_path, filename or "reconstruction.json")

    def reconstruction_exists(self, filename: Optional[str] = None) -> bool:
        return self.io_handler.isfile(self._reconstruction_file(filename))

    def load_reconstruction(
        self, filename: Optional[str] = None
    ) -> List[types.Reconstruction]:
        with self.io_handler.open_rt(self._reconstruction_file(filename)) as fin:
            reconstructions = io.reconstructions_from_json(io.json_load(fin))
        return reconstructions

    def save_reconstruction(
        self,
        reconstruction: List[types.Reconstruction],
        filename: Optional[str] = None,
        minify: bool = False,
    ) -> None:
        with self.io_handler.open_wt(self._reconstruction_file(filename)) as fout:
            io.json_dump(io.reconstructions_to_json(
                reconstruction), fout, minify)

    def _reference_lla_path(self) -> str:
        return os.path.join(self.data_path, "reference_lla.json")

    def init_reference(self, images: Optional[List[str]] = None) -> None:
        """Initializes the dataset reference if not done already."""
        if not self.reference_exists():
            reference = invent_reference_from_gps_and_gcp(self, images)
            self.save_reference(reference)

    def save_reference(self, reference: geo.TopocentricConverter) -> None:
        reference_lla = {
            "latitude": reference.lat,
            "longitude": reference.lon,
            "altitude": reference.alt,
        }

        with self.io_handler.open_wt(self._reference_lla_path()) as fout:
            io.json_dump(reference_lla, fout)

    def load_reference(self) -> geo.TopocentricConverter:
        """Load reference as a topocentric converter."""
        with self.io_handler.open_rt(self._reference_lla_path()) as fin:
            lla = io.json_load(fin)

        return geo.TopocentricConverter(
            lla["latitude"], lla["longitude"], lla["altitude"]
        )

    def reference_exists(self) -> bool:
        return self.io_handler.isfile(self._reference_lla_path())

    def _camera_models_file(self) -> str:
        """Return path of camera model file"""
        return os.path.join(self.data_path, "camera_models.json")

    def load_camera_models(self) -> Dict[str, pygeometry.Camera]:
        """Return camera models data"""
        with self.io_handler.open_rt(self._camera_models_file()) as fin:
            obj = json.load(fin)
            return io.cameras_from_json(obj)

    def save_camera_models(self, camera_models: Dict[str, pygeometry.Camera]) -> None:
        """Save camera models data"""
        with self.io_handler.open_wt(self._camera_models_file()) as fout:
            obj = io.cameras_to_json(camera_models)
            io.json_dump(obj, fout)

    def _camera_models_overrides_file(self) -> str:
        """Path to the camera model overrides file."""
        return os.path.join(self.data_path, "camera_models_overrides.json")

    def camera_models_overrides_exists(self) -> bool:
        """Check if camera overrides file exists."""
        return self.io_handler.isfile(self._camera_models_overrides_file())

    def load_camera_models_overrides(self) -> Dict[str, pygeometry.Camera]:
        """Load camera models overrides data."""
        with self.io_handler.open_rt(self._camera_models_overrides_file()) as fin:
            obj = json.load(fin)
            return io.cameras_from_json(obj)

    def save_camera_models_overrides(
        self, camera_models: Dict[str, pygeometry.Camera]
    ) -> None:
        """Save camera models overrides data"""
        with self.io_handler.open_wt(self._camera_models_overrides_file()) as fout:
            obj = io.cameras_to_json(camera_models)
            io.json_dump(obj, fout)

    def _exif_overrides_file(self) -> str:
        """Path to the EXIF overrides file."""
        return os.path.join(self.data_path, "exif_overrides.json")

    def exif_overrides_exists(self) -> bool:
        """Check if EXIF overrides file exists."""
        return self.io_handler.isfile(self._exif_overrides_file())

    def load_exif_overrides(self) -> Dict[str, Any]:
        """Load EXIF overrides data."""
        with self.io_handler.open_rt(self._exif_overrides_file()) as fin:
            return json.load(fin)

    def save_exif_overrides(self, exif_overrides: Dict[str, Any]) -> None:
        """Load EXIF overrides data."""
        with self.io_handler.open_wt(self._exif_overrides_file()) as fout:
            io.json_dump(exif_overrides, fout)

    def _rig_cameras_file(self) -> str:
        """Return path of rig models file"""
        return os.path.join(self.data_path, "rig_cameras.json")

    def load_rig_cameras(self) -> Dict[str, pymap.RigCamera]:
        """Return rig models data"""
        all_rig_cameras = rig.default_rig_cameras(self.load_camera_models())
        if not self.io_handler.exists(self._rig_cameras_file()):
            return all_rig_cameras
        with self.io_handler.open_rt(self._rig_cameras_file()) as fin:
            rig_cameras = io.rig_cameras_from_json(json.load(fin))
            for rig_camera_id, rig_camera in rig_cameras.items():
                all_rig_cameras[rig_camera_id] = rig_camera
        return all_rig_cameras

    def save_rig_cameras(self, rig_cameras: Dict[str, pymap.RigCamera]) -> None:
        """Save rig models data"""
        with self.io_handler.open_wt(self._rig_cameras_file()) as fout:
            io.json_dump(io.rig_cameras_to_json(rig_cameras), fout)

    def _rig_assignments_file(self) -> str:
        """Return path of rig assignments file"""
        return os.path.join(self.data_path, "rig_assignments.json")

    def load_rig_assignments(self) -> Dict[str, List[Tuple[str, str]]]:
        """Return rig assignments  data"""
        if not self.io_handler.exists(self._rig_assignments_file()):
            return {}
        with self.io_handler.open_rt(self._rig_assignments_file()) as fin:
            assignments = json.load(fin)

        # Backward compatibility.
        # Older versions of the file were stored as a list of instances without id.
        if isinstance(assignments, list):
            assignments = {str(i): v for i, v in enumerate(assignments)}

        return assignments

    def save_rig_assignments(
        self, rig_assignments: Dict[str, List[Tuple[str, str]]]
    ) -> None:
        """Save rig assignments  data"""
        with self.io_handler.open_wt(self._rig_assignments_file()) as fout:
            io.json_dump(rig_assignments, fout)

    def append_to_profile_log(self, content: str) -> None:
        """Append content to the profile.log file."""
        path = os.path.join(self.data_path, "profile.log")
        with self.io_handler.open_at(path) as fp:
            fp.write(content)

    def _report_path(self) -> str:
        return os.path.join(self.data_path, "reports")

    def load_report(self, path: str) -> str:
        """Load a report file as a string."""
        with self.io_handler.open_rt(os.path.join(self._report_path(), path)) as fin:
            return fin.read()

    def save_report(self, report_str: str, path: str) -> None:
        """Save report string to a file."""
        filepath = os.path.join(self._report_path(), path)
        self.io_handler.mkdir_p(os.path.dirname(filepath))
        with self.io_handler.open_wt(filepath) as fout:
            fout.write(report_str)

    def _ply_file(self, filename: Optional[str]) -> str:
        return os.path.join(self.data_path, filename or "reconstruction.ply")

    def save_ply(
        self,
        reconstruction: types.Reconstruction,
        tracks_manager: pymap.TracksManager,
        filename: Optional[str] = None,
        no_cameras: bool = False,
        no_points: bool = False,
        point_num_views: bool = False,
    ) -> None:
        """Save a reconstruction in PLY format."""
        ply = io.reconstruction_to_ply(
            reconstruction, tracks_manager, no_cameras, no_points, point_num_views
        )
        with self.io_handler.open_wt(self._ply_file(filename)) as fout:
            fout.write(ply)

    def _ground_control_points_file(self) -> str:
        return os.path.join(self.data_path, "ground_control_points.json")

    def _gcp_list_file(self) -> str:
        return os.path.join(self.data_path, "gcp_list.txt")

    def load_ground_control_points(self) -> List[pymap.GroundControlPoint]:
        """Load ground control points."""
        exif = {image: self.load_exif(image) for image in self.images()}

        cdn_enabled = self.config["proj_cdn_enabled"]
        grid_cache_dir = self.config["proj_grid_cache_dir"]

        gcp = []
        if self.io_handler.isfile(self._gcp_list_file()):
            with self.io_handler.open_rt(self._gcp_list_file()) as fin:
                gcp = io.read_gcp_list(fin, exif, cdn_enabled, grid_cache_dir)

        pcs = []
        if self.io_handler.isfile(self._ground_control_points_file()):
            with self.io_handler.open_rt(self._ground_control_points_file()) as fin:
                pcs, _ = io.read_ground_control_points(
                    fin, cdn_enabled, grid_cache_dir)

        return gcp + pcs

    def load_gcp_coordinate_system(self) -> Optional[str]:
        """Return the CRS string from ground_control_points.json or gcp_list.txt."""
        crs = None
        cdn_enabled = self.config["proj_cdn_enabled"]
        grid_cache_dir = self.config["proj_grid_cache_dir"]
        has_gcp_json = self.io_handler.isfile(
            self._ground_control_points_file())
        if has_gcp_json:
            with self.io_handler.open_rt(self._ground_control_points_file()) as fin:
                _, crs = io.read_ground_control_points(
                    fin, cdn_enabled, grid_cache_dir)

        has_gcp_txt = self.io_handler.isfile(self._gcp_list_file())
        if not crs and has_gcp_txt:
            with self.io_handler.open_rt(self._gcp_list_file()) as fin:
                proj = io.read_gcp_projection_string(fin)
                crs = proj

        if not has_gcp_json and not has_gcp_txt:
            return None

        # None means identity / WGS84
        result = crs if crs is not None else "WGS84"
        geo.log_vertical_datum(result)
        return result

    def output_coordinate_system(self) -> str:
        """CRS used for georeferenced products (LAS/LAZ, DSM/ortho, report).

        Centralizes the choice: the GCP coordinate system when it is projected,
        otherwise the UTM zone of the reconstruction reference (see
        ``geo.decide_output_crs``).  Returned as an EPSG/proj/WKT string usable by
        both pyproj and OSR.
        """
        return geo.decide_output_crs(
            self.load_gcp_coordinate_system(), self.load_reference()
        )

    def save_ground_control_points(
        self,
        points: List[pymap.GroundControlPoint],
    ) -> None:
        crs = self.load_gcp_coordinate_system() or "WGS84"
        cdn_enabled = self.config["proj_cdn_enabled"]
        grid_cache_dir = self.config["proj_grid_cache_dir"]
        with self.io_handler.open_wt(self._ground_control_points_file()) as fout:
            io.write_ground_control_points(
                points, fout, crs, cdn_enabled, grid_cache_dir)

    def _stats_file(self) -> str:
        return os.path.join(self.data_path, "stats", "stats.json")

    def load_stats(self) -> Dict[str, Any]:
        """Load statistics from stats/stats.json."""
        stats_file = self._stats_file()
        if self.io_handler.isfile(stats_file):
            with self.io_handler.open_rt(stats_file) as fin:
                return io.json_load(fin)
        return {}

    def save_stats(self, stats: Dict[str, Any]) -> None:
        """Save statistics to stats/stats.json."""
        stats_file = self._stats_file()
        self.io_handler.mkdir_p(os.path.dirname(stats_file))
        with self.io_handler.open_wt(stats_file) as fout:
            io.json_dump(stats, fout)

    def image_as_array(self, image: str) -> NDArray:
        logger.warning(
            "image_as_array() is deprecated. Use load_image() instead.")
        return self.load_image(image)

    def mask_as_array(self, image: str) -> Optional[NDArray]:
        logger.warning(
            "mask_as_array() is deprecated. Use load_mask() instead.")
        return self.load_mask(image)

    def subset(self, name: str, images_subset: List[str]) -> "DataSet":
        """Create a subset of this dataset by symlinking input data."""
        subset_dataset_path = os.path.join(self.data_path, name)
        self.io_handler.mkdir_p(subset_dataset_path)

        folders = ["images", "segmentations", "masks"]
        for folder in folders:
            self.io_handler.mkdir_p(os.path.join(subset_dataset_path, folder))
        subset_dataset = DataSet(subset_dataset_path, self.io_handler)

        files = []
        for method in [
            "_camera_models_file",
            "_config_file",
            "_camera_models_overrides_file",
            "_exif_overrides_file",
        ]:
            files.append(
                (
                    getattr(self, method)(),
                    getattr(subset_dataset, method)(),
                )
            )
        for image in images_subset:
            files.append(
                (
                    self._image_file(image),
                    os.path.join(subset_dataset_path, "images", image),
                )
            )
            files.append(
                (
                    self._segmentation_file(image),
                    os.path.join(subset_dataset_path,
                                 "segmentations", image + ".png"),
                )
            )
            if image in self.mask_files:
                files.append(
                    (
                        self.mask_files[image],
                        os.path.join(subset_dataset_path,
                                     "masks", image + ".png"),
                    )
                )

        for src, dst in files:
            if not self.io_handler.exists(src):
                continue
            self.io_handler.rm_if_exist(dst)
            self.io_handler.symlink(src, dst)

        return DataSet(subset_dataset_path, self.io_handler)

    def undistorted_dataset(self) -> "UndistortedDataSet":
        return UndistortedDataSet(
            self, os.path.join(self.data_path, "undistorted"), self.io_handler
        )


class UndistortedDataSet:
    """Accessors to the undistorted data of a dataset.

    Data include undistorted images, masks, and segmentation as well
    the undistorted reconstruction, tracks graph and computed depth maps.

    All data is stored inside the single folder ``undistorted_data_path``.
    By default, this path is set to the ``undistorted`` subfolder.
    """

    base: DataSetBase
    config: Dict[str, Any] = {}
    data_path: str
    io_handler: io.IoFilesystemBase = io.IoFilesystemDefault()

    def __init__(
        self,
        base_dataset: DataSetBase,
        undistorted_data_path: str,
        io_handler: Optional[io.IoFilesystemBase] = None,
    ) -> None:
        """Init dataset associated to a folder."""
        self.base = base_dataset
        self.config = self.base.config
        self.data_path = undistorted_data_path
        if io_handler is not None:
            self.io_handler = io_handler

    def load_undistorted_shot_ids(self) -> Dict[str, List[str]]:
        filename = os.path.join(self.data_path, "undistorted_shot_ids.json")
        with self.io_handler.open_rt(filename) as fin:
            return io.json_load(fin)

    def save_undistorted_shot_ids(self, ushot_dict: Dict[str, List[str]]) -> None:
        filename = os.path.join(self.data_path, "undistorted_shot_ids.json")
        self.io_handler.mkdir_p(self.data_path)
        with self.io_handler.open_wt(filename) as fout:
            io.json_dump(ushot_dict, fout, minify=False)

    def _undistorted_image_path(self) -> str:
        return os.path.join(self.data_path, "images")

    def _undistorted_image_file(self, image: str) -> str:
        """Path of undistorted version of an image."""
        p, ext = os.path.splitext(image)
        if ' ' in image:
            # NOTE: Downstream tools in ODM (ie OpenMVS) choke on spaces in image paths.
            image = p.replace(' ', '') + hashlib.md5(image.encode('utf8')).hexdigest() + ext
        return os.path.join(self._undistorted_image_path(), image)

    def load_undistorted_image(self, image: str) -> NDArray:
        """Load undistorted image pixels as a numpy array."""
        return self.io_handler.imread(self._undistorted_image_file(image))

    def save_undistorted_image(self, image: str, array: NDArray) -> None:
        """Save undistorted image pixels."""
        self.io_handler.mkdir_p(self._undistorted_image_path())
        self.io_handler.imwrite(self._undistorted_image_file(image), array)

    def undistorted_image_size(self, image: str) -> Tuple[int, int]:
        """Height and width of the undistorted image."""
        return self.io_handler.image_size(self._undistorted_image_file(image))

    def _undistorted_mask_path(self) -> str:
        return os.path.join(self.data_path, "masks")

    def _undistorted_mask_file(self, image: str) -> str:
        """Path of undistorted version of a mask."""
        return os.path.join(self._undistorted_mask_path(), image + ".png")

    def undistorted_mask_exists(self, image: str) -> bool:
        """Check if the undistorted mask file exists."""
        return self.io_handler.isfile(self._undistorted_mask_file(image))

    def load_undistorted_mask(self, image: str) -> NDArray:
        """Load undistorted mask pixels as a numpy array."""
        return self.io_handler.imread(
            self._undistorted_mask_file(image), grayscale=True
        )

    def save_undistorted_mask(self, image: str, array: NDArray) -> None:
        """Save the undistorted image mask."""
        self.io_handler.mkdir_p(self._undistorted_mask_path())
        self.io_handler.imwrite(self._undistorted_mask_file(image), array)

    def _undistorted_validity_mask_path(self) -> str:
        return os.path.join(self.data_path, "validity_masks")

    def _undistorted_validity_mask_file(self, image: str) -> str:
        """Path of undistorted validity mask for an image."""
        return os.path.join(
            self._undistorted_validity_mask_path(), image + ".png"
        )

    def undistorted_validity_mask_exists(self, image: str) -> bool:
        """Check if the undistorted validity mask file exists."""
        return self.io_handler.isfile(
            self._undistorted_validity_mask_file(image)
        )

    def load_undistorted_validity_mask(self, image: str) -> NDArray:
        """Load undistorted validity mask as a numpy array."""
        return self.io_handler.imread(
            self._undistorted_validity_mask_file(image), grayscale=True
        )

    def save_undistorted_validity_mask(
        self, image: str, array: NDArray
    ) -> None:
        """Save the undistorted validity mask."""
        self.io_handler.mkdir_p(self._undistorted_validity_mask_path())
        self.io_handler.imwrite(
            self._undistorted_validity_mask_file(image), array
        )

    def _undistorted_segmentation_path(self) -> str:
        return os.path.join(self.data_path, "segmentations")

    def _undistorted_segmentation_file(self, image: str) -> str:
        """Path of undistorted version of a segmentation."""
        return os.path.join(self._undistorted_segmentation_path(), image + ".png")

    def undistorted_segmentation_exists(self, image: str) -> bool:
        """Check if the undistorted segmentation file exists."""
        return self.io_handler.isfile(self._undistorted_segmentation_file(image))

    def load_undistorted_segmentation(self, image: str) -> NDArray:
        """Load an undistorted image segmentation."""
        segmentation_file = self._undistorted_segmentation_file(image)
        with self.io_handler.open_rb(segmentation_file) as fp:
            with PngImageFile(fp) as png_image:
                # TODO: We do not write a header tag in the metadata. Might be good safety check.
                data = np.array(png_image)
                if data.ndim == 2:
                    return data
                elif data.ndim == 3:
                    return data[:, :, 0]

                    # TODO we can optionally return also the instances and scores:
                    # instances = (
                    #     data[:, :, 1].astype(np.int16) + data[:, :, 2].astype(np.int16) * 256
                    # )
                    # scores = data[:, :, 3].astype(np.float32) / 256.0
                else:
                    raise IndexError

    def save_undistorted_segmentation(self, image: str, array: NDArray) -> None:
        """Save the undistorted image segmentation."""
        self.io_handler.mkdir_p(self._undistorted_segmentation_path())
        self.io_handler.imwrite(
            self._undistorted_segmentation_file(image), array)

    def load_undistorted_segmentation_mask(self, image: str) -> Optional[NDArray]:
        """Build a mask from the undistorted segmentation.

        The mask is non-zero only for pixels with segmentation
        labels not in undistorted_segmentation_ignore_values.

        If there are no undistorted_segmentation_ignore_values in the config,
        the segmentation_ignore_values are used instead.
        """
        ignore_values = self.base.undistorted_segmentation_ignore_values(image)
        if not ignore_values:
            return None

        segmentation = self.load_undistorted_segmentation(image)
        if segmentation is None:
            return None

        return masking.mask_from_segmentation(segmentation, ignore_values)

    def load_undistorted_combined_mask(self, image: str) -> Optional[NDArray]:
        """Combine undistorted binary mask with segmentation mask.

        Return a mask that is non-zero only where the binary
        mask and the segmentation mask are non-zero.
        """
        mask = None
        if self.undistorted_mask_exists(image):
            mask = self.load_undistorted_mask(image)
        smask = None
        if self.undistorted_segmentation_exists(image):
            smask = self.load_undistorted_segmentation_mask(image)
        return masking.combine_masks(mask, smask)

    def equalization_file(self) -> str:
        return os.path.join(self.data_path, "equalization.json")

    def equalization_exists(self) -> bool:
        return self.io_handler.isfile(self.equalization_file())

    def load_equalization(self) -> Dict[str, Any]:
        """Per-image radiometric corrections produced by ``dense_equalize``:
        ``{shot_id: {gain, vignette, pp, rmax, vignette_order}}``."""
        with self.io_handler.open_rt(self.equalization_file()) as fin:
            return io.json_load(fin)

    def save_equalization(self, equalization: Dict[str, Any]) -> None:
        with self.io_handler.open_wt(self.equalization_file()) as fout:
            io.json_dump(equalization, fout)

    def clusters_file(self) -> str:
        return os.path.join(self.data_path, "clusters.json")

    def clusters_points_file(self) -> str:
        return os.path.join(self.data_path, "clusters_points.json")

    def load_clusters(self) -> List[List[str]]:
        with self.io_handler.open_rt(self.clusters_file()) as fin:
            cluster_dict = io.json_load(fin)
        return [cluster_dict[ci] for ci in sorted(cluster_dict.keys())]

    def save_clusters(self, clusters: List[List[str]]) -> None:
        cluster_dict = {str(i): cluster for i, cluster in enumerate(clusters)}
        with self.io_handler.open_wt(self.clusters_file()) as fout:
            io.json_dump(cluster_dict, fout)

    def load_clusters_points(self) -> List[pysfm.SuperPoint]:
        with self.io_handler.open_rt(self.clusters_points_file()) as fin:
            clusters_points_list = io.json_load(fin)["clusters_points"]
            clusters_points = []
            for points_dict in clusters_points_list:
                superpoint = pysfm.SuperPoint()
                superpoint.coord = np.array(
                    points_dict["coord"], dtype=np.float32)
                superpoint.vis = points_dict["vis"]
                superpoint.tracks = points_dict["tracks"]
                clusters_points.append(superpoint)
        return clusters_points

    def save_clusters_points(self, cluster_points: List[pysfm.SuperPoint]) -> None:
        with self.io_handler.open_wt(self.clusters_points_file()) as fout:
            clusters_points_list = []
            for points in cluster_points:
                clusters_points_list.append({
                    "coord": points.coord.tolist(),
                    "vis": points.vis,
                    "tracks": points.tracks,
                })
            io.json_dump({"clusters_points": clusters_points_list}, fout)

    def _neighbors_best_file(self) -> str:
        return os.path.join(self.data_path, "neighbors_best.json")

    def _neighbors_all_file(self) -> str:
        return os.path.join(self.data_path, "neighbors_all.json")

    def _depth_ranges_file(self) -> str:
        return os.path.join(self.data_path, "depth_ranges.json")

    def neighbors_exist(self) -> bool:
        return self.io_handler.isfile(
            self._neighbors_best_file()
        ) and self.io_handler.isfile(self._neighbors_all_file())

    def save_neighbors_best(self, neighbors: Dict[str, List[str]]) -> None:
        with self.io_handler.open_wt(self._neighbors_best_file()) as fout:
            io.json_dump(neighbors, fout)

    def load_neighbors_best(self) -> Dict[str, List[str]]:
        with self.io_handler.open_rt(self._neighbors_best_file()) as fin:
            return io.json_load(fin)

    def save_neighbors_all(self, neighbors: Dict[str, List[str]]) -> None:
        with self.io_handler.open_wt(self._neighbors_all_file()) as fout:
            io.json_dump(neighbors, fout)

    def load_neighbors_all(self) -> Dict[str, List[str]]:
        with self.io_handler.open_rt(self._neighbors_all_file()) as fin:
            return io.json_load(fin)

    def depth_ranges_exist(self) -> bool:
        return self.io_handler.isfile(self._depth_ranges_file())

    def save_depth_ranges(
        self, depth_ranges: Dict[str, Tuple[float, float]]
    ) -> None:
        payload = {
            sid: [float(dr[0]), float(dr[1])]
            for sid, dr in depth_ranges.items()
        }
        with self.io_handler.open_wt(self._depth_ranges_file()) as fout:
            io.json_dump(payload, fout)

    def load_depth_ranges(self) -> Dict[str, Tuple[float, float]]:
        with self.io_handler.open_rt(self._depth_ranges_file()) as fin:
            payload = io.json_load(fin)
        return {sid: (float(v[0]), float(v[1])) for sid, v in payload.items()}

    def _cluster_bboxes_file(self) -> str:
        return os.path.join(self.data_path, "cluster_bboxes.json")

    def cluster_bboxes_exist(self) -> bool:
        return self.io_handler.isfile(self._cluster_bboxes_file())

    def save_cluster_bboxes(
        self, bboxes: List[Tuple[NDArray, NDArray]]
    ) -> None:
        payload = [
            {"min": np.asarray(mn).tolist(), "max": np.asarray(mx).tolist()}
            for mn, mx in bboxes
        ]
        with self.io_handler.open_wt(self._cluster_bboxes_file()) as fout:
            io.json_dump({"cluster_bboxes": payload}, fout)

    def load_cluster_bboxes(self) -> List[Tuple[NDArray, NDArray]]:
        with self.io_handler.open_rt(self._cluster_bboxes_file()) as fin:
            payload = io.json_load(fin)["cluster_bboxes"]
        return [
            (
                np.array(b["min"], dtype=np.float64),
                np.array(b["max"], dtype=np.float64),
            )
            for b in payload
        ]

    def crop_hull_file(self) -> str:
        return os.path.join(self.data_path, "dense_crop_hull.json")

    def crop_hull_exists(self) -> bool:
        return self.io_handler.isfile(self.crop_hull_file())

    def save_crop_hull(self, vertices: NDArray, percentile: float) -> None:
        """Persist the dense crop hull (CCW topocentric XY polygon).

        Saved by ``dense_clustering`` and consumed by ``dense_merging`` to clip
        the point cloud and mask the DSM/ortho to the surveyed footprint.
        """
        payload = {
            "frame": "topocentric",
            "percentile": float(percentile),
            "vertices": np.asarray(vertices, dtype=float).tolist(),
        }
        with self.io_handler.open_wt(self.crop_hull_file()) as fout:
            io.json_dump(payload, fout)

    def load_crop_hull(self) -> NDArray:
        with self.io_handler.open_rt(self.crop_hull_file()) as fin:
            payload = io.json_load(fin)
        return np.array(payload["vertices"], dtype=np.float64)

    def _depthmap_path(self) -> str:
        return os.path.join(self.data_path, "depthmaps")

    def depthmap_file(self, image: str, suffix: str) -> str:
        """Path to the depthmap file"""
        return os.path.join(self._depthmap_path(), image + "." + suffix)

    def point_cloud_file(self, filename: str = "merged.ply") -> str:
        return os.path.join(self._depthmap_path(), filename)

    def load_point_cloud(
        self, filename: str = "merged.ply"
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        with self.io_handler.open_rb(self.point_cloud_file(filename)) as fp:
            return io.point_cloud_from_ply(fp)

    def save_point_cloud(
        self,
        points: NDArray,
        normals: NDArray,
        colors: NDArray,
        labels: NDArray,
        filename: str = "merged.ply",
    ) -> None:
        self.io_handler.mkdir_p(self._depthmap_path())
        with self.io_handler.open_wb(self.point_cloud_file(filename)) as fp:
            io.point_cloud_to_ply(points, normals, colors, labels, fp)

    def mesh_file(self, filename: str = "mesh.ply") -> str:
        return os.path.join(self._depthmap_path(), filename)

    def load_mesh(
        self, filename: str = "mesh.ply"
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Load a dense triangle mesh PLY → (vertices, normals, colors, faces)."""
        with self.io_handler.open_rb(self.mesh_file(filename)) as fp:
            return io.load_mesh_from_ply(fp)

    def save_mesh(
        self,
        vertices: NDArray,
        normals: NDArray,
        colors: NDArray,
        faces: NDArray,
        filename: str = "mesh.ply",
    ) -> None:
        """Save a dense triangle mesh as a binary PLY (xyz+normal+rgb + faces)."""
        self.io_handler.mkdir_p(self._depthmap_path())
        with self.io_handler.open_wb(self.mesh_file(filename)) as fp:
            io.mesh_to_ply(vertices, normals, colors, faces, fp)

    def dsm_file(self) -> str:
        return os.path.join(self._depthmap_path(), "dsm.tif")

    def save_dsm(
        self,
        grid: NDArray,
        origin_x: float,
        origin_y: float,
        gsd: float,
        srs_wkt: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        """Save a DSM grid as a GeoTIFF (see ``geo.save_dsm_geotiff``).

        ``srs_wkt`` tags the raster's CRS; None leaves it untagged (topocentric).
        ``path`` defaults to the final ``dsm.tif`` (override for per-cluster
        debug tiles).
        """
        self.io_handler.mkdir_p(self._depthmap_path())
        geo.save_dsm_geotiff(
            path or self.dsm_file(), grid, origin_x, origin_y, gsd, srs_wkt
        )

    def load_dsm(self) -> Tuple[NDArray, Tuple[float, ...], str]:
        """Load DSM from GeoTIFF (see ``geo.load_dsm_geotiff``)."""
        return geo.load_dsm_geotiff(self.dsm_file())

    def ortho_file(self) -> str:
        return os.path.join(self._depthmap_path(), "ortho.tif")

    def save_ortho(
        self,
        image: NDArray,
        origin_x: float,
        origin_y: float,
        gsd: float,
        srs_wkt: Optional[str] = None,
        nodata_mask: Optional[NDArray] = None,
        path: Optional[str] = None,
    ) -> None:
        """Save an ortho image as a GeoTIFF (see ``geo.save_ortho_geotiff``).

        ``srs_wkt`` tags the raster's CRS; None leaves it untagged (topocentric).
        ``path`` defaults to the final ``ortho.tif`` (override for per-cluster
        tiles).
        """
        self.io_handler.mkdir_p(self._depthmap_path())
        geo.save_ortho_geotiff(
            path or self.ortho_file(), image, origin_x, origin_y, gsd,
            srs_wkt, nodata_mask=nodata_mask,
        )

    def save_dsm_ortho_streamed(
        self,
        gh: int,
        gw: int,
        origin_x: float,
        origin_y: float,
        gsd: float,
        fill_band: "Any",
        band_rows: int,
        reference: Optional["geo.TopocentricConverter"] = None,
        output_crs: Optional[str] = None,
    ) -> int:
        """Write dsm.tif + ortho.tif band-by-band, never holding the full grid.

        When ``output_crs`` is given (with ``reference``), the rasters are
        reprojected to that CRS, north-up (see
        ``geo.save_dsm_ortho_streamed_georeferenced``).  Otherwise the grid stays
        in topocentric coordinates with no CRS tag.  Returns the valid-cell count.
        """
        self.io_handler.mkdir_p(self._depthmap_path())
        if output_crs is not None and reference is not None:
            return geo.save_dsm_ortho_streamed_georeferenced(
                self.dsm_file(), self.ortho_file(), gh, gw, origin_x, origin_y,
                gsd, fill_band, band_rows, reference, output_crs,
                tmp_dsm_path=self.dsm_file() + ".topo.tif",
                tmp_ortho_path=self.ortho_file() + ".topo.tif",
            )
        return geo.save_dsm_ortho_streamed_geotiff(
            self.dsm_file(), self.ortho_file(), gh, gw, origin_x, origin_y,
            gsd, fill_band, band_rows, srs_wkt=None,
        )

    # ── Per-cluster DSM/ortho tiles (composited into the final raster) ──
    def dsm_ortho_batch_file(self, batch_num: int) -> str:
        return os.path.join(
            self._depthmap_path(), f"dsm_ortho_batch_{batch_num:04d}.npz"
        )

    def dsm_cluster_file(self, batch_num: int) -> str:
        """Path of a per-cluster debug DSM GeoTIFF (dsm_save_cluster_tiles)."""
        return os.path.join(
            self._depthmap_path(), f"dsm_cluster_{batch_num:04d}.tif"
        )

    def ortho_cluster_file(self, batch_num: int) -> str:
        """Path of a per-cluster debug ortho GeoTIFF (dsm_save_cluster_tiles)."""
        return os.path.join(
            self._depthmap_path(), f"ortho_cluster_{batch_num:04d}.tif"
        )

    def dsm_ortho_batch_exists(self, batch_num: int) -> bool:
        return self.io_handler.isfile(self.dsm_ortho_batch_file(batch_num))

    def list_batch_indices(self, prefix: str, suffix: str) -> List[int]:
        """Discover fusion batch indices by listing the depthmap folder.

        Returns the sorted indices ``i`` for every file named exactly
        ``{prefix}{i:04d}{suffix}`` (e.g. ``fused_batch_0003.ply``).  Names with
        anything between the number and suffix (``fused_batch_0003_debug.ply``)
        are ignored.  The fusion stage emits one batch per KD-tree chunk, so the
        merge derives the chunk count from disk rather than a persisted marker.
        """
        path = self._depthmap_path()
        if not self.io_handler.isdir(path):
            return []
        indices: List[int] = []
        for name in self.io_handler.ls(path):
            base = os.path.basename(name)
            if base.startswith(prefix) and base.endswith(suffix):
                middle = base[len(prefix): len(base) - len(suffix)]
                if middle.isdigit():
                    indices.append(int(middle))
        return sorted(indices)

    def save_dsm_ortho_batch(
        self,
        batch_num: int,
        dsm_grid: NDArray,
        ortho_grid: NDArray,
        origin_x: float,
        origin_y: float,
        gsd: float,
        base_offset: Tuple[int, int] = (0, 0),
        global_shape: Optional[Tuple[int, int]] = None,
        confidence: Optional[NDArray] = None,
    ) -> None:
        """Save one cluster's finished DSM+ortho as a compact tile.

        The passed grids may already be a territory-sized WINDOW into the shared
        global grid (re-fitted per cluster to bound RAM).  We crop them further
        to the tight bounding box of valid (non-NaN) DSM cells and store the
        row/col offset *within the global grid* — ``base_offset`` is the window's
        own offset, added to the local crop.  The tiles are composited back by
        max-z in ``dense.merge.merge_dsm_ortho_batches``.  Writes nothing for an
        empty footprint.

        Args:
            batch_num: cluster index (matches the ``fused_batch`` numbering).
            dsm_grid: (H, W) float32, NaN = no surface (window or full extent).
            ortho_grid: (H, W, 3) uint8 (same window as ``dsm_grid``).
            origin_x/origin_y/gsd: GLOBAL grid georeference (identical across
                clusters); stored so the merge can write the final GeoTIFF.
            base_offset: (row, col) offset of the passed window within the global
                grid.  ``(0, 0)`` when the grid already spans the full extent.
            global_shape: (H, W) of the global grid; defaults to the passed
                grid's shape (full-extent case).
            confidence: optional (H, W) bool/uint8, True where the DSM cell is a
                REAL reconstruction (vs a hole-fill).  Stored (cropped identically
                to the DSM) so the merge can favour real heights over interpolated
                ones in tile overlaps.  Omitted → the merge treats the tile as all
                real (legacy behaviour).
        """
        self.io_handler.mkdir_p(self._depthmap_path())
        valid = ~np.isnan(dsm_grid)
        if not valid.any():
            return
        rows = np.where(valid.any(axis=1))[0]
        cols = np.where(valid.any(axis=0))[0]
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        c0, c1 = int(cols[0]), int(cols[-1]) + 1
        dsm_win = np.ascontiguousarray(
            dsm_grid[r0:r1, c0:c1], dtype=np.float32)
        ortho_win = np.ascontiguousarray(
            ortho_grid[r0:r1, c0:c1], dtype=np.uint8
        )
        base_r, base_c = base_offset
        gshape = (
            global_shape if global_shape is not None
            else (int(dsm_grid.shape[0]), int(dsm_grid.shape[1]))
        )
        arrays = dict(
            dsm=dsm_win,
            ortho=ortho_win,
            offset=np.array([base_r + r0, base_c + c0], dtype=np.int64),
            global_shape=np.array(gshape, dtype=np.int64),
            geo=np.array([origin_x, origin_y, gsd], dtype=np.float64),
        )
        if confidence is not None:
            arrays["conf"] = np.ascontiguousarray(
                confidence[r0:r1, c0:c1], dtype=np.uint8
            )
        buf = BytesIO()
        np.savez(buf, **arrays)
        with self.io_handler.open_wb(self.dsm_ortho_batch_file(batch_num)) as f:
            f.write(lz4_frame.compress(buf.getvalue()))

    def load_dsm_ortho_batch(
        self, batch_num: int
    ) -> Tuple[NDArray, NDArray, int, int, Tuple[int, int],
               Tuple[float, float, float], Optional[NDArray]]:
        """Load a DSM/ortho tile.

        Returns ``(dsm_win, ortho_win, row0, col0, (global_h, global_w),
        (origin_x, origin_y, gsd), conf)`` where ``conf`` is the (H, W) uint8
        real-reconstruction mask (1 = real, 0 = hole-fill), or ``None`` for a
        legacy tile saved without it.
        """
        with self.io_handler.open_rb(
            self.dsm_ortho_batch_file(batch_num)
        ) as f:
            raw = lz4_frame.decompress(f.read())
        o = np.load(BytesIO(raw))
        dsm = o["dsm"]
        ortho = o["ortho"]
        r0, c0 = int(o["offset"][0]), int(o["offset"][1])
        gh, gw = int(o["global_shape"][0]), int(o["global_shape"][1])
        ox, oy, gsd = (float(o["geo"][0]), float(o["geo"][1]),
                       float(o["geo"][2]))
        conf = o["conf"] if "conf" in o.files else None
        o.close()
        dsm = np.where(
            np.isfinite(dsm) & (dsm != geo.DSM_NODATA), dsm, np.nan
        ).astype(np.float32)
        return dsm, ortho, r0, c0, (gh, gw), (ox, oy, gsd), conf

    def raw_depthmap_exists(self, image: str) -> bool:
        return self.io_handler.isfile(self.depthmap_file(image, "raw.npz"))

    def save_raw_depthmap(
        self,
        image: str,
        depth: NDArray,
        plane: NDArray,
        score: NDArray,
        nghbr: NDArray,
        nghbrs: List[str],
        confidence: Optional[NDArray] = None,
    ) -> None:
        self.io_handler.mkdir_p(self._depthmap_path())
        self._write_raw_depthmap(image, depth, plane, score, nghbr, nghbrs,
                                 confidence)

    def _write_raw_depthmap(
        self,
        image: str,
        depth: NDArray,
        plane: NDArray,
        score: NDArray,
        nghbr: NDArray,
        nghbrs: List[str],
        confidence: Optional[NDArray] = None,
    ) -> None:
        filepath = self.depthmap_file(image, "raw.npz")
        buf = BytesIO()
        # ``score`` (PatchMatch cost), ``nghbr`` and ``nghbrs`` are accepted
        # for call-site compatibility but not persisted: nothing ever reads
        # them back (cleaning consumes only depth, plane/normal, confidence).
        arrays = dict(depth=depth, plane=plane)
        if confidence is not None:
            arrays["confidence"] = confidence
        np.savez(buf, **arrays)
        with self.io_handler.open_wb(filepath) as f:
            f.write(lz4_frame.compress(buf.getvalue()))

    def load_raw_depthmap(
        self, image: str
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, List[str]]:
        with self.io_handler.open_rb(self.depthmap_file(image, "raw.npz")) as f:
            raw = lz4_frame.decompress(f.read())
        o = np.load(BytesIO(raw))
        depth = o["depth"]
        plane = o["plane"]
        # score/nghbr/nghbrs are legacy fields no longer written; tolerate
        # both old files (present) and new ones (absent).
        score = o["score"] if "score" in o else None
        nghbr = o["nghbr"] if "nghbr" in o else None
        nghbrs = o["nghbrs"] if "nghbrs" in o else None
        confidence = o["confidence"] if "confidence" in o else None
        o.close()
        return depth, plane, score, nghbr, nghbrs, confidence

    def clean_depthmap_exists(self, image: str) -> bool:
        return self.io_handler.isfile(self.depthmap_file(image, "clean.npz"))

    def save_clean_depthmap(
        self, image: str, depth: NDArray, plane: NDArray, score: NDArray,
        confidence: Optional[NDArray] = None,
    ) -> None:
        self.io_handler.mkdir_p(self._depthmap_path())
        self._write_clean_depthmap(image, depth, plane, score, confidence)

    def _write_clean_depthmap(
        self, image: str, depth: NDArray, plane: NDArray, score: NDArray,
        confidence: Optional[NDArray] = None,
    ) -> None:
        filepath = self.depthmap_file(image, "clean.npz")
        buf = BytesIO()
        arrays = dict(depth=depth, plane=plane, score=score)
        if confidence is not None:
            arrays["confidence"] = confidence
        np.savez(buf, **arrays)
        with self.io_handler.open_wb(filepath) as f:
            f.write(lz4_frame.compress(buf.getvalue()))

    def load_clean_depthmap(
        self, image: str
    ) -> Tuple[NDArray, NDArray, NDArray, Optional[NDArray]]:
        with self.io_handler.open_rb(self.depthmap_file(image, "clean.npz")) as f:
            raw = lz4_frame.decompress(f.read())
        o = np.load(BytesIO(raw))
        depth = o["depth"]
        plane = o["plane"]
        score = o["score"]
        confidence = o["confidence"] if "confidence" in o else None
        o.close()
        return depth, plane, score, confidence

    def save_raw_depthmaps_parallel(
        self,
        items: List[
            Tuple[str, NDArray, NDArray, NDArray, NDArray, List[str],
                  Optional[NDArray]]
        ],
    ) -> None:
        """Save multiple raw depthmaps in parallel.

        Each item is (image, depth, plane, score, nghbr, nghbrs, confidence).
        LZ4 compression and NPZ serialization run in a thread pool.
        """
        self.io_handler.mkdir_p(self._depthmap_path())

        def _save_one(
            item: Tuple[str, NDArray, NDArray, NDArray, NDArray, List[str],
                        Optional[NDArray]],
        ) -> None:
            self._write_raw_depthmap(*item)

        with ThreadPoolExecutor(max_workers=self.config["io_processes"]) as pool:
            list(pool.map(_save_one, items))

    def load_raw_depthmaps_parallel(
        self,
        images: List[str],
    ) -> List[Tuple[NDArray, NDArray, NDArray, NDArray, List[str],
                    Optional[NDArray]]]:
        """Load multiple raw depthmaps in parallel."""

        def _load_one(
            image: str,
        ) -> Tuple[NDArray, NDArray, NDArray, NDArray, List[str],
                   Optional[NDArray]]:
            return self.load_raw_depthmap(image)

        with ThreadPoolExecutor(max_workers=self.config["io_processes"]) as pool:
            return list(pool.map(_load_one, images))

    def save_clean_depthmaps_parallel(
        self,
        items: List[Tuple[str, NDArray, NDArray, NDArray, Optional[NDArray]]],
    ) -> None:
        """Save multiple clean depthmaps in parallel.

        Each item is (image, depth, plane, score, confidence).
        """
        self.io_handler.mkdir_p(self._depthmap_path())

        def _save_one(
            item: Tuple[str, NDArray, NDArray, NDArray, Optional[NDArray]],
        ) -> None:
            self._write_clean_depthmap(*item)

        with ThreadPoolExecutor(max_workers=self.config["io_processes"]) as pool:
            list(pool.map(_save_one, items))

    def load_clean_depthmaps_parallel(
        self,
        images: List[str],
    ) -> List[Tuple[NDArray, NDArray, NDArray, Optional[NDArray]]]:
        """Load multiple clean depthmaps in parallel."""

        def _load_one(
            image: str,
        ) -> Tuple[NDArray, NDArray, NDArray, Optional[NDArray]]:
            return self.load_clean_depthmap(image)

        with ThreadPoolExecutor(max_workers=self.config["io_processes"]) as pool:
            return list(pool.map(_load_one, images))

    def load_undistorted_images_parallel(
        self,
        images: List[str],
    ) -> Dict[str, NDArray]:
        """Load multiple undistorted images in parallel.

        Returns a dict mapping image id → pixel array.
        """
        if not images:
            return {}

        def _load_one(image: str) -> Tuple[str, NDArray]:
            return image, self.load_undistorted_image(image)

        with ThreadPoolExecutor(max_workers=self.config["io_processes"]) as pool:
            return dict(pool.map(_load_one, images))

    def load_undistorted_validity_masks_parallel(
        self,
        images: List[str],
    ) -> Dict[str, NDArray]:
        """Load validity masks for multiple images in parallel.

        Returns a dict mapping image id → mask array, only for images
        that have a validity mask on disk.
        """
        existing = [
            img for img in images
            if self.undistorted_validity_mask_exists(img)
        ]
        if not existing:
            return {}

        def _load_one(image: str) -> Tuple[str, NDArray]:
            return image, self.load_undistorted_validity_mask(image)

        with ThreadPoolExecutor(max_workers=self.config["io_processes"]) as pool:
            return dict(pool.map(_load_one, existing))

    # ── Fusion progress checkpointing ─────────────────────────────────

    def _fusion_progress_file(self) -> str:
        return os.path.join(self._depthmap_path(), "fusion_progress.json")

    def save_fusion_progress(
        self, fused_shot_ids: set, fused_batches: List[int]
    ) -> None:
        """Persist which shots have been fused and which batch PLYs exist."""
        self.io_handler.mkdir_p(self._depthmap_path())
        payload = {
            "fused_shot_ids": sorted(fused_shot_ids),
            "fused_batches": sorted(fused_batches),
        }
        with self.io_handler.open_wt(self._fusion_progress_file()) as f:
            json.dump(payload, f, indent=2)

    def load_fusion_progress(self) -> Tuple[set, List[int]]:
        """Load fusion checkpoint.  Returns ``(fused_shot_ids, fused_batches)``.

        If no checkpoint exists, returns empty containers.
        """
        path = self._fusion_progress_file()
        if not self.io_handler.isfile(path):
            return set(), []
        with self.io_handler.open_rt(path) as f:
            data = json.load(f)
        return set(data.get("fused_shot_ids", [])), list(
            data.get("fused_batches", [])
        )

    def load_undistorted_tracks_manager(self) -> pymap.TracksManager:
        filename = os.path.join(self.data_path, "tracks.csv")
        if isinstance(self.io_handler, io.IoFilesystemDefault):
            return pymap.TracksManager.instanciate_from_file(filename)
        with self.io_handler.open_rt(filename) as f:
            return pymap.TracksManager.instanciate_from_string(f.read())

    def save_undistorted_tracks_manager(
        self, tracks_manager: pymap.TracksManager
    ) -> None:
        filename = os.path.join(self.data_path, "tracks.csv")
        if isinstance(self.io_handler, io.IoFilesystemDefault):
            tracks_manager.write_to_file(filename)
            return
        with self.io_handler.open_wt(filename) as fw:
            fw.write(tracks_manager.as_string())

    def load_undistorted_reconstruction(self) -> List[types.Reconstruction]:
        filename = os.path.join(self.data_path, "reconstruction.json")
        with self.io_handler.open_rt(filename) as fin:
            return io.reconstructions_from_json(io.json_load(fin))

    def save_undistorted_reconstruction(
        self, reconstruction: List[types.Reconstruction]
    ) -> None:
        filename = os.path.join(self.data_path, "reconstruction.json")
        self.io_handler.mkdir_p(self.data_path)
        with self.io_handler.open_wt(filename) as fout:
            io.json_dump(io.reconstructions_to_json(
                reconstruction), fout, minify=True)


def invent_reference_from_gps_and_gcp(
    data: DataSetBase, images: Optional[List[str]] = None
) -> geo.TopocentricConverter:
    """Invent the reference from the weighted average of lat/lon measurements.
    Most of the time the altitude provided in the metadata is inaccurate, thus
    the reference altitude is set equal to 0 regardless of the altitude measurements.
    """
    lat, lon = 0.0, 0.0
    wlat, wlon = 0.0, 0.0
    if images is None:
        images = data.images()
    for image in images:
        d = data.load_exif(image)
        if "gps" in d and "latitude" in d["gps"] and "longitude" in d["gps"]:
            w = 1.0 / max(0.01, d["gps"].get("dop", 15))
            lat += w * d["gps"]["latitude"]
            lon += w * d["gps"]["longitude"]
            wlat += w
            wlon += w

    if not wlat and not wlon:
        for gcp in data.load_ground_control_points():
            if gcp.lla:
                lat += gcp.lla["latitude"]
                lon += gcp.lla["longitude"]
                wlat += 1.0
                wlon += 1.0

    if wlat:
        lat /= wlat
    if wlon:
        lon /= wlon

    return geo.TopocentricConverter(lat, lon, 0)
