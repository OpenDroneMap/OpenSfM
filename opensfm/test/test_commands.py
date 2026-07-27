# pyre-strict
import argparse
from os.path import join
from types import ModuleType
from typing import Any, List

import pytest

from opensfm import commands, dataset, reconstruction
from opensfm.test import data_generation, utils


def run_command(command: ModuleType, args: List[str]) -> None:
    parser = argparse.ArgumentParser()
    command.add_arguments(parser)
    parsed_args = parser.parse_args(args)
    command.run(dataset.DataSet(parsed_args.dataset), parsed_args)


# If that one is too slow : uncomment
# @pytest.mark.slow
def test_run_all(tmpdir: Any) -> None:
    data = data_generation.create_berlin_test_folder(tmpdir)
    data_generation.save_config({"matcher_type": "FLANN"}, data.data_path)
    run_all_commands = [
        commands.extract_metadata,
        commands.detect_features,
        commands.match_features,
        commands.create_tracks,
        commands.reconstruct,
        commands.bundle,
        commands.reconstruct_from_prior,
        commands.mesh,
        commands.undistort,
        commands.dense_clustering,
        commands.compute_depthmaps,
        commands.fuse_depthmaps,
        commands.dense_merging,
        commands.export_ply,
        commands.export_visualsfm,
        commands.export_openmvs,
        commands.export_pmvs,
        commands.export_bundler,
        commands.export_colmap,
        commands.compute_statistics,
        commands.export_report,
        commands.export_geocoords,
    ]

    output_rec_path = join(data.data_path, "rec_prior.json")
    command_options = {
        commands.reconstruct_from_prior: [
            "--input",
            join(data.data_path, "reconstruction.json"),
            "--output",
            output_rec_path,
        ],
        commands.export_geocoords: [
            "--proj",
            "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs",
            "--offset-x",
            "100",
            "--offset-y",
            "200",
            "--mode",
            "projected",
            "--reconstruction"
        ]
    }

    for module in run_all_commands:
        command = module.Command()
        options = command_options.get(module, [])
        run_command(command, [data.data_path] + options)

    check_reconstruction(data)
    check_prior(data, output_rec_path)
    check_export_geocoords(data)


def check_reconstruction(data: dataset.DataSet) -> None:
    reconstruction = data.load_reconstruction()
    assert len(reconstruction[0].shots) == 3
    assert len(reconstruction[0].points) > 1000


def check_prior(data: dataset.DataSet, output_rec_path: str) -> None:
    reconstruction = data.load_reconstruction()  # load old reconstruction
    prior_rec = data.load_reconstruction(output_rec_path)
    for shot_id, shot in reconstruction[0].shots.items():
        utils.assert_shots_equal(shot, prior_rec[0].shots[shot_id])

    assert len(prior_rec[0].points) > 1000


def check_export_geocoords(data: dataset.DataSet) -> None:
    import os
    geocoords_path = join(data.data_path, "reconstruction.geocoords.json")
    assert os.path.exists(geocoords_path)
    
    # Load and check if points/shots exist in the exported file
    geocoords_rec = data.load_reconstruction("reconstruction.geocoords.json")
    assert len(geocoords_rec) > 0
    assert len(geocoords_rec[0].shots) == 3
    assert len(geocoords_rec[0].points) > 1000


def test_command_reconstruct_config(tmpdir: Any, monkeypatch: Any) -> None:
    """
    Mocks Reconstruct command algorithm setting from config.py::reconstruction_algorithm
    """
    data = data_generation.create_berlin_test_folder(tmpdir)
    # Set the config to use triangulation
    data_generation.save_config({"reconstruction_algorithm": "triangulation"}, data.data_path)
    
    called_algorithm = None

    def mock_triangulation(dataset: Any, tracks: Any) -> Any:
        nonlocal called_algorithm
        called_algorithm = "triangulation"
        return {}, []

    def mock_incremental(dataset: Any, tracks: Any) -> Any:
        nonlocal called_algorithm
        called_algorithm = "incremental"
        return {}, []

    # Mock the actual algorithms so it doesn't take forever to run
    monkeypatch.setattr(reconstruction, "triangulation_reconstruction", mock_triangulation)
    monkeypatch.setattr(reconstruction, "incremental_reconstruction", mock_incremental)
    monkeypatch.setattr(dataset.DataSet, "load_tracks_manager", lambda self, *args, **kwargs: None)

    # Run the command with no --algorithm flag, it should fall back to config
    command = commands.reconstruct.Command()
    run_command(command, [data.data_path])
    
    assert called_algorithm == "triangulation"
