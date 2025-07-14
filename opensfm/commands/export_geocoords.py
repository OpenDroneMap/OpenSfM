from opensfm.actions import export_geocoords

from . import command
import argparse
from opensfm.dataset import DataSet


class Command(command.CommandBase):
    name = "export_geocoords"
    help = "Export reconstructions in geographic coordinates"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        export_geocoords.run_dataset(
            dataset,
            args.proj,
            args.transformation,
            args.image_positions,
            args.reconstruction,
            args.dense,
            args.output,
            (args.offset_x, args.offset_y),
            args.mode
        )

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--proj", help="PROJ.4 projection string", required=True)
        parser.add_argument(
            "--transformation",
            help="Print cooordinate transformation matrix",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--image-positions",
            help="Export image positions",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--reconstruction",
            help="Export reconstruction.json",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--dense",
            help="Export dense point cloud (depthmaps/merged.ply)",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--output", help="Path of the output file relative to the dataset"
        )
        parser.add_argument(
            "--offset-x", type=float, help="Value to add to the final translation X axis", default=0.0
        )
        parser.add_argument(
            "--offset-y", type=float, help="Value to add to the final translation Y axis", default=0.0
        )
        parser.add_argument(
            "--mode",
            help="Georeferencing method",
            type=str,
            choices=["affine", "projected"],
            default="affine",
        )
